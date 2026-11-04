"""Opt-in source-independent graph products using the existing checked RHS owner.

No production default changes. The experiment clones the graph entry with two
explicit AST connection points: group discovery before builder creation and a
group dispatch before ordinary product dispatch. Parsing, numeric recognition,
ABI construction, source task attribution and terminal encoding stay shared.
Only graphs of zero-seeded independent rank-two i8 products with ABI-input
operands and distinct returned i32 outputs admit reordering. No attention names
or geometry tables participate in admission.
"""
from contextlib import contextmanager
from pathlib import Path
import ast
import hashlib
import inspect
import json


def group_tasks(tasks, geometry, layouts, fused_readout, skipped, block, returned,
                source_identity, *, max_members=8):
    """Source independence permits moving whole tasks, including initializers."""
    import gemmini_graph as graph
    if not 2 <= max_members <= 8:
        raise ValueError('bounded shared RHS groups require 2..8 members')
    from merlin.compile.tuning_knobs import knob
    package=Path(__file__).resolve().parent
    if knob('dense_schedule',default='compact',values=('compact','panel_drained','panel_epochs','ready_readout','segmented_ready','auto','auto_ready','array_epochs'),root=package) not in ('auto','auto_ready'):
        return tasks,{}
    if knob('gemv_column_blocks',default='off',values=('off',16,32,64),root=package)!='off':return tasks,{}
    records=[]
    for kind,owned in tasks:
        roots=[op for op in owned if geometry.get(op) is not None]
        if kind!='device' or len(roots)!=1:return tasks,{}
        root=roots[0]
        if (root in layouts or root in fused_readout or not graph._zero(root.operands[-1])
                or any(op is not root and op not in skipped for op in owned)
                or any(v not in block.args for v in root.operands[:2])
                or root.results[0] not in returned.operands
                or str(root.results[0].type.element_type)!='i32'):
            return tasks,{}
        records.append((root,owned))
    if len({root.results[0] for root,_ in records})!=len(records):return tasks,{}
    # All reads are original immutable ABI inputs and all writes are distinct
    # fresh tensor results. No crossed operation can read a group live-out.
    buckets={}
    for root,owned in records:
        buckets.setdefault((root.operands[1],geometry[root]),[]).append((root,owned))
    groups={};new_tasks=[]
    for bucket in buckets.values():
        for start in range(0,len(bucket),max_members):
            chunk=bucket[start:start+max_members]
            if len(chunk)<2:
                new_tasks.extend(('device',owned) for _,owned in chunk);continue
            root=chunk[0][0];m,k,n=geometry[root]
            # Existing owner reserves half of operand storage for complete RHS.
            from gemmini_dense import geometry as target_geometry
            edge,operand,operand_rows,_,_=target_geometry()
            if ((k+edge-1)//edge)*((n+edge-1)//edge)*edge>operand_rows-operand_rows//2:
                new_tasks.extend(('device',owned) for _,owned in chunk);continue
            groups[root]=dict(roots=[r for r,_ in chunk],source_identity=source_identity,
                source_geometry=(m,k,n),source_independence='ABI-input reads; distinct returned fresh tensor writes')
            new_tasks.append(('device',[op for _,owned in chunk for op in owned]))
    return (new_tasks,groups) if groups else (tasks,{})


def emit_group(fb,group,ptrs,names,boundary_name,cb):
    from gemmini_dense import emit_resident_group
    from gemmini_graph import _emit_dot
    from gemmini_encode import fence
    roots=group['roots'];m,k,n=group['source_geometry']
    receipt=emit_resident_group(fb,[ptrs[r.operands[0]] for r in roots],
        ptrs[roots[0].operands[1]],[ptrs[r.results[0]] for r in roots],m,k,n,
        attrs={'output_dtype':'i32'},source_identity=group['source_identity'])
    if receipt is None:
        # Library refuses before touching the builder. Preserve each original
        # product and original encoder, now under the merged source task.
        for root in roots:
            _emit_dot(fb,ptrs[root.operands[0]],ptrs[root.operands[1]],ptrs[root.results[0]],
                (m,k,n),None,output_dtype='i32')
            fence(fb)
        receipt={'fallback':'existing shared RHS owner refused before emission'}
    receipt['source_independence']=group['source_independence']
    cb['params'].setdefault('attention_schedule_experiment',[]).append(receipt)
    return list(dict.fromkeys(boundary_name(v) for r in roots for v in r.operands[:2])),[names[r.results[0]] for r in roots]


def connected_compile(original, *, max_members=8):
    """Build a guarded opt-in graph connection without editing the shared owner."""
    tree=ast.parse(inspect.getsource(original));function=tree.body[0]
    count=0
    for i,node in enumerate(function.body):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fb' for t in node.targets):
            addition=ast.parse('tasks, attention_groups = _attention_group_tasks(tasks, geometry, layouts, fused_readout, skipped, block, returned, inventory["source_sha256"])').body
            function.body[i:i]=addition;count+=1;break
    for node in ast.walk(function):
        if isinstance(node,ast.If) and ast.unparse(node.test)=='root in reuses':
            old=ast.If(test=node.test,body=node.body,orelse=node.orelse)
            node.test=ast.parse('root in attention_groups',mode='eval').body
            node.body=ast.parse('reads, writes = _attention_emit_group(fb, attention_groups[root], ptrs, names, boundary_name, cb)').body
            node.orelse=[old];count+=1;break
    if count!=2:raise ValueError('graph experiment connection points changed; safe refusal')
    namespace=dict(original.__globals__)
    namespace.update(_attention_group_tasks=lambda *args:group_tasks(*args,max_members=max_members),_attention_emit_group=emit_group)
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename,'exec'),namespace)
    return namespace[original.__name__]


@contextmanager
def install(records, *, max_members=8, capture=None):
    import gemmini_graph as graph
    import merlin.compile.scheduling.dense_shared_rhs as shared
    from gemmini_dense_pipeline import ordering_premises
    from gemmini_dense import FACTS_PATH
    from merlin.compile.scheduling.dense_panel_pipeline import DenseRetainedReductionAdmission
    from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw
    original=graph.compile_linalg;scheduler=shared.schedule_dense_shared_rhs
    def schedule(*args,**kwargs):
        adapter=args[3]
        flags,pending,ordering=ordering_premises(FACTS_PATH)
        admission=DenseRetainedReductionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
            'selected-command-ordering:'+ordering['ordering_sha256']+':gemmini-encoder-v1',
            4096,4096,pending['execute']//2,pending['load'],max_issued_readouts=64,
            pending_readouts=pending['store'],readout_configuration='ordered_snapshot_before_store_issue')
        law=DensePanelEpochLaw(adapter.configuration_identity,adapter.numerical_policy_identity,
            'selected-command-ordering:'+ordering['ordering_sha256'],**flags)
        refusal=None
        try:result=scheduler(*args,**kwargs,epoch_admission=admission,epoch_law=law)
        except ValueError as error:
            # Library checks precede builder emission, so refusal is reversible.
            result=None;refusal=str(error)
        records.append(dict(source_identity=kwargs['source_identity'],members=len(args[0]),
            problem=args[0][0].__dict__,checked_kernel_sha256=None if result is None else result.kernel.digest(),
            ordering=ordering,library='schedule_dense_shared_rhs',epoch_law=law.__dict__,
            terminal='existing emit_resident_group -> dense_terminal',refusal=refusal,timing='UNKNOWN'))
        if capture is not None and result is not None:capture(result,args,kwargs)
        return result
    graph.compile_linalg=connected_compile(original,max_members=max_members)
    shared.schedule_dense_shared_rhs=schedule
    try:yield
    finally:graph.compile_linalg=original;shared.schedule_dense_shared_rhs=scheduler


def main():
    import runpy,sys
    output=next((x.split('=',1)[1] for x in sys.argv if x.startswith('--attention-schedule-report=')),None)
    sys.argv=[x for x in sys.argv if not x.startswith('--attention-schedule-report=')]
    records=[]
    with install(records):runpy.run_path(str(Path(__file__).resolve().parent/'gemmini-opt-original'),run_name='__main__')
    if output:Path(output).write_text(json.dumps(records,indent=2)+'\n')
