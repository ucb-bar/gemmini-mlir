"""Original-pointer row batching: source-only proof, no prepared storage ABI.

The constructive source owner proves only independent integer row algebra. Its
concat candidate is never lowered. Projected dense effects are split into an
explicit bijection onto original rows; all external pointers are original ABI
arguments. Unknown source effects, shapes, exits or ownership refuse.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('original_row_batching',default='off',values=('off','separate_rows'),root=Path(__file__).parent)


def _guard(cb):
    return json.dumps({k:cb[k] for k in ('tensors','commands','kernel_abi')}|{'source':{k:v for k,v in cb['params'].items() if k in ('tensor_source_map','source_sha256','source_task_definitions','source_op_count')}},sort_keys=True)


def _group_guard(groups):
    return json.dumps([{k:v for k,v in g.items() if k!='schedule'}|{'kernel':g['schedule'].kernel.digest()} for g in groups],sort_keys=True)


@dataclass
class OriginalRows:
    source_owner: object
    cb: dict
    guard: str
    groups: tuple
    group_guard: str
    def revalidate(self):
        self.source_owner.revalidate()
        if _group_guard(self.groups)!=self.group_guard:raise ValueError('original row lane binding or schedule changed')
        if _guard(self.cb)!=self.guard:raise ValueError('original row source/pointer/ownership graph changed')
        for g in self.groups:g['schedule'].revalidate()
        return self
    def record(self):
        self.revalidate()
        return {'schema':'gemmini.original_row_batching.v1','source_sha256':self.cb['params']['source_sha256'],
            'groups':[{'lhs':g['lhs'],'rhs':g['rhs'],'outputs':g['outputs'],'commits':g['commits'],'shape':[len(g['lhs']),g['k'],g['n']],
            'kernel_sha256':g['schedule'].kernel.digest()} for g in self.groups],
            'additional_tensors':0,'additional_storage_bytes':0,'source_candidate_emitted':False,
            'ownership':'complete original pure source region; original SSA and ABI bindings retained'}


def prepare(source,cb,*,max_members=8):
    """Return original tensors/commands with one checked region task and schedules.

    Task ranges must be coalesced: a physical M-group compute simultaneously
    serves several source results. Exact original source identities, commands,
    tensor bindings and public writes are retained; no preparation plan is used.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.contract.linalg_iface import is_linalg_on_tensors
    from merlin.transforms.input_row_packing import plan_shared_rhs_grouping
    from xdsl.dialects.func import FuncOp
    # The optimization is a partial adapter. Check the declared grammar and
    # constructive source opportunity before asserting its exact lowering join.
    # Unsupported source routes retain their ordinary compiler behavior.
    if not is_linalg_on_tensors(source):return None
    module=parse_mlir_text(source)
    owners=[op for op in module.body.block.ops if type(op) is FuncOp and op.sym_name.data=='forward' and len(op.body.blocks)==1]
    if len(owners)!=1:return None
    proof=plan_shared_rhs_grouping(module,max_members=max_members,max_rows=max_members)
    if proof is None:return None
    proof.revalidate()
    from linalg_lowering import lower_linalg
    from source_attribution import prepare_source_bindings
    original=lower_linalg(source);prepare_source_bindings(source,original)
    if (any(cb.get(key)!=original.get(key) for key in ('tensors','commands','kernel_abi'))
            or cb['params'].get('tensor_source_map')!=original['params']['tensor_source_map']
            or cb['params'].get('source_op_count')!=original['params']['source_op_count']
            or cb['params'].get('source_sha256')!=sha256(source.encode()).hexdigest()):
        raise ValueError('original row grouping requires exact reconstructed source lowering and ABI')
    from dense import _geometry,_adapter
    from merlin.transforms.dense_planning import DenseProblem,DenseResources
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,schedule_compact_dense
    if cb['params'].get('source_transport_tasks') or cb['params'].get('compiler_temporaries'):
        return None
    # Original immutable SSA args are the only inputs; selected whole-program
    # ABI admits disjoint public outputs. Never infer sharing from equal bytes.
    names={v['arg_index']:n for n,v in cb['params']['tensor_source_map'].items() if set(v)=={'arg_index'}}
    products=[];resident={};acc={}
    for i,c in enumerate(cb['commands']):
        o=c.get('operands',{});a=c.get('attributes',{});kind=c['opcode']
        if kind=='RES_PACK':resident[o['dst']]=o['src']
        elif kind=='EVICT':resident.pop(o['handle'],None)
        elif kind=='MATMUL_RESIDENT':
            if a.get('rhs_transposed') or a.get('epilogue') or a.get('bias') or a.get('output_dtype','i32')!='i32':return None
            acc[o['dst']]=(o['lhs'],resident.get(o['rhs']),i)
        elif kind=='COMMIT':
            if o.get('bias') or a.get('epilogue'):return None
            if o['src'] not in acc:return None
            lhs,rhs,command=acc[o['src']];products.append((lhs,rhs,o['dst'],command,i))
        else:return None
    out=deepcopy(cb);edge,bank,sp,accbytes=_geometry()
    storage=DenseLoopStorage(DenseMemoryView('a','spad',edge,0,bank),DenseMemoryView('b','spad',edge,bank,bank),DenseMemoryView('c','acc',edge*4,0,accbytes//(edge*4)))
    resources=DenseResources(sp,accbytes,max_m_blocks=1,max_k_blocks=1,max_n_blocks=1,partial_output_tiles=True,partial_output_blocks=True)
    groups=[];covered=set()
    for g in proof.groups:
        lhs=[names[i] for i in g['lhs_args']];rhs=names[g['rhs_arg']];rows=[]
        if len(lhs)>edge:return None
        if cb['tensors'][rhs]!={'shape':[g['k'],g['n']],'dtype':'i8','role':'input'}:return None
        for name in lhs:
            hits=[r for r in products if r[0]==name and r[1]==rhs]
            if len(hits)!=1:return None
            r=hits[0]
            if cb['tensors'][name]!={'shape':[1,g['k']],'dtype':'i8','role':'input'}:return None
            if cb['tensors'][r[2]]!={'shape':[1,g['n']],'dtype':'i32','role':'output'}:return None
            if r[4] in covered:return None
            rows.append(r);covered.add(r[4])
        if len(set(lhs+[rhs]+[r[2] for r in rows]))!=2*len(lhs)+1:return None
        p=DenseProblem(len(lhs),g['k'],g['n'],edge,edge,edge,1,1,4)
        schedule=schedule_compact_dense(p,resources,storage,_adapter(),name='original_rows',dtypes=('i8','i8','i32'),static_budget=128,
            external_buffers_disjoint=True,verification_timeout_ms=5000)
        if schedule is None:return None
        groups.append({'lhs':lhs,'rhs':rhs,'outputs':[r[2] for r in rows],'commits':[r[4] for r in rows],
            'k':g['k'],'n':g['n'],'schedule':schedule})
    # Closed complete grouping makes movement across interleaved products legal;
    # no unknown command, output observer or side effect is skipped.
    if covered!={r[4] for r in products}:return None
    ops=list(range(out['params']['source_op_count']))
    for c in out['commands']:
        c['attributes'].update(source_task_index=0,source_op_indices=ops)
    out['params'].pop('command_source_attribution',None)
    out['params']['source_task_definitions']=[{'task_index':0,'kind':'source_product_group','source_op_indices':ops,
        'command_indices':list(range(len(out['commands'])))}]
    result=OriginalRows(proof,out,_guard(out),tuple(groups),_group_guard(groups)).revalidate()
    return out,result


def emit(builder,owner,group,pointers):
    """Consume checked abstract row effects through exact original pointer lanes."""
    from dense import _IndexBridge
    from dense_residency import invalidate_residency
    from merlin.compile.scheduling.llvm_program import emit_index_program
    owner.revalidate()
    if not any(group is g for g in owner.groups):raise ValueError('row effect group is not owned by the checked original source')
    count=len(group['lhs']);k=group['k'];n=group['n']
    expected=set(group['lhs']+[group['rhs']]+group['outputs'])
    from backend import boundary
    actual_abi={arg.name:builder.args[f'arg{i}'] for i,arg in enumerate(boundary(owner.cb))}
    if (set(pointers)!=expected or len(set(pointers.values()))!=len(expected)
            or any(pointers[name]!=actual_abi[name] for name in expected)):
        raise ValueError('original row pointer roster/ABI identity/alias changed')
    invalidate_residency(builder,reason='checked original row group acquires private SP/ACC')
    builder.config_ex(dataflow=1);builder.config_ld(k,id=0);builder.config_ld(n,id=1);builder.config_st(n*4)
    def lower(name,a):
        if name in ('lhs_load','store'):
            # Singleton M-tile is a required geometric predicate, not a guessed
            # dynamic lane. Scheduler independently checks all K/N tails.
            if type(a['r']) is not int or a['r']!=0 or type(a['rows']) is not int or a['rows']!=count:
                raise ValueError('row transport lacks exact complete constant lane coverage')
            for lane in range(count):
                if name=='lhs_load':
                    builder.mvin(builder.ptr_offset(pointers[group['lhs'][lane]],a['c']),builder.add(a['local'],lane),cols=a['cols'],rows=1,id=0)
                else:
                    builder.mvout(builder.ptr_offset(pointers[group['outputs'][lane]],builder.mul(a['c'],4)),builder.or_(builder.add(a['local'],lane),0xa0000000),cols=a['cols'],rows=1)
                # Conservative lane completion: require no additional DMA
                # independence law beyond the ordinary synchronous adapter.
                builder.fence()
        elif name=='weight_load':
            builder.mvin(builder.ptr_offset(pointers[group['rhs']],builder.add(builder.mul(a['r'],n),a['c'])),a['local'],cols=a['cols'],rows=a['rows'],id=1);builder.fence()
        elif name=='preload':
            builder.preload(a['weight'],builder.or_(builder.or_(a['acc'],0x80000000),builder.shl(a['continue'],30)),b_cols=a['cols'],b_rows=a['depth'],c_cols=a['cols'],c_rows=a['rows'])
        elif name=='compute':
            builder.compute(a['lhs'],0xffffffff,a_cols=a['depth'],a_rows=a['rows']);builder.fence()
        else:raise ValueError('unknown original row effect '+name)
    bridge=_IndexBridge(builder,'tiny')
    receipt=emit_index_program(group['schedule'].kernel,bridge,lower)
    owner.revalidate()
    return receipt
