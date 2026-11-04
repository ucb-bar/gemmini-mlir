"""Projection scheduling laws derived from the selected command-ordering facts.

The full source, local addresses and every removed boundary are checked by the
library. The ordering card supplies hardware premises; fit never supplies them.
"""
from pathlib import Path
from hashlib import sha256
import json
import sys


def epoch_law(adapter):
    from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw
    root = Path(__file__).resolve().parent
    payload = (root / 'command_ordering.json').read_bytes()
    selected = json.loads(payload)
    facts = (root / 'rtl_facts/facts.json').read_bytes()
    if selected['base_facts']['sha256'] != sha256(facts).hexdigest():
        raise ValueError('selected command ordering is not bound to current resource facts')
    ordering = selected['ordering']
    if ordering.get('status') != 'derived':
        raise ValueError('selected command ordering is not derived')
    edges = {(r['older'],r['younger'],r['space']):r for r in ordering['cross_queue']}
    reads = ('COMPUTE_PRELOADED.a_spad','COMPUTE_ACCUMULATE.a_spad','PRELOAD.weight_spad')
    writes = ('MVIN.spad_addr','MVIN2.spad_addr','MVIN3.spad_addr')
    def waits(older,younger,space):
        return edges.get((older,younger,space),{}).get('overlap') == 'completion'
    classes = json.loads(facts)['facts']['timing_structure']['command_classes']['classes']
    queues = [classes[name]['command_queue'] for name in ('execute','load','store')]
    flags = dict(
        local_read_after_write=all(waits(w,r,'scratchpad') for w in writes for r in reads),
        local_write_after_read=all(waits(r,w,'scratchpad') for w in writes for r in reads),
        ordered_accumulator_writes=ordering['completion_order']['execute']['order']=='in_order',
        accumulator_result_visibility=waits('PRELOAD.c_addr','MVOUT.acc_addr','accumulator'),
        # One fixed EX/LD/ST configuration is established before this invocation;
        # the rewrite moves no configuration and no data operation.
        ordered_configuration_capture=True,
        finite_queue_progress=all(q['status']=='derived' and q['entries_total']>0 for q in queues),
        complete_device_drain=True,
        exclusive_private_state=True,
    )
    return DensePanelEpochLaw(adapter.configuration_identity,adapter.numerical_policy_identity,
        'selected-gemmini-ordering:'+sha256(payload).hexdigest(),**flags)


def apply_epochs(result, problem, block, resources, storage, adapter, *, checked_source=None):
    from merlin.compile.scheduling.dense_panel_epochs import schedule_dense_panel_epochs
    try:
        law = epoch_law(adapter)
        return schedule_dense_panel_epochs(result,problem,block,resources,storage,adapter,
            (result.admission if checked_source is None else checked_source.admission),law,static_budget=128,timeout_ms=30000,
            max_read_instances=16384,checked_source=checked_source)
    except ValueError as error:
        print('projection-epoch-fallback: '+str(error),file=sys.stderr)
        return result


from dataclasses import dataclass, replace

@dataclass(frozen=True)
class WideProjectionResult:
    spatial: object
    temporal: object
    kernel: object
    realization: dict
    project: object
    encode_band: object
    problem: object
    storage: object
    adapter: object
    source_hash: str

    @property
    def admission(self):
        return self.spatial.admission

    def revalidate(self):
        from merlin.compile.scheduling.panel_tile_loads import expand_dense_panel_tile_loads
        self.spatial.revalidate()
        self.temporal.revalidate()
        expanded, _ = expand_dense_panel_tile_loads(self.kernel,self.problem,self.storage,self.adapter,
            project=self.project,encode_band=self.encode_band,max_static_nodes=4096)
        if expanded != self.temporal.kernel or sha256(Path(__file__).read_bytes()).hexdigest()!=self.source_hash:
            raise ValueError('wide projection expansion differs from checked temporal owner')

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
        self.revalidate()
        report=emit_index_effect_program(self.kernel,control,lower_effect,
            complete_epoch=complete_epoch,completion_instruction=DRAIN)
        self.revalidate()
        return report


def partition_load_tails(kernel, adapter, *, budget=128):
    """Use ordered library partitions to expose constant complete load tiles.

    Only load-only nests are candidates. A predicate at the last iteration
    becomes an exact complete prefix plus its original one-iteration tail.
    No command or completion crosses a neighbor. Final source/hazard checking
    still consumes every actual band, including the unchanged partial tile.
    """
    from merlin.sched.ir import Call, Const, Loop, Scaled, Select, Sum
    from merlin.sched.ir.expr import add, select
    from merlin.compile.scheduling.loop_schedule import partition_counted_loop
    from merlin.compile.scheduling.block_schedule import Load
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
    actual=kernel
    records=[]
    blocked=set()
    def selectors(x):
        if type(x) is Select:
            return {(x.var,x.equals)}|selectors(x.then)|selectors(x.other)
        if type(x) is Scaled:return selectors(x.expr)
        if type(x) is Sum:return set().union(*(selectors(t) for t in x.terms))
        return set()
    def pure_loads(body):
        calls=[]
        for s in body:
            if type(s) is Loop:
                nested=pure_loads(s.body)
                if nested is None:return None
                calls.extend(nested)
            elif s.instr!=DRAIN and type(adapter.project(s)) is Load:
                calls.append(s)
            else:return None
        return calls
    def candidate(body,path=()):
        for statement in body:
            if type(statement) is not Loop:continue
            route=(*path,statement.var)
            calls=pure_loads(statement.body)
            if route not in blocked and calls and statement.extent>1 and any(
                (statement.var,statement.extent-1) in selectors(v)
                for c in calls for _,v in c.args):
                return route,statement.extent
            found=candidate(statement.body,route)
            if found:return found
        return None
    # Each cut removes one last-iteration selector from a load-only nest;
    # ordinary exact domain simplification prevents revisiting its prefix.
    def simplify(body,domains=()):
        known=dict(domains)
        def expr(x):
            if type(x) is Select:
                yes,no=expr(x.then),expr(x.other)
                extent=known.get(x.var)
                if extent is not None:
                    if not 0<=x.equals<extent:return no
                    if extent==1:return yes
                return select(x.var,x.equals,yes,no)
            if type(x) is Scaled:return expr(x.expr)*x.factor
            if type(x) is Sum:return add(*(expr(v) for v in x.terms))
            return x
        output=[]
        for statement in body:
            if type(statement) is Loop:
                output.append(replace(statement,body=simplify(statement.body,(*domains,(statement.var,statement.extent)))))
            else:output.append(replace(statement,args=tuple((k,expr(v)) for k,v in statement.args)))
        return tuple(output)
    actual=replace(actual,body=simplify(actual.body))
    for _ in range(16):
        found=candidate(actual.body)
        if found is None:return actual,records
        path,extent=found
        try:
            cut=partition_counted_loop(actual,path,cuts=(extent-1,),max_static_operations=budget)
            cut.verify_current(actual,cut.kernel)
        except ValueError as error:
            blocked.add(path)
            records.append({'path':path,'extent':extent,'refused':str(error),'route':'retain original loop'})
            print('projection-tail-refusal: '+str(error)+'; retain original load loop',file=sys.stderr)
            continue
        records.append({'path':path,'extent':extent,'cut':extent-1,'ordered_partition':True})
        actual=replace(cut.kernel,body=simplify(cut.kernel.body))
    raise ValueError('load-tail partition exceeds bounded 16-cut proof policy')


def apply_wide(result, problem, block, resources, storage, adapter, *, epochs=True):
    """Consume every wide band, then join its unchanged epoch names to the temporal proof.

    The physical command expands exactly to the checked primitive command stream.
    No operand or completion moves across another operation. The temporal owner
    checks this expanded source, not the precursor's requested source operations.
    """
    from merlin.compile.scheduling.tile_load_coalescing import (
        TileLoadAddressLaw, TileLoadMapping, propose_tile_load_coalescing)
    from merlin.compile.scheduling.panel_tile_loads import (
        admit_panel_tile_loads, expand_dense_panel_tile_loads)
    from merlin.compile.scheduling.block_schedule import Load
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
    from merlin.sched.ir import Call, Const, Loop
    # MvinRs2's selected 7-bit columns field allows at most seven complete
    # 16-lane bands. Four bands bound each DMA request to 64 columns.
    edge=problem.tile_k
    root=Path(__file__).resolve().parent
    card=json.loads((root/'command_ordering.json').read_bytes())
    extents=card['ordering']['extents']
    limits=[]
    for name in ('MVIN.spad_addr','MVIN2.spad_addr'):
        e=extents[name]
        if e['status']!='derived' or e['blocks']['block']!=edge:
            raise ValueError('wide loads lack selected tiled transfer extents')
        limits.append(((1<<e['blocks']['cols']['width'])-1)//edge)
    max_tiles=min(4,*limits)
    if max_tiles<2:
        return result
    lawid='gemmini-mvin-exact-i8-block-stride:'+sha256((root/'command_ordering.json').read_bytes()).hexdigest()
    def project(call):
        if call.instr not in ('gemmini_lhs_load','gemmini_weight_load'):
            return None
        op=adapter.project(call)
        left=op.role=='lhs'
        pitch=problem.k if left else problem.n
        view=storage.lhs if left else storage.rhs
        law=TileLoadAddressLaw(edge,edge,max_tiles*edge,1,pitch,edge,lawid)
        return TileLoadMapping(op.role,view.backing,op.dram_row*pitch+op.dram_col,
            op.row,op.rows,op.cols,law)
    def encode(call,columns):
        args=dict(call.args);args['cols']=Const(columns)
        return replace(call,args=tuple(args.items()))
    def encode_band(call,band):
        op=adapter.project(call)
        from merlin.compile.scheduling.panel_tile_loads import _offset_form
        from merlin.sched.ir.expr import add
        delta=band.source_byte+(op.dram_row*(problem.k if op.role=='lhs' else problem.n)+op.dram_col)*-1
        constant,terms=_offset_form(delta)
        if constant%edge or any(v%edge for v in terms.values()):
            raise ValueError('wide band source column is not an exact tile displacement')
        tile_delta=add(Const(constant//edge),*(atom*(v//edge) for atom,v in terms.items()))
        args=dict(call.args)
        args['col']=op.dram_col+delta
        args['tile1']=args['tile1']+tile_delta
        args['dst']=band.local_row;args['rows']=band.rows;args['cols']=band.columns
        return replace(call,args=tuple(args.items()))
    try:
        original=getattr(result,'checked',result)
        prepared, partitions=partition_load_tails(original.kernel,adapter)
        proposal, coalescing=propose_tile_load_coalescing(prepared,
            project=project,encode=encode,max_tiles=max_tiles,coalesce_tail=True)
        if proposal==original.kernel:
            return apply_epochs(result,problem,block,resources,storage,adapter,
                checked_source=original if original is not result else None) if epochs else result
        spatial=admit_panel_tile_loads(original,proposal,problem,block,resources,storage,adapter,
            project=project,encode_band=encode_band,static_budget=128,timeout_ms=30000)
        expanded, projection=expand_dense_panel_tile_loads(proposal,problem,storage,adapter,
            project=project,encode_band=encode_band)
        # admit_panel_tile_loads has just checked ALL effects of this exact
        # expansion against the original source and explicit completion law.
        report=dict(spatial.realization,kernel_sha256=expanded.digest())
        checked=replace(original,kernel=expanded,realization=report,
            checked_kernel_sha256=expanded.digest())
        checked.revalidate()
        temporal=apply_epochs(checked,problem,block,resources,storage,adapter) if epochs else checked
        def drain_names(kernel):
            names=set()
            def walk(body):
                for s in body:
                    if type(s) is Loop:
                        if s.body and s.body[-1]==Call(DRAIN,()): names.add(s.var)
                        walk(s.body)
            walk(kernel.body)
            return names
        removed=drain_names(expanded)-drain_names(temporal.kernel)
        def rewrite(body):
            output=[]
            for s in body:
                if type(s) is Loop:
                    body=s.body[:-1] if s.var in removed else s.body
                    s=replace(s,body=rewrite(body))
                output.append(s)
            return tuple(output)
        physical=replace(proposal,body=rewrite(proposal.body))
        finalreport=dict(temporal.realization,wide_coalescing=coalescing,
            wide_projection=projection,load_tail_partitions=partitions,kernel_sha256=physical.digest(),
            expanded_kernel_sha256=temporal.kernel.digest())
        final=WideProjectionResult(spatial,temporal,physical,finalreport,project,encode_band,
            problem,storage,adapter,sha256(Path(__file__).read_bytes()).hexdigest())
        final.revalidate()
        return final
    except ValueError as error:
        print('projection-wide-fallback: '+str(error),file=sys.stderr)
        return apply_epochs(result,problem,block,resources,storage,adapter,
            checked_source=getattr(result,'checked',None)) if epochs else result
