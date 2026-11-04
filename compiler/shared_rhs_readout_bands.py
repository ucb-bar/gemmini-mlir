"""Group final-ready columns, prove their actual products, then widen MVOUT.

The ordinary epoch verifier consumes every primitive band of the actual new
frontier, without a synthetic source-order inverse. Addressing expansion is
separate from the selected causal completion premise and never removes drains.
"""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
from merlin.sched.ir import Call, Loop, Const, Select, Sum, Scaled, Ptr, add
from merlin.compile.scheduling.block_schedule import Store, Preload, Compute
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN, _remap, _calls, _verify_dense_epochs
from merlin.compile.scheduling.tile_load_coalescing import (
    TileStoreAddressLaw, TileStoreMapping, propose_tile_store_coalescing,
    expand_tile_store_mapping)
from dense_transfer import _trace_equal
from merlin.compile.scheduling.tile_load_coalescing import _replace_call


def _specialize(items, var, extent):
    def expr(x):
        if type(x) is Select:
            return expr(x.other) if x.var == var and not 0 <= x.equals < extent else replace(x, then=expr(x.then), other=expr(x.other))
        if type(x) is Sum:return add(*(expr(t) for t in x.terms))
        if type(x) is Scaled:return expr(x.expr)*x.factor
        if type(x) is Ptr:return replace(x, offset=expr(x.offset))
        return x
    return tuple(replace(s,body=_specialize(s.body,var,extent)) if type(s)is Loop else replace(s,args=tuple((k,expr(v))for k,v in s.args))for s in items)


def group_ready_columns(kernel, adapter, *, max_tiles):
    """Keep all original products, stores and drains; delay stores by <=7 bands."""
    if type(max_tiles) is not int or max_tiles<2:
        raise ValueError('ready grouping requires a selected multi-tile bound')
    records=[]; names={s.var for s in _loops(kernel.body)}
    def fresh(base):
        while base in names:base+='_' 
        names.add(base);return base
    def walk(items):
        out=[]
        for s in items:
            if type(s)is not Loop:out.append(s);continue
            if (s.extent>1 and len(s.body)==2 and type(s.body[0])is Loop and type(s.body[1])is Call
                and type(adapter.project(s.body[1]))is Store
                and {type(adapter.project(c))for c in _calls((s.body[0],))}=={Preload,Compute}):
                if (s.iter_args or s.yields or s.results
                    or any(x.iter_args or x.yields or x.results for x in _loops(s.body))):
                    raise ValueError('ready grouping cannot replace carried loop state')
                width=min(max_tiles,s.extent)
                if width<2:out.append(s);continue
                for first in range(0,s.extent,width):
                    count=min(width,s.extent-first)
                    cv=fresh('readout_compute_band');sv=fresh('readout_store_band')
                    compute=_specialize(_remap((s.body[0],),s.var,cv,offset=first),cv,count)
                    store=_specialize(_remap((s.body[1],),s.var,sv,offset=first),sv,count)
                    out.extend((Loop(cv,count,compute),Loop(sv,count,store)))
                records.append(dict(original_axis=s.var,extent=s.extent,max_group=width,
                    parent_sha256=kernel.digest(),rule='complete original final-K bands before their same original stores'))
            else:out.append(replace(s,body=walk(s.body)))
        return tuple(out)
    return replace(kernel,body=walk(kernel.body)),records


def _loops(items):
    for s in items:
        if type(s)is Loop:
            yield s;yield from _loops(s.body)


def ready_store_address_law(problem):
    """Decode selected non-pooling raw-MVOUT tile geometry; no issue grant."""
    root=Path(__file__).parent
    ordering_path=root/'data/command_ordering.json'
    ordering=json.loads(ordering_path.read_text())['ordering']
    e=ordering['extents']['MVOUT.acc_addr'];blocks=e['blocks']
    # This route is only the incumbent raw-i32 path. The caller already refuses
    # every narrowing/activation readout; do not infer another conversion law.
    if (e['status']!='derived' or blocks['block']!=problem.tile_n
        or blocks['stride'].get('constant')!=problem.tile_m
        or problem.tile_n!=16 or problem.tile_m!=16
        or not any(c.get('code')==2 and c.get('field')=='ConfigMvoutRs1.pool_stride' and c.get('reset')==0 for c in e['conditions'])):
        raise ValueError('ready store bands lack selected non-pooling MVOUT tile geometry')
    identity='selected-mvout:'+sha256(ordering_path.read_bytes()).hexdigest()
    return TileStoreAddressLaw(problem.tile_n,min(problem.tile_m,(1<<e['rows']['width'])-1),
        ((1<<blocks['cols']['width'])-1)//problem.tile_n*problem.tile_n,
        4,problem.n*4,problem.tile_m,'original-raw-i32-unit-scale-no-relu',identity)


def expand_ready_store_bands(op, problem, storage):
    """Decode every ACC tile read by one actual original raw wide Store.

    These are address/effect bands only. The original ready proof owns its
    producers and numerical state; the input transport retains that custody.
    """
    if type(op) is not Store or op.readout is not None:
        raise ValueError('ready store bands require the original raw-i32 descriptor')
    law=ready_store_address_law(problem)
    if type(op.rows) is not Const or not 0<op.rows.value<=law.max_rows:
        raise ValueError('ready store bands exceed selected original row geometry')
    mapping=TileStoreMapping(storage.accumulator.backing,'output',
        (op.dram_row*problem.n+op.dram_col)*4,op.accumulator_row,op.rows,op.cols,law)
    return tuple(replace(op,accumulator_row=band.local_row,
        dram_col=op.dram_col+index*law.tile_columns,cols=band.columns)
        for index,band in enumerate(expand_tile_store_mapping(mapping)))


def widen_ready_member(kernel, problem, resources, storage, adapter, bound, admission):
    """Actual grouped source proof plus actual MVOUT all-band projection."""
    law=ready_store_address_law(problem)
    grouped,changes=group_ready_columns(kernel,adapter,max_tiles=law.max_columns//law.tile_columns)
    if not changes:raise ValueError('no adjacent original final-ready columns')
    # No checking projection: this verifies actual regrouped products, coverage,
    # latest writers, assigned views, all ACC producers before their readout,
    # queue budgets and unchanged explicit completion epochs directly.
    proof=_verify_dense_epochs(grouped,problem,resources,storage,adapter,bound,
        static_budget=256,timeout_ms=20000,ordered_accumulator_commits=True,
        source_phase_fission=True,readout_compute_admission=admission)
    def project(call):
        if call.instr==DRAIN:return None
        op=adapter.project(call)
        if type(op)is not Store:return None
        if op.readout is not None or adapter.encode(op)!=call:
            raise ValueError('wide ready store changed original raw numerical descriptor')
        return TileStoreMapping(storage.accumulator.backing,'output',
            (op.dram_row*problem.n+op.dram_col)*4,op.accumulator_row,op.rows,op.cols,law)
    def encode(call,cols):return adapter.encode(replace(adapter.project(call),cols=Const(cols)))
    actual,coalescing=propose_tile_store_coalescing(grouped,project=project,encode=encode,
        max_tiles=law.max_columns//law.tile_columns,coalesce_tail=True)
    def expand(items):
        out=[]
        for s in items:
            if type(s)is Loop:out.append(replace(s,body=expand(s.body)));continue
            mapping=project(s)
            if mapping is None:out.append(s);continue
            op=adapter.project(s)
            for band in expand_ready_store_bands(op,problem,storage):
                out.append(adapter.encode(band))
        return tuple(out)
    expanded=replace(actual,body=expand(actual.body))
    # Coalescing may strip a singleton store loop. Compare complete actual
    # expanded trace only AFTER proving the new grouped frontier independently.
    def normalize_store_domains(items):
        # Grouping bounds each pure-store domain by the selected seven-band
        # limit. Enumerate only these tiny domains, preserving every surrounding
        # producer loop, drain and order; compare the complete decoded calls.
        out=[]
        def pure(body):
            return all(pure(s.body) if type(s) is Loop else project(s) is not None for s in body)
        def specialize(body,var,value):
            return tuple(replace(s,body=specialize(s.body,var,value)) if type(s) is Loop else
                _replace_call(s,var,Const(value)) for s in body)
        for s in items:
            if type(s) is Loop:
                if pure(s.body):
                    if s.iter_args or s.yields or s.results or s.extent>max(1,law.max_columns//law.tile_columns):
                        raise ValueError('store trace normalization exceeds original bounded plain domain')
                    for i in range(s.extent):
                        out.extend(normalize_store_domains(specialize(s.body,s.var,i)))
                else:out.append(replace(s,body=normalize_store_domains(s.body)))
            else:out.append(s)
        return tuple(out)
    if not _trace_equal(normalize_store_domains(grouped.body),normalize_store_domains(expanded.body)):
        raise ValueError('actual MVOUT expansion differs from the proved grouped frontier')
    if not coalescing['changes']:raise ValueError('grouped ready stores do not admit a physical wider MVOUT')
    return actual,dict(grouping=changes,actual_grouped_kernel_sha256=grouped.digest(),
        actual_expanded_kernel_sha256=expanded.digest(),direct_actual_source_and_epoch_proof=proof,
        actual_band_projection=coalescing,mvout_address_law=repr(law))
