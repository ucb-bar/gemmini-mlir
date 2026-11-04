"""Expose exact singleton LHS rows to the shared physical-band coalescer.

The complete checked wide-decode source owner remains authoritative. This
transport specializes pure-load singleton domains in the ACTUAL schedule, then
uses the existing address-only coalescer. Expanded actual bands must retain the
complete original counted trace including every completion boundary.
"""
from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import json
from merlin.sched.ir import Const,Var,add
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw,TileLoadMapping,propose_tile_load_coalescing)
from merlin.compile.scheduling.panel_tile_loads import (
    expand_panel_tile_loads,_canonical_load_runs,_offset_form)
from dense_transfer import DenseTransferResult,_singleton_load_nests,_trace_equal


def coalesce_wide_gemv_lhs(problem,original):
    """Return the original owner unless its actual M1 LHS commands coalesce."""
    from dense import dense_adapter
    if problem.m!=1 or 'wide_decode_frontier'not in original.realization:return original
    original.revalidate()
    # Preserve the typed storage/adapter from the original checked ready owner.
    ready=original.original
    storage=ready._storage;adapter=ready._adapter
    root=Path(__file__).parent
    evidence=root/'data/command_ordering.json'
    extent=json.loads(evidence.read_text())['ordering']['extents']['MVIN.spad_addr']
    bands=extent['blocks'];tile=problem.tile_k
    if (extent['status']!='derived' or bands['block']!=tile or bands['stride']['code']!=1
            or bands['stride']['state']['value']!=0 or problem.lhs_bytes!=1):
        raise ValueError('wide GEMV LHS requires selected original MVIN address bands')
    law=TileLoadAddressLaw(tile,min(tile,(1<<extent['rows']['width'])-1),
        ((1<<bands['cols']['width'])-1)//tile*tile,1,problem.k,problem.tile_m,
        'selected-mvin-multiblock:'+sha256(evidence.read_bytes()).hexdigest())
    def project(call):
        if call.instr!='lhs_load':return None
        op=adapter.project(call)
        if type(op)is not Load or op.role!='lhs' or op.step!=Const(1):
            raise ValueError('wide GEMV LHS requires original contiguous source rows')
        return TileLoadMapping('lhs',storage.lhs.backing,op.dram_row*problem.k+op.dram_col,
            op.row,op.rows,op.cols,law)
    def encode(call,columns):
        return replace(call,args=tuple((name,Const(columns)if name=='cols'else value)for name,value in call.args))
    def encode_band(call,band):
        mapping=project(call);delta=band.source_byte+mapping.source_byte*-1
        constant,terms=_offset_form(delta)
        if constant%tile or any(v%tile for v in terms.values()):
            raise ValueError('wide GEMV LHS actual band is not source tile aligned')
        tiles=add(Const(constant//tile),*(atom*(value//tile)for atom,value in terms.items()))
        fields=dict(call.args);fields.update(src_col=fields['src_col']+delta,dst=band.local_row,
            cols=band.columns,b=fields['b']+tiles)
        return replace(call,args=tuple(fields.items()))
    actual,specializations=_singleton_load_nests(original.kernel,project)
    actual,proposal=propose_tile_load_coalescing(actual,project=project,encode=encode,
        max_tiles=law.max_columns//tile,coalesce_tail=True)
    if not proposal['changes']:return original
    normalized,_=_canonical_load_runs(_singleton_load_nests(original.kernel,project)[0],project)
    paths=('wide_gemv_lhs.py','dense.py','dense_transfer.py','dense_wide_decode.py','encoding.py',
        'data/command_ordering.json','compiler_library/merlin/compile/scheduling/tile_load_coalescing.py',
        'compiler_library/merlin/compile/scheduling/panel_tile_loads.py')
    owners={name:sha256((root/name).read_bytes()).hexdigest()for name in paths}
    def check(kernel):
        if any(sha256((root/name).read_bytes()).hexdigest()!=identity for name,identity in owners.items()):
            raise ValueError('wide GEMV LHS source or selected address law changed')
        if kernel!=actual:raise ValueError('wide GEMV LHS actual checked product changed')
        expanded,projection=expand_panel_tile_loads(kernel,project=project,encode_band=encode_band)
        expanded,_=_singleton_load_nests(expanded,project)
        expanded,_=_canonical_load_runs(expanded,project)
        if (replace(expanded,body=())!=replace(normalized,body=())or
                not _trace_equal(normalized.body,expanded.body)):
            raise ValueError('wide GEMV LHS bands differ from original complete source/effect trace')
        return projection
    projection=check(actual)
    result=DenseTransferResult(original,actual,{**original.realization,'kernel_sha256':actual.digest(),
        'wide_gemv_lhs':dict(schema='gemmini.wide_gemv_lhs.v1',
            original_kernel_sha256=original.kernel.digest(),actual_kernel_sha256=actual.digest(),
            actual_singleton_specializations=specializations,coalescing=proposal,
            actual_band_projection=projection,source_owners=owners,
            proof='exact actual all-band counted trace, including every original source coordinate, local access and drain',
            completion='unchanged',timing='UNKNOWN')},check)
    result.revalidate()
    return result
