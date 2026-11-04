"""Exploratory wide-output frontier over the checked segmented library.

Retain the complete original LHS across N panels. Interleave each future RHS
segment between current K tiles, using the library's checked source and storage
mapping and retaining every original segment drain. The N guard bounds this
experimental alternative; it is not an established performance crossover.
"""
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from merlin.transforms.dense_planning import DenseProblem,DenseResources
from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView
from merlin.compile.scheduling.dense_segmented_pipeline import (
    dense_segmented_blocks,DenseSegmentedAdmission)
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan


@lru_cache(maxsize=128)
def plan_wide_decode_frontier(m,k,n,*,bias=False):
    """Return (problem, checked actual transport), or None outside this region.

    The source caller must supply the ordinary immutable contiguous row-major
    signed i8*i8 product ABI with independent output storage. Existing complete
    source, storage and selected execution checks are retained, not recreated.
    """
    from dense import _capacities,_execution_laws,dense_adapter
    from dense_readout import resident_readout_laws
    from dense_transfer import coalesce_dense_epochs
    if bias or any(type(x) is not int or x<=0 for x in (m,k,n)) or m!=1:return None
    kr,edge,operand,rows,rowbytes,acc,accrows,accbytes=_capacities()
    if kr!=edge or n < rows:return None
    root=Path(__file__).parent/'data'
    ordering=json.loads((root/'command_ordering.json').read_text())['ordering']
    extent=ordering['extents']['MVIN2.spad_addr'];bands=extent['blocks']
    if (extent['status']!='derived' or bands['block']!=edge
        or bands['stride']['code']!=1 or bands['stride']['state']['value']!=1):
        raise ValueError('decode transport requires selected original RHS MVIN band mapping')
    max_columns=((1<<bands['cols']['width'])-1)//edge*edge
    selected=json.loads((root/'facts.json').read_text())
    request_bytes=selected['facts']['timing_structure']['memory_clients']['clients']['dma_read']['max_request_bytes']
    # This alternative is meaningful only when a legal physical command spans
    # more than one DMA request. A width is not a bandwidth or overlap grant.
    if max_columns<=request_bytes or n<2*max_columns:return None
    half=rows//2
    if k*n<=half*rowbytes:return None
    p=DenseProblem(m,k,n,edge,kr,edge,1,1,4)
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand,rowbytes,0,half),
        DenseMemoryView('rhs',operand,rowbytes,half,rows-half),
        DenseMemoryView('acc',acc,accbytes,0,accrows))
    if p.tiles[1]*edge>half or p.tiles[1]<2:return None
    r=DenseResources(rows*rowbytes,accrows*accbytes,max_m_blocks=1,
        max_n_blocks=min(p.tiles[2],max_columns//edge),max_k_blocks=p.tiles[1]-1,
        partial_output_tiles=True,partial_output_blocks=True)
    legal=[b for b in dense_segmented_blocks(p,r,storage)
           if b.k<p.tiles[1] and b.n==max_columns//edge]
    if not legal:return None
    block=min(legal,key=lambda b:(b.loads,-b.k,b.m,b.n))
    adapter=dense_adapter();overlap,reduction,epoch=_execution_laws(adapter)
    if epoch.unknown():raise ValueError('decode transport needs selected combined execution ordering')
    reduction,readout=resident_readout_laws(adapter,block,reduction,epoch)
    overlap=replace(overlap,max_issued_readouts=reduction.max_issued_readouts,
        pending_readouts=reduction.pending_readouts,readout_configuration=reduction.readout_configuration)
    admission=DenseSegmentedAdmission(overlap,reduction,
        'selected-combined-segmented:'+epoch.qualification_identity)
    source=sha256(repr((m,k,n,bias,'row-major-i8-product-i32')).encode()).hexdigest()
    decision=make_dense_schedule_plan(p,r,storage,adapter,source_identity=source,
        dtypes=('i8','i8','i32'),static_budget=256,
        block_tiles=(block.m,block.k,block.n),outer_order=('m','n'),
        pipeline_prefetch=True,pipeline_kind='retained_lhs_segmented_k',
        pipeline_issue_order='prefetch_rhs_between_k_tiles')
    original=schedule_dense_pipeline(p,r,storage,adapter,name='dense',
        dtypes=('i8','i8','i32'),external_buffers_disjoint=True,
        schedule_plan=decision,source_identity=source,pipeline_admission=admission,
        readout_mode='ready',readout_admission=readout,static_budget=256,timeout_ms=20000)
    actual=coalesce_dense_epochs(original,p,storage,adapter,request_bound=False)
    actual.realization.update(wide_decode_frontier=dict(schema='gemmini.wide_decode_frontier.v1',
        source_layout='original contiguous K,N; no packing',max_command_columns=max_columns,
        dma_request_bytes=request_bytes,block_tiles=[block.m,block.k,block.n],
        transport='hardware-column-bound coalescing; all scalar bands checked',
        issue_order='prefetch_rhs_between_k_tiles',
        region='single source row; N at least hardware operand-store rows',
        timing='UNKNOWN; no bandwidth benefit inferred from width',
        owner_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    actual.revalidate()
    return p,actual
