"""Optional source-derived thin-M storage, using the checked retained pipeline.

Only the complete source M extent is compacted. Hardware rows still contain DIM
lanes; neither K/N primitive geometry nor source views or completion laws change.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.compact_dense import DenseBiasSource, emit_compact_dense
from merlin.compile.scheduling.dense_panel_pipeline import (
    DenseRetainedReductionAdmission, dense_retained_operand_hoists,
    dense_retained_compute_order)
from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from . import isa
from .dense_storage_partition import select_full_k_partition


@lru_cache(maxsize=128)
def thin_storage_schedule(m, k, n, bias=False, column_scales=False):
    if column_scales:
        return None, 'thin storage preserves the incumbent per-column readout'
    if not 0 < m < isa.DIM:
        return None, 'thin storage requires complete source M smaller than DIM'
    from .dense import _adapter
    from .dense_pipeline import _checked_pipeline
    a = _adapter()
    p = DenseProblem(m,k,n,m,isa.DIM,isa.DIM,1,1,4,bias=bias)
    # Keep the incumbent wide panel bound. The only new freedom is physical
    # row occupancy and the consequent disjoint scratchpad split.
    r = DenseResources(isa.SPAD_ROWS*isa.DIM,isa.ACC_ROWS*4*isa.DIM,
        max_m_blocks=8,max_n_blocks=8,max_k_blocks=p.tiles[1],
        partial_output_tiles=True,partial_output_blocks=True)
    partition = select_full_k_partition(p,r,compact_m_rows=True)
    if partition is None:
        return None, 'thin full-K retained panel exceeds physical storage'
    s,block,receipt = partition
    evidence = json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if evidence['completion_order']['execute']['order'] != 'in_order':
        return None, 'selected execute completion order is unknown'
    law = DenseRetainedReductionAdmission(a.configuration_identity,
        a.numerical_policy_identity,evidence['identity'],
        max(1,block.m*block.k*block.n),max(1,(block.m+block.n)*block.k),2,8)
    bias_source = DenseBiasSource(n,'i32') if bias else None
    raw = emit_compact_dense(p,block,s,a,name='dense_thin_storage',
        dtypes=('i8','i8','i32'),bias_source=bias_source)
    transforms = dense_retained_operand_hoists(raw,a,role='lhs')
    transforms += dense_retained_compute_order(
        propose_loop_transforms(raw,transforms),a,order=('k','n','m'))
    source = hashlib.sha256(json.dumps(dict(m=m,k=k,n=n,bias=bias,
        dtype=['i8','i8','i32']),sort_keys=True).encode()).hexdigest()
    plan = make_dense_schedule_plan(p,r,s,a,source_identity=source,
        dtypes=('i8','i8','i32'),static_budget=128,
        block_tiles=(block.m,block.k,block.n),loop_transforms=transforms,
        pipeline_kind='retained_full_k',pipeline_prefetch=False,bias_source=bias_source)
    result, refusal = _checked_pipeline(p,r,s,a,name='dense_thin_storage',
        dtypes=('i8','i8','i32'),source_identity=source,
        external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=law,
        bias_source=bias_source,readout_mode='separate',static_budget=128)
    if result is not None:
        result.realization['thin_storage'] = dict(selection='compact_m',
            source_extent=[m,k,n],primitive_shape=[m,isa.DIM,isa.DIM],
            physical_row_bytes=dict(lhs=isa.DIM,rhs=isa.DIM,accumulator=4*isa.DIM),
            storage_partition=receipt,
            completion='unchanged retained ownership drains and in-order execute law')
    return result, refusal
