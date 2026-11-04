"""Full-K immutable RHS residency across counted M panels.

Only storage load placement and outer loop order change. The ordinary dense
adapter and terminal emitter still own every instruction and full drain.
Array STAY, widened DMA and load/compute overlap have no admission here.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView, DenseBiasSource
from merlin.compile.scheduling.dense_panel_pipeline import (
    DenseRetainedReductionAdmission, dense_retained_choices, schedule_dense_m_panels,
)
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from merlin.compile.tuning_knobs import knob
from . import isa


WEIGHT_PANEL_GEOMETRIES={'8x8':(8,8),'8x32':(8,32)}


def weight_panel_schedule(m,k,n,bias,column_scales=False,*,geometry=None,choice_policy='traffic'):
    """Resolve the declared resource bound before the geometry-keyed cache."""
    if geometry is None:
        selection=knob('DENSE_WEIGHT_PANEL_GEOMETRY',default='8x8',
            values=tuple(WEIGHT_PANEL_GEOMETRIES),root=Path(__file__).resolve().parents[1])
        geometry=WEIGHT_PANEL_GEOMETRIES[selection]
    return _weight_panel_schedule(m,k,n,bias,column_scales,
        geometry=tuple(geometry),choice_policy=choice_policy)


@lru_cache(maxsize=256)
def _weight_panel_schedule(m,k,n,bias,column_scales=False,*,geometry=(8,8),choice_policy='traffic'):
    """Return a checked RHS alternative or an explicit original-domain refusal.

    Geometry bounds are explicit trial parameters, never case-name dispatch.
    Capacity and invariant hoists are selected by the shared scheduling library.
    """
    if choice_policy not in ('traffic','amortized_m'):
        raise ValueError('weight panel choice_policy must be traffic or amortized_m')
    if column_scales:
        return None,'RHS panel trial excludes one-column recomputation readouts'
    from .dense import _adapter
    a=_adapter()
    d=isa.DIM
    p=DenseProblem(m,k,n,d,d,d,1,1,4,bias=bias)
    s=DenseLoopStorage(
        DenseMemoryView('lhs','spad',d,0,isa.SPAD_ROWS//2),
        DenseMemoryView('rhs','spad',d,isa.SPAD_ROWS//2,isa.SPAD_ROWS//2),
        DenseMemoryView('acc','acc',4*d,0,isa.ACC_ROWS))
    r=DenseResources(isa.SPAD_ROWS*d,isa.ACC_ROWS*4*d,
        max_m_blocks=geometry[0],max_n_blocks=geometry[1],max_k_blocks=p.tiles[1],
        partial_output_tiles=True,partial_output_blocks=True)
    choices=dense_retained_choices(p,r,s,roles=('weight',))
    if not choices:
        return None,'full-K RHS panel exceeds original private storage assignments'
    choice=choices[0]
    if choice_policy=='amortized_m':
        # Preserve the traffic-ranked N width. Within that same width, avoid
        # shrinking M to a divisor merely to eliminate a partial-family RHS
        # reload: each extra M panel pays its own load/compute/store epochs.
        # This ranks legal original choices, with no runtime profiling or new
        # capacity assignment; paired complete cycles still decide promotion.
        choice=max((item for item in choices if item.block.n==choice.block.n),
                   key=lambda item:item.block.m)
    evidence=json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if evidence['completion_order']['execute']['order']!='in_order':
        return None,'selected execute completion order is unknown'
    b=choice.block
    law=DenseRetainedReductionAdmission(a.configuration_identity,a.numerical_policy_identity,
        evidence['identity'],max(1,b.m*b.k*b.n),max(1,(b.m+b.n)*b.k),2,8)
    source=hashlib.sha256(json.dumps({'m':m,'k':k,'n':n,'bias':bias,
        'dtype':['i8','i8','i32']},sort_keys=True).encode()).hexdigest()
    attempt=schedule_dense_m_panels(p,r,s,a,law,n_tiles=b.n,m_tiles=b.m,
        name='dense_weight_panels',dtypes=('i8','i8','i32'),source_identity=source,
        external_buffers_disjoint=True,static_budget=128,timeout_ms=30000,
        bias_source=DenseBiasSource(n,'i32') if bias else None)
    if attempt.result is None:
        return None,json.dumps(attempt.feedback,sort_keys=True)
    result=attempt.result
    result.revalidate()
    result.realization['weight_panels']={
        **attempt.feedback,'retained_role':'weight','outer_order':['n','m'],
        'selected_block_tiles':[b.m,b.k,b.n],
        'geometry_bound':list(geometry),'array_weight_retention':False,
        'completion':'original pure load/compute epochs and full device drains',
        'source_identity':source}
    # Preserve the original default receipt and checked artifact byte for byte.
    if choice_policy!='traffic':
        result.realization['weight_panels']['choice_policy']=choice_policy
    return result,None
