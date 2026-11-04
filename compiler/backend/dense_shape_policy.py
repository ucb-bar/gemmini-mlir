"""Registered dense alternatives with explicit empirical and legality guards.

``auto`` uses the shared assessed policy and keeps the incumbent while exact
paired cutoff feedback is missing. ``frontier_class`` explicitly requests the
coarse short-row/long-reduction/wide-output class; it is an opt-in experiment,
not a claim that missing boundary observations passed. Both execute real current
schedule constructors and return their checked product to the common encoder.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from merlin.compile.tuning_knobs import knob

from merlin.transforms.pass_exploration import ExplorationGeometry, ValidationPartition
from merlin.transforms.pipeline_policy import (
    DimensionRef, FeaturePredicate, PipelinePolicy, PipelineVariant, PolicyFacts,
    RegisteredPass, ResourcePredicate, ResourceRef, RatioFeature, ShapePredicate, TileFeature, execute_policy,
)
from merlin.transforms.pipeline_state import PipelineContext, PipelineSession, PassSpec
from merlin.transforms.shape_policy import ShapeScheduleClass, assess_shape_policy
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from . import isa

LEVEL = 'gemmini_dense_capacity_source'
OUTPUT = 'dense_checked_schedule'
# Freeze the shape panel before collecting held-out cycles. Values are source
# geometry only: no capsule/model identifiers participate in dispatch.
TRAINING = ((1,2048,32000), (16,5632,2048))
HELD_OUT = ((1,2048,2048), (16,2048,2048),
            (16,2032,2048), (16,2064,2048),
            (16,2048,2032), (16,2048,2064),
            (17,2048,2048), (15,2033,2033))


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class DenseSource:
    m: int
    k: int
    n: int
    bias: bool
    column_scales: bool

    def verify(self):
        if any(type(v) is not int or v <= 0 for v in (self.m,self.k,self.n)):
            raise ValueError('dense policy requires positive source dimensions')
        if type(self.bias) is not bool or type(self.column_scales) is not bool:
            raise ValueError('dense policy requires explicit numerical boundary flags')


def _geometry(shape):
    m,k,n=shape
    return ExplorationGeometry('dense',((m,k),(k,n)))


def class_predicates():
    tile=ResourceRef('primitive_dim')
    return (FeaturePredicate(TileFeature(DimensionRef(0,0),tile),at_most=(1,1)),
            FeaturePredicate(TileFeature(DimensionRef(0,1),tile),at_least=(128,1)),
            FeaturePredicate(TileFeature(DimensionRef(1,1),tile),at_least=(128,1)))


def alternatives():
    return (PipelineVariant('frontier',('dense_frontier',),
                (ResourcePredicate('segmented_placement',1),)),
            PipelineVariant('retained',('dense_retained',)))


def assessed_policy(feedback=()):
    group=ShapeScheduleClass('short_rows_long_reduction_wide_output',class_predicates(),
        'retained',ValidationPartition(tuple(map(_geometry,TRAINING)),tuple(map(_geometry,HELD_OUT))))
    return assess_shape_policy((group,),alternatives(),tuple(feedback),fallback='retained')


def configured_issue_order(override=None):
    configured=knob('DENSE_FRONTIER_ISSUE_ORDER',default='prefetch_rhs_between_k_tiles',
        values=('prefetch_rhs_between_k_tiles','prefetch_rhs_between_columns','aspect'),
        root=Path(__file__).resolve().parents[1])
    issue=(override if override is not None else
           os.environ.get('DENSE_FRONTIER_ISSUE_ORDER',configured))
    if issue not in ('prefetch_rhs_between_k_tiles','prefetch_rhs_between_columns','aspect'):
        raise ValueError('unsupported dense class frontier issue order')
    return issue


def configured_frontier_panel():
    # Resource bounds only. Both choices run the same shared segmented planner,
    # ownership/source checks and terminal encoder. No new DMA mapping law.
    return knob('DENSE_FRONTIER_PANEL',default='wide',values=('wide','deep','order'),
        root=Path(__file__).resolve().parents[1])


def column_aspect_predicate():
    # Offline issue-order hypothesis frozen before the next holdout panel.
    # Exact dimension ratio; this empirical bound is separate from storage fit.
    return FeaturePredicate(RatioFeature(DimensionRef(1,1),DimensionRef(0,1)),at_most=(3,1))


def binding(m,k,n,bias,column_scales=False,*,frontier_issue_order=None):
    """Current source, public resource facts, and registered real schedule passes."""
    from .dense import _schedule
    from .dense_pipeline import _retained_schedule, RETAINED_CAPACITY_REFUSAL
    from .dense_storage_partition import select_segmented_partition
    issue=configured_issue_order(frontier_issue_order)
    if issue=='aspect':
        issue='prefetch_rhs_between_k_tiles'
    panel=configured_frontier_panel()
    max_n=32 if panel=='deep' else 64
    geometry='skinny_deep' if panel=='deep' else 'skinny'
    source=DenseSource(m,k,n,bias,column_scales)
    source.verify()
    context=PipelineContext(hardware='gemmini-fire-aligned-dim16-ws',
        numerical_policy='i8-i8-i32-modular',
        resources=_sha([isa.DIM,isa.SPAD_ROWS,isa.ACC_ROWS]),
        execution='selected-rtl-ordered-accumulator-commit',
        library='selected-dense-scheduling-library')
    session=PipelineSession(source,context=context,ir_level=LEVEL,
        fingerprint=lambda src:_sha(asdict(src)),verify_ir=lambda src:src.verify())
    p=DenseProblem(m,k,n,isa.DIM,isa.DIM,1 if column_scales else isa.DIM,1,1,4,bias=bias)
    resources=DenseResources(isa.SPAD_ROWS*isa.DIM,isa.ACC_ROWS*4*isa.DIM,
        max_m_blocks=1,max_n_blocks=max_n,max_k_blocks=max(1,min(64,p.tiles[1]-1)),
        partial_output_tiles=True,partial_output_blocks=True)
    placement=(None if column_scales or p.tiles[1]<2 else select_segmented_partition(p,resources))
    deep_placement=(None if panel!='order' or column_scales or p.tiles[1]<2 else
        select_segmented_partition(p,replace(resources,max_n_blocks=32)))
    facts=PolicyFacts(session.stamp(),shapes=((m,k),(k,n)),
        element_types=('i8','i8'),layouts=('row-major','row-major'),
        operations=frozenset({'dense'}),resources=(('primitive_dim',isa.DIM),
        ('segmented_placement',int(placement is not None)),
        ('segmented_deep_placement',int(deep_placement is not None))))

    def retained(sess):
        s=sess.module
        if (not s.column_scales and s.m<=isa.DIM and s.k<=isa.DIM
                and s.n<=isa.DIM):
            return {OUTPUT:(_schedule(s.m,s.k,s.n,s.bias,s.column_scales),
                'compact','single primitive product has no retained operand reuse or reduction overlap')}
        result,refusal=_retained_schedule(s.m,s.k,s.n,s.bias,s.column_scales,
            False,'wide','half')
        selected='retained'
        if result is None and refusal==RETAINED_CAPACITY_REFUSAL:
            from .dense_streaming import streaming_schedule
            result,refusal=streaming_schedule(s.m,s.k,s.n,s.bias,s.column_scales)
            selected='streaming'
        if result is None:
            result=_schedule(s.m,s.k,s.n,s.bias,s.column_scales)
            selected='compact'
        return {OUTPUT:(result,selected,refusal)}

    def frontier(sess,selected_issue=None,selected_geometry=None):
        from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
        s=sess.module
        try:
            result,refusal=_retained_schedule(s.m,s.k,s.n,s.bias,s.column_scales,
                True,selected_geometry or geometry,'adaptive',64,selected_issue or issue)
        except DenseVerificationUnavailable as unavailable:
            # UNKNOWN cannot establish the transformed program. The incumbent
            # still receives its complete independent source/ownership proof.
            incumbent,selected,prior_refusal=retained(sess)[OUTPUT]
            reason='segmented proof unavailable; checked incumbent selected: '+str(unavailable)
            if prior_refusal:
                reason+='; '+prior_refusal
            return {OUTPUT:(incumbent,selected,reason)}
        if result is None:
            raise ValueError('admitted dense frontier failed current schedule construction: '+str(refusal))
        return {OUTPUT:(result,'segmented',refusal)}

    registry={name:RegisteredPass(PassSpec(name,0,produces=(OUTPUT,),mutates_ir=False,ir_level=LEVEL),action)
              for name,action in (('dense_retained',retained),('dense_frontier',frontier),
                  ('dense_frontier_columns',lambda sess:frontier(sess,'prefetch_rhs_between_columns')),
                  ('dense_frontier_deep',lambda sess:frontier(sess,selected_geometry='skinny_deep')))}
    return session,registry,facts


def select_dense_schedule(m,k,n,bias,column_scales=False,*,mode='auto',feedback=(),domain=None,frontier_issue_order=None):
    if mode not in ('auto','frontier_class'):
        raise ValueError('unknown dense shape policy mode')
    session,registry,facts=binding(m,k,n,bias,column_scales,frontier_issue_order=frontier_issue_order)
    assessed=assessed_policy(feedback)
    if mode=='auto':
        result=assessed.execute(session,registry,facts,output=OUTPUT,domain=domain)
    else:
        candidate,incumbent=alternatives()
        explicit=PipelineVariant(candidate.name,candidate.passes,
            (*class_predicates(),*candidate.predicates))
        variants=(explicit,incumbent)
        if (configured_frontier_panel()=='order'
                and configured_issue_order(frontier_issue_order)!='prefetch_rhs_between_columns'):
            # Reuse the existing issue-order partition. In aspect mode the
            # original columns variant below has priority, so this pass only
            # consumes the K-prefetch region. No new aspect cutoff is learned.
            deep=PipelineVariant('frontier_deep',('dense_frontier_deep',),
                (*class_predicates(),ResourcePredicate('segmented_deep_placement',1),
                 ShapePredicate(0,0,at_least=2)))
            variants=(deep,*variants)
        if configured_issue_order(frontier_issue_order)=='aspect':
            columns=PipelineVariant('frontier_columns',('dense_frontier_columns',),
                (*class_predicates(),*candidate.predicates,column_aspect_predicate()))
            variants=(columns,*variants)
        result=execute_policy(session,PipelinePolicy(variants,'retained'),
            registry,facts,output=OUTPUT)
    schedule,selected,refusal=result.artifact
    receipt=dict(result.receipt,requested_mode=mode,
        actual_selected_schedule=selected,
        configured_frontier_issue_order=configured_issue_order(frontier_issue_order),
        configured_frontier_panel=configured_frontier_panel(),
        assessment=assessed.report,
        evidence='explicit class experiment' if mode=='frontier_class' else 'assessed exact paired feedback',
        automatic_promotion=mode=='auto' and selected=='segmented')
    return schedule,selected,refusal,receipt
