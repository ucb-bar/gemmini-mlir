"""Explicit spatial-group policy over current checked convolution artifacts.

The regional policy is an experimental geometry class with an unvalidated
boundary. It performs no timing, caches no training kernel and grants no storage
or ordering authority. Ordinary convolution preparation admits both alternatives.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

LEVEL='authored_convolution_group_source'
OUTPUT='prepared_convolution_group'
SELECTIONS=('single','group4','capacity_guarded','group8')


@dataclass(frozen=True)
class ConvGroupSource:
    source_identity: str
    original_state_identity: str
    pixels: int
    primitive: int
    group4_blocks: int
    rhs_rows: int
    rhs_base: int
    column_blocks: int
    operand_capacity_rows: int
    accumulator_capacity_rows: int

    def __post_init__(self):
        if (any(type(v)is not int or v<=0 for k,v in asdict(self).items()
                if k not in ('source_identity','original_state_identity'))
                or any(type(v)is not str or len(v)!=64
                    for v in (self.source_identity,self.original_state_identity))):
            raise ValueError('convolution group policy requires exact admitted positive source resources')


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def prepare_group_policy(ctx, ifm, rhs, dst, attributes, *, selection,
                         physical_views=None, allow_wide=False):
    """Return actual ``(prepared_or_None, policy_receipt)`` before ctx mutation.

    Capacity-guarded uses complete primitive spatial tiles, not ceil-rounded
    tails: eight means at least 128 original output pixels for the selected DIM.
    The common planner independently bounds every private LHS/RHS/ACC interval.
    Explicit group4 is available for experiments on smaller spatial domains.
    Opt-in group8 tries eight independently admitted row blocks; its fallback
    is precisely the incumbent capacity-guarded group4/single policy.
    """
    if selection not in SELECTIONS:
        raise ValueError('unknown convolution spatial row-group policy')
    from .conv_spatial_panels import _prepare
    single=_prepare(ctx,ifm,rhs,dst,attributes,physical_views,
        allow_wide=allow_wide,max_row_blocks=1)
    if single is None:
        return None,dict(schema='authored_conv_group_policy.v1',selection=selection,
            admission='refused before target mutation')
    group4=_prepare(ctx,ifm,rhs,dst,attributes,physical_views,
        allow_wide=allow_wide,max_row_blocks=4)
    if group4 is None:
        # The group bound only narrows planning, but retain a complete original
        # refusal if future common admission adds another grouped requirement.
        group4=single
    group8=None
    if selection=='group8':
        group8=_prepare(ctx,ifm,rhs,dst,attributes,physical_views,
            allow_wide=allow_wide,max_row_blocks=8)
    panel=single.group.panel
    pixels=single.source.plan.window.input_shape[0]
    for extent in single.source.plan.window.output_spatial:pixels*=extent
    state=dict(source=single.source.identity,operands=[ifm,rhs,dst],
        physical_views=None if physical_views is None else asdict(physical_views),
        attributes=single.attributes,
        rhs_pitch=single.rhs_pitch,output_pitch=single.output_pitch,
        zero_seed=single.zero_seed,ordering=single.ordering['ordering_sha256'])
    source=ConvGroupSource(single.source.identity,_digest(state),pixels,panel.tile,
        group4.group.row_blocks,panel.rhs_rows,panel.rhs_base,panel.column_blocks,
        panel.operand_capacity_rows,panel.accumulator_capacity_rows)
    source.__post_init__()
    def verify(value):
        if type(value)is not ConvGroupSource:raise ValueError('convolution policy source changed type')
        value.__post_init__()
    context=PipelineContext(hardware='gemmini-fire-aligned-b8ff28edc096392c',
        numerical_policy='signed-i8-products-i32-source-ordered-readout',
        resources='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af',
        execution=single.ordering['ordering_sha256'])
    session=PipelineSession(source,context=context,ir_level=LEVEL,
        fingerprint=lambda value:_digest(asdict(value)),verify_ir=verify)
    resources=dict(pixels=pixels,m_tiles=(pixels+panel.tile-1)//panel.tile,
        complete_m_tiles=pixels//panel.tile,admitted_group4_blocks=source.group4_blocks,
        primitive=panel.tile,rhs_rows=panel.rhs_rows,rhs_base=panel.rhs_base,
        column_blocks=panel.column_blocks,spad_rows=panel.operand_capacity_rows,
        acc_rows=panel.accumulator_capacity_rows)
    if group8 is not None:
        resources['admitted_group8_blocks']=group8.group.row_blocks
    facts=PolicyFacts(session.stamp(),operations=frozenset({'convolution'}),
        layouts=('nhwc',),element_types=('i8','i8'),resources=tuple(sorted(resources.items())))
    def entry(name,artifact):
        key='conv_spatial_group_'+name
        def action(active):
            if active.module!=source:raise ValueError('convolution group artifact belongs to another source')
            artifact.gather.__post_init__();artifact.group.__post_init__()
            if artifact.source.identity!=source.source_identity:
                raise ValueError('convolution group source identity changed')
            return {OUTPUT:artifact}
        return key,RegisteredPass(PassSpec(key,0,produces=(OUTPUT,),
            mutates_ir=False,ir_level=LEVEL),action)
    registry=dict((entry('single',single),entry('group4',group4)))
    if group8 is not None:registry.update((entry('group8',group8),))
    fallback=PipelineVariant('single',('conv_spatial_group_single',))
    if selection=='single':variants=(fallback,)
    else:
        predicates=(ResourcePredicate('admitted_group4_blocks',4),)
        if selection in ('capacity_guarded','group8'):
            predicates+=(ResourcePredicate('complete_m_tiles',8),)
        variants=(PipelineVariant('group4',('conv_spatial_group_group4',),predicates),fallback)
        if selection=='group8' and group8 is not None:
            variants=(PipelineVariant('group8',('conv_spatial_group_group8',),
                (ResourcePredicate('admitted_group8_blocks',8),
                 ResourcePredicate('complete_m_tiles',8))),)+variants
    result=execute_policy(session,PipelinePolicy(variants,'single'),registry,facts,output=OUTPUT)
    receipt=dict(schema='authored_conv_group_policy.v1',selection=selection,
        actual_spatial_group_blocks=result.artifact.group.row_blocks,policy=result.receipt,
        performance_class=dict(resource='complete_m_tiles',at_least=8,
            boundary_status='UNKNOWN',missing_neighbor_values=[7,8],
            observed_complete_m_tiles=[3,49,784],
            observations='complete same-model Spike call pairs; 3 tiles regressed 0.239%; 49/784 tiles improved 29.13%/41.28%',
            automatic_default_promotion=False),
        qualification='current-source checked planning and policy execution; independent numerics/timing required')
    if selection=='group8':
        receipt['incumbent_performance_class']=receipt.pop('performance_class')
        receipt['experiment']=dict(requested_row_blocks=8,
            admitted_row_blocks=None if group8 is None else group8.group.row_blocks,
            fallback='unchanged capacity_guarded group4/single',
            timing_status='UNKNOWN',promotion_authority=False)
    return result.artifact,receipt
