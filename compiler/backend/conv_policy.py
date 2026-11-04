"""Explicit experimental convolution proposals over shared typed guards.

``auto`` is an opt-in proposal, not an assessed empirical default. The shared
shape-policy assessment deliberately retains the incumbent until completed,
identity-bound paired observations cover the frozen design and boundaries.
No simulator, profiling, workload name or observation lookup runs at compile
time. Capacity/epoch admission remains the responsibility of each real plan.
"""
from dataclasses import asdict
from functools import lru_cache

from merlin.transforms.pass_exploration import ExplorationGeometry, ValidationPartition
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass, ResourcePredicate,
    ShapePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import AnalysisStamp, PassSpec, PipelineContext, PipelineSession
from merlin.transforms.shape_policy import ShapeScheduleClass, assess_shape_policy

from . import isa


@lru_cache(maxsize=1)
def proposals():
    # Output NHWC and input NHWC are explicit geometry operands. These are
    # prospective tests, never keys into a schedule or timing lookup table.
    def geometry(width,channels,height,outputs):
        return ExplorationGeometry('convolution_nhwc',
            ((1,height,width,outputs),(1,height,width,channels),(3,3,channels,outputs)))
    # Freeze the refined half-row rule before using the fresh disjoint panel.
    # The previous 16 source geometries supply development coverage only.
    design=ValidationPartition(
        tuple(geometry(w,c,h,129)for h in (4,8)for w in (8,15,16,17)for c in (256,512)),
        tuple(geometry(w,c,h,97)for h in (5,9)for w in (7,9,13,18)for c in (128,384,640)))
    classes=(
        ShapeScheduleClass('narrow_output_row',
            (ShapePredicate(0,2,at_least=1,at_most=isa.DIM//2),),'complete_k',design),
        ShapeScheduleClass('bounded_spatial_revisit',
            (ResourcePredicate('spatial_group_credit',1),),'complete_k',design),
    )
    alternatives=(PipelineVariant('complete_k',('complete_k_panels',)),
        PipelineVariant('hoisted_gather',('hoisted_gather_panels',)),
        PipelineVariant('reduction_panels',('one_k_rhs_stages',)))
    # Native public replies currently omit identity pins required to construct
    # ShapeCaseFeedback. Do not manufacture them from geometry or cycle rows.
    assessment=assess_shape_policy(classes,alternatives,(),fallback='complete_k')
    return classes,assessment.report


def _execute_proposal(source,context,facts,shape_class,candidate,artifact,incumbent):
    """Make the shared pass manager own the explicit proposal selection."""
    output='convolution_schedule_choice';level=facts.stamp.ir_level
    def verify(value):
        value.__post_init__()
        if value.identity!=source.identity:raise ValueError('changed convolution policy source')
    session=PipelineSession(source,context=context,ir_level=level,
        fingerprint=lambda value:value.identity,verify_ir=verify)
    def entry(name,value):
        return RegisteredPass(PassSpec(name,0,produces=(output,),mutates_ir=False,ir_level=level),
            lambda session:{output:value})
    registry={candidate:entry(candidate,artifact),'complete_k':entry('complete_k',incumbent)}
    policy=PipelinePolicy((PipelineVariant(candidate,(candidate,),shape_class.predicates),
        PipelineVariant('complete_k',('complete_k',))),'complete_k')
    result=execute_policy(session,policy,registry,facts,output=output)
    return result.artifact,result.receipt


def select(prepared,stage,hoist_request,reduction_request):
    """Apply explicitly requested proposals through existing shape predicates."""
    p=prepared.source.plan;w=p.window;oh,ow=w.output_spatial
    pixels=w.input_shape[0]*oh*ow
    ntiles=(p.output_channels+isa.DIM-1)//isa.DIM
    old_phases=(ntiles+prepared.group.panel.column_blocks-1)//prepared.group.panel.column_blocks
    groups=(pixels+stage.block.m*isa.DIM-1)//(stage.block.m*isa.DIM)if stage else None
    # Credit is nonnegative and reaches the exact shared >=1 cutoff iff the
    # transformed M traversal count does not exceed incumbent N phase count.
    credit=max(0,old_phases-groups+1)if groups is not None else 0
    context=PipelineContext(hardware=prepared.epoch.hardware_identity,
        numerical_policy=prepared.epoch.numerical_policy_identity,
        resources='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af')
    facts=PolicyFacts(AnalysisStamp(prepared.source.identity,context,'counted_convolution'),
        shapes=((w.input_shape[0],oh,ow,p.output_channels),tuple(w.input_shape),
            (*w.kernel,w.input_shape[-1],p.output_channels)),
        resources=(('spatial_group_credit',credit),('incumbent_n_phases',old_phases),
            ('primitive_rows',isa.DIM),('spad_rows',isa.SPAD_ROWS),('acc_rows',isa.ACC_ROWS)))
    classes,assessment=proposals()
    matches=tuple(all(predicate.match(facts)is True for predicate in c.predicates)for c in classes)
    passes=[]
    hoist=hoist_request=='on'
    if hoist_request=='auto':
        hoist,receipt=_execute_proposal(prepared.source,context,facts,classes[0],
            'hoisted_gather',True,False)
        passes.append(receipt)
    chosen=stage if reduction_request=='on'else None
    if reduction_request=='auto':
        chosen,receipt=_execute_proposal(prepared.source,context,facts,classes[1],
            'reduction_panels',stage,None)
        passes.append(receipt)
    return hoist,chosen,dict(
        mode='explicit opt-in geometry proposals; defaults unchanged',
        source_sha256=prepared.source.identity,shapes=facts.shapes,
        classes=[dict(name=c.name,predicates=[asdict(p)for p in c.predicates],matches=m)
            for c,m in zip(classes,matches)],
        resources=dict(facts.resources),transformed_spatial_groups=groups,
        requested_gather_hoist=hoist_request,gather_hoist_applied=hoist,
        requested_reduction_panels=reduction_request,reduction_panels_applied=chosen is not None,
        executed_proposal_policies=passes,
        empirical_assessment=assessment)
