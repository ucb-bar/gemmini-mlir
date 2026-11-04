"""Optional geometry proposal over existing checked spatial group products.

Capacity and selected ordering remain the group planner's responsibility.
The performance region is an experimental hypothesis, not a legality law or
an assessed default: large spatial work, short K and narrow N may amortize a
four-output completion epoch. Missing native boundary coverage remains UNKNOWN.
"""
from dataclasses import asdict
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PipelineContext, PipelineSession, PassSpec
from merlin.transforms.shape_policy import ShapeScheduleClass
from merlin.transforms.pass_exploration import ExplorationGeometry, ValidationPartition
from . import isa


def select(prepared):
    """Return the actual checked two/four group artifact, never a shadow plan."""
    from .conv_spatial_groups import plan
    p=prepared.source.plan;w=p.window;oh,ow=w.output_spatial
    # Compiler performance bounds in primitive-tile units. These do not grant
    # physical storage, queue capacity, halo omission or reordered reductions.
    predicates=(ResourcePredicate('spatial_pixels',4*isa.DIM*isa.DIM),
        ResourcePredicate('short_reduction_credit',1),
        ResourcePredicate('narrow_output_credit',1),
        ResourcePredicate('regular_window',1))
    def geometry(h,width,ci,co,stride=1):
        return ExplorationGeometry('convolution_nhwc',((1,h,width,ci),(9*ci,co),
            (1,(h-1)//stride+1,(width-1)//stride+1,co)))
    design=ValidationPartition((geometry(56,56,64,64),geometry(28,28,128,128)),
        tuple(geometry(*g)for g in ((31,33,64,64),(32,32,64,64),(33,32,64,64),
            (32,32,113,64),(32,32,114,64),(32,32,64,63),(32,32,64,65),(64,64,64,64,2))))
    shape_class=ShapeScheduleClass('large_spatial_short_reduction_narrow_output',predicates,'two',design)
    resources=(('spatial_pixels',oh*ow),
        ('short_reduction_credit',int(w.kernel[0]*w.kernel[1]*w.input_shape[-1]<=4*isa.DIM*isa.DIM)),
        ('narrow_output_credit',int(p.output_channels<=4*isa.DIM)),
        ('regular_window',int(w.kernel==(3,3)and w.stride==(1,1)and w.dilation==(1,1))),
        ('spad_rows',isa.SPAD_ROWS),('acc_rows',isa.ACC_ROWS))
    level='counted_convolution';output='selected_checked_spatial_group'
    context=PipelineContext(hardware=prepared.epoch.hardware_identity,
        numerical_policy=prepared.epoch.numerical_policy_identity,
        resources=prepared.epoch.hardware_identity)
    session=PipelineSession(prepared.source,context=context,ir_level=level,
        fingerprint=lambda source:source.identity,verify_ir=lambda source:source.__post_init__())
    facts=PolicyFacts(session.stamp(),shapes=(w.input_shape,(oh,ow,p.output_channels)),resources=resources)
    def entry(name):
        return RegisteredPass(PassSpec(name,0,produces=(output,),
            mutates_ir=False,ir_level=level),lambda session:{output:plan(prepared,name)})
    registry={name:entry(name)for name in ('two','four')}
    policy=PipelinePolicy((PipelineVariant('four',('four',),shape_class.predicates),
        PipelineVariant('two',('two',))),'two')
    result=execute_policy(session,policy,registry,facts,output=output)
    chosen=result.artifact
    # The performance predicate is separate from actual admission. A refused
    # four-output product grants no effect and falls back to checked two.
    fallback=False
    if chosen is None and all(predicate.match(facts)is True for predicate in predicates):
        chosen=plan(prepared,'two');fallback=True
    receipt=dict(selection='large_spatial',shape_class=shape_class.name,frozen_validation_design=design.manifest(),source_sha256=prepared.source.identity,
        resources=dict(resources),predicates=[asdict(x)for x in predicates],
        predicate_matches=[x.match(facts)for x in predicates],pipeline=result.receipt,
        four_admission_fallback=fallback,
        selected_group=chosen.receipt['actual_row_blocks']if chosen else 1,
        hypothesis='large spatial >=4*DIM^2 pixels/image; K<=4*DIM^2; CO<=4*DIM; regular 3x3 stride1 dilation1',
        legality='actual shared group planner plus unchanged selected ordering; no new target premises',
        empirical_status='optional unassessed proposal; boundary/model cycles UNKNOWN')
    if chosen is not None:chosen.receipt['geometry_policy']=receipt
    return chosen
