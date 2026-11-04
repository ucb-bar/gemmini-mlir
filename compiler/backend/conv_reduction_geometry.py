"""Finite joint M/N exploration through real shared capacity planners.

These explicit alternatives bound compiler expansion, never physical storage.
Every choice enumerates dense_blocks against the same selected SPAD/ACC facts.
PassFamily binds those real planners; the requested pass artifact is consumed
by convolution emission. No static score or case identity selects a geometry.
"""
from dataclasses import asdict

from merlin.transforms.dense_planning import DenseProblem,DenseResources,dense_blocks
from merlin.transforms.pass_exploration import ExplorationGeometry,PassFamily,ValidationPartition,generate_candidates
from merlin.transforms.pipeline_policy import PipelinePolicy,PolicyFacts,RegisteredPass,execute_policy
from merlin.transforms.pipeline_state import PassSpec,PipelineContext,PipelineSession

from . import isa

GEOMETRIES={'n_wide':(2,32),'balanced':(8,8),'m_tall':(16,4)}


def validation_design():
    def geometry(batch,height,channels,stride):
        out=(height-1)//stride+1
        return ExplorationGeometry('convolution_nhwc',((batch,height,height,channels),
            (9*channels,channels),(batch*out*out,channels)))
    return ValidationPartition((geometry(4,56,64,1),geometry(4,28,128,1)),
        (geometry(1,56,128,2),geometry(4,14,256,1)))


def _capacity(prepared,geometry):
    p=prepared.source.plan;w=p.window
    pixels=w.input_shape[0]*w.output_spatial[0]*w.output_spatial[1]
    chunks=(w.input_shape[-1]+isa.DIM-1)//isa.DIM
    problem=DenseProblem(pixels,w.kernel[0]*w.kernel[1]*chunks*isa.DIM,p.output_channels,
        isa.DIM,isa.DIM,isa.DIM,1,1,4,bias=p.bias)
    m,n=GEOMETRIES[geometry]
    resources=DenseResources(isa.SPAD_ROWS*isa.DIM,isa.ACC_ROWS*isa.DIM*4,
        max_m_blocks=m,max_n_blocks=n,max_k_blocks=1,
        partial_output_tiles=True,partial_output_blocks=True)
    candidates=dense_blocks(problem,resources)
    block=max(candidates,key=lambda b:(b.n,b.m))if candidates else None
    return block,dict(geometry=geometry,problem=asdict(problem),resources=asdict(resources),
        enumerated_resource_legal_blocks=len(candidates),
        selected_block=asdict(block)if block else None,
        within_geometry_rule='max N then M over shared resource-legal enumeration')


def plan_geometry(prepared,geometry):
    if geometry not in GEOMETRIES:raise ValueError('unknown convolution reduction geometry')
    output='capacity_checked_convolution_geometry';level='counted_convolution'
    def bind(options):
        name=options['geometry']
        return RegisteredPass(PassSpec('capacity_'+name,0,produces=(output,),
            mutates_ir=False,ir_level=level),lambda session:{output:_capacity(prepared,name)})
    family=PassFamily('convolution_reduction_capacity',bind,
        (('geometry',tuple(GEOMETRIES)),))
    space=generate_candidates((family,),output=output,max_candidates=3)
    selected=next(v for v in space.variants
        if space.bindings[v.passes[0]]['knobs']['geometry']==geometry)
    context=PipelineContext(hardware=prepared.epoch.hardware_identity,
        numerical_policy=prepared.epoch.numerical_policy_identity,
        resources='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af')
    session=PipelineSession(prepared.source,context=context,ir_level=level,
        fingerprint=lambda source:source.identity,verify_ir=lambda source:source.__post_init__())
    facts=PolicyFacts(session.stamp(),shapes=(tuple(prepared.source.plan.window.input_shape),))
    result=execute_policy(session,PipelinePolicy((selected,),selected.name),space.registry,facts,output=output)
    block,receipt=result.artifact
    return block,dict(receipt,pass_family_bindings=dict(space.bindings),pipeline=result.receipt,
        frozen_validation_design=validation_design().manifest(),
        empirical_selection='explicit knob only; no automatic promotion; incumbent unchanged')
