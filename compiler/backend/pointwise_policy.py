"""Bounded real schedule alternatives for offline strided-pointwise exploration.

This module grants no performance admission. Compilation uses the explicitly
configured alternative; complete native timings decide any later geometry rule.
"""
from merlin.transforms.pass_exploration import PassFamily,ExplorationGeometry,ValidationPartition
from merlin.transforms.pipeline_policy import RegisteredPass
from merlin.transforms.pipeline_state import PassSpec

ALTERNATIVES=('single','wide','rhs_stationary')


def schedule_family():
    """Each option executes the shared source/target-checked dense lowering."""
    def bind(options):
        def lower(session):
            from .pointwise import _strided_schedule
            from merlin.compile.scheduling.window_transfer import Window2D
            source=session.module
            window=Window2D(tuple(source['input_shape']),(1,1),tuple(source['stride']),(1,1),(0,0,0,0))
            width=window.output_spatial[1]
            from .isa import DIM
            tile=max(d for d in range(1,min(DIM,width)+1) if width%d==0)
            schedule,problem,advanced=_strided_schedule(window,source['channels_out'],tile,
                source.get('seeded',False),source.get('mixed_scales',False),panel_choice=options['panel'])
            return {'program':{'realization':dict(schedule.realization),'advanced':advanced,
                'dimensions':(problem.m,problem.k,problem.n)}}
        return RegisteredPass(PassSpec('strided_pointwise_schedule',0,produces=('program',),
            mutates_ir=False,ir_level='strided_pointwise_source'),lower)
    return PassFamily('strided_pointwise_schedule',bind,(('panel',ALTERNATIVES),))


def frozen_partition():
    """Freeze original source geometry; numeric variants share one identity."""
    def geometry(h,ci,co):
        return ExplorationGeometry('strided_pointwise',((1,h,h,ci),(ci,co),(((h+1)//2)**2,co)))
    return ValidationPartition((geometry(56,256,512),),
        (geometry(28,512,1024),geometry(14,1024,2048),geometry(5,12,24)))


def rhs_stationary_geometry(input_shape,channels_out,stride=(2,2)):
    """Necessary geometry conditions for a future measured retention policy.

    This diagnostic neither selects a lowering nor grants performance admission.
    Full-weight residency avoids the known input rereads across N phases; more
    than one spatial panel is necessary to reuse the retained weights. Actual
    lowering must still pass the shared source, storage and target verifiers.
    """
    from . import isa
    from merlin.compile.scheduling.window_transfer import Window2D
    if (len(input_shape)!=4 or len(stride)!=2 or any(type(x) is not int or x<1
            for x in (*input_shape,channels_out,*stride)) or tuple(stride)==(1,1)):
        raise ValueError('expected positive original NHWC strided pointwise geometry')
    window=Window2D(tuple(input_shape),(1,1),tuple(stride),(1,1),(0,0,0,0))
    oh,ow=window.output_spatial
    tile_m=max(d for d in range(1,min(isa.DIM,ow)+1) if ow%d==0)
    m=input_shape[0]*oh*ow
    mt=(m+tile_m-1)//tile_m
    kt=(input_shape[-1]+isa.DIM-1)//isa.DIM
    nt=(channels_out+isa.DIM-1)//isa.DIM
    half_spad=isa.SPAD_ROWS//2
    weight_rows=kt*nt*isa.DIM
    # Match the actual alternative's bounded expansion and physical storage.
    n_phase=min(32,nt,half_spad//(kt*isa.DIM))
    block_m=min(8,mt,half_spad//(kt*isa.DIM),
                isa.ACC_ROWS//(nt*isa.DIM))
    reasons=[]
    if weight_rows>half_spad or n_phase!=nt:
        reasons.append('weights_require_multiple_n_phases')
    if block_m<1:
        reasons.append('full_weight_phase_has_no_accumulator_panel')
    elif mt<=block_m:
        reasons.append('no_spatial_panel_weight_reuse')
    return {'schema':'merlin.strided_pointwise_geometry_candidate.v1',
        'input_shape':tuple(input_shape),'channels_out':channels_out,'stride':tuple(stride),
        'matrix_dimensions':(m,input_shape[-1],channels_out),'row_tile':tile_m,
        'matrix_tiles':(mt,kt,nt),'full_weight_rows':weight_rows,
        'rhs_capacity_rows':half_spad,'accumulator_capacity_rows':isa.ACC_ROWS,
        'bounded_n_phase_tiles':n_phase,'full_weight_m_panel_tiles':block_m,
        'candidate_geometry':not reasons,'refusals':reasons,
        'performance_qualified':False,
        'qualification':'necessary resource/reuse conditions; no held-out timing admission'}
