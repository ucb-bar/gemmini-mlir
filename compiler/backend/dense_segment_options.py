"""Finite segmented schedule options for offline actual-pass exploration.

Each family constructs the same checked schedules as ordinary dense lowering.
It supplies no static performance score and does not select an online default.
"""
SEGMENT_K_LIMITS=(8,16,32,64)
SEGMENT_PARTITIONS=('legacy','balanced_short')


def resolve_segment_partition(selection=None):
    from pathlib import Path
    import os
    from merlin.compile.tuning_knobs import knob
    configured=knob('DENSE_SEGMENT_PARTITION',default='legacy',values=SEGMENT_PARTITIONS,
        root=Path(__file__).resolve().parent.parent)
    selection=selection if selection is not None else os.environ.get('DENSE_SEGMENT_PARTITION',configured)
    if selection not in SEGMENT_PARTITIONS:
        raise ValueError('DENSE_SEGMENT_PARTITION must be legacy or balanced_short')
    return selection


def segment_k_cap(k_tiles, limit, partition, frontier):
    if (type(k_tiles) is not int or k_tiles < 1 or limit not in SEGMENT_K_LIMITS
            or partition not in SEGMENT_PARTITIONS):
        raise ValueError('segmented cap requires exact source depth and declared options')
    if frontier=='output_groups':
        return k_tiles
    cap=min(limit,k_tiles-1)
    if partition=='balanced_short' and k_tiles<=limit:
        cap=min(cap,(k_tiles+1)//2)
    return cap


def resolve_segment_geometry(geometry):
    from pathlib import Path
    import os
    from merlin.compile.tuning_knobs import knob
    configured=knob('DENSE_SEGMENT_PANEL',default='inherit',values=('inherit','wide_16'),
        root=Path(__file__).resolve().parent.parent)
    selection=os.environ.get('DENSE_SEGMENT_PANEL',configured)
    if selection not in ('inherit','wide_16'):
        raise ValueError('DENSE_SEGMENT_PANEL must be inherit or wide_16')
    return geometry if selection=='inherit' else 'wide_16'


def _segmented_pass(options):
    from merlin.transforms.pipeline_policy import RegisteredPass
    from merlin.transforms.pipeline_state import PassSpec
    from .dense_pipeline_policy import DensePipelineSource,LEVEL
    from .dense_pipeline import _retained_schedule
    def action(session):
        source=session.module
        if type(source) is not DensePipelineSource:
            raise ValueError('segmented family requires the typed dense source')
        source.__post_init__()
        limit=options.get('segment_k_limit',source.segment_k_limit)
        partition=options.get('partition',source.segment_partition)
        geometry=source.segmented_geometry or source.geometry
        if 'panel' in options:
            geometry=source.geometry if options['panel']=='inherit' else options['panel']
        result,refusal=_retained_schedule(source.m,source.k,source.n,
            source.bias,source.column_scales,True,geometry,
            source.storage_split,source.frontier,'lhs',limit,partition,options.get('rhs_residency'))
        if result is None:
            raise ValueError('segmented option refused: '+str(refusal))
        result.revalidate()
        if (result.realization['segment_k_limit']!=limit
                or result.realization['segment_partition']!=partition
                or result.realization['panel_geometry']['selection']!=geometry):
            raise ValueError('segmented option lost its selected resource bounds')
        return {'program':result}
    return RegisteredPass(PassSpec('dense_segment',0,produces=('program',),
        mutates_ir=False,ir_level=LEVEL),action)


def segmented_k_limit_family():
    from merlin.transforms.pass_exploration import PassFamily
    return PassFamily('dense_segment_k',_segmented_pass,(("segment_k_limit",SEGMENT_K_LIMITS),))


def segmented_panel_family():
    from merlin.transforms.pass_exploration import PassFamily
    return PassFamily('dense_segment_panel',_segmented_pass,(("panel",('inherit','wide_16')),))


def segmented_partition_family():
    from merlin.transforms.pass_exploration import PassFamily
    return PassFamily('dense_segment_partition',_segmented_pass,(("partition",SEGMENT_PARTITIONS),))


def segmented_residency_family():
    from merlin.transforms.pass_exploration import PassFamily
    return PassFamily('dense_segment_rhs_residency',_segmented_pass,
        (("rhs_residency",('off','boundary')),))
