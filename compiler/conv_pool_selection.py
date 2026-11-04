"""Default-off, source/resource selection of existing pooled-convolution routes.

The finite policy is authored from complete paired Spike timing, not command
counts. Observations below are provenance only: ordinary compilation reads no
live feedback and invokes no profiler. The original emitters retain all source,
physical storage, numerical and completion checks.
"""
from hashlib import sha256
from math import isfinite, prod
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan
from merlin.transforms.pipeline_policy import (PipelinePolicy, PipelineVariant,
    PolicyFacts, PropertyPredicate, RegisteredPass, execute_policy)
from merlin.transforms.pipeline_state import PipelineContext, PipelineSession, PassSpec

VALUES = ('off', 'supported_geometry')
DECLARATION = dict(values=list(VALUES), mechanism='resource-bound pooled convolution route policy',
    note='Off preserves explicit pool settings. Supported geometry chooses existing checked epochs or quantized-once routes; no runtime profiling.')
ROUTES = {
    'incumbent': dict(conv_pool_schedule='off', conv_pool_materialization='off'),
    'epochs': dict(conv_pool_schedule='retained_input_pairs16', conv_pool_materialization='off'),
    'quantized_once': dict(conv_pool_schedule='off', conv_pool_materialization='quantized_once'),
}
# Original observations retain their own compiler/source identities. These
# summaries do not purport to be strict assess_shape_policy identity pins.
OBSERVATIONS = (
    dict(panel='pool_epochs018', baseline=302932816, epochs=101661758,
         scope='complete modeled whole-call', numeric='pass'),
    dict(panel='pool023_small_edge', baseline=329111, epochs=126650, quantized_once=144661,
         scope='complete modeled whole-call', numeric='pass'),
    dict(panel='pool023_small_residual', baseline=692353, epochs=276092, quantized_once=309687,
         scope='complete modeled whole-call', numeric='pass'),
    dict(panel='pool023_large_once', quantized_once=51469086,
         request='15_1793277761028747069_b5b75c00e44debe43e48af7170852cef',
         detail_sha256='296d2a098a96b2c6881b85f5ee7346e91facdd6fcc7d9588fb542974552a7027',
         model_sha256='b3ce9aaf7e8e2227e781f5d0bb3bd597f73ab12cd8c67aff64a5cd7f54347b11',
         scope='complete modeled whole-call', numeric='pass'),
)


def selected():
    return knob('conv_pool_selection', default='off', values=VALUES, root=Path(__file__).parent)


def geometry(plan, attrs, *, workspace_limit_bytes=None):
    """Check the original source and actual selected shared-route prerequisites."""
    if type(plan) is not CountedConvolutionPlan:
        raise ValueError('pool selection requires the original counted convolution plan')
    plan.__post_init__()
    from conv_policy import completion_laws
    from conv_pool_policy import plan_quantized_once
    w = plan.window
    oh, ow = w.output_spatial
    edge = plan.primitive
    tiles = (plan.output_channels + edge - 1)//edge
    scales = attrs.get('acc_scale_by_column')
    scalar = attrs.get('acc_scale', 1.0)
    stages = attrs.get('epilogue', ())
    conditions = dict(
        original_layout=str(attrs.get('layout', 'nhwc')).lower() == 'nhwc',
        original_quantized_output=attrs.get('output_dtype') == 'i8',
        original_scalar_readout=scales is None and isfinite(float(scalar)) and float(scalar) > 0,
        original_epilogue=set(stages) == {'bias_add', 'acc_scale', 'relu', 'maxpool'} and bool(attrs.get('bias')),
        observed_spatial_family=w.kernel == (7, 7) and w.stride == (2, 2)
            and w.dilation == (1, 1) and w.padding == (3, 3, 3, 3),
        observed_pool_family=tuple(attrs.get('pool_size', ())) == (3, 3)
            and tuple(attrs.get('pool_stride', ())) == (2, 2)
            and tuple(attrs.get('pool_padding', ())) == (1, 1, 1, 1)
            and attrs.get('pool_pad_value', -128) == -128,
        exact_prepool_shape=tuple(attrs.get('pool_in_dims', (oh, ow))) == (oh, ow),
        primitive_column_partition=plan.column_tile == edge,
        bounded_channel_expansion=1 <= tiles <= 8,
        complete_patch_capacity=plan.accumulator_row + tiles*9 <= plan.accumulator_rows,
        source_bias=plan.bias,
        actual_ordering_laws=completion_laws(plan, plan.identity, tiles, mode='pairs16') is not None,
    )
    epochs = all(conditions.values())
    required = prod(w.input_shape[:1]) * oh * ow * plan.output_channels
    materialized = None
    refusal = None
    # The empirical crossover separates complete prepool images that outgrow
    # the ACC spatial footprint from the measured small private-patch class.
    # Multi-column tiles also amplify overlapping-patch recomputation. This is
    # resource-relative, not a fixed image-size or capsule-name threshold.
    large = oh*ow > plan.accumulator_rows and tiles >= 2
    if epochs and large:
        if type(workspace_limit_bytes) is not int or workspace_limit_bytes < required:
            refusal = 'caller_mapped_workspace_capacity_unavailable'
        else:
            try:
                materialized = plan_quantized_once(batches=w.input_shape[0], input_hw=(oh, ow),
                    channels=plan.output_channels, pool_size=3, pool_stride=2,
                    padding=(1, 1, 1, 1), pad_value=-128, relu=True)
            except ValueError as error:
                refusal = str(error)
    return dict(conditions=conditions, exclusions=tuple(k for k,v in conditions.items() if not v),
        capabilities=dict(pool_epochs_supported=epochs, pool_once_supported=materialized is not None),
        resources=dict(channel_tiles=tiles, patch_accumulator_rows=tiles*9,
            accumulator_rows=plan.accumulator_rows, prepool_spatial_rows=oh*ow,
            workspace_bytes_required=required, workspace_limit_bytes=workspace_limit_bytes),
        geometry_class='prepool_exceeds_accumulator' if large else 'bounded_prepool_image',
        materialization_refusal=refusal)


def select(plan, attrs, *, emit_route=None, mode=None, workspace_limit_bytes=None,
           explicit_options=None):
    """Select real existing options, optionally invoking their actual emitter.

    ``emit_route(plan, conv_pool_schedule=..., conv_pool_materialization=...)``
    is the parent adapter to the unchanged existing route emitters. If omitted,
    the returned ``options`` is a planning result, not an emitted-work claim.
    Explicit settings have precedence. A materialized selection additionally
    requires a caller-declared mapped workspace byte capacity.
    """
    mode = selected() if mode is None else mode
    if mode not in VALUES:
        raise ValueError('unknown pool selection mode')
    import conv_pool_policy, conv_policy
    current = dict(conv_pool_schedule=conv_pool_policy.selected(),
                   conv_pool_materialization=conv_pool_policy.materialization_selected())
    if explicit_options is not None:
        current.update(explicit_options)
    explicit = explicit_options is not None or current != ROUTES['incumbent'] or (conv_policy.selected(), conv_policy.epoch_selection()) != ('synchronous', 'off')
    def realize(options):
        return emit_route(plan, **options) if emit_route is not None else None
    if mode == 'off' or explicit:
        return dict(mode=mode, reason='explicit_options' if explicit else 'disabled',
                    options=current, emitted=emit_route is not None, actual=realize(current))
    g = geometry(plan, attrs, workspace_limit_bytes=workspace_limit_bytes)
    pin = sha256(Path(__file__).with_name('facts.json').read_bytes()).hexdigest()
    context = PipelineContext(hardware=pin, resources=pin, numerical_policy='original_signed_i8_pool_readout')
    session = PipelineSession(plan, context=context, ir_level='counted-convolution',
        fingerprint=lambda p:p.identity, verify_ir=lambda p:p.__post_init__())
    facts = PolicyFacts(session.stamp(), capabilities=tuple(sorted(g['capabilities'].items())),
        resources=tuple((k,v) for k,v in sorted(g['resources'].items()) if v is not None))
    variants = tuple(PipelineVariant(name, (name,), (PropertyPredicate('capability', cap),))
        for name,cap in (('quantized_once','pool_once_supported'),('epochs','pool_epochs_supported')))
    policy = PipelinePolicy((*variants, PipelineVariant('incumbent', ('incumbent',))), 'incumbent')
    def action(name):
        options = dict(ROUTES[name])
        return lambda _: {'pool_route': dict(options=options, actual=realize(options))}
    registry = {name:RegisteredPass(PassSpec(name, 0, produces=('pool_route',),
        mutates_ir=False, ir_level='counted-convolution'), action(name)) for name in ROUTES}
    result = execute_policy(session, policy, registry, facts, output='pool_route')
    return dict(mode=mode, reason='shared_geometry_policy', options=result.artifact['options'],
        emitted=emit_route is not None, actual=result.artifact['actual'], geometry=g,
        policy=result.receipt, observations=OBSERVATIONS,
        qualification='reviewed generic applicability from complete modeled pairs; no strict identity-pin or RTL promotion claim')
