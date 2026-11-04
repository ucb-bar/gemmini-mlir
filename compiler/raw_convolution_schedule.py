"""Optional retained-input epochs for an unchanged full-width raw CONV producer.

This adapter does not own the per-column narrowing boundary. It selects and
consumes the existing checked conv_policy route, with the same source window,
primitive effects and terminal encoder as ordinary convolution lowering.
"""
from copy import deepcopy
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan

VALUES = ('off', 'retained_input_pairs16', 'supported_geometry')
DECLARATION = dict(values=list(VALUES),
    mechanism='full-width raw pointwise convolution retained-input compute epochs',
    note='Default off. Explicit epochs preserve broad legal admission; supported_geometry narrows to aligned multi-tile CI/CO and at least array_edge squared output rows. Original quantized readout remains outside producer; provisional modeled evidence requires paired coverage.')


def selected():
    return knob('raw_convolution_schedule', default='off', values=VALUES, root=Path(__file__).parent)



# Parent-reported completed paired observation; no missing identity/domain pins
# are invented, and ordinary compilation never reads live feedback. This is a
# provisional modeled result, not an RTL measurement or a coverage certificate.
PROVISIONAL_OBSERVATION = dict(panel='raw026', baseline_cycles=13473245,
    retained_input_pairs16_cycles=9563355, numeric='pass',
    evidence='modeled: Spike timing model (PROVISIONAL)',
    scope='complete modeled whole-call including original column readout')


def geometry_resources(plan):
    """Performance class, separate from options()' independent legality checks.

    Aligned at-least-two-tile channels provide useful input reuse across real
    N tiles and multiple K slices. Requiring at least one primitive square of
    output rows avoids promoting tiny calls from a large-call observation.
    These coarse bounds are hypotheses awaiting paired neighbor coverage; they
    grant neither storage capacity nor permission to remove a completion.
    """
    if type(plan) is not CountedConvolutionPlan:
        raise ValueError('raw shape guard requires original counted convolution plan')
    plan.__post_init__()
    e = plan.primitive
    ci = plan.window.input_shape[-1]
    co = plan.output_channels
    oh, ow = plan.window.output_spatial
    rows = plan.window.input_shape[0]*oh*ow
    conditions = dict(aligned_input_channels=ci % e == 0,
        aligned_output_channels=co % e == 0,
        multiple_reduction_tiles=ci >= 2*e,
        multiple_output_tiles=co >= 2*e,
        nontrivial_output_rows=rows >= e*e)
    return dict(conditions=conditions, supported=all(conditions.values()),
        exclusions=tuple(k for k,v in conditions.items() if not v),
        resources=dict(array_edge=e,input_channels=ci,output_channels=co,
            input_channel_tiles=(ci+e-1)//e,output_channel_tiles=(co+e-1)//e,
            output_rows=rows,minimum_output_rows=e*e),
        qualification='coarse performance applicability only; independent shared schedule legality still required')

def options(plan, command, *, source_attributes=None, mode=None, explicit_options=None):
    """Return existing route options/receipt, or None before any target effect.

    ``command`` is the actual private raw producer command, not its original i8
    consumer. ``source_attributes`` optionally records that original boundary;
    its presence never grants raw arithmetic or moves scales into this stage.
    Explicit parent schedule options take precedence over this optional policy.
    """
    mode = selected() if mode is None else mode
    if mode not in VALUES:
        raise ValueError('unknown raw convolution schedule mode')
    if mode == 'off' or explicit_options is not None:
        return None
    if type(plan) is not CountedConvolutionPlan:
        raise ValueError('raw producer schedule requires original counted convolution plan')
    plan.__post_init__()
    geometry = geometry_resources(plan) if mode == 'supported_geometry' else None
    if geometry is not None and not geometry['supported']:
        return None
    if command.get('opcode') not in ('CONV', 'CONV2D'):
        return None
    attrs = command.get('attributes', {})
    w = plan.window
    if (str(attrs.get('layout', 'nhwc')).lower() != 'nhwc'
        or w.kernel != (1, 1) or w.stride not in ((1, 1), (2, 2))
        or w.padding != (0, 0, 0, 0) or w.dilation != (1, 1)):
        return None
    kernel = tuple(attrs.get('kernel', ()))
    if (kernel[:2] != w.kernel or len(kernel) not in (2, 4)
        or len(kernel) == 4 and kernel[2:] != (w.input_shape[-1], plan.output_channels)
        or tuple(attrs.get('stride', (1, 1))) != w.stride
        or tuple(attrs.get('padding', (0, 0, 0, 0))) != w.padding
        or tuple(attrs.get('dilation', (1, 1))) != w.dilation):
        return None
    stages = attrs.get('epilogue', ())
    if (attrs.get('output_dtype') != 'i32' or set(stages)-{'bias', 'bias_add'}
        or 'acc_scale_by_column' in attrs or float(attrs.get('acc_scale', 1.0)) != 1.0
        or 'requant_shift' in attrs or any(k.startswith('pool_') for k in attrs)
        or any(attrs.get(k, False) for k in ('rhs_transposed','transpose','a_transpose','b_transpose'))):
        return None
    edge = plan.primitive
    tiles = (plan.output_channels + edge - 1)//edge
    if plan.column_tile != edge or not 1 <= tiles <= 32:
        return None
    required = tiles*plan.row_tile
    if plan.accumulator_row + required > plan.accumulator_rows:
        return None
    from conv_policy import completion_laws, selected as current_storage, epoch_selection
    if (current_storage(), epoch_selection()) != ('synchronous', 'off'):
        return None
    # Exact same ordering/source law binding consumed by emit_alternative.
    if completion_laws(plan, plan.identity, tiles, mode='pairs16') is None:
        return None
    return dict(schedule='retained_input', completion_epochs='pairs16',
        receipt=dict(mechanism='full_width_raw_convolution_retained_epochs',
            selection_mode=mode, geometry=geometry,
            performance_observation=PROVISIONAL_OBSERVATION if mode=='supported_geometry' else None,
            plan_sha256=plan.identity, original_window=dict(input_shape=w.input_shape,
                kernel=w.kernel,stride=w.stride,dilation=w.dilation,padding=w.padding),
            raw_attributes=deepcopy(attrs), source_attributes=deepcopy(source_attributes),
            column_tiles=tiles, private_accumulator_rows=required,
            accumulator_rows=plan.accumulator_rows,
            numerical_boundary='unchanged source i8 products and i32 bias/reduction; all original per-column scaling/activation remains after raw producer',
            source_effects='same original window row runs and HWIO weights; complete full-K i32 outputs',
            timing='unmeasured optional schedule; complete-call paired comparison required'))


def emit_selected(plan, builder, lower, *, command, issue, complete,
                  source_attributes=None, mode=None, explicit_options=None):
    """Consume the selected real shared schedule with the parent's effect encoder.

    ``issue`` is the existing non-draining convolution terminal callback;
    ``complete`` is its actual fence callback. Unsupported inputs return None
    before emission so the parent can retain ordinary convolution lowering.
    """
    choice = options(plan, command, source_attributes=source_attributes,
                     mode=mode, explicit_options=explicit_options)
    if choice is None:
        return None
    from conv_policy import emit_alternative
    actual = emit_alternative(plan, builder, lower, issue=issue, complete=complete,
        selection=choice['schedule'], completion_mode=choice['completion_epochs'])
    if actual is None:
        raise ValueError('admitted full-width raw schedule unexpectedly refused shared emission')
    return dict(actual, raw_convolution_schedule=choice['receipt'])
