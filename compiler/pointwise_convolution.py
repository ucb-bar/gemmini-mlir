"""Optional source-proven NHWC 1x1 convolution view into the shared dense emitter.

Admission precedes every target effect. The library retains the original source
window and proves the no-copy spatial row/reduction correspondence. This module
supplies only logical ABI bindings; dense.py retains target storage, completion
and exact integer readout ownership.
"""
from dataclasses import dataclass
from copy import deepcopy
from math import prod, isfinite
from pathlib import Path

from merlin.compile.scheduling.window_transfer import Window2D
from merlin.compile.scheduling.pointwise_convolution import pointwise_dense_view
from merlin.compile.tuning_knobs import knob

VALUES = ('off', 'dense_view')
DECLARATION = dict(values=list(VALUES), mechanism='pointwise_convolution_dense_view',
    note='Source-proven zero-copy NHWC 1x1 view; shared dense scheduling and encoder. Off preserves spatial lowering.')


def selected():
    return knob('pointwise_convolution', default='off', values=VALUES, root=Path(__file__).parent)


@dataclass(frozen=True)
class PointwiseBinding:
    view: object
    lhs: str
    rhs: str
    output: str
    bias: str | None
    attributes: dict
    original_attributes: dict

    def record(self):
        return dict(view=self.view.record(), bindings=dict(lhs=self.lhs, rhs=self.rhs,
            output=self.output, bias=self.bias), original_attributes=self.original_attributes,
            axes=dict(matrix_row='(batch*H+y)*W+x', reduction='input_channel',
                matrix_column='output_channel'),
            byte_equations=dict(input='(((batch*H+y)*W+x)*CI+c) == m*CI+c',
                weight='((0*1+0)*CI+c)*CO+n == c*CO+n', output='m*CO+n'),
            source_effects='same original input/weight/bias reads and complete output writes; no payload copy')


def admit(command, tensors, pointers, *, residents=None):
    """Return a checked target binding, or None without issuing commands."""
    if command.get('opcode') not in ('CONV', 'CONV2D'):
        return None
    a = command.get('attributes', {})
    o = command.get('operands', {})
    if str(a.get('layout', 'nhwc')).lower() != 'nhwc':
        return None
    if any(a.get(k, False) for k in ('rhs_transposed', 'transpose', 'a_transpose', 'b_transpose')):
        return None
    if 'maxpool' in a.get('epilogue', ()) or any(k.startswith('pool_') for k in a):
        return None
    if tuple(a.get('kernel', ()))[:2] != (1, 1) or tuple(a.get('stride', (1, 1))) != (1, 1):
        return None
    if tuple(a.get('padding', (0, 0, 0, 0))) != (0, 0, 0, 0) or tuple(a.get('dilation', (1, 1))) != (1, 1):
        return None
    def operand(*keys):
        return next((o[k] for k in keys if k in o), None)
    lhs = operand('ifm', 'input', 'A', 'lhs', 'src')
    rhs = operand('weight', 'W', 'B', 'rhs')
    output = operand('dst', 'output', 'C', 'Y')
    seen = set()
    while rhs in (residents or {}):
        if rhs in seen:
            return None
        seen.add(rhs)
        rhs = residents[rhs]
        if isinstance(rhs, dict):
            rhs = rhs.get('src', rhs.get('source'))
    if any(x not in tensors or x not in pointers for x in (lhs, rhs, output)):
        return None
    inp, weight, out = (tensors[x] for x in (lhs, rhs, output))
    shape = tuple(inp['shape'])
    if len(shape) != 4 or inp['dtype'] != 'i8' or weight['dtype'] != 'i8':
        return None
    kernel = tuple(a['kernel'])
    if len(kernel) not in (2, 4):
        return None
    ci, co = shape[-1], out['shape'][-1]
    if len(kernel) == 4 and kernel[2:] != (ci, co):
        return None
    # Both accepted physical layouts have exactly the same CI-major/CO-minor
    # byte indexing; a same-size transposed or blocked shape is not enough.
    if tuple(weight['shape']) not in ((ci, co), (1, 1, ci, co)):
        return None
    try:
        window = Window2D(shape, kernel[:2], (1, 1), (1, 1), (0, 0, 0, 0))
        view = pointwise_dense_view(window, co, output_shape=tuple(out['shape']))
    except (ValueError, TypeError):
        return None
    if view is None:
        return None
    dtype = a.get('output_dtype', out['dtype'])
    stages = list(a.get('epilogue', ()))
    if dtype != out['dtype'] or dtype not in ('i8', 'i32'):
        return None
    if set(stages) - {'bias', 'bias_add', 'acc_scale', 'relu'} or 'requant_shift' in a:
        return None
    bias = a.get('bias', o.get('bias'))
    if ('bias' in stages or 'bias_add' in stages) and not bias:
        return None
    if bias and (bias not in pointers or bias not in tensors or tensors[bias]['dtype'] != 'i32'
                 or prod(tensors[bias]['shape']) != co):
        return None
    scales = a.get('acc_scale_by_column')
    if scales is not None and (len(scales) != co or 'acc_scale' in a):
        return None
    values = scales if scales is not None else (a.get('acc_scale', 1.0),)
    if not all(isfinite(float(x)) for x in values):
        return None
    if dtype == 'i32' and ('acc_scale' in stages or 'relu' in stages or any(float(x) != 1.0 for x in values)):
        return None
    attrs = deepcopy(a)
    attrs['output_dtype'] = dtype
    attrs['epilogue'] = [s for s in stages if s not in ('bias', 'bias_add')]
    if bias:
        attrs['epilogue'].insert(0, 'bias_add')
    # Spatial lowering applies its explicit gain even when a producer omitted
    # the redundant stage spelling. Dense's readout uses the stage as authority,
    # so realize that same source gain rather than silently dropping it.
    if dtype == 'i8' and (scales is not None or any(float(x) != 1.0 for x in values)):
        if 'acc_scale' not in attrs['epilogue']:
            at = attrs['epilogue'].index('relu') if 'relu' in attrs['epilogue'] else len(attrs['epilogue'])
            attrs['epilogue'].insert(at, 'acc_scale')
    return PointwiseBinding(view, lhs, rhs, output, bias, attrs, deepcopy(a))


def emit(builder, command, tensors, pointers, *, residents=None,
         schedule_options=None, resident_key=None, enabled=None):
    """Emit the actual shared dense product, or None before all target effects.

    ``enabled=True`` lets an enclosing registered alternative select this route;
    otherwise the module's default-off knob decides. Explicit dense options are
    forwarded unchanged and retain the shared dense policy's precedence.
    """
    if enabled is None:
        enabled = selected() == 'dense_view'
    if not enabled:
        return None
    binding = admit(command, tensors, pointers, residents=residents)
    if binding is None:
        return None
    binding.view.revalidate()
    from dense import lower_dense
    m, k, n = binding.view.dimensions
    attrs = deepcopy(binding.attributes)
    bias_pointer = pointers[binding.bias] if binding.bias else None
    initializer = 'original_bias' if binding.bias else 'overwrite'
    from conv_policy import accumulator_init_selection
    from isa import DIM
    init = accumulator_init_selection()
    if bias_pointer is None and (init == 'zero_accumulate' or init == 'zero_single_slice' and k <= DIM):
        # Real source reduction identity, consumed before the first product.
        bias_pointer = builder.constant_i32_table([0] * n)
        attrs['epilogue'].insert(0, 'bias_add')
        initializer = 'explicit_source_zero'
    result = lower_dense(builder, pointers[binding.lhs], pointers[binding.rhs], pointers[binding.output],
        m, k, n, attrs=attrs, bias_ptr=bias_pointer,
        rhs_transposed=False, schedule_options=schedule_options, resident_key=resident_key)
    return dict(route='pointwise_convolution_dense_view', source=binding.record(),
                source_initializer_realization=initializer,
                dense=result, target_execution=False, timing=None)


lower_pointwise = emit
