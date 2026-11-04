"""Standalone elementwise and movement routes using Gemmini DMA/accumulator.

Only address arithmetic executes on the host. Residual load/readout scales are
admitted by an exhaustive proof over the two signed-byte input domains against
the source's single-rounding f32 reference and its declared error bound.
"""
from functools import lru_cache
from math import isfinite, prod
import struct

try:
    from .isa import ACC, ACCUMULATE, FULL, DIM
except ImportError:
    from isa import ACC, ACCUMULATE, FULL, DIM


def _f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def _clamp(value, relu=False):
    return max(0 if relu else -128, min(127, value))


@lru_cache(maxsize=256)
def residual_plan(lhs_scale, rhs_scale, bound_lsb, relu=False):
    """Return (lhs load scale, rhs load scale, readout scale, proved error).

    A byte load scales, rounds half-even and saturates before the accumulator
    sums it. The readout repeats that rounding after the integer sum. Factoring
    out a shared multiplier can avoid premature load saturation. No stimulus,
    expected outputs, or capsule identity participates in this decision.
    """
    if not all(isfinite(x) for x in (lhs_scale, rhs_scale)):
        raise ValueError('residual scales must be finite')
    if not isinstance(bound_lsb, int) or isinstance(bound_lsb, bool) or bound_lsb < 0:
        raise ValueError('residual bound_lsb must be a declared nonnegative integer')
    sa, sb = _f32(lhs_scale), _f32(rhs_scale)
    values = range(-128, 128)
    ref_a = [_f32(a * sa) for a in values]
    ref_b = [_f32(b * sb) for b in values]
    candidates = [1.0, max(1.0, abs(sa), abs(sb)), 2.0, 4.0, 8.0]
    seen = set()
    for factor in candidates:
        factor = _f32(factor)
        if factor in seen:
            continue
        seen.add(factor)
        la, lb = _f32(sa / factor), _f32(sb / factor)
        load_a = [_clamp(round(_f32(a * la))) for a in values]
        load_b = [_clamp(round(_f32(b * lb))) for b in values]
        output = {s: _clamp(round(_f32(s * factor)), relu) for s in range(-256, 255)}
        worst = 0
        legal = True
        for ai, av in enumerate(load_a):
            if not legal:
                break
            for bi, bv in enumerate(load_b):
                expected = _clamp(round(_f32(ref_a[ai] + ref_b[bi])), relu)
                error = abs(output[av + bv] - expected)
                worst = max(worst, error)
                if error > bound_lsb:
                    legal = False
                    break
        if legal:
            return la, lb, factor, worst
    raise ValueError('no accumulator operand-sum route satisfies declared residual bound_lsb')


def _tiles(builder, size, callback):
    """Compact contiguous DMA rectangles, each at most one array tile."""
    block = DIM * DIM
    bulk = size // block * block
    if bulk:
        with builder.counted_loop(0, bulk, block) as offset:
            callback(offset, DIM, DIM)
    remain = size - bulk
    rows = remain // DIM
    if rows:
        callback(bulk, DIM, rows)
    cols = remain % DIM
    if cols:
        callback(bulk + rows * DIM, cols, 1)


def _spec(cb, name):
    spec = cb.get('tensors', {}).get(name)
    if not isinstance(spec, dict) or 'shape' not in spec or 'dtype' not in spec:
        raise ValueError(f'elementwise tensor {name!r} has no shape/dtype declaration')
    if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in spec['shape']):
        raise ValueError('elementwise requires positive static extents')
    return spec


def _lower_bias_add(builder, cb, command, ptrs):
    """Broadcast a stored i32 column bias through accumulator DMA addition.

    Both matrix and bias loads use the exact full-width path. A zero DMA
    stride repeats the same column vector for every row of a tile. Loading
    into a fresh accumulator then accumulating bias preserves the published
    source matrix even when both source and result are externally visible.
    """
    ops, attrs = command.get('operands', {}), command.get('attributes', {})
    src, bias, dst = (_spec(cb, ops[key]) for key in ('src', 'bias', 'dst'))
    if src['dtype'] != 'i32' or bias['dtype'] != 'i32':
        raise ValueError('bias add requires an i32 accumulator source and i32 bias')
    if src['shape'] != dst['shape'] or not src['shape']:
        raise ValueError('bias add output must retain the source matrix shape')
    n = src['shape'][-1]
    m = prod(src['shape'][:-1])
    if prod(bias['shape']) != n or any(d != 1 for d in bias['shape'][:-1]):
        raise ValueError('bias add requires one broadcast i32 bias per final-axis column')
    stages = attrs.get('epilogue', [])
    if set(stages) - {'bias_add', 'bias', 'acc_scale', 'relu'}:
        raise ValueError('unsupported bias-add readout stage')
    dtype = attrs.get('output_dtype', dst['dtype'])
    if dtype != dst['dtype'] or dtype not in ('i32', 'i8'):
        raise ValueError('bias add requires an i32 or i8 output declaration')
    if dtype == 'i32' and any(stage in stages for stage in ('acc_scale', 'relu')):
        raise ValueError('full-width bias readout cannot apply scale or activation')
    scale = attrs['acc_scale'] if 'acc_scale' in stages else 1.0
    if 'acc_scale_by_column' in attrs:
        raise ValueError('standalone bias add does not yet support per-column scales')
    width = 4 if dtype == 'i32' else 1
    builder.config_ex(activation=int('relu' in stages), scale=scale)
    builder.config_ld(n * 4, shrink=False, id=0)
    builder.config_ld(0, shrink=False, id=1)
    builder.config_st(n * width, activation=int('relu' in stages), scale=scale)
    with builder.counted_loop(0, m, DIM) as row:
        rows = builder.umin(DIM, builder.sub(m, row))
        with builder.counted_loop(0, n, DIM) as col:
            cols = builder.umin(DIM, builder.sub(n, col))
            index = builder.add(builder.mul(row, n), col)
            builder.mvin(builder.ptr_offset(ptrs[ops['src']], builder.mul(index, 4)),
                         ACC, cols, rows, id=0)
            builder.fence()
            builder.mvin(builder.ptr_offset(ptrs[ops['bias']], builder.mul(col, 4)),
                         ACC | ACCUMULATE, cols, rows, id=1)
            builder.fence()
            builder.mvout(builder.ptr_offset(ptrs[ops['dst']], builder.mul(index, width)),
                          ACC | (FULL if dtype == 'i32' else 0), cols, rows)
            builder.fence()
    return True


def _lower_vector_map(builder, cb, command, ptrs):
    """Exact signed-i32 pointwise sum through accumulator load addition.

    Accumulator addition retains the source integer width, including modular
    overflow. The full-width readout copies those bits without f32 conversion,
    scaling, activation or saturation. DMA rectangles cover contiguous logical
    storage independently of the tensor's rank.
    """
    ops, attrs = command.get('operands', {}), command.get('attributes', {})
    combine = attrs.get('combine')
    activation = attrs.get('activation') or []
    if activation:
        raise ValueError('vector map full-width readout cannot apply activation')
    lhs, dst = _spec(cb, ops['lhs']), _spec(cb, ops['dst'])
    if combine == 'identity':
        if lhs['shape'] != dst['shape'] or lhs['dtype'] != dst['dtype']:
            raise ValueError('vector identity requires unchanged shape and dtype')
        return lower_elementwise(builder, cb, {
            'opcode': 'MOVEMENT',
            'operands': {'src': ops['lhs'], 'dst': ops['dst']},
            'attributes': {'semantic': 'copy', 'output_dtype': dst['dtype']},
        }, ptrs)
    if combine != 'add':
        raise ValueError('selected accumulator vector map route supports add or identity')
    rhs = _spec(cb, ops['rhs'])
    if any(spec['dtype'] != 'i32' for spec in (lhs, rhs, dst)):
        raise ValueError('exact vector add requires i32 operands and output')
    if not (lhs['shape'] == rhs['shape'] == dst['shape']):
        raise ValueError('vector add requires equally shaped tensors')
    if attrs.get('output_dtype', 'i32') != 'i32':
        raise ValueError('vector add cannot implicitly narrow the output')
    builder.config_ex()
    builder.config_ld(DIM * 4, shrink=False, id=0)
    builder.config_ld(DIM * 4, shrink=False, id=1)
    builder.config_st(DIM * 4)
    def add(offset, cols, rows):
        byte_offset = builder.mul(offset, 4)
        builder.mvin(builder.ptr_offset(ptrs[ops['lhs']], byte_offset), ACC, cols, rows, id=0)
        builder.fence()
        builder.mvin(builder.ptr_offset(ptrs[ops['rhs']], byte_offset), ACC | ACCUMULATE, cols, rows, id=1)
        builder.fence()
        builder.mvout(builder.ptr_offset(ptrs[ops['dst']], byte_offset), ACC | FULL, cols, rows)
        builder.fence()
    _tiles(builder, prod(dst['shape']), add)
    return True


def lower_elementwise(builder, cb, command, ptrs):
    """Emit supported movement/elementwise commands; return False for other opcodes.

    ptrs maps logical tensor names to builder LLVM pointer values. Output
    declarations are required; unsupported numeric contracts fail closed.
    """
    opcode = command.get('opcode')
    if opcode == 'VECTOR_MAP':
        return _lower_vector_map(builder, cb, command, ptrs)
    if opcode == 'BIAS_ADD':
        return _lower_bias_add(builder, cb, command, ptrs)
    if opcode not in ('MOVEMENT', 'RESIDUAL_ADD'):
        return False
    ops, attrs = command.get('operands', {}), command.get('attributes', {})
    dst = _spec(cb, ops['dst'])
    out_dtype = attrs.get('output_dtype', dst['dtype'])
    if out_dtype != dst['dtype']:
        raise ValueError('elementwise output declaration disagrees with output_dtype')
    size = prod(dst['shape'])
    if opcode == 'MOVEMENT':
        src = _spec(cb, ops['src'])
        if prod(src['shape']) != size or src['dtype'] not in ('i8', 'i32') or out_dtype not in ('i8', 'i32'):
            raise ValueError('movement supports equally sized i8/i32 contiguous tensors')
        if src['dtype'] == 'i32' and out_dtype == 'i8':
            raise ValueError('movement does not implicitly narrow i32 tensors')
        semantic = attrs.get('semantic', 'copy')
        if semantic not in ('copy', 'mvin_mvout'):
            raise ValueError(f'unsupported movement semantic {semantic!r}')
        in_bytes, out_bytes = (1 if src['dtype'] == 'i8' else 4), (1 if out_dtype == 'i8' else 4)
        # i8 copies can use scratchpad. Conversion/full-width copies use acc.
        local = 0 if out_dtype == 'i8' else ACC
        builder.config_ld(DIM * in_bytes, shrink=(in_bytes == 1 and local == ACC))
        builder.config_st(DIM * out_bytes)
        builder.config_ex()
        def move(offset, cols, rows):
            builder.mvin(builder.ptr_offset(ptrs[ops['src']], builder.mul(offset, in_bytes)), local, cols, rows)
            builder.mvout(builder.ptr_offset(ptrs[ops['dst']], builder.mul(offset, out_bytes)), local | (FULL if out_dtype == 'i32' else 0), cols, rows)
        _tiles(builder, size, move)
    else:
        lhs, rhs = _spec(cb, ops['lhs']), _spec(cb, ops['rhs'])
        if lhs['shape'] != rhs['shape'] or lhs['shape'] != dst['shape'] or (lhs['dtype'], rhs['dtype'], out_dtype) != ('i8', 'i8', 'i8'):
            raise ValueError('residual add requires equally shaped signed-byte tensors')
        epilogue = attrs.get('epilogue', [])
        if any(stage != 'relu' for stage in epilogue):
            raise ValueError('residual add only admits relu after its sum')
        relu = 'relu' in epilogue
        la, lb, factor, _ = residual_plan(attrs['lhs_scale'], attrs['rhs_scale'], attrs['bound_lsb'], relu)
        # Serialize accumulator ownership across overwrite, add and readout.
        builder.config_ld(DIM, la, shrink=True, id=0)
        builder.config_ld(DIM, lb, shrink=True, id=1)
        builder.config_ex(activation=int(relu), scale=factor)
        builder.config_st(DIM, activation=int(relu), scale=factor)
        def residual(offset, cols, rows):
            builder.mvin(builder.ptr_offset(ptrs[ops['lhs']], offset), ACC, cols, rows, id=0)
            builder.fence()
            builder.mvin(builder.ptr_offset(ptrs[ops['rhs']], offset), ACC | ACCUMULATE, cols, rows, id=1)
            builder.fence()
            builder.mvout(builder.ptr_offset(ptrs[ops['dst']], offset), ACC, cols, rows)
            builder.fence()
        _tiles(builder, size, residual)
    builder.fence()
    return True
