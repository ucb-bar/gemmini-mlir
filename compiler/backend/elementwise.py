"""Device movement and standalone operand sums through accumulator DMA.

No tensor arithmetic is performed on the host: the host loops issue DMA commands.
The numeric admission check enumerates the public i8 domain at compile time.
"""
from __future__ import annotations

from functools import lru_cache
import math
import struct

from . import isa


class ElementwiseError(ValueError):
    pass


def _f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def _sat(x):
    return max(-128, min(127, int(x)))


@lru_cache(maxsize=256)
def residual_numeric_screen(lhs_scale, rhs_scale, bound_lsb, relu=False):
    """Prove separately rounded/saturated DMA loads fit the declared source bound.

    Source residual arithmetic rounds once after the f32 sum. Hardware rounds
    and saturates each operand before accumulating. These are distinct contracts.
    """
    a, b = _f32(lhs_scale), _f32(rhs_scale)
    if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < 0:
        raise ElementwiseError('operand-sum requires finite nonnegative f32 scales')
    if type(bound_lsb) is not int or bound_lsb < 0:
        raise ElementwiseError('operand-sum requires an explicit nonnegative integer error bound')
    readout = max(1.0, a, b)
    load_a, load_b = _f32(a / readout), _f32(b / readout)
    ap = [_f32(x * a) for x in range(-128, 128)]
    bp = [_f32(x * b) for x in range(-128, 128)]
    aq = [_sat(round(_f32(x * load_a))) for x in range(-128, 128)]
    bq = [_sat(round(_f32(x * load_b))) for x in range(-128, 128)]
    worst = 0
    for i, ax in enumerate(ap):
        for j, bx in enumerate(bp):
            reference = round(_f32(ax + bx))
            device = aq[i] + bq[j]
            if relu:
                reference, device = max(0, reference), max(0, device)
            device = round(_f32(device * readout))
            error = abs(_sat(reference) - _sat(device))
            worst = max(worst, error)
            if error > bound_lsb:
                raise ElementwiseError(f'operand-sum load rounding exceeds declared bound {bound_lsb}: pair {(i-128,j-128)} error {error}')
    return {'max_error_lsb': worst, 'pairs_checked': 65536, 'lhs_load_f32': load_a,
            'rhs_load_f32': load_b, 'readout_scale_f32': readout}


def _shape(ctx, name):
    shape = ctx.spec(name)['shape']
    if not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ElementwiseError('elementwise lowering requires positive static tensor extents')
    return tuple(shape)


def _dtype(ctx, name):
    return ctx.spec(name)['dtype']


def _tiles(ctx, count, body):
    """Use contiguous flattened tiles; rank and logical row boundaries are immaterial."""
    full = count // 256 * 256
    if full:
        ctx.fb.for_loop(0, full, 256, lambda offset: body(offset, 16, 16), retain_loop=True)
    rows, cols = divmod(count - full, 16)
    if rows:
        body(full, rows, 16)
    if cols:
        body(full + rows * 16, 1, cols)


def _validate_epilogue(attributes, *, allow_scale=True):
    epilogue = attributes.get('epilogue', [])
    allowed = {'relu', 'acc_scale'} if allow_scale else {'relu'}
    if any(stage not in allowed for stage in epilogue):
        raise ElementwiseError(f'unsupported standalone epilogue {epilogue!r}')
    if len(epilogue) != len(set(epilogue)):
        raise ElementwiseError('repeated standalone epilogue stages require a numeric proof')
    scale = attributes.get('acc_scale', 1.0) if 'acc_scale' in epilogue else 1.0
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
        raise ElementwiseError('standalone readout requires a finite nonnegative scalar scale')
    return float(scale), int('relu' in epilogue)


def lower_movement(ctx, operands, attributes):
    src, dst = operands['src'], operands['dst']
    if _shape(ctx, src) != _shape(ctx, dst):
        raise ElementwiseError('movement requires matching source and output shapes')
    if attributes.get('semantic', 'mvin_mvout') not in ('mvin_mvout', 'copy'):
        raise ElementwiseError('movement semantic has no selected hardware form')
    inp, out = _dtype(ctx, src), _dtype(ctx, dst)
    if inp not in ('i8', 'i32') or out not in ('i8', 'i32'):
        raise ElementwiseError(f'movement dtype route {inp}->{out} is unsupported')
    scale, activation = _validate_epilogue(attributes)
    if out == 'i32' and (activation or scale != 1):
        raise ElementwiseError('raw i32 readout does not apply scale or activation')
    from .strided_elementwise import plan_strided_inputs, emit_strided_movement
    strided = plan_strided_inputs(ctx, (src,), dst, _shape(ctx, src), attributes=attributes)
    if strided is not None:
        receipt = emit_strided_movement(ctx, src, dst, strided, scale, activation)
        ctx.metadata.setdefault('strided_movement', []).append(receipt)
        return receipt
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    epochs = knob('ELEMENTWISE_EPOCHS', default='off', values=('off', 'on'),
                  root=Path(__file__).resolve().parents[1])
    if epochs == 'on':
        from .elementwise_movement_epochs import emit_batched_movement
        receipt = emit_batched_movement(ctx, src, dst, _shape(ctx, src), scale, activation)
        if receipt is not None:
            ctx.metadata.setdefault('movement_epochs', []).append(receipt)
            return receipt
    ctx.emit(*isa.config_ld(16 * (1 if inp == 'i8' else 4), shrink=inp == 'i8'))
    ctx.emit(*isa.config_st(16 * (1 if out == 'i8' else 4), scale=scale, activation=activation))
    def tile(offset, rows, cols):
        ctx.emit(*isa.mvin(ctx.ptr(src, offset), isa.acc_addr(0), rows=rows, cols=cols))
        ctx.fence()  # Finish the accumulator write before readout reads it.
        ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(0, full=out == 'i32'), rows=rows, cols=cols))
        ctx.fence()  # Finish readout before the next tile reuses these rows.
    _tiles(ctx, math.prod(_shape(ctx, src)), tile)
    ctx.fence()


def lower_residual(ctx, operands, attributes):
    lhs, rhs, dst = operands['lhs'], operands['rhs'], operands['dst']
    shape = _shape(ctx, lhs)
    if _shape(ctx, rhs) != shape or _shape(ctx, dst) != shape:
        raise ElementwiseError('operand-sum requires identical contiguous tensor shapes')
    if any(_dtype(ctx, name) != 'i8' for name in (lhs, rhs, dst)):
        raise ElementwiseError('selected standalone operand-sum supports i8 operands and output')
    _, activation = _validate_epilogue(attributes, allow_scale=False)
    proof = residual_numeric_screen(float(attributes.get('lhs_scale', 1.0)),
                                    float(attributes.get('rhs_scale', 1.0)),
                                    attributes.get('bound_lsb', 0), bool(activation))
    from .strided_elementwise import plan_strided_inputs, emit_strided_residual
    strided = plan_strided_inputs(ctx, (lhs, rhs), dst, shape, attributes=attributes)
    if strided is not None:
        original = {'lhs_scale': float(attributes.get('lhs_scale', 1.0)),
                    'rhs_scale': float(attributes.get('rhs_scale', 1.0)),
                    'bound_lsb': attributes.get('bound_lsb', 0), 'relu': bool(activation)}
        receipt = emit_strided_residual(ctx, lhs, rhs, dst, strided, proof, activation, original)
        ctx.metadata.setdefault('strided_residual', []).append(receipt)
        return dict(proof, strided=receipt)
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    epochs = knob('ELEMENTWISE_EPOCHS', default='off', values=('off', 'on'),
                  root=Path(__file__).resolve().parents[1])
    if epochs == 'on':
        from .elementwise_epochs import emit_batched_residual
        original_proof = dict(proof, original_contract={
            'lhs_scale': float(attributes.get('lhs_scale', 1.0)),
            'rhs_scale': float(attributes.get('rhs_scale', 1.0)),
            'bound_lsb': attributes.get('bound_lsb', 0), 'relu': bool(activation)})
        receipt = emit_batched_residual(ctx, lhs, rhs, dst, shape, original_proof, activation)
        if receipt is not None:
            ctx.metadata.setdefault('residual_epochs', []).append(receipt)
            return dict(proof, epochs=receipt)
    ctx.emit(*isa.config_ld(16, channel=0, scale=proof['lhs_load_f32'], shrink=True))
    ctx.emit(*isa.config_ld(16, channel=1, scale=proof['rhs_load_f32'], shrink=True))
    ctx.emit(*isa.config_st(16, scale=proof['readout_scale_f32'], activation=activation))
    def tile(offset, rows, cols):
        ctx.emit(*isa.mvin(ctx.ptr(lhs, offset), isa.acc_addr(0), rows=rows, cols=cols, channel=0))
        # Independent DMA queues may release younger loads at issue. The seed
        # must complete before an accumulating load writes the same rows.
        ctx.fence()
        ctx.emit(*isa.mvin(ctx.ptr(rhs, offset), isa.acc_addr(0, accumulate=True), rows=rows, cols=cols, channel=1))
        ctx.fence()
        ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(0), rows=rows, cols=cols))
        ctx.fence()  # Readout completes before local-row reuse on the next tile.
    _tiles(ctx, math.prod(shape), tile)
    ctx.fence()
    return proof


def lower_bias(ctx, operands, attributes):
    """Add an i32 column vector with accumulator DMA, preserving i32 wrap.

    Full-width loads pass their lanes through unchanged. The second load uses
    stride zero to broadcast one bias row and accumulator metadata to add it.
    Raw full-width readout preserves the selected i32 addition semantics.
    """
    src, bias, dst = operands['src'], operands['bias'], operands['dst']
    shape = _shape(ctx, src)
    if len(shape) != 2 or _shape(ctx, dst) != shape:
        raise ElementwiseError('standalone bias requires matching rank-two source and output shapes')
    if _shape(ctx, bias) != (shape[-1],):
        raise ElementwiseError('standalone bias requires one bias per source column')
    if any(_dtype(ctx, name) != 'i32' for name in (src, bias, dst)):
        raise ElementwiseError('standalone exact bias route requires i32 source, bias and result')
    scale, activation = _validate_epilogue(attributes)
    if activation or scale != 1 or attributes.get('output_dtype', 'i32') != 'i32':
        raise ElementwiseError('standalone full-width bias output cannot apply a narrowing epilogue')
    from .strided_elementwise import plan_strided_inputs
    strided = plan_strided_inputs(ctx, (src,), dst, shape, attributes=attributes)
    if strided is not None:
        from .strided_bias import emit_strided_bias
        receipt = emit_strided_bias(ctx, src, bias, dst, strided)
        ctx.metadata.setdefault('strided_bias', []).append(receipt)
        return receipt
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    epochs = knob('BIAS_EPOCHS', default='off', values=('off', 'on'),
                  root=Path(__file__).resolve().parent.parent)
    if epochs == 'on':
        from .elementwise_readout_epochs import emit_batched_bias
        receipt = emit_batched_bias(ctx, src, bias, dst, shape, scale, activation,
                                    attributes.get('output_dtype', 'i32'))
        if receipt is not None:
            ctx.metadata.setdefault('bias_epochs', []).append(receipt)
            return receipt
    m, n = shape
    ctx.fence()
    ctx.emit(*isa.config_ld(n * 4, channel=0, shrink=False))
    ctx.emit(*isa.config_ld(0, channel=1, shrink=False))
    ctx.emit(*isa.config_st(n * 4))
    fb = ctx.fb
    def emit_tile(row, col, rows, cols):
        row = fb.const(row) if isinstance(row, int) else row
        col = fb.const(col) if isinstance(col, int) else col
        offset = fb.add_i(fb.mul_i(row, fb.const(n)), col)
        ctx.emit(*isa.mvin(ctx.ptr(src, offset), isa.acc_addr(0), rows=rows, cols=cols, channel=0))
        ctx.fence()
        ctx.emit(*isa.mvin(ctx.ptr(bias, col), isa.acc_addr(0, accumulate=True), rows=rows, cols=cols, channel=1))
        ctx.fence()
        ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(0, full=True), rows=rows, cols=cols))
        ctx.fence()
    def row_tile(row, rows):
        full_n = n // 16 * 16
        if full_n:
            fb.for_loop(0, full_n, 16, lambda col: emit_tile(row, col, rows, 16), retain_loop=True)
        if n % 16:
            emit_tile(row, full_n, rows, n % 16)
    full_m = m // 16 * 16
    if full_m:
        fb.for_loop(0, full_m, 16, lambda row: row_tile(row, 16), retain_loop=True)
    if m % 16:
        row_tile(full_m, m % 16)
    ctx.fence()


def lower_readout(ctx, operands, attributes):
    """Standalone accumulator-domain scale, ReLU and saturating conversion."""
    if _dtype(ctx, operands['src']) != 'i32':
        raise ElementwiseError('standalone accumulator readout requires i32 input')
    if attributes.get('acc_scale_by_column') is not None:
        from .elementwise_readout_epochs import lower_column_readout
        receipt = lower_column_readout(ctx, operands['src'], operands['dst'],
                                      _shape(ctx, operands['src']), attributes)
        ctx.metadata.setdefault('standalone_column_readout', []).append(receipt)
        return receipt
    return lower_movement(ctx, operands, attributes)
