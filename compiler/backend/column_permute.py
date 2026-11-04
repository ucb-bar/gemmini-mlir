"""Device-only column layout for exact per-column accumulator scales.

Source constants select a permutation; tensor payloads move only through DMA.
The common matmul lowerer computes in that private layout and completes before
an inverse DMA copy restores every output column to its logical position.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from xdsl.dialects.builtin import i8, i32

from . import isa
from .column_schedule import _lookup
from .tile_bias import has_bias, select_bias
from .tile_epilogue import select_readout


@dataclass(frozen=True)
class ColumnPermutation:
    groups: tuple[tuple[int, ...], ...]
    scale_bits: tuple[int, ...]

    @property
    def columns(self):
        return tuple(column for group in self.groups for column in group)

    @property
    def scales(self):
        return [struct.unpack('<f', struct.pack('<I', bits))[0]
                for group, bits in zip(self.groups, self.scale_bits)
                for _ in group]


def select_permutation(ctx, m, k, n, attrs):
    """Prove semantics before allocating or emitting any preparation command.

    The common matmul API supplies signed-i8 row-major A/B pointers. Named bias
    operands additionally retain their declared signed-i32 vector type proof.
    Prepared/resident sources and nonzero initial destinations keep their
    existing lowering. This explicit experiment has no profitability cutoff.
    """
    if (any(type(d) is not int or d <= 0 for d in (m, k, n))
            or attrs.get('output_dtype', 'i32') != 'i8'
            or attrs.get('_column_layout_prepared')
            or any(attrs.get(key) is not None for key in
                   ('_resident_weight', '_activation', '_compact_columns'))):
        return None
    boundary = attrs.get('source_numeric_boundary')
    if isinstance(boundary, dict) and boundary.get('destination_constants', [0]) != [0]:
        return None
    stride = attrs.get('_lhs_row_stride', k)
    if type(stride) is not int or stride < k or n * 4 >= 1 << 32:
        return None
    bias = None
    if has_bias(attrs):
        if not bool(attrs.get('bias_broadcast', True)):
            return None
        bias = select_bias(attrs, m, n, ctx.cb['tensors'])
        if bias is None:
            return None
    scales = attrs.get('acc_scale_by_column')
    if not isinstance(scales, (list, tuple)) or len(scales) != n:
        return None
    scalar = dict(attrs)
    scalar.pop('acc_scale_by_column')
    if select_readout(scalar, n, bias=bias) is None:
        return None
    try:
        values = tuple(float(value) for value in scales)
        bits = tuple(isa.f32_bits(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not math.isfinite(value) or word & 0x7F800000 == 0x7F800000
           for value, word in zip(values, bits)):
        return None
    grouped = {}
    for column, word in enumerate(bits):
        grouped.setdefault(word, []).append(column)
    permutation = ColumnPermutation(tuple(tuple(group) for group in grouped.values()),
                                    tuple(grouped))
    if permutation.columns == tuple(range(n)):
        return None
    return permutation



def should_permute(ctx, m, k, n, attrs):
    """Pure auto policy over source geometry and exact scale-bit groups.

    Semantic admission remains in select_permutation. The coarse profitability
    hypothesis requires a large contraction and at least one complete tile per
    group; explicit permutation is independent of this auto policy.
    """
    if any(type(d) is not int or d < 128 for d in (m, k, n)):
        return False
    permutation = select_permutation(ctx, m, k, n, attrs)
    return (permutation is not None
            and all(len(group) >= isa.DIM for group in permutation.groups))


def _private(ctx, stem, shape, dtype):
    """Fresh invocation-local storage; no cross-call physical identity cache."""
    counter = getattr(ctx, '_column_permutation_count', 0)
    name = '__column_permute_' + stem + '_' + str(counter)
    while name in ctx.cb['tensors'] or name in getattr(ctx, 'pointers', {}):
        name += '_'
    return ctx.alloc(name, shape, dtype)


def _copy_columns(f, source, target, rows, n, permutation, *, scatter=False):
    """Copy signed bytes by columns, preserving original row-major stride N."""
    isa.config_ld(f, n, state=0)
    isa.config_st(f, n, scale=1.0, activation=0)
    offset = 0
    for group in permutation.groups:
        start = offset
        def column(index, group=group, start=start):
            logical = _lookup(f, index, group)
            physical = f.add_i(f.const(start), index)
            source_column, target_column = ((physical, logical) if scatter
                                            else (logical, physical))
            def row(row):
                count = f.smin(f.const(isa.DIM), f.sub_i(f.const(rows), row))
                base = f.mul_i(row, f.const(n))
                src = f.add_i(base, source_column)
                dst = f.add_i(base, target_column)
                isa.mvin(f, f.gep(source, src, i8), 0, 1, count, state=0)
                # Raw scratchpad readout copies signed bytes without applying
                # accumulator scale/saturation/ReLU a second time.
                isa.mvout(f, f.gep(target, dst, i8), 0, 1, count)
            f.for_loop(0, rows, isa.DIM, row, retain_loop=True)
        f.for_loop(0, len(group), 1, column, retain_loop=True)
        offset += len(group)


def _copy_bias(f, source, target, n, permutation):
    """Preserve all32 bias bits using nonshrinking accumulator DMA/readout."""
    isa.config_ld(f, n * 4, state=2, shrink=False)
    isa.config_st(f, n * 4, scale=1.0, activation=0)
    offset = 0
    for group in permutation.groups:
        start = offset
        def column(index, group=group, start=start):
            logical = _lookup(f, index, group)
            physical = f.add_i(f.const(start), index)
            isa.mvin(f, f.gep(source, logical, i32), 0x80000000, 1, 1, state=2)
            isa.mvout(f, f.gep(target, physical, i32), 0xA0000000, 1, 1)
        f.for_loop(0, len(group), 1, column, retain_loop=True)
        offset += len(group)


@dataclass(frozen=True)
class PreparedColumns:
    """One explicitly owned, immutable RHS layout within a lowering context.

    No global or pointer-identity cache exists. The caller must ensure that its
    source RHS/broadcast bias stay read-only across the descriptor's consumers.
    Preparation's completion fence publishes both private copies before use.
    """
    owner: object
    permutation: ColumnPermutation
    rhs: object
    k: int
    n: int
    attributes: dict


def prepare_rhs(ctx, b, m, k, n, attrs):
    """Gather immutable RHS/bias once, or return None without IR effects."""
    permutation = select_permutation(ctx, m, k, n, attrs)
    if permutation is None:
        return None
    attrs = dict(attrs)
    f = ctx.f
    packed_b = _private(ctx, 'rhs', [k, n], 'i8')
    _copy_columns(f, b, packed_b, k, n, permutation)
    if has_bias(attrs):
        source = ctx.ptr(attrs['bias'])
        packed_bias = _private(ctx, 'bias', [n], 'i32')
        _copy_bias(f, source, packed_bias, n, permutation)
        attrs['bias'] = packed_bias
    isa.fence(f)
    ctx._column_permutation_count = getattr(ctx, '_column_permutation_count', 0) + 1
    attrs['acc_scale_by_column'] = permutation.scales
    attrs['_column_layout_prepared'] = True
    return PreparedColumns(ctx, permutation, packed_b, k, n, attrs)


def lower_prepared(ctx, a, c, m, prepared, *, lhs_row_stride=None):
    """Compute/scatter one row block using an explicitly supplied RHS layout.

    Each consumer completes its private output and scatter before returning;
    packed-A preparation and later consumers may therefore reuse local storage.
    The descriptor never escapes its originating function/context.
    """
    if (not isinstance(prepared, PreparedColumns) or prepared.owner is not ctx
            or type(m) is not int or m <= 0):
        raise ValueError('prepared column layout requires its owning context and positive M')
    from .matmul import lower_matmul
    attrs = dict(prepared.attributes)
    if lhs_row_stride is not None:
        if type(lhs_row_stride) is not int or lhs_row_stride < prepared.k:
            raise ValueError('prepared column layout has invalid LHS row stride')
        attrs['_lhs_row_stride'] = lhs_row_stride
    packed_c = _private(ctx, 'output', [m, prepared.n], 'i8')
    lower_matmul(ctx, a, prepared.rhs, packed_c, m, prepared.k, prepared.n, attrs)
    _copy_columns(ctx.f, packed_c, c, m, prepared.n,
                  prepared.permutation, scatter=True)
    isa.fence(ctx.f)


def lower_permuted(ctx, a, b, c, m, k, n, attrs):
    """Emit preparation, common matmul and scatter; False has no IR effects."""
    prepared = prepare_rhs(ctx, b, m, k, n, attrs)
    if prepared is None:
        return False
    lower_prepared(ctx, a, c, m, prepared)
    return True
