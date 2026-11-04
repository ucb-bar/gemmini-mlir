"""Compact command parameters represented entirely by integer constants.

The CPU selects configuration words, never tensor values. Keeping parameters in
integer expressions avoids scalar reads of a private metadata array and retains
one accelerator contraction CFG even for long per-column scale schedules.
"""
from dataclasses import dataclass
from . import isa

COMPACT_GROUP_THRESHOLD = 16


def _lookup(f, index, values):
    """Select an exact nonnegative integer from a finite source-derived table.

    Affine tables need no lookup. Other tables use periodic packed bit fields;
    an arbitrary table remains supported by finite masked constant-word selection.
    All resulting values are consumed only as addresses or command parameters.
    """
    values = tuple(values)
    first = values[0]
    step = values[1] - first if len(values) > 1 else 0
    if all(value == first + i * step for i, value in enumerate(values)):
        if step == 0:
            return f.const(first)
        offset = index if step == 1 else f.mul_i(index, f.const(step))
        return offset if first == 0 else f.add_i(f.const(first), offset)
    period = next(p for p in range(1, len(values) + 1)
                  if all(value == values[i % p] for i, value in enumerate(values)))
    if period < len(values):
        index = f.urem_i(index, f.const(period))
        values = values[:period]
    base = min(values)
    deltas = tuple(value - base for value in values)
    combined = 0
    for delta in deltas:
        combined |= delta
    shift = (combined & -combined).bit_length() - 1
    fields = tuple(delta >> shift for delta in deltas)
    bits = max(fields).bit_length()
    per_word = 64 // bits
    words = []
    for start in range(0, len(fields), per_word):
        word = sum(value << (bits * i) for i, value in enumerate(fields[start:start + per_word]))
        words.append(word if word < (1 << 63) else word - (1 << 64))
    if len(words) == 1:
        word = f.const(words[0])
        lane = index
    else:
        chunk = f.udiv_i(index, f.const(per_word))
        lane = f.urem_i(index, f.const(per_word))
        word = f.const(0)
        for i, constant in enumerate(words):
            difference = f.sub_i(chunk, f.const(i))
            negative = f.sub_i(f.const(0), difference)
            mask = f.mask_nonneg(f.or_i(difference, negative))
            word = f.or_i(word, f.and_i(f.const(constant), mask))
    amount = f.mul_i(lane, f.const(bits))
    field = f.and_i(f.lshr_v(word, amount), f.const((1 << bits) - 1))
    delta = f.shl_i(field, shift) if shift else field
    return f.add_i(f.const(base), delta) if base else delta


@dataclass(frozen=True)
class ColumnSchedule:
    columns: tuple
    widths: tuple
    scale_bits: tuple

    def emit(self, f, n, dtype, relu, body):
        def tile(index):
            start = _lookup(f, index, self.columns)
            width = _lookup(f, index, self.widths)
            scale_bits = _lookup(f, index, self.scale_bits)
            isa.config_st(f, n * (4 if dtype == 'i32' else 1),
                          scale_bits=scale_bits, activation=1 if relu else 0)
            body(start, f.add_i(start, width))
        # Peel one actual output tile into a constant parameter path. Besides
        # specializing its configuration, this keeps a consumed nonidentity
        # scale visible to command decoders that cannot resolve integer LUTs.
        # The remaining output tiles stay in at most two compact CPU loops.
        peeled = next((i for i, bits in enumerate(self.scale_bits)
                       if bits != isa.f32_bits(1.0)), 0)
        isa.config_st(f, n * (4 if dtype == 'i32' else 1),
                      scale_bits=self.scale_bits[peeled], activation=1 if relu else 0)
        body(f.const(self.columns[peeled]),
             f.const(self.columns[peeled] + self.widths[peeled]))
        f.for_loop(0, peeled, 1, tile, retain_loop=True)
        f.for_loop(peeled + 1, len(self.columns), 1, tile, retain_loop=True)


def prepare_columns(ctx, scales, groups):
    """Retain one contraction CFG above the small unrolled-schedule threshold."""
    if scales is None or len(groups) <= COMPACT_GROUP_THRESHOLD:
        return None
    records = [(column, min(isa.DIM, end - column), isa.f32_bits(scale))
               for start, end, scale in groups
               for column in range(start, end, isa.DIM)]
    return ColumnSchedule(*(tuple(record[field] for record in records) for field in range(3)))
