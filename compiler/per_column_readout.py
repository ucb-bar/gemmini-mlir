"""Device-only per-column f32 scaled readout of a row-major raw i32 matrix.

The caller owns the raw producer and its storage lifetime. No tensor values
are read by the host: ordinary accumulator DMA copies each column group and
Gemmini performs its selected half-even, saturating i8 scaled readout.
"""
from math import isfinite

from isa_encoder import ACC_ADDR, config_load, config_store, float_bits, pack_matrix


def emit_per_column_readout(e, raw_i32_ptr, out_i8_ptr, M, N, scales,
                            activation=0, out_stride=None, loop_repeated=False,
                            accumulator_panel=False):
    """Read raw[M,N] into strided i8 output using explicit source column gains.

    ``activation=1`` admits source post-scale ReLU only for nonnegative gains,
    where it commutes with the device's pre-scale ReLU. ``out_stride`` is in
    bytes (i8 elements); raw rows are contiguous N-element i32 rows. Adjacent
    equal f32 gains share a <=16-column DMA/readout, without crossing a tile.
    All validation precedes emission. Empty M emits no device commands.

    This helper clobbers load state 2, store configuration, and accumulator
    rows 0..15, or rows 0..1023 with ``accumulator_panel=True``. The caller
    must end any earlier accumulator lifetime first.
    The publication fence precedes raw input DMA, and each tile's completion
    fence makes accumulator reuse and the final DRAM output safe on return.
    Panel mode gathers disjoint M tiles before reading them out and fences
    once per at-most-1024-row panel; it does not change per-column gains.
    """
    if type(accumulator_panel) is not bool:
        raise ValueError('accumulator_panel must be a boolean')
    if type(M) is not int or M < 0 or type(N) is not int or N <= 0:
        raise ValueError('per-column readout requires static M>=0 and N>0')
    if type(activation) is not int or activation not in (0, 1):
        raise ValueError('per-column readout supports identity or ReLU activation')
    out_stride = N if out_stride is None else out_stride
    if (type(out_stride) is not int or not N <= out_stride < 1 << 32
            or N * 4 >= 1 << 32):
        raise ValueError('per-column readout byte strides must fit CONFIG fields')
    if not isinstance(scales, (list, tuple)) or len(scales) != N:
        raise ValueError('per-column readout needs exactly N explicit gains')
    bits = []
    for scale in scales:
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not isfinite(scale) or (activation and scale < 0)):
            raise ValueError('per-column readout gain violates finite f32/ReLU domain')
        try:
            encoded = float_bits(scale)
        except (OverflowError, ValueError) as error:
            raise ValueError('per-column readout gain is not representable in f32') from error
        if encoded & 0x7f800000 == 0x7f800000:
            raise ValueError('per-column readout gain is not finite f32')
        bits.append(encoded)
    groups = []
    start = 0
    while start < N:
        end = start + 1
        limit = min(N, (start // 16 + 1) * 16)
        while end < limit and bits[end] == bits[start]:
            end += 1
        groups.append((start, end - start, scales[start]))
        start = end
    if M == 0:
        return

    b, val = e.b, e.value
    e.fence()
    e.insn('config', *config_load(4 * N, state_id=2, shrink=False))
    def column_group(column, width, gain):
        e.insn('config', *config_store(out_stride, scale=gain,
                                      activation=activation))
        if accumulator_panel:
            # Selected accumulator geometry: 1024 rows, 16 i32 lanes per row.
            # Every tile has its own contiguous row interval. The fence after
            # all readouts ends every consumer before any interval is reused.
            def panel(first_row, panel_count):
                def tile(local_row, count, load):
                    row = b.add_i(first_row, local_row)
                    packed = b.add_i(val(pack_matrix(ACC_ADDR, count, width)), local_row)
                    if load:
                        source_index = b.add_i(b.mul_i(row, val(N)), e.value(column))
                        e.insn('mvin3', e.address(e.ptr(raw_i32_ptr, source_index, 'i32')), packed)
                    else:
                        target_index = b.add_i(b.mul_i(row, val(out_stride)), e.value(column))
                        e.insn('mvout', e.address(e.ptr(out_i8_ptr, target_index)), packed)
                full_rows = panel_count // 16 * 16
                for load in (True, False):
                    e.for_loop(0, full_rows, 16, lambda row: tile(row, 16, load))
                    if full_rows != panel_count:
                        tile(val(full_rows), panel_count - full_rows, load)
                e.fence()
            full_panels = M // 1024 * 1024
            e.for_loop(0, full_panels, 1024, lambda row: panel(row, 1024))
            if full_panels != M:
                panel(val(full_panels), M - full_panels)
            return
        def rows(row, count):
            source_index = b.add_i(b.mul_i(row, val(N)), e.value(column))
            target_index = b.add_i(b.mul_i(row, val(out_stride)), e.value(column))
            e.insn('mvin3', e.address(e.ptr(raw_i32_ptr, source_index, 'i32')),
                   pack_matrix(ACC_ADDR, count, width))
            e.insn('mvout', e.address(e.ptr(out_i8_ptr, target_index)),
                   pack_matrix(ACC_ADDR, count, width))
            e.fence()
        full = M // 16 * 16
        e.for_loop(0, full, 16, lambda row: rows(row, 16))
        if M != full:
            rows(val(full), M - full)

    from column_group_schedule import emit_column_groups
    if loop_repeated:
        emit_column_groups(e, groups, column_group, loop_repeated=True)
    else:
        # Preserve original constant construction order on the default path.
        for column, width, gain in groups:
            column_group(column, width, gain)
