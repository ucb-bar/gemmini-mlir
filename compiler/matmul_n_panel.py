"""Bounded K-major N panels for a single M tile.

The panel retains independent output tiles in 1024 accumulator rows.
The optional compact layout gives each tile exactly M consecutive rows.
Each reduction tile loads A once, then consumes it with each streamed B tile.
All tensor arithmetic stays on the ordinary WS command path. This changes
traffic and control counts; it does not assert a complete-call cycle benefit.
"""
from math import isfinite
from xdsl.dialects.builtin import StringAttr

from isa_encoder import (ACC_ADDR, ACC_ACCUMULATE, ACC_FULL, GARBAGE_ADDR,
                         config_ex, config_load, config_store, float_bits)
from matmul_lowering import extent, index2, packed


ACC_ROWS = 1024
TILE_ROWS = 16
PANEL_COLUMNS = ACC_ROWS  # 64 N tiles, with a conservative 16-row slot each.


def try_emit_n_panel(e, A, B, C, M, K, N, attrs=None, bias=None,
                     lhs_stride=None, rhs_stride=None, out_stride=None,
                     rhs_resident=None, compute_provenance=None,
                     compact_accumulator=False, retain_lhs=False, wide_rhs=False):
    """Emit a streamed-RHS N panel, or decline before changing the emitter.

    Supported domain: i8 A/B, 1<=M<=16, K>=0, N>0, raw i32 or scalar i8
    readout, optional i32 column bias and nonnegative-scale ReLU. Per-column
    gains, pooling and already resident RHS lifetimes retain their caller's
    existing schedule. Strides use the shared matmul's element units.
    retain_lhs caches the full-K activation sequence above live scratch weights
    when it fits; insufficient capacity preserves the existing streamed loads.
    wide_rhs prepares the bounded N panel in scratchpad with demonstrated
    64-column ordinary MVIN2 commands plus <=16-column tail commands. The
    conservative command width follows the public att_pv reference, rather
    than treating the DMA transaction maximum as a command-width limit.
    """
    attrs = attrs or {}
    if (any(type(x) is not int for x in (M, K, N))
            or not 1 <= M <= TILE_ROWS or K < 0 or N <= 0
            or rhs_resident is not None or 'acc_scale_by_column' in attrs):
        return False
    stages = attrs.get('epilogue', [])
    if not isinstance(stages, (list, tuple)) or any(
            stage not in ('bias', 'bias_add', 'acc_scale', 'relu') for stage in stages):
        return False
    dtype = attrs.get('output_dtype', 'i32')
    if dtype not in ('i8', 'i32') or (dtype == 'i32' and any(
            stage in stages for stage in ('acc_scale', 'relu'))):
        return False
    if any(stage in stages for stage in ('bias', 'bias_add')) and bias is None:
        return False
    scale = attrs.get('acc_scale', 1.0)
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or ('relu' in stages and scale < 0)):
        return False
    try:
        if not isfinite(scale):
            return False
        float_bits(scale)
    except (OverflowError, ValueError):
        return False
    lhs_stride = K if lhs_stride is None else lhs_stride
    rhs_stride = N if rhs_stride is None else rhs_stride
    out_stride = N if out_stride is None else out_stride
    if (any(type(x) is not int for x in (lhs_stride, rhs_stride, out_stride))
            or lhs_stride < K or rhs_stride < N or out_stride < N
            or max(lhs_stride, rhs_stride, out_stride * (4 if dtype == 'i32' else 1)) >= 1 << 32):
        return False

    panel_columns = TILE_ROWS * (ACC_ROWS // M) if compact_accumulator else PANEL_COLUMNS
    b, val = e.b, e.value

    def accumulator_offset(local_n):
        if not compact_accumulator or M == TILE_ROWS:
            return local_n
        return b.mul_i(b.udiv_i(local_n, val(TILE_ROWS)), val(M))

    provenance = compute_provenance or {}
    e.insn('config', *config_ex(dataflow=1))
    e.insn('config', *config_load(lhs_stride, state_id=0))
    e.insn('config', *config_load(rhs_stride, state_id=1))
    e.insn('config', *config_load(0, state_id=2))
    e.insn('config', *config_store(out_stride * (4 if dtype == 'i32' else 1),
                                  scale=scale, activation=int('relu' in stages)))
    overwrite_first = bias is None and K > 0 and not attrs.get('_initialize_accumulator', False)
    lhs_cache = None
    if retain_lhs and K > 0:
        # Scratch rows 0..15 and 16..31 remain the streamed A/B slots.
        # Other source resident lifetimes occupy everything below high water.
        base = max(32, getattr(e, '_gemmini_resident_end', 32))
        tiles = (K + TILE_ROWS - 1) // TILE_ROWS
        if base + tiles * M <= 16384:
            lhs_cache = base
            def fill(k, columns, clear=False):
                address = b.add_i(val(base), b.mul_i(b.udiv_i(k, val(TILE_ROWS)), val(M)))
                if clear:
                    # Clear precisely these M rows, preserving neighboring
                    # cached tiles even when M is smaller than TILE_ROWS.
                    e.insn('mvin', 0, packed(e, address, M, TILE_ROWS))
                    e.fence()
                e.insn('mvin', e.address(e.ptr(A, k)), packed(e, address, M, columns))
            full_k = K // TILE_ROWS * TILE_ROWS
            e.for_loop(0, full_k, TILE_ROWS, lambda k: fill(k, val(TILE_ROWS)))
            if K != full_k:
                fill(val(full_k), val(K - full_k), clear=True)
            # All N panels consume the same completed scratchpad sequence.
            e.fence()

    rhs_panel_base = None
    if wide_rhs and K > 0:
        occupied = max(32, getattr(e, '_gemmini_resident_end', 32))
        if lhs_cache is not None:
            occupied = max(occupied, lhs_cache + ((K + 15) // 16) * M)
        # Keep the demonstrated 16-row block pitch and align the first block.
        base = (occupied + 15) // 16 * 16
        available_columns = (16384 - base) // 16 * 16
        if available_columns >= 64 and N > 16:
            rhs_panel_base = base
            panel_columns = min(panel_columns, available_columns)

    def panel(first_column, width):
        # Local addresses select accumulator rows; dimensions remain M x nn.
        # Compact slots are disjoint [tile*M, (tile+1)*M), and the panel
        # bound keeps every addressed row strictly below ACC_ROWS.
        def initialize(local_n):
            nn = extent(e, width, local_n)
            global_n = b.add_i(first_column, local_n)
            source = 0 if bias is None else e.address(e.ptr(bias, global_n, 'i32'))
            destination = b.add_i(val(ACC_ADDR), accumulator_offset(local_n))
            e.insn('mvin3', source, packed(e, destination, M, nn))
        if not overwrite_first:
            e.for_loop(0, width, TILE_ROWS, initialize)

        def reduction(k, first=False, clear_tail=False):
            kk = extent(e, K, k)
            local_a = 0
            if lhs_cache is None:
                if clear_tail:
                    # Partial loads retain inactive physical lanes. Clear the A
                    # tile before the final K tail, as in the shared schedule.
                    e.insn('mvin', 0, packed(e, 0, TILE_ROWS, TILE_ROWS))
                    e.fence()
                e.insn('mvin', e.address(e.ptr(A, k)), packed(e, 0, M, kk))
            else:
                local_a = b.add_i(val(lhs_cache), b.mul_i(b.udiv_i(k, val(TILE_ROWS)), val(M)))
            if rhs_panel_base is not None:
                def fill_rhs(local_n, columns):
                    global_n = b.add_i(first_column, local_n)
                    source = e.ptr(B, index2(e, k, rhs_stride, global_n))
                    address = b.add_i(val(rhs_panel_base), local_n)
                    e.insn('mvin2', e.address(source), packed(e, address, kk, columns))
                full_width = width // 64 * 64
                e.for_loop(0, full_width, 64, lambda n: fill_rhs(n, val(64)))
                e.for_loop(full_width, width, 16, lambda n: fill_rhs(n, extent(e, width, n)))
                # Publish every block before consuming its real local rows.
                e.fence()
            def columns(local_n):
                nn = extent(e, width, local_n)
                if rhs_panel_base is None:
                    global_n = b.add_i(first_column, local_n)
                    source = e.ptr(B, index2(e, k, rhs_stride, global_n))
                    e.insn('mvin2', e.address(source), packed(e, TILE_ROWS, kk, nn))
                    local_b = TILE_ROWS
                else:
                    local_b = b.add_i(val(rhs_panel_base), local_n)
                destination = b.add_i(val(ACC_ADDR | (0 if first else ACC_ACCUMULATE)), accumulator_offset(local_n))
                e.insn('preload', packed(e, local_b, kk, nn),
                       packed(e, destination, M, nn), **provenance)
                command = e.insn('compute', packed(e, local_a, M, kk),
                                 packed(e, GARBAGE_ADDR, TILE_ROWS, TILE_ROWS), **provenance)
                if lhs_cache is not None:
                    command.attributes['merlin.lhs_residency'] = StringAttr('n_panel_full_k')
                    command.attributes['merlin.lhs_cache'] = StringAttr(
                        f'base={lhs_cache};pitch={M};tiles={(K + TILE_ROWS - 1) // TILE_ROWS}')
            e.for_loop(0, width, TILE_ROWS, columns)
            if rhs_panel_base is not None:
                # Every consumer finishes before the next K tile reuses B rows.
                e.fence()

        start_k = 0
        if overwrite_first and K >= TILE_ROWS:
            reduction(val(0), first=True)
            start_k = TILE_ROWS
        full_k = K // TILE_ROWS * TILE_ROWS
        e.for_loop(start_k, full_k, TILE_ROWS, reduction)
        if K != full_k:
            reduction(val(full_k), first=overwrite_first and full_k == 0, clear_tail=True)

        def readout(local_n):
            nn = extent(e, width, local_n)
            global_n = b.add_i(first_column, local_n)
            source = b.add_i(val(ACC_ADDR | (ACC_FULL if dtype == 'i32' else 0)), accumulator_offset(local_n))
            destination = e.ptr(C, global_n, dtype)
            e.insn('mvout', e.address(destination), packed(e, source, M, nn))
        e.for_loop(0, width, TILE_ROWS, readout)
        # Finish all consumers before reusing accumulator slots for the next
        # N panel, and publish the complete output before returning.
        e.fence()

    full_n = N // panel_columns * panel_columns
    e.for_loop(0, full_n, panel_columns, lambda first: panel(first, panel_columns))
    if N != full_n:
        panel(val(full_n), N - full_n)
    return True
