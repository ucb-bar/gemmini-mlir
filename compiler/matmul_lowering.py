"""Ordinary-command weight-stationary matrix schedules.

All loops are scalar runtime control; arithmetic on tensor values is performed
by Gemmini. Ragged operands use the command's explicit row/column extents.
"""
from pathlib import Path
from xdsl.dialects.builtin import StringAttr
from merlin.compile.tuning_knobs import knob
from schedule_policy import select_schedule

from isa_encoder import (ACC_ADDR, ACC_ACCUMULATE, ACC_FULL, GARBAGE_ADDR,
                         config_ex, config_load, config_store, float_bits)


def packed(e, addr, rows=16, cols=16):
    b = e.b
    v = lambda x: e.value(x)
    return b.or_i(v(addr), b.or_i(b.shl_i(v(cols), 32), b.shl_i(v(rows), 48)))


def extent(e, total, iv):
    """Extent at an in-range tile start, aligned to the 16-row/column tile."""
    if total > 0 and total % 16 == 0:
        return e.value(16)
    return e.b.smin(e.b.sub_i(e.value(total), iv), e.value(16))


def index2(e, row, stride, col):
    return e.b.add_i(e.b.mul_i(e.value(row), e.value(stride)), e.value(col))


def load_resident_rhs(e, B, K, N, base=32):
    """Pack a complete RHS into tiled scratchpad rows, once per resident lifetime."""
    nt = (N + 15) // 16
    e.insn('config', *config_load(N, state_id=1))
    def columns(n):
        nn = extent(e, N, n)
        def reduction(k):
            kk = extent(e, K, k)
            tile = e.b.add_i(e.b.mul_i(e.b.udiv_i(k, e.value(16)), e.value(nt)),
                             e.b.udiv_i(n, e.value(16)))
            local = e.b.add_i(e.value(base), e.b.mul_i(tile, e.value(16)))
            ptr = e.ptr(B, index2(e, k, N, n))
            e.insn('mvin2', e.address(ptr), packed(e, local, kk, nn))
        e.for_loop(0, K, 16, reduction)
    e.for_loop(0, N, 16, columns)
    return (base, nt)


def emit_matmul(e, A, B, C, M, K, N, attrs=None, bias=None,
                lhs_stride=None, rhs_stride=None, out_stride=None,
                rhs_resident=None, compute_provenance=None, lhs_panel_loader=None):
    """Emit A[M,K] @ B[K,N] and its admitted integer readout epilogue.

    Pointers are LLVM SSA values; strides are static element strides. Resident
    RHS is (scratchpad_base_row, number_of_column_tiles), as returned above.
    Bias is i32[N], broadcast through DMA's zero row stride.
    An optional lhs_panel_loader(e, base, first_m, rows, k_tiles) fills the
    existing M-panel scratch layout before its device compute consumers.
    """
    attrs = attrs or {}
    initialize_accumulator = bool(attrs.get('_initialize_accumulator', False))
    thin_order = knob('thin_matmul_order', default='incumbent', values=('incumbent', 'n_panel', 'regional', 'compact_n_panel', 'cached_n_panel', 'cached_compact_n_panel', 'wide_n_panel', 'wide_cached_n_panel'), root=Path(__file__).parent)
    thin_dtype = attrs.get('output_dtype', 'i32')
    regional_readout = ((thin_dtype == 'i32' and (N <= 4096 or (K >= 2048 and N <= 32768)))
                        or (thin_dtype == 'i8' and M == 16 and N <= 16384)
                        or (thin_dtype == 'i8' and M == 1 and K >= 2048 and 2048 <= N <= 16384))
    use_n_panel = thin_order in ('n_panel', 'compact_n_panel', 'cached_n_panel', 'cached_compact_n_panel', 'wide_n_panel', 'wide_cached_n_panel') or (
        thin_order == 'regional' and 1 <= M <= 16 and K > 0 and N > 16
        and regional_readout and M * K * N >= 2**20)
    if use_n_panel and lhs_panel_loader is None:
        from matmul_n_panel import try_emit_n_panel
        retain_n_lhs = thin_order in ('cached_n_panel', 'cached_compact_n_panel', 'wide_cached_n_panel') or (
            thin_order == 'regional' and ((thin_dtype == 'i32' and N > 1024)
                                        or (K >= 2048 and N >= 1024)))
        if try_emit_n_panel(e, A, B, C, M, K, N, attrs, bias=bias,
                            lhs_stride=lhs_stride, rhs_stride=rhs_stride,
                            out_stride=out_stride, rhs_resident=rhs_resident,
                            compute_provenance=compute_provenance,
                            compact_accumulator=thin_order in ('compact_n_panel', 'cached_compact_n_panel'),
                            retain_lhs=retain_n_lhs,
                            wide_rhs=(thin_order in ('wide_n_panel', 'wide_cached_n_panel')
                                      or (thin_order == 'regional' and K >= 2048 and N >= 1024))):
            return
    if 'maxpool' in attrs.get('epilogue', []):
        if lhs_panel_loader is not None:
            raise ValueError('prepared LHS callback does not admit pooled readout')
        from pooling_lowering import emit_pooled_matmul
        return emit_pooled_matmul(e, A, B, C, M, K, N, attrs, bias=bias,
                                  lhs_stride=lhs_stride, rhs_stride=rhs_stride,
                                  out_stride=out_stride, rhs_resident=rhs_resident,
                                  compute_provenance=compute_provenance)
    compute_provenance = compute_provenance or {}
    stages = attrs.get('epilogue', [])
    unknown = set(stages) - {'bias', 'bias_add', 'acc_scale', 'relu'}
    if unknown:
        raise ValueError(f'unsupported matmul readout stages: {sorted(unknown)}')
    dtype = attrs.get('output_dtype', 'i32')
    if dtype not in ('i8', 'i32'):
        raise ValueError(f'unsupported matrix output dtype {dtype}')
    if ('bias' in stages or 'bias_add' in stages) and bias is None:
        raise ValueError('matmul bias epilogue requires its source operand')
    if 'acc_scale' in stages and dtype != 'i8':
        raise ValueError('scaled accumulator readout requires i8 result')
    if dtype == 'i32' and ('relu' in stages or 'acc_scale' in stages):
        raise ValueError('full-width readout cannot apply activation or scale')
    scales = attrs.get('acc_scale_by_column')
    if scales is not None:
        if len(scales) != N:
            raise ValueError('per-column readout scale extent differs from N')
        if 'acc_scale' not in stages or dtype != 'i8':
            raise ValueError('per-column scales require i8 acc_scale readout')
        if 'relu' in stages and any(float(x) < 0 for x in scales):
            raise ValueError('negative scale and post-scale relu cannot fuse on readout')
        column_groups = _column_scale_groups(scales)
        # Each constant-gain group contracts into accumulator column zero.
        # Constants remove runtime scale-table dispatch and expose address
        # arithmetic to ordinary LLVM constant folding.
        rhs_resident = None
    scale = attrs.get('acc_scale', 1.0)
    if not isinstance(scale, (int, float)):
        raise ValueError('matrix scalar readout requires scalar acc_scale')
    if 'relu' in stages and float(scale) < 0:
        raise ValueError('negative scale and post-scale relu cannot fuse on readout')
    lhs_stride = K if lhs_stride is None else lhs_stride
    rhs_stride = N if rhs_stride is None else rhs_stride
    out_stride = N if out_stride is None else out_stride
    column_order = knob('column_readout', default='grouped',
                        values=('grouped', 'staged', 'grouped_loops', 'staged_loops', 'regional', 'staged_panel'), root=Path(__file__).parent)
    if column_order == 'regional':
        # Independent gains narrower than a physical tile repeat contraction
        # work. For nontrivial K, stage the raw result once before exact device
        # readout. Uniform full-tile gains and short reductions retain grouped.
        stage_columns = (scales is not None and K >= 16
                         and len(column_groups) > (N + 15) // 16
                         and 0 < M * N * 4 <= 64 * 1024 * 1024)
        column_order = ('staged_panel' if M > 16 else 'staged_loops') if stage_columns else 'grouped'
    if scales is not None and column_order in ('staged', 'staged_loops', 'staged_panel') and 0 < M * N * 4 <= 64 * 1024 * 1024:
        # Form every output column in one ordinary matrix contraction. Exact
        # i32 publication then permits the device to gather column groups into
        # accumulator column zero for their independent scaled readouts.
        # The intermediate remains device-produced and device-consumed.
        raw = e.scratch(M * N, 'i32')
        raw_attrs = dict(attrs)
        raw_attrs.pop('acc_scale_by_column')
        raw_attrs.pop('acc_scale', None)
        raw_attrs['output_dtype'] = 'i32'
        raw_attrs['epilogue'] = [stage for stage in stages if stage in ('bias', 'bias_add')]
        emit_matmul(e, A, B, raw, M, K, N, attrs=raw_attrs, bias=bias,
                    lhs_stride=lhs_stride, rhs_stride=rhs_stride, out_stride=N,
                    rhs_resident=rhs_resident, compute_provenance=compute_provenance,
                    lhs_panel_loader=lhs_panel_loader)
        from per_column_readout import emit_per_column_readout
        emit_per_column_readout(e, raw, C, M, N, scales,
                                activation=int('relu' in stages), out_stride=out_stride,
                                loop_repeated=column_order in ('staged_loops', 'staged_panel'),
                                accumulator_panel=column_order == 'staged_panel')
        return
    if lhs_panel_loader is not None:
        # Admission precedes RHS loading and every optional preparation effect.
        # The caller retains its source-specific fallback when this cannot fit.
        prepared_base = max(32, getattr(e, '_gemmini_resident_end', 32))
        if rhs_resident is not None:
            rhs_base, rhs_nt = rhs_resident
            prepared_base = max(prepared_base, rhs_base + ((K + 15) // 16) * rhs_nt * 16)
        if M <= 16 or N <= 16 or K <= 0 or K % 16 or prepared_base + ((K + 15) // 16) * 16 > 16384:
            raise ValueError('prepared LHS M-panel does not fit its aligned scratch contract')
    if rhs_resident is None and scales is None and M > 16 and rhs_stride == N:
        residency = knob('rhs_residency', default='full', values=('full', 'stream'),
                         root=Path(__file__).parent)
        base = getattr(e, '_gemmini_resident_end', 32)
        if lhs_panel_loader is not None:
            base = max(32, base)
        cache_rows = ((K + 15) // 16) * ((N + 15) // 16) * 16
        reserved_lhs = ((K + 15) // 16) * 16 if lhs_panel_loader is not None else 0
        if residency == 'full' and base + cache_rows + reserved_lhs <= 16384:
            rhs_resident = load_resident_rhs(e, B, K, N, base)
    e.insn('config', *config_ex(dataflow=1))
    e.insn('config', *config_load(lhs_stride, state_id=0))
    e.insn('config', *config_load(rhs_stride, state_id=1))
    e.insn('config', *config_load(0, state_id=2))
    e.insn('config', *config_store(out_stride * (4 if dtype == 'i32' else 1),
                                  scale=scale, activation=int('relu' in stages)))
    order = knob('matmul_order', default='output_tile',
                 values=('output_tile', 'm_panel'), root=Path(__file__).parent)
    lhs_cache = None
    lhs_mode = knob('lhs_residency', default='stream', values=('stream', 'panel'),
                    root=Path(__file__).parent)
    policy = knob('schedule_policy', default='incumbent', values=('incumbent', 'regional'),
                  root=Path(__file__).parent)
    selection = select_schedule(M, K, N, policy=policy, matmul_order=order,
                                lhs_residency=lhs_mode)
    order, lhs_mode = selection['matmul_order'], selection['lhs_residency']
    if lhs_panel_loader is not None:
        order, lhs_mode = 'm_panel', 'panel'
    if lhs_mode == 'panel' and N > 16 and K > 0:
        # The RHS cache may have been created transiently by this invocation;
        # account for it in addition to semantic resident lifetimes.
        cache_base = getattr(e, '_gemmini_resident_end', 32)
        if lhs_panel_loader is not None:
            cache_base = max(32, cache_base)
        if rhs_resident is not None:
            rhs_base, rhs_nt = rhs_resident
            cache_base = max(cache_base, rhs_base + ((K + 15) // 16) * rhs_nt * 16)
        row_pitch = min(16, M)
        cache_rows = ((K + 15) // 16) * row_pitch
        if cache_base + cache_rows <= 16384:
            lhs_cache = (cache_base, row_pitch)
    if order == 'm_panel' and M > 16:
        panel_limit = 1024
        if lhs_cache is not None:
            # A K panel needs ceil(K/16) scratch rows per retained M row.
            # Keep complete 16-row tiles and the independent accumulator cap.
            kt = (K + 15) // 16
            panel_limit = min(1024, ((16384 - lhs_cache[0]) // (kt * 16)) * 16)
        return _emit_m_panel(e, A, B, C, M, K, N, dtype=dtype, stages=stages,
                             bias=bias, lhs_stride=lhs_stride, rhs_stride=rhs_stride,
                             out_stride=out_stride, rhs_resident=rhs_resident,
                             scales=scales, column_groups=column_groups if scales is not None else None,
                             compute_provenance=compute_provenance,
                             lhs_cache=lhs_cache, panel_limit=panel_limit, selection=selection,
                             initialize_accumulator=initialize_accumulator,
                             lhs_panel_loader=lhs_panel_loader,
                             column_loop=column_order == 'grouped_loops')
    def rows(m, row_extent=None):
        mm = extent(e, M, m) if row_extent is None else e.value(row_extent)
        if lhs_cache is not None:
            _load_lhs_panel(e, A, m, mm, K, lhs_stride, lhs_cache)
        def columns(n, group_width=None, group_scale=None):
            nn = extent(e, N, n) if scales is None else e.value(group_width)
            if scales is not None:
                e.insn('config', *config_store(out_stride, scale=group_scale,
                                              activation=int('relu' in stages)))
            # Address zero is the selected load path's zero writer, not a host buffer.
            bias_addr = 0 if bias is None else e.address(e.ptr(bias, n, 'i32'))
            overwrite_first = bias is None and K > 0 and not initialize_accumulator
            if not overwrite_first:
                e.insn('mvin3', bias_addr, packed(e, ACC_ADDR, mm, nn))
            def reduction(k, first=False):
                kk = extent(e, K, k)
                if lhs_cache is None:
                    a = e.ptr(A, index2(e, m, lhs_stride, k))
                    e.insn('mvin', e.address(a), packed(e, 0, mm, kk))
                    local_a = e.value(0)
                else:
                    a_base, a_pitch = lhs_cache
                    local_a = e.b.add_i(e.value(a_base), e.b.mul_i(
                        e.b.udiv_i(k, e.value(16)), e.value(a_pitch)))
                if rhs_resident is None:
                    local_b = 16
                    bp = e.ptr(B, index2(e, k, rhs_stride, n))
                    e.insn('mvin2', e.address(bp), packed(e, local_b, kk, nn))
                else:
                    base, nt = rhs_resident
                    tile = e.b.add_i(e.b.mul_i(e.b.udiv_i(k, e.value(16)), e.value(nt)),
                                     e.b.udiv_i(n, e.value(16)))
                    local_b = e.b.add_i(e.value(base), e.b.mul_i(tile, e.value(16)))
                e.insn('preload', packed(e, local_b, kk, nn),
                       packed(e, ACC_ADDR | (0 if first else ACC_ACCUMULATE), mm, nn), **compute_provenance)
                command = e.insn('compute', packed(e, local_a, mm, kk),
                                 packed(e, GARBAGE_ADDR, 16, 16), **compute_provenance)
                _record_selection(command, selection, 'output_tile', lhs_cache,
                                  min(M, 16), M, K, N)
            # A bias-free product begins a fresh accumulator lifetime: the
            # first device compute overwrites it instead of DMA-loading zero.
            start_k = 0
            if overwrite_first and K >= 16:
                reduction(e.value(0), first=True)
                start_k = 16
            if K % 16:
                full_k = K // 16 * 16
                e.for_loop(start_k, full_k, 16, reduction)
                # The selected zero writer clears stale scratchpad lanes before
                # the final partial A load. No tensor arithmetic runs on host.
                if lhs_cache is None:
                    e.insn('mvin', 0, packed(e, 0, 16, 16))
                    e.fence()
                reduction(e.value(full_k), first=overwrite_first and full_k == 0)
            else:
                e.for_loop(start_k, K, 16, reduction)
            out = e.ptr(C, index2(e, m, out_stride, n), dtype)
            addr = ACC_ADDR | (ACC_FULL if dtype == 'i32' else 0)
            e.insn('mvout', e.address(out), packed(e, addr, mm, nn))
        if scales is None:
            e.for_loop(0, N, 16, columns)
        elif column_order == 'grouped_loops':
            from column_group_schedule import emit_column_groups
            emit_column_groups(e, column_groups, columns, loop_repeated=True)
        else:
            for start, width, gain in column_groups:
                columns(e.value(start), width, gain)
        if lhs_cache is not None:
            # Complete every consumer before recycling the panel's local rows.
            e.fence()
    if order == 'output_tile' and M > 32 and M % 16 and N == 16:
        # Separate the single ragged tail from the full-row loop. Ordinary
        # commands retain their order and dimensions; full-tile accumulator
        # operands no longer depend on an induction-variable extent. Keep the
        # incumbent for partial-column tiles, where paired physical evidence
        # did not support the extra host tail body.
        full_m = M // 16 * 16
        e.for_loop(0, full_m, 16, lambda m: rows(m, 16))
        rows(e.value(full_m), M - full_m)
    else:
        e.for_loop(0, M, 16, rows)
    e.fence()


def _emit_m_panel(e, A, B, C, M, K, N, *, dtype, stages, bias,
                  lhs_stride, rhs_stride, out_stride, rhs_resident,
                  scales, column_groups, compute_provenance,
                  lhs_cache=None, panel_limit=1024, selection=None, lhs_panel_loader=None,
                  initialize_accumulator=False, column_loop=False):
    """Retain at most 1024 accumulator rows and share WS weights across M.

    For each N tile and K tile, preload the RHS on the first M tile and flip
    it into the active WS mesh. Later M tiles preload a garbage RHS address
    (only changing the accumulator destination) and COMPUTE_AND_STAY reuses
    the active weights. Each tile retains its independent i32 accumulator.
    """
    b = e.b
    def panel(first_row, panel_rows):
        kt = (K + 15) // 16
        if lhs_cache is not None:
            if lhs_panel_loader is not None:
                lhs_panel_loader(e, lhs_cache[0], first_row, panel_rows, kt)
            else:
                def fill(local_m):
                    mm = extent(e, panel_rows, local_m)
                    global_m = b.add_i(first_row, local_m)
                    local_base = b.add_i(e.value(lhs_cache[0]), b.mul_i(local_m, e.value(kt)))
                    _load_lhs_panel(e, A, global_m, mm, K, lhs_stride, (local_base, 16), complete=False)
                e.for_loop(0, panel_rows, 16, fill)
            e.fence()
        def columns(n, group_width=None, group_scale=None):
            nn = extent(e, N, n) if scales is None else e.value(group_width)
            if scales is not None:
                e.insn('config', *config_store(out_stride, scale=group_scale,
                                              activation=int('relu' in stages)))
            bias_addr = 0 if bias is None else e.address(e.ptr(bias, n, 'i32'))
            def initialize(local_m):
                mm = extent(e, panel_rows, local_m)
                acc = b.add_i(e.value(ACC_ADDR), local_m)
                e.insn('mvin3', bias_addr, packed(e, acc, mm, nn))
            overwrite_first = bias is None and K > 0 and not initialize_accumulator
            if not overwrite_first:
                e.for_loop(0, panel_rows, 16, initialize)
            def reduction(k, clear_tail=False, overwrite=False):
                kk = extent(e, K, k)
                if rhs_resident is None:
                    local_b = e.value(16)
                    bp = e.ptr(B, index2(e, k, rhs_stride, n))
                    e.insn('mvin2', e.address(bp), packed(e, local_b, kk, nn))
                else:
                    base, nt = rhs_resident
                    tile = b.add_i(b.mul_i(b.udiv_i(k, e.value(16)), e.value(nt)),
                                   b.udiv_i(n, e.value(16)))
                    local_b = b.add_i(e.value(base), b.mul_i(tile, e.value(16)))
                def compute(local_m, first):
                    mm = extent(e, panel_rows, local_m)
                    global_m = b.add_i(first_row, local_m)
                    if lhs_cache is None:
                        ap = e.ptr(A, index2(e, global_m, lhs_stride, k))
                        if clear_tail:
                            e.insn('mvin', 0, packed(e, 0, 16, 16))
                            e.fence()
                        e.insn('mvin', e.address(ap), packed(e, 0, mm, kk))
                        local_a = e.value(0)
                    else:
                        local_a = b.add_i(e.value(lhs_cache[0]), b.add_i(
                            b.mul_i(local_m, e.value(kt)),
                            b.mul_i(b.udiv_i(k, e.value(16)), e.value(16))))
                    acc = b.add_i(e.value(ACC_ADDR | (0 if overwrite else ACC_ACCUMULATE)), local_m)
                    weights = local_b if first else e.value(GARBAGE_ADDR)
                    e.insn('preload', packed(e, weights, kk, nn),
                           packed(e, acc, mm, nn), **compute_provenance)
                    command = e.insn('compute' if first else 'compute_and_stay',
                                     packed(e, local_a, mm, kk), packed(e, GARBAGE_ADDR, 16, 16),
                                     **compute_provenance)
                    if lhs_cache is not None:
                        command.attributes['merlin.schedule_path'] = StringAttr('m_panel_lhs_panel')
                        command.attributes['merlin.schedule_geometry'] = StringAttr(
                            f'rows={panel_rows};scratch_base={lhs_cache[0]};k_tiles={kt}')
                    _record_selection(command, selection, 'm_panel', lhs_cache,
                                      panel_rows, M, K, N)
                compute(e.value(0), True)
                if panel_rows > 16:
                    e.for_loop(16, panel_rows, 16, lambda m: compute(m, False))
            start_k = 0
            if overwrite_first and K >= 16:
                reduction(e.value(0), overwrite=True)
                start_k = 16
            if K % 16:
                full_k = K // 16 * 16
                e.for_loop(start_k, full_k, 16, reduction)
                reduction(e.value(full_k), True, overwrite=overwrite_first and full_k == 0)
            else:
                e.for_loop(start_k, K, 16, reduction)
            def store(local_m):
                mm = extent(e, panel_rows, local_m)
                global_m = b.add_i(first_row, local_m)
                out = e.ptr(C, index2(e, global_m, out_stride, n), dtype)
                acc = b.add_i(e.value(ACC_ADDR | (ACC_FULL if dtype == 'i32' else 0)), local_m)
                e.insn('mvout', e.address(out), packed(e, acc, mm, nn))
            e.for_loop(0, panel_rows, 16, store)
        if scales is None:
            e.for_loop(0, N, 16, columns)
        elif column_loop:
            from column_group_schedule import emit_column_groups
            emit_column_groups(e, column_groups, columns, loop_repeated=True)
        else:
            for start, width, gain in column_groups:
                columns(e.value(start), width, gain)
        if lhs_cache is not None:
            # Finish all slab consumers before recycling its scratch rows.
            e.fence()
    full = M // panel_limit * panel_limit
    if full:
        e.for_loop(0, full, panel_limit, lambda m: panel(m, panel_limit))
    if M != full:
        panel(e.value(full), M - full)
    e.fence()


def _column_scale_groups(scales):
    """Compile-time groups preserve every explicit f32 column gain."""
    groups = []
    start = 0
    while start < len(scales):
        end = start + 1
        bits = float_bits(scales[start])
        limit = min(len(scales), (start // 16 + 1) * 16)
        while end < limit and float_bits(scales[end]) == bits:
            end += 1
        groups.append((start, end - start, scales[start]))
        start = end
    return groups


def _load_lhs_panel(e, A, m, mm, K, lhs_stride, cache, *, complete=True):
    """Load nonoverlapping K tiles once, including an explicitly cleared tail."""
    base, pitch = cache
    def load(k, cols, clear=False):
        local = e.b.add_i(e.value(base), e.b.mul_i(
            e.b.udiv_i(k, e.value(16)), e.value(pitch)))
        if clear:
            e.insn('mvin', 0, packed(e, local, mm, 16))
            e.fence()
        ptr = e.ptr(A, index2(e, m, lhs_stride, k))
        e.insn('mvin', e.address(ptr), packed(e, local, mm, cols))
    full_k = K // 16 * 16
    e.for_loop(0, full_k, 16, lambda k: load(k, e.value(16)))
    if K != full_k:
        load(e.value(full_k), e.value(K - full_k), True)
    # Separate asynchronous panel fills from all subsequent compute consumers.
    if complete:
        e.fence()


def _record_selection(command, selection, order, lhs_cache, rows, M, K, N):
    """Describe the emitted consumer path without asserting timing or receipts."""
    if selection is None or selection['policy'] != 'regional':
        return
    path = order + ('_lhs_panel' if lhs_cache is not None else '_lhs_stream')
    command.attributes['merlin.schedule_policy'] = StringAttr('regional')
    command.attributes['merlin.schedule_reason'] = StringAttr(selection['reason'])
    command.attributes['merlin.schedule_path'] = StringAttr(path)
    base = 'none' if lhs_cache is None else str(lhs_cache[0])
    command.attributes['merlin.schedule_geometry'] = StringAttr(
        f'M={M};K={K};N={N};rows={rows};scratch_base={base};k_tiles={(K + 15) // 16}')
