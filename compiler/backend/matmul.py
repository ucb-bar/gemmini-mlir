"""General row-major int8 Gemmini matmul using ordinary CPU tile loops.

No target loop FSM instructions are emitted. Each tile consists of explicit
load, preload, execute, and store commands, with the command scoreboard carrying
scratchpad and accumulator dependencies.
"""
from __future__ import annotations

from pathlib import Path
from merlin.compile.tuning_knobs import knob
from xdsl.dialects.builtin import i8, i32
from . import isa
from .row_policy import select_row_tiles, unroll_full_rows
from .resident import prepare as prepare_resident
from .column_schedule import prepare_columns
from .activation_prepare import prepare as prepare_activation
from .reduction import emit_reduction

# Selected RTL facts: AccumulatorMem io_write_bits_addr is i9 per bank,
# with sixteen i32 lanes per row. Stay within one bank-sized row window
# (0..511), conservatively independent of the bank selection mapping.
_ACCUMULATOR_BANK_ROWS = 1 << 9


def lower_matmul(ctx, lhs, rhs, dst, m, k, n, attrs=None):
    """Lower C[m,n] = A[m,k] @ B[k,n], optionally bias/scale/ReLU.

    Operands are tensor names or LLVM pointers. Bias is a broadcast i32[n]
    tensor; a full matrix bias is selected with ``bias_broadcast=False``.
    The output dtype is i32 (full accumulator) or i8 (scaled and saturated).
    """
    attrs = dict(attrs or {})
    m, k, n = int(m), int(k), int(n)
    if min(m, k, n) <= 0:
        raise ValueError("Gemmini matmul dimensions must be positive")
    lhs_row_stride = int(attrs.get("_lhs_row_stride", k))
    if lhs_row_stride < k:
        raise ValueError("matmul LHS row stride must be at least K")
    dtype = str(attrs.get("output_dtype", "i32"))
    if dtype not in ("i8", "i32"):
        raise ValueError(f"unsupported Gemmini matmul output dtype {dtype}")
    epilogue = attrs.get("epilogue", "none")
    epilogues = [str(x).lower() for x in epilogue] if isinstance(epilogue, (list, tuple)) else [str(epilogue).lower()]
    relu = "relu" in epilogues or bool(attrs.get("relu", False))
    if any(x not in ("none", "identity", "", "relu", "relu6", "bias", "bias_add", "acc_scale", "saturate") for x in epilogues):
        raise ValueError(f"unsupported matmul epilogue {epilogue}")
    # The physical readout pipeline is bias preload -> accumulator scale ->
    # i8 saturation -> ReLU. Preserve that order rather than treating the
    # source's ordered epilogue as an unordered set of requested features.
    stage_aliases = {"bias_add": "bias"}
    stages = [stage_aliases.get(x, x) for x in epilogues
              if x not in ("none", "identity", "")]
    stage_order = {"bias": 0, "acc_scale": 1, "saturate": 2, "relu": 3, "relu6": 3}
    if any(stage_order[a] > stage_order[b] for a, b in zip(stages, stages[1:])):
        raise ValueError(f"matmul epilogue order is not implementable by readout: {epilogue}")
    if stages.count("bias") > 1 or stages.count("acc_scale") > 1:
        raise ValueError(f"repeated matmul arithmetic epilogue stage: {epilogue}")
    if "relu6" in epilogues:
        raise ValueError("relu6 needs an explicit quantized clipping bound")
    if dtype == "i32" and relu:
        raise ValueError("raw full-width accumulator readout cannot apply ReLU")
    scale = float(attrs.get("acc_scale", 1.0))
    if dtype == "i32" and scale != 1.0:
        raise ValueError("raw full-width accumulator readout cannot scale")
    if dtype == "i32" and attrs.get("acc_scale_by_column") is not None:
        if any(float(x) != 1.0 for x in attrs["acc_scale_by_column"]):
            raise ValueError("raw full-width accumulator readout cannot scale columns")
    f = ctx.f
    pointer = lambda x: ctx.ptr(x) if isinstance(x, str) else x
    a, b, c = pointer(lhs), pointer(rhs), pointer(dst)
    bias = attrs.get("bias")
    if bias in (False, "none", ""):
        bias = None
    if bias is not None:
        bias = pointer(bias)
    broadcast_bias = bool(attrs.get("bias_broadcast", True))
    prepared_columns = bool(attrs.get('_column_layout_prepared', False))
    column_choice = (4 if prepared_columns else
                     knob("matmul_column_tiles", default=1,
                          values=(1, 2, 4, "auto", "permute"),
                          root=Path(__file__).resolve().parents[1]))
    if column_choice == "auto":
        from .column_permute import should_permute
        if should_permute(ctx, m, k, n, attrs):
            column_choice = "permute"
    if column_choice == "permute":
        from .column_permute import lower_permuted
        if lower_permuted(ctx, a, b, c, m, k, n, attrs):
            return
        column_choice = 1
    if column_choice != 1:
        from .tile_schedule import lower_tiled, select_column_tiles
        maximum = knob("matmul_row_tiles", default=1,
                       values=(1, 2, 4, 8, 16, 32),
                       root=Path(__file__).resolve().parents[1])
        maximum = select_row_tiles(m, k, maximum,
                                   root=Path(__file__).resolve().parents[1])
        selected_columns = select_column_tiles(m, k, n, maximum, column_choice)
        lower = lower_tiled
        if prepared_columns:
            # The private permutation already proves this explicit experiment;
            # the terminal run scheduler must not reread the global choice.
            from .tile_column_readout import lower_column_tiled
            lower = lower_column_tiled
        if (selected_columns in (2, 4)
                and lower(ctx, a, b, c, m, k, n, attrs, maximum,
                          selected_columns)):
            return
    # DMA state 0 is A, state 1 is B, state 2 is full-width bias.
    isa.config_ex(f, dataflow=1, activation=0, scale=1.0)
    isa.config_ld(f, lhs_row_stride, state=0)
    activation = prepare_activation(ctx, a, m, k, n, attrs) if lhs_row_stride == k else None
    isa.config_ld(f, n, state=1)
    resident = prepare_resident(ctx, b, k, n, attrs)
    if bias is not None:
        isa.config_ld(f, 0 if broadcast_bias else n * 4, state=2, shrink=False)
    isa.config_st(f, n * (4 if dtype == "i32" else 1), scale=scale,
                  activation=1 if relu else 0)

    scales = attrs.get("acc_scale_by_column")
    column_groups = []
    if scales is None:
        column_groups = [(0, n, scale)]
    else:
        if len(scales) != n:
            raise ValueError("acc_scale_by_column must have N entries")
        start = 0
        while start < n:
            end = start + 1
            while end < n and float(scales[end]) == float(scales[start]):
                end += 1
            column_groups.append((start, end, float(scales[start])))
            start = end

    compact_columns = prepare_columns(ctx, scales, column_groups)

    row_tiles = knob("matmul_row_tiles", default=1, values=(1, 2, 4, 8, 16, 32),
                     root=Path(__file__).resolve().parents[1])
    row_tiles = select_row_tiles(m, k, row_tiles, root=Path(__file__).resolve().parents[1])
    if row_tiles * isa.DIM > _ACCUMULATOR_BANK_ROWS:
        raise ValueError("matmul row group exceeds selected accumulator address window")
    if row_tiles != 1:
        _grouped_rows(f, a, b, c, bias, broadcast_bias, m, k, n, dtype,
                      scales, column_groups, relu, row_tiles, resident, compact_columns, activation, lhs_row_stride)
        isa.fence(f)
        return

    def output_rows(mi):
        tm = f.smin(f.const(16), f.sub_i(f.const(m), mi))
        def output_columns(nj, end):
            tn = f.smin(f.const(16), f.sub_i(f.const(end) if isinstance(end, int) else end, nj))
            if bias is not None:
                bias_index = nj if broadcast_bias else f.add_i(f.mul_i(mi, f.const(n)), nj)
                isa.mvin(f, f.gep(bias, bias_index, i32), 0x80000000, tn, tm, state=2)
            def reduction(kk, first):
                tk = f.smin(f.const(16), f.sub_i(f.const(k), kk))
                if activation is None:
                    a_index = f.add_i(f.mul_i(mi, f.const(lhs_row_stride)), kk)
                if resident is None:
                    b_index = f.add_i(f.mul_i(kk, f.const(n)), nj)
                if activation is None:
                    isa.mvin(f, f.gep(a, a_index, i8), 0, tk, tm, state=0)
                if resident is None:
                    isa.mvin(f, f.gep(b, b_index, i8), 16, tn, tk, state=1)
                # Address bit 30 requests accumulation. First reduction tile
                # overwrites unless bias has initialized the accumulator.
                if bias is not None or first is False:
                    acc_address = 0xC0000000
                elif first is True:
                    acc_address = 0x80000000
                else:
                    subsequent = f.smin(kk, f.const(1))
                    acc_address = f.or_i(f.const(0x80000000), f.shl_i(subsequent, 30))
                b_local = 16 if resident is None else resident.address(f, kk, nj)
                isa.preload(f, b_local, acc_address, tn, tk, tn, tm)
                a_local = 0 if activation is None else activation.address(f, kk)
                if activation is not None and m > 16:
                    a_local = f.add_i(a_local, mi)
                isa.compute(f, a_local, isa.GARBAGE_ADDR, tk, tm, tn, tm, flip=True)
            emit_reduction(f, k, reduction)
            out_index = f.add_i(f.mul_i(mi, f.const(n)), nj)
            out_ptr = f.gep(c, out_index, i32 if dtype == "i32" else i8)
            isa.mvout(f, out_ptr, 0xA0000000 if dtype == "i32" else 0x80000000, tn, tm)
        if compact_columns is not None:
            compact_columns.emit(f, n, dtype, relu, output_columns)
        else:
            for start, end, column_scale in column_groups:
                if scales is not None:
                    isa.config_st(f, n * (4 if dtype == "i32" else 1),
                                  scale=column_scale, activation=1 if relu else 0)
                f.for_loop(start, end, 16, lambda nj: output_columns(nj, end), retain_loop=True)
    f.for_loop(0, m, 16, output_rows, retain_loop=True)
    isa.fence(f)


def _grouped_rows(f, a, b, c, bias, broadcast_bias, m, k, n, dtype,
                  scales, column_groups, relu, row_tiles, resident=None, compact_columns=None, activation=None, lhs_row_stride=None):
    """Keep several output row tiles in distinct accumulators across K.

    Each reduction step DMA-loads B once, then reuses that scratchpad tile
    for every active row tile. A uses scratch rows0:16 and B rows16:32.
    """
    lhs_row_stride = k if lhs_row_stride is None else lhs_row_stride
    group_rows = 16 * row_tiles
    unroll = unroll_full_rows(root=Path(__file__).resolve().parents[1])
    def group(mi, count, full_group=False):
        def each_row(body):
            def row(ri):
                row_start = f.add_i(mi, f.mul_i(ri, f.const(16)))
                tm = f.smin(f.const(16), f.sub_i(f.const(m), row_start))
                local = f.mul_i(ri, f.const(16))
                body(row_start, tm, local)
            if unroll and full_group and type(count) is int and count <= 16:
                for ri in range(count):
                    row(f.const(ri))
            else:
                f.for_loop(0, count, 1, row, retain_loop=True)
        def columns(nj, end):
            tn = f.smin(f.const(16), f.sub_i(f.const(end) if isinstance(end, int) else end, nj))
            if bias is not None:
                def load_bias(row_start, tm, local):
                    index = nj if broadcast_bias else f.add_i(f.mul_i(row_start, f.const(n)), nj)
                    isa.mvin(f, f.gep(bias, index, i32), f.or_i(f.const(0x80000000), local), tn, tm, state=2)
                each_row(load_bias)
            def reduction(kk, first):
                tk = f.smin(f.const(16), f.sub_i(f.const(k), kk))
                if resident is None:
                    b_index = f.add_i(f.mul_i(kk, f.const(n)), nj)
                    isa.mvin(f, f.gep(b, b_index, i8), 16, tn, tk, state=1)
                if bias is not None or first is False:
                    acc_flags = f.const(0xC0000000)
                elif first is True:
                    acc_flags = f.const(0x80000000)
                else:
                    acc_flags = f.or_i(f.const(0x80000000), f.shl_i(f.smin(kk, f.const(1)), 30))
                def compute_row(row_start, tm, local):
                    a_index = f.add_i(f.mul_i(row_start, f.const(lhs_row_stride)), kk)
                    if activation is None:
                        isa.mvin(f, f.gep(a, a_index, i8), 0, tk, tm, state=0)
                    b_local = 16 if resident is None else resident.address(f, kk, nj)
                    isa.preload(f, b_local, f.or_i(acc_flags, local), tn, tk, tn, tm)
                    a_local = 0 if activation is None else activation.address(f, kk)
                    if activation is not None and m > 16:
                        a_local = f.add_i(a_local, row_start)
                    isa.compute(f, a_local, isa.GARBAGE_ADDR, tk, tm, tn, tm, flip=True)
                each_row(compute_row)
            emit_reduction(f, k, reduction)
            def store_row(row_start, tm, local):
                index = f.add_i(f.mul_i(row_start, f.const(n)), nj)
                out = f.gep(c, index, i32 if dtype == "i32" else i8)
                isa.mvout(f, out, f.or_i(f.const(0xA0000000 if dtype == "i32" else 0x80000000), local), tn, tm)
            each_row(store_row)
        if compact_columns is not None:
            compact_columns.emit(f, n, dtype, relu, columns)
        else:
            for start, end, column_scale in column_groups:
                if scales is not None:
                    isa.config_st(f, n * (4 if dtype == "i32" else 1),
                                  scale=column_scale, activation=1 if relu else 0)
                f.for_loop(start, end, 16, lambda nj: columns(nj, end), retain_loop=True)
    full_rows = m // group_rows * group_rows
    f.for_loop(0, full_rows, group_rows,
               lambda mi: group(mi, row_tiles, full_group=True), retain_loop=True)
    if full_rows < m:
        group(f.const(full_rows), (m - full_rows + 15) // 16)
