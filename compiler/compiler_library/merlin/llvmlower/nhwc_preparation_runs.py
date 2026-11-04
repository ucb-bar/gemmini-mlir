"""Checked contiguous kernel-row runs for original NHWC window preparation.

Only unit horizontal dilation admits contiguous tap/channel bytes. Both ends
must be valid original samples; otherwise the original individual tap guards
and zero padding remain. This changes host transport, never payload arithmetic.
"""
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i8, i64
from xdsl.ir import Block, SSAValue

from merlin.compile.scheduling.window_transfer import Window2D
from merlin.llvmlower.host_builder import FnBuilder, PTR
from merlin.llvmlower.convolution_packing import _WindowBuilder, _panel_receipt


def emit_nhwc_window_runs(builder, window, source, destination, row_start, rows,
                          *, source_destination_disjoint, maximum_panel_bytes):
    """Materialize the original [M, KH*KW*CI] panel using checked byte runs."""
    if type(builder) is not FnBuilder or type(window) is not Window2D:
        raise ValueError("window runs require typed builder and original window")
    window.__post_init__()
    if window.dilation[1] != 1:
        raise ValueError("contiguous kernel-row preparation requires horizontal dilation one")
    if source_destination_disjoint is not True:
        raise ValueError("window runs require disjoint source and private destination")
    if any(not isinstance(p, SSAValue) or p.type != PTR for p in (source, destination)):
        raise ValueError("window runs require byte-addressable pointers")
    if not isinstance(row_start, SSAValue) or row_start.type != i64:
        raise ValueError("window runs require an i64 original output-row start")
    kh, kw = window.kernel
    ci = window.input_shape[-1]
    reduction = kh * kw * ci
    if (type(rows) is not int or rows <= 0 or type(maximum_panel_bytes) is not int
            or not 0 < maximum_panel_bytes < 1 << 63
            or rows * reduction > maximum_panel_bytes):
        raise ValueError("window runs exceed bounded original panel storage")
    ops = _WindowBuilder(builder)
    oh, ow = window.output_spatial
    start_valid = ops.cmp("ult", row_start, window.input_shape[0] * oh * ow)

    def branch(condition, yes, no):
        valid, padding, after = Block(), Block(), Block()
        for block in (valid, padding, after):
            builder.region.add_block(block)
        builder.add(llvm.CondBrOp(condition, valid, [], padding, []))
        builder.blk = valid
        yes()
        builder.add(llvm.BrOp(after))
        builder.blk = padding
        no()
        builder.add(llvm.BrOp(after))
        builder.blk = after

    def copy(first, packed, count):
        def byte(index):
            src = builder.gep(source, ops.op("add", first, index), i8)
            dst = builder.gep(destination, ops.op("add", packed, index), i8)
            value = builder.add(llvm.LoadOp(src, i8)).results[0]
            builder.add(llvm.StoreOp(value, dst))
        builder.for_loop(0, count, 1, byte, retain_loop=True)

    def zero(packed, count):
        def byte(index):
            dst = builder.gep(destination, ops.op("add", packed, index), i8)
            builder.add(llvm.StoreOp(builder.const(0, i8), dst))
        builder.for_loop(0, count, 1, byte, retain_loop=True)

    def panel_row(pr):
        row = ops.op("add", row_start, pr)
        def kernel_row(ky):
            first = window.emit_row_run(ops, row, 1, ky, 0, 0, ci)
            last = window.emit_row_run(ops, row, 1, ky, kw - 1, 0, ci)
            valid = ops.op("and", start_valid,
                ops.op("and", first["predicate"], last["predicate"], "i1"), "i1")
            packed = ops.op("add", ops.op("mul", pr, reduction), ops.op("mul", ky, kw * ci))
            def boundary():
                def tap(kx):
                    run = window.emit_row_run(ops, row, 1, ky, kx, 0, ci)
                    target = ops.op("add", packed, ops.op("mul", kx, ci))
                    branch(ops.op("and", start_valid, run["predicate"], "i1"),
                        lambda: copy(run["first_element"], target, ci), lambda: zero(target, ci))
                builder.for_loop(0, kw, 1, tap, retain_loop=True)
            branch(valid, lambda: copy(first["first_element"], packed, kw * ci), boundary)
        builder.for_loop(0, kh, 1, kernel_row, retain_loop=True)
    builder.for_loop(0, rows, 1, panel_row, retain_loop=True)
    return dict(_panel_receipt(rows, reduction, rows * reduction),
        transport="checked_contiguous_kernel_rows",
        contiguous_run_bytes=kw * ci,
        guard="original first and last tap valid; horizontal dilation one",
        boundary="original per-tap predicate and byte-copy/zero loops")
