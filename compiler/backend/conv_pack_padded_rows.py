"""Exact NHWC row-group DMA with bounded zero/valid horizontal spans."""
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, i8, i64
from xdsl.ir import Block
from . import isa


def pack_padded_rows(f, source, destination, zero, row_start, count: int, *,
                     height: int, width: int, ci: int, kh: int, kw: int,
                     ho: int, wo: int, sh: int, sw: int, dh: int, dw: int,
                     padding):
    """Pack a validated convolution row block using address-only CPU control.

    Each group stays within its output scanline and block. For source start
    x, valid row offsets q satisfy ceil(-x/SW) <= q < ceil((W-x)/SW).
    Clamp both endpoints to [0,rows] before constructing the three spans.
    Y-invalid groups use only immutable zero16. Caller fences before use.
    """
    if any(type(x) is not int or x <= 0 for x in
           (count, height, width, ci, kh, kw, ho, wo, sh, sw, dh, dw)):
        raise ValueError('padded row packing requires positive static geometry')
    if len(padding) != 4 or any(type(x) is not int or x < 0 for x in padding):
        raise ValueError('padded row packing requires nonnegative padding')
    pt, pl, pb, pr = padding
    if (ho != (height + pt + pb - (dh * (kh - 1) + 1)) // sh + 1 or
            wo != (width + pl + pr - (dw * (kw - 1) + 1)) // sw + 1):
        raise ValueError('padded row packing output geometry mismatch')
    k = kh * kw * ci
    isa.config_st(f, k, scale=1.0, activation=0)

    def cmp(a, b, pred):
        return f.add(llvm.ICmpOp(a, b, IntegerAttr(pred.to_int(), i64))).results[0]

    def branch(cond, yes, no=None):
        inside, outside, done = Block(), Block(), Block()
        for bb in (inside, outside, done):
            f.region.add_block(bb)
        f.add(llvm.CondBrOp(cond, inside, [], outside, []))
        f.blk = inside
        yes()
        f.add(llvm.BrOp(done))
        f.blk = outside
        if no is not None:
            no()
        f.add(llvm.BrOp(done))
        f.blk = done

    def row(r):
        logical = f.add_i(row_start, r)
        ox = f.urem_i(logical, f.const(wo))
        alignment = f.urem_i(ox, f.const(16))
        active = f.or_i(cmp(r, f.const(0), llvm.ICmpPredicateFlag.EQ),
                         cmp(alignment, f.const(0), llvm.ICmpPredicateFlag.EQ))

        def group():
            rows = f.smin(f.sub_i(f.const(16), alignment),
                          f.smin(f.sub_i(f.const(wo), ox), f.sub_i(f.const(count), r)))
            oy = f.urem_i(f.udiv_i(logical, f.const(wo)), f.const(ho))
            bn = f.udiv_i(logical, f.const(ho * wo))
            by = f.sub_i(f.mul_i(oy, f.const(sh)), f.const(pt))
            bx = f.sub_i(f.mul_i(ox, f.const(sw)), f.const(pl))

            def kernel_y(ky):
                iy = f.add_i(by, f.mul_i(ky, f.const(dh)))
                yvalid = f.and_i(cmp(iy, f.const(0), llvm.ICmpPredicateFlag.SGE),
                                 cmp(iy, f.const(height), llvm.ICmpPredicateFlag.SLT))

                def kernel_x(kx):
                    ix = f.add_i(bx, f.mul_i(kx, f.const(dw)))
                    target_base = f.add_i(f.mul_i(r, f.const(k)),
                        f.mul_i(f.add_i(f.mul_i(ky, f.const(kw)), kx), f.const(ci)))

                    def zero_span(start, length):
                        def emit():
                            isa.config_ld(f, 0, state=0, block_stride=16)
                            target = f.add_i(target_base, f.mul_i(start, f.const(k)))
                            def channel(c):
                                cols = f.smin(f.const(16), f.sub_i(f.const(ci), c))
                                isa.mvin(f, zero, 0, cols, length, state=0)
                                isa.mvout(f, f.gep(destination, f.add_i(target, c), i8),
                                          0, cols, length)
                            f.for_loop(0, ci, 16, channel, retain_loop=True)
                        branch(cmp(f.const(0), length, llvm.ICmpPredicateFlag.SLT), emit)

                    def valid_y():
                        # max(0,numerator) permits unsigned division while
                        # implementing signed ceil followed by lower clamping.
                        def endpoint(limit):
                            numerator = f.sub_i(f.const(limit + sw - 1), ix)
                            return f.smin(rows, f.udiv_i(f.smax(f.const(0), numerator), f.const(sw)))
                        left, end = endpoint(0), endpoint(width)
                        zero_span(f.const(0), left)
                        length = f.sub_i(end, left)

                        def valid_span():
                            isa.config_ld(f, sw * ci, state=0, block_stride=16)
                            x = f.add_i(ix, f.mul_i(left, f.const(sw)))
                            source_base = f.mul_i(f.add_i(f.mul_i(f.add_i(
                                f.mul_i(bn, f.const(height)), iy), f.const(width)), x), f.const(ci))
                            target = f.add_i(target_base, f.mul_i(left, f.const(k)))
                            def channel(c, cols):
                                isa.mvin(f, f.gep(source, f.add_i(source_base, c), i8),
                                         0, cols, length, state=0)
                                for offset in range(0, cols, 16):
                                    index = f.add_i(target, f.add_i(c, f.const(offset)))
                                    isa.mvout(f, f.gep(destination, index, i8), offset,
                                              min(16, cols - offset), length)
                            full = ci // 64 * 64
                            if full:
                                f.for_loop(0, full, 64, lambda c: channel(c, 64), retain_loop=True)
                            if full < ci:
                                channel(f.const(full), ci - full)
                        branch(cmp(f.const(0), length, llvm.ICmpPredicateFlag.SLT), valid_span)
                        zero_span(end, f.sub_i(rows, end))
                    branch(yvalid, valid_y, lambda: zero_span(f.const(0), rows))
                f.for_loop(0, kw, 1, kernel_x, retain_loop=True)
            f.for_loop(0, kh, 1, kernel_y, retain_loop=True)
        branch(active, group)
    f.for_loop(0, count, 1, row, retain_loop=True)
