"""Multirow NHWC im2col DMA for unpadded scratchpad preparation.

Each group stays within one output scanline and the current im2col row
block. LD0 advances SW*CI source bytes per output row; ST advances K bytes.
Wide channel blocks occupy scratchpad rows0:16,16:32,32:48,48:64.
"""
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, i8, i64
from xdsl.ir import Block
from . import isa


def supported(prepare, pack_kind, padding):
    """Pure selector, evaluated before any preparation effects."""
    return prepare == 'auto' and pack_kind == 'scratchpad' and not any(padding)


def pack_rows(f, source, destination, row_start, count: int, *, height: int,
              width: int, ci: int, kh: int, kw: int, ho: int, wo: int,
              sh: int, sw: int, dh: int, dw: int):
    """Pack ``count`` logical output rows from a validated unpadded conv.

    Caller guarantees positive geometry, the standard output-shape formula,
    source NHWC i8 layout, destination [count,Kh*Kw*Ci], and a completion
    fence before consuming destination. No padding or tensor values are
    computed on the host. At most64 scratchpad rows are occupied.
    """
    if any(type(x) is not int or x <= 0 for x in
           (count, height, width, ci, kh, kw, ho, wo, sh, sw, dh, dw)):
        raise ValueError('row packing requires positive static geometry')
    if (ho != (height - (dh * (kh - 1) + 1)) // sh + 1 or
            wo != (width - (dw * (kw - 1) + 1)) // sw + 1):
        raise ValueError('row packing requires unpadded convolution geometry')
    k = kh * kw * ci
    isa.config_ld(f, sw * ci, state=0, block_stride=16)
    isa.config_st(f, k, scale=1.0, activation=0)

    def eq(a, b):
        return f.add(llvm.ICmpOp(a, b, IntegerAttr(
            llvm.ICmpPredicateFlag.EQ.to_int(), i64))).results[0]

    def row(r):
        logical = f.add_i(row_start, r)
        ox = f.urem_i(logical, f.const(wo))
        offset = f.urem_i(ox, f.const(16))
        active = f.or_i(eq(r, f.const(0)), eq(offset, f.const(0)))
        inside, done = Block(), Block()
        f.region.add_block(inside)
        f.region.add_block(done)
        f.add(llvm.CondBrOp(active, inside, [], done, []))
        f.blk = inside
        rows = f.smin(f.sub_i(f.const(16), offset),
                      f.smin(f.sub_i(f.const(wo), ox), f.sub_i(f.const(count), r)))
        oy = f.urem_i(f.udiv_i(logical, f.const(wo)), f.const(ho))
        bn = f.udiv_i(logical, f.const(ho * wo))
        by = f.mul_i(oy, f.const(sh))
        bx = f.mul_i(ox, f.const(sw))

        def kernel_y(ky):
            iy = f.add_i(by, f.mul_i(ky, f.const(dh)))

            def kernel_x(kx):
                ix = f.add_i(bx, f.mul_i(kx, f.const(dw)))
                source_base = f.mul_i(f.add_i(f.mul_i(f.add_i(
                    f.mul_i(bn, f.const(height)), iy), f.const(width)), ix), f.const(ci))
                target_base = f.add_i(f.mul_i(r, f.const(k)),
                    f.mul_i(f.add_i(f.mul_i(ky, f.const(kw)), kx), f.const(ci)))

                def channels(channel, length):
                    isa.mvin(f, f.gep(source, f.add_i(source_base, channel), i8),
                             0, length, rows, state=0)
                    for offset in range(0, length, 16):
                        index = f.add_i(target_base, f.add_i(channel, f.const(offset)))
                        isa.mvout(f, f.gep(destination, index, i8), offset,
                                  min(16, length - offset), rows)
                full = ci // 64 * 64
                if full:
                    f.for_loop(0, full, 64, lambda c: channels(c, 64), retain_loop=True)
                if full < ci:
                    channels(f.const(full), ci - full)
            f.for_loop(0, kw, 1, kernel_x, retain_loop=True)
        f.for_loop(0, kh, 1, kernel_y, retain_loop=True)
        f.add(llvm.BrOp(done))
        f.blk = done
    f.for_loop(0, count, 1, row, retain_loop=True)
