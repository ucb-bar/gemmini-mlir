"""Device maxpool over a materialized, epilogue-converted NHWC i8 matrix.

The host computes addresses and supplies explicit constant padding. Each window
is staged in scratchpad and reduced by StoreController's pooling readout.
"""
from xdsl.dialects import llvm
from xdsl.dialects.builtin import DenseArrayBase, IntegerAttr, i8, i64
from xdsl.ir import Block, Region
from merlin.compile.logical_abi import pool_params, pool_out_dims
from . import isa



def _padding_buffer(ctx, value):
    """Read-only compiler constant; no runtime host work on tensor elements."""
    from merlin.llvmlower.host_builder import PTR, iconst
    name = '__pool_padding_' + str(value)
    if name in ctx.pointers:
        return ctx.pointers[name]
    ty = llvm.LLVMArrayType(16, i8)
    init = Block()
    zero = llvm.ZeroOp(result_types=[ty])
    scalar = iconst(value, i8)
    init.add_ops([zero, scalar])
    aggregate = zero.results[0]
    for lane in range(16):
        element = llvm.InsertValueOp(DenseArrayBase.from_list(i64, [lane]),
                                     aggregate, scalar.results[0])
        init.add_op(element)
        aggregate = element.results[0]
    init.add_op(llvm.ReturnOp(aggregate))
    symbol = 'scratch_' + str(len(ctx.globals))
    ctx.globals.append(llvm.GlobalOp(ty, symbol, 'internal', constant=True,
                                   alignment=64, body=Region(init)))
    address = llvm.AddressOfOp(symbol, PTR)
    address.attributes['merlin.global_task'] = IntegerAttr(-1, i64)
    terminator = ctx.f.entry.last_op
    if terminator is not None and terminator.has_trait(llvm.IsTerminator):
        ctx.f.entry.insert_op_before(address, terminator)
    else:
        ctx.f.entry.add_op(address)
    ctx.pointers[name] = address.results[0]
    return address.results[0]


def config_pool(builder, *, stride, size, channels, activation=0, scale=1.0):
    """Configure one square input patch -> one output cell, without implicit padding."""
    if size not in (1, 2, 3):
        raise ValueError('hardware pooling supports square sizes 1..3')
    # ConfigMvoutRs1: cmd/activation, pool_stride=1, pool_size=size,
    # pool_out_dim=1, porows=pocols=1, orows=ocols=size, upad=lpad=0.
    rs1 = (2 | (activation << 2) | (1 << 4) | (size << 6) |
           (1 << 24) | (1 << 32) | (1 << 40) | (size << 48) | (size << 56))
    rs2 = isa._pack(builder, [(stride, 0, 32), (isa.f32_bits(scale), 32, 32)])
    return isa.emit(builder, 0, rs1, rs2)



def _retained_plane(ctx, source, destination, batch, h, w, ho, wo, channels, size, stride):
    """Pool a complete unpadded plane with one hidden-footprint readout per tile."""
    f = ctx.f
    plane, out_plane = h * w, ho * wo
    rs1 = (2 | (stride << 4) | (size << 6) | (wo << 24) |
           (ho << 32) | (wo << 40) | (h << 48) | (w << 56))
    isa.config_ld(f, channels, state=0)
    isa.emit(f, 0, rs1, (isa.f32_bits(1.0) << 32) | channels)
    def one_batch(bn):
        def channel_tile(channel, count):
            def load_rows(row, count_rows):
                index = f.add_i(f.mul_i(f.add_i(f.mul_i(bn, f.const(plane)), row),
                                            f.const(channels)), channel)
                isa.mvin(f, f.gep(source, index, i8), row, count, count_rows)
            full = plane // 16 * 16
            if plane <= 16:
                load_rows(f.const(0), plane)
            else:
                f.for_loop(0, full, 16, lambda row: load_rows(row, 16), retain_loop=True)
                if full != plane:
                    load_rows(f.const(full), plane - full)
            out_index = f.add_i(f.mul_i(bn, f.const(out_plane * channels)), channel)
            isa.fence(f)
            isa.mvout(f, f.gep(destination, out_index, i8), 0, count, 0)
            isa.fence(f)
        if channels <= 16:
            channel_tile(f.const(0), channels)
        else:
            f.for_loop(0, channels, 16, lambda channel: channel_tile(channel,
                f.smin(f.const(16), f.sub_i(f.const(channels), channel))), retain_loop=True)
    if batch == 1:
        one_batch(f.const(0))
    else:
        f.for_loop(0, batch, 1, one_batch, retain_loop=True)
    return batch * out_plane, channels


def device_pool(ctx, src, dst, rows, cols, attrs):
    """Pool already scaled/clamped i8 [batch*H*W,C] to [batch*Ho*Wo,C].

    `pool_in_dims`, `pool_size`, `pool_stride`, and `pool_padding` use the
    public logical ABI. Source and destination may be names or LLVM pointers.
    Return (number of output rows, C). No arithmetic is performed on host data.
    """
    p = pool_params(dict(attrs), op='device maxpool')
    h, w = p['pool_in_dims']
    ph, pw = p['pool_size']
    sh, sw = p['pool_stride']
    pt, pl, pb, pr = p['pool_padding']
    if min(h, w, ph, pw, sh, sw, rows, cols) <= 0 or min(pt, pl, pb, pr) < 0:
        raise ValueError('invalid pooling geometry')
    if ph != pw or ph > 3:
        raise ValueError('device pooling requires a square window of size <=3')
    if rows % (h * w):
        raise ValueError('pool_in_dims must divide source row count')
    ho, wo = pool_out_dims(h, w, (ph, pw), (sh, sw), (pt, pl, pb, pr))
    if min(ho, wo) <= 0:
        raise ValueError('pooling has an empty output')
    batch = rows // (h * w)
    pad_value = -128 if p['pad_value'] is None else p['pad_value']
    if not -128 <= pad_value <= 127:
        raise ValueError('i8 pooling padding must fit signed i8')
    f = ctx.f
    source = ctx.ptr(src) if isinstance(src, str) else src
    destination = ctx.ptr(dst) if isinstance(dst, str) else dst
    # Whole planes avoid repeating spatial address arithmetic per pooled pixel.
    # Geometry fields are eight bits and scratchpad has 16384 rows. Each command
    # carries at most sixteen rows, while the pooling readout walks the full plane.
    if not any((pt, pl, pb, pr)) and sh == sw and sh <= 3 and max(h, w, ho, wo) <= 255 and h * w <= 16384:
        return _retained_plane(ctx, source, destination, batch, h, w, ho, wo, cols, ph, sh)
    preparation = getattr(ctx, '_pool_prepare', None)
    if preparation is None:
        from pathlib import Path
        from merlin.compile.tuning_knobs import knob
        if knob('conv_prepare', default='generic',
                values=('generic', 'auto', 'structured', 'auto_pool'),
                root=Path(__file__).resolve().parents[1]) == 'auto_pool':
            preparation = 'retained_padded'
    if preparation == 'retained_padded':
        from .pool_retain_padded import try_retained_padded
        retained = try_retained_padded(ctx, source, destination, batch, h, w, ho, wo,
                                       cols, ph, sh, sw, (pt, pl, pb, pr), pad_value)
        if retained is not None:
            return retained
    padding = _padding_buffer(ctx, pad_value)
    isa.fence(f)
    isa.config_ld(f, cols, state=0)
    config_pool(f, stride=cols, size=ph, channels=cols)

    def compare(a, b, predicate):
        return f.add(llvm.ICmpOp(a, b, IntegerAttr(predicate.to_int(), i64))).results[0]

    def channel_tile(channel):
        count = f.smin(f.const(16), f.sub_i(f.const(cols), channel))
        def output(pixel):
            bn = f.udiv_i(pixel, f.const(ho * wo))
            oy = f.urem_i(f.udiv_i(pixel, f.const(wo)), f.const(ho))
            ox = f.urem_i(pixel, f.const(wo))
            base_y = f.sub_i(f.mul_i(oy, f.const(sh)), f.const(pt))
            base_x = f.sub_i(f.mul_i(ox, f.const(sw)), f.const(pl))
            for ky in range(ph):
                for kx in range(pw):
                    iy = f.add_i(base_y, f.const(ky))
                    ix = f.add_i(base_x, f.const(kx))
                    valid = f.and_i(compare(iy, f.const(0), llvm.ICmpPredicateFlag.SGE),
                                    compare(iy, f.const(h), llvm.ICmpPredicateFlag.SLT))
                    valid = f.and_i(valid, compare(ix, f.const(0), llvm.ICmpPredicateFlag.SGE))
                    valid = f.and_i(valid, compare(ix, f.const(w), llvm.ICmpPredicateFlag.SLT))
                    inside, outside, done = Block(), Block(), Block()
                    for block in (inside, outside, done):
                        f.region.add_block(block)
                    f.add(llvm.CondBrOp(valid, inside, [], outside, []))
                    f.blk = inside
                    index = f.add_i(f.mul_i(f.add_i(f.mul_i(f.add_i(f.mul_i(bn,
                        f.const(h)), iy), f.const(w)), ix), f.const(cols)), channel)
                    isa.mvin(f, f.gep(source, index, i8), ky * pw + kx, count, 1)
                    f.add(llvm.BrOp(done))
                    f.blk = outside
                    isa.mvin(f, padding, ky * pw + kx, count, 1)
                    f.add(llvm.BrOp(done))
                    f.blk = done
            index = f.add_i(f.mul_i(pixel, f.const(cols)), channel)
            # rows=0 selects a multirow pool traversal, whose full footprint
            # is not expressed by the ordinary MVOUT shape to the scoreboard.
            # Finish tap loads before reading, then finish pooled readout before
            # reusing the same scratchpad rows for the next output window.
            isa.fence(f)
            isa.mvout(f, f.gep(destination, index, i8), 0, count, 0)
            isa.fence(f)
        f.for_loop(0, batch * ho * wo, 1, output, retain_loop=True)
    f.for_loop(0, cols, 16, channel_tile, retain_loop=True)
    isa.fence(f)
    return batch * ho * wo, cols
