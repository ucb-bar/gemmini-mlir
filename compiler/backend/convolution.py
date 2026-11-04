"""NHWC convolution via bounded im2col preparation and device matmul.

Auto preparation uses device DMA movement for all tensor payload bytes.
The host controls addresses and commands; scalar copy paths are tuning baselines.
"""
from __future__ import annotations

from pathlib import Path

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, i8, i64
from xdsl.ir import Block, Region

from .matmul import lower_matmul
from . import isa


def _zero_buffer(ctx):
    """Immutable DMA source: zero bytes are initialized in the object, never host-stored."""
    from merlin.llvmlower.host_builder import PTR
    name = '__conv_device_zero'
    if name in ctx.pointers:
        return ctx.pointers[name]
    ty = llvm.LLVMArrayType(16, i8)
    init = Block()
    zero = llvm.ZeroOp(result_types=[ty])
    init.add_ops([zero, llvm.ReturnOp(zero.results[0])])
    symbol = 'scratch_' + str(len(ctx.globals))
    ctx.globals.append(llvm.GlobalOp(ty, symbol, 'internal', constant=True,
                                   alignment=64, body=Region(init)))
    address = llvm.AddressOfOp(symbol, PTR)
    terminator = ctx.f.entry.last_op
    if terminator is not None and terminator.has_trait(llvm.IsTerminator):
        ctx.f.entry.insert_op_before(address, terminator)
    else:
        ctx.f.entry.add_op(address)
    ctx.pointers[name] = address.results[0]
    return address.results[0]


def _positive(values, length, label):
    if len(values) != length or any(type(x) is not int or x <= 0 for x in values):
        raise ValueError(f"CONV2D {label} must contain {length} positive integers")
    return tuple(values)


def _copy_if_in_bounds(f, src, dst, src_index, dst_index, h, w, height, width):
    """Branch before loading, so padded taps never form an invalid load."""
    def cmp(a, b, predicate):
        return f.add(llvm.ICmpOp(a, b, IntegerAttr(predicate.to_int(), i64))).results[0]
    slt = llvm.ICmpPredicateFlag.SLT
    sge = llvm.ICmpPredicateFlag.SGE
    valid = f.and_i(cmp(h, f.const(0), sge), cmp(h, f.const(height), slt))
    valid = f.and_i(valid, cmp(w, f.const(0), sge))
    valid = f.and_i(valid, cmp(w, f.const(width), slt))
    inside, outside, done = Block(), Block(), Block()
    for block in (inside, outside, done):
        f.region.add_block(block)
    f.add(llvm.CondBrOp(valid, inside, [], outside, []))
    f.blk = inside
    value = f.add(llvm.LoadOp(f.gep(src, src_index, i8), i8)).results[0]
    f.add(llvm.StoreOp(value, f.gep(dst, dst_index, i8)))
    f.add(llvm.BrOp(done))
    f.blk = outside
    f.add(llvm.StoreOp(f.add(llvm.TruncOp(f.const(0), i8)).results[0],
                       f.gep(dst, dst_index, i8)))
    f.add(llvm.BrOp(done))
    f.blk = done


def lower_conv(ctx, command, residents):
    """Lower a canonical command-buffer CONV2D, independent of source identity."""
    op = command['operands']
    attrs = dict(command.get('attributes', {}))
    tensors = ctx.cb['tensors']
    ifm_name = op['ifm']
    weight_name = op.get('weight', op.get('weights'))
    weight_name = residents.get(weight_name, weight_name)
    dst_name = op['dst']
    if attrs.get('layout', 'nhwc').lower() != 'nhwc':
        raise ValueError('CONV2D requires NHWC layout')
    kh, kw, ci, co = _positive(attrs['kernel'], 4, 'kernel')
    shape = tensors[ifm_name]['shape']
    if len(shape) != 4:
        raise ValueError('CONV2D input must have NHWC rank four')
    batch, height, width, channels = _positive(shape, 4, 'input shape')
    if ci != channels:
        raise ValueError('CONV2D kernel/input channel mismatch')
    if tensors[ifm_name]['dtype'] != 'i8' or tensors[weight_name]['dtype'] != 'i8':
        raise ValueError('CONV2D target lowering requires signed i8 operands')
    sh, sw = _positive(attrs.get('stride', [1, 1]), 2, 'stride')
    dh, dw = _positive(attrs.get('dilation', [1, 1]), 2, 'dilation')
    padding = attrs.get('padding', [0, 0, 0, 0])
    if len(padding) != 4 or any(type(p) is not int or p < 0 for p in padding):
        raise ValueError('CONV2D padding must contain four nonnegative integers')
    pt, pl, pb, pr = padding
    ho = (height + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    wo = (width + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    if ho <= 0 or wo <= 0:
        raise ValueError('CONV2D has empty output geometry')
    k = kh * kw * ci
    m = batch * ho * wo
    if list(tensors[weight_name]['shape']) != [k, co]:
        raise ValueError('CONV2D weights must be pre-im2col row-major [Kh*Kw*Ci,Co]')
    epilogue = attrs.get('epilogue', [])
    if isinstance(epilogue, str):
        epilogue = [] if epilogue == 'none' else [epilogue]
    pooling_attrs = None
    if 'maxpool' in epilogue:
        if epilogue[-1] != 'maxpool' or epilogue.count('maxpool') != 1:
            raise ValueError('CONV2D pooling lowering requires maxpool as the last epilogue')
        if list(attrs.get('pool_in_dims', [])) != [ho, wo]:
            raise ValueError('CONV2D pool_in_dims disagrees with convolution geometry')
        pooling_attrs = dict(attrs)
        attrs['epilogue'] = epilogue[:-1]
    dst_shape = list(tensors[dst_name]['shape'])
    expected_shapes = ([m, co], [batch, ho, wo, co])
    if pooling_attrs is not None:
        from merlin.compile.logical_abi import pool_params, pool_out_dims
        pool = pool_params(pooling_attrs, op='CONV2D')
        hp, wp = pool_out_dims(ho, wo, pool['pool_size'], pool['pool_stride'], pool['pool_padding'])
        expected_shapes = ([batch * hp * wp, co], [batch, hp, wp, co])
    if dst_shape not in expected_shapes:
        raise ValueError('CONV2D destination shape disagrees with source geometry')
    output_dtype = attrs.get('output_dtype', tensors[dst_name]['dtype'])
    if output_dtype != tensors[dst_name]['dtype']:
        raise ValueError('CONV2D output dtype disagrees with destination')
    attrs['output_dtype'] = output_dtype
    # A single reduction tile still has explicit accumulator initialization:
    # load an immutable zero addend, then accumulate the first product onto it.
    # Reuse the shared bias DMA path without adding a numerical epilogue stage.
    if k <= 16 and attrs.get('bias') in (None, False, 'none', ''):
        attrs['bias'] = ctx.alloc(f'__conv_zero_acc_{dst_name}_{co}', [co], 'i32',
                                  constant=True)
        attrs['bias_broadcast'] = True
    from merlin.compile.tuning_knobs import knob
    prepare = getattr(ctx, 'conv_prepare', None) or knob(
        'conv_prepare', default='generic', values=('generic', 'auto', 'structured', 'auto_pool'),
        root=Path(__file__).resolve().parents[1])
    if prepare == 'auto_pool':
        prepare = 'auto'
    if prepare not in ('auto', 'generic', 'structured'):
        raise ValueError('conv_prepare must be auto, generic, or structured')
    src, rhs, dst = ctx.ptr(ifm_name), ctx.ptr(weight_name), ctx.ptr(dst_name)
    f = ctx.f
    if pooling_attrs is not None:
        if output_dtype != 'i8':
            raise ValueError('CONV2D device pooling currently requires i8 readout')
        final_dst = dst
        dst = ctx.alloc(f'__conv_prepool_{dst_name}', [m, co], 'i8')
    def finish_pool():
        if pooling_attrs is not None:
            from .pooling import device_pool
            device_pool(ctx, dst, final_dst, m, co, pooling_attrs)
    # NHWC 1x1/s1 is already exactly the logical [N*H*W,Ci] lhs.
    # Keep generic available for direct same-source baseline comparisons.
    if (prepare == 'auto' and kh == kw == dh == dw == sh == sw == 1
            and not any(padding)):
        lower_matmul(ctx, src, rhs, dst, m, k, co, attrs)
        finish_pool()
        return
    # This CONV2D writes only its output/private im2col storage. Its original
    # named weights and broadcast bias remain read-only throughout the row
    # loop, so an explicit column layout may gather them before that loop.
    # No descriptor is cached across commands or source generations.
    prepared_columns = None
    column_choice = knob('matmul_column_tiles', default=1,
                         values=(1, 2, 4, 'auto', 'permute'),
                         root=Path(__file__).resolve().parents[1])
    if column_choice == 'auto':
        from .column_permute import should_permute
        if should_permute(ctx, m, k, co, attrs):
            column_choice = 'permute'
    if column_choice == 'permute':
        from .column_permute import prepare_rhs, lower_prepared
        bias_source = attrs.get('bias')
        def root_pointer(pointer):
            while isinstance(getattr(pointer, 'owner', None), llvm.GEPOp):
                pointer = pointer.owner.operands[0]
            return pointer
        # Named storage identities and normalized SSA roots establish that the
        # only user-visible writes cannot change either preparation source.
        readonly = weight_name != dst_name and root_pointer(rhs) is not root_pointer(dst)
        if bias_source not in (None, False, 'none', ''):
            bias_pointer = ctx.ptr(bias_source)
            readonly = (readonly and bias_source != dst_name
                        and root_pointer(bias_pointer) is not root_pointer(dst))
        if readonly:
            prepared_columns = prepare_rhs(ctx, rhs, m, k, co, attrs)
    pointwise_source = getattr(ctx, 'conv_pointwise_source', None) or knob(
        'conv_pointwise_source', default='packed', values=('packed', 'strided'),
        root=Path(__file__).resolve().parents[1])
    if pointwise_source not in ('packed', 'strided'):
        raise ValueError('conv_pointwise_source must be packed or strided')
    if (prepare == 'auto' and pointwise_source == 'strided'
            and kh == kw == dh == dw == 1 and not any(padding)):
        # Each output scanline is a strided row-major matrix of real input
        # channels. Stop at OW so DMA never crosses the source scanline gap.
        # The output formula guarantees (OW-1)*sw < width, and likewise for H.
        from merlin.llvmlower.host_builder import INT_TYPES
        def batch_rows(bn):
            def scanline(oy):
                input_row = f.add_i(f.mul_i(bn, f.const(height)),
                                    f.mul_i(oy, f.const(sh)))
                source_offset = f.mul_i(input_row, f.const(width * ci))
                output_row = f.add_i(f.mul_i(bn, f.const(ho)), oy)
                output_offset = f.mul_i(output_row, f.const(wo * co))
                strided = dict(attrs)
                strided['_lhs_row_stride'] = sw * ci
                bias = attrs.get('bias')
                if bias not in (None, False, 'none', '') and not attrs.get('bias_broadcast', True):
                    pointer = ctx.ptr(bias) if isinstance(bias, str) else bias
                    strided['bias'] = f.gep(pointer, output_offset, INT_TYPES['i32'])
                lhs = f.gep(src, source_offset, i8)
                output = f.gep(dst, output_offset, INT_TYPES[output_dtype])
                if prepared_columns is not None:
                    lower_prepared(ctx, lhs, output, wo, prepared_columns,
                                   lhs_row_stride=sw * ci)
                else:
                    lower_matmul(ctx, lhs, rhs, output, wo, k, co, strided)
            f.for_loop(0, ho, 1, scanline, retain_loop=True)
        f.for_loop(0, batch, 1, batch_rows, retain_loop=True)
        finish_pool()
        return
    # Bounded row blocks avoid materializing the entire activation-window matrix.
    # The scratch allocation is shared across sequential row-block iterations.
    row_tiles = 1 if prepare == 'generic' else min(8, knob(
        'matmul_row_tiles', default=1, values=(1, 2, 4, 8, 16, 32),
        root=Path(__file__).resolve().parents[1]))
    tile_rows = min(16 * row_tiles, m)
    scratch_name = f'__conv_im2col_{dst_name}'
    scratch = ctx.alloc(scratch_name, [tile_rows, k], 'i8')
    zero = _zero_buffer(ctx) if prepare == 'auto' else None
    pack_kind = getattr(ctx, 'conv_pack', None) or knob(
        'conv_pack', default='scratchpad', values=('scratchpad', 'accumulator'),
        root=Path(__file__).resolve().parents[1])
    if pack_kind not in ('scratchpad', 'accumulator'):
        raise ValueError('conv_pack must be scratchpad or accumulator')
    pack_dma = getattr(ctx, 'conv_pack_dma', None) or knob(
        'conv_pack_dma', default='tile', values=('tile', 'wide', 'rows', 'padded_rows'),
        root=Path(__file__).resolve().parents[1])
    from .conv_pack_rows import supported as rows_supported
    pack_row_groups = pack_dma in ('rows', 'padded_rows') and rows_supported(prepare, pack_kind, padding)
    pack_wide = pack_kind == 'scratchpad' and ci > 16 and pack_dma in ('wide', 'rows', 'padded_rows')
    pack_local = 0x80000000 if pack_kind == 'accumulator' else 0
    def device_rows(row_start, count):
        if (pack_dma == 'padded_rows' and pack_kind == 'scratchpad'
                and any(padding) and count > 1 and wo > 1):
            from .conv_pack_padded_rows import pack_padded_rows
            pack_padded_rows(f, src, scratch, zero, row_start, count,
                            height=height, width=width, ci=ci, kh=kh, kw=kw,
                            ho=ho, wo=wo, sh=sh, sw=sw, dh=dh, dw=dw, padding=padding)
            isa.fence(f)
            return
        if pack_row_groups and count > 1 and wo > 1:
            from .conv_pack_rows import pack_rows
            pack_rows(f, src, scratch, row_start, count, height=height, width=width,
                      ci=ci, kh=kh, kw=kw, ho=ho, wo=wo, sh=sh, sw=sw, dh=dh, dw=dw)
            isa.fence(f)
            return
        # Both choices move exact i8 values. Accumulator packing sign-extends
        # each source byte on load, then scale1/activation0 readout restores it:
        # every signed i8 value is exactly representable through this pipeline.
        # DMA scoreboard dependencies protect row0; the completion fence makes
        # packed DRAM visible before the shared matmul input DMA.
        isa.config_ld(f, 0, state=0, shrink=pack_kind == 'accumulator')
        isa.config_st(f, k, scale=1.0, activation=0)
        def row(local_row):
            logical = f.add_i(row_start, local_row)
            ox = f.urem_i(logical, f.const(wo))
            oy = f.urem_i(f.udiv_i(logical, f.const(wo)), f.const(ho))
            bn = f.udiv_i(logical, f.const(ho * wo))
            by = f.sub_i(f.mul_i(oy, f.const(sh)), f.const(pt))
            bx = f.sub_i(f.mul_i(ox, f.const(sw)), f.const(pl))
            def ky_loop(ky):
                iy = f.add_i(by, f.mul_i(ky, f.const(dh)))
                def kx_loop(kx):
                    ix = f.add_i(bx, f.mul_i(kx, f.const(dw)))
                    source_base = f.mul_i(f.add_i(f.mul_i(f.add_i(
                        f.mul_i(bn, f.const(height)), iy), f.const(width)), ix), f.const(ci))
                    target_base = f.add_i(f.mul_i(local_row, f.const(k)),
                        f.mul_i(f.add_i(f.mul_i(ky, f.const(kw)), kx), f.const(ci)))
                    if pack_wide:
                        from .conv_pack_load import pack_tap
                        valid = None
                        if any(padding):
                            def cmp(a, b, pred):
                                return f.add(llvm.ICmpOp(a, b, IntegerAttr(pred.to_int(), i64))).results[0]
                            valid = f.and_i(cmp(iy, f.const(0), llvm.ICmpPredicateFlag.SGE),
                                            cmp(iy, f.const(height), llvm.ICmpPredicateFlag.SLT))
                            valid = f.and_i(valid, cmp(ix, f.const(0), llvm.ICmpPredicateFlag.SGE))
                            valid = f.and_i(valid, cmp(ix, f.const(width), llvm.ICmpPredicateFlag.SLT))
                        pack_tap(f, src, scratch, zero, source_base, target_base, ci, valid)
                        return
                    def channels(channel):
                        length = f.smin(f.const(16), f.sub_i(f.const(ci), channel))
                        source = f.gep(src, f.add_i(source_base, channel), i8)
                        target = f.gep(scratch, f.add_i(target_base, channel), i8)
                        if any(padding):
                            def cmp(a, b, pred):
                                return f.add(llvm.ICmpOp(a, b, IntegerAttr(pred.to_int(), i64))).results[0]
                            valid = f.and_i(cmp(iy, f.const(0), llvm.ICmpPredicateFlag.SGE),
                                            cmp(iy, f.const(height), llvm.ICmpPredicateFlag.SLT))
                            valid = f.and_i(valid, cmp(ix, f.const(0), llvm.ICmpPredicateFlag.SGE))
                            valid = f.and_i(valid, cmp(ix, f.const(width), llvm.ICmpPredicateFlag.SLT))
                            inside, outside, done = Block(), Block(), Block()
                            for bb in (inside, outside, done):
                                f.region.add_block(bb)
                            f.add(llvm.CondBrOp(valid, inside, [], outside, []))
                            f.blk = inside
                            isa.mvin(f, source, pack_local, length, 1)
                            f.add(llvm.BrOp(done))
                            f.blk = outside
                            isa.mvin(f, zero, pack_local, length, 1)
                            f.add(llvm.BrOp(done))
                            f.blk = done
                        else:
                            isa.mvin(f, source, pack_local, length, 1)
                        isa.mvout(f, target, pack_local, length, 1)
                    f.for_loop(0, ci, 16, channels, retain_loop=True)
                f.for_loop(0, kw, 1, kx_loop, retain_loop=True)
            f.for_loop(0, kh, 1, ky_loop, retain_loop=True)
        f.for_loop(0, count, 1, row, retain_loop=True)
        isa.fence(f)

    def block(row_start, count):
        def gather(flat):
            row = f.add_i(row_start, f.udiv_i(flat, f.const(k)))
            tap = f.urem_i(flat, f.const(k))
            channel = f.urem_i(tap, f.const(ci))
            kernel_x = f.urem_i(f.udiv_i(tap, f.const(ci)), f.const(kw))
            kernel_y = f.udiv_i(tap, f.const(ci * kw))
            out_x = f.urem_i(row, f.const(wo))
            out_y = f.urem_i(f.udiv_i(row, f.const(wo)), f.const(ho))
            bn = f.udiv_i(row, f.const(ho * wo))
            iy = f.sub_i(f.add_i(f.mul_i(out_y, f.const(sh)),
                                  f.mul_i(kernel_y, f.const(dh))), f.const(pt))
            ix = f.sub_i(f.add_i(f.mul_i(out_x, f.const(sw)),
                                  f.mul_i(kernel_x, f.const(dw))), f.const(pl))
            index = f.add_i(f.mul_i(f.add_i(f.mul_i(f.add_i(f.mul_i(bn,
                f.const(height)), iy), f.const(width)), ix), f.const(ci)), channel)
            if any(padding):
                _copy_if_in_bounds(f, src, scratch, index, flat, iy, ix, height, width)
            else:
                value = f.add(llvm.LoadOp(f.gep(src, index, i8), i8)).results[0]
                f.add(llvm.StoreOp(value, f.gep(scratch, flat, i8)))
        def structured_row(local_row):
            row = f.add_i(row_start, local_row)
            out_x = f.urem_i(row, f.const(wo))
            out_y = f.urem_i(f.udiv_i(row, f.const(wo)), f.const(ho))
            bn = f.udiv_i(row, f.const(ho * wo))
            row_y = f.sub_i(f.mul_i(out_y, f.const(sh)), f.const(pt))
            row_x = f.sub_i(f.mul_i(out_x, f.const(sw)), f.const(pl))
            def kernel_row(ky):
                iy = f.add_i(row_y, f.mul_i(ky, f.const(dh)))
                def kernel_column(kx):
                    ix = f.add_i(row_x, f.mul_i(kx, f.const(dw)))
                    source_base = f.mul_i(f.add_i(f.mul_i(f.add_i(
                        f.mul_i(bn, f.const(height)), iy), f.const(width)), ix), f.const(ci))
                    target_base = f.add_i(f.mul_i(local_row, f.const(k)),
                        f.mul_i(f.add_i(f.mul_i(ky, f.const(kw)), kx), f.const(ci)))
                    def copy_channels(channel):
                        source_index = f.add_i(source_base, channel)
                        target_index = f.add_i(target_base, channel)
                        value = f.add(llvm.LoadOp(f.gep(src, source_index, i8), i8)).results[0]
                        f.add(llvm.StoreOp(value, f.gep(scratch, target_index, i8)))
                    if any(padding):
                        # Validity is spatial, shared by all channels of a tap.
                        # Branch once per tap, then run a pure copy/zero loop.
                        def cmp(a, b, pred):
                            return f.add(llvm.ICmpOp(a, b, IntegerAttr(pred.to_int(), i64))).results[0]
                        valid = f.and_i(cmp(iy, f.const(0), llvm.ICmpPredicateFlag.SGE),
                                        cmp(iy, f.const(height), llvm.ICmpPredicateFlag.SLT))
                        valid = f.and_i(valid, cmp(ix, f.const(0), llvm.ICmpPredicateFlag.SGE))
                        valid = f.and_i(valid, cmp(ix, f.const(width), llvm.ICmpPredicateFlag.SLT))
                        inside, outside, done = Block(), Block(), Block()
                        for bb in (inside, outside, done):
                            f.region.add_block(bb)
                        f.add(llvm.CondBrOp(valid, inside, [], outside, []))
                        f.blk = inside
                        f.for_loop(0, ci, 1, copy_channels, retain_loop=True)
                        f.add(llvm.BrOp(done))
                        f.blk = outside
                        def zero_channels(channel):
                            target_index = f.add_i(target_base, channel)
                            zero = f.add(llvm.TruncOp(f.const(0), i8)).results[0]
                            f.add(llvm.StoreOp(zero, f.gep(scratch, target_index, i8)))
                        f.for_loop(0, ci, 1, zero_channels, retain_loop=True)
                        f.add(llvm.BrOp(done))
                        f.blk = done
                    else:
                        f.for_loop(0, ci, 1, copy_channels, retain_loop=True)
                f.for_loop(0, kw, 1, kernel_column, retain_loop=True)
            f.for_loop(0, kh, 1, kernel_row, retain_loop=True)
        if prepare == 'auto':
            device_rows(row_start, count)
        elif prepare == 'generic':
            f.for_loop(0, count * k, 1, gather, retain_loop=True)
        else:
            f.for_loop(0, count, 1, structured_row, retain_loop=True)
        from merlin.llvmlower.host_builder import INT_TYPES
        output = f.gep(dst, f.mul_i(row_start, f.const(co)), INT_TYPES[output_dtype])
        matmul_attrs = dict(attrs)
        bias = attrs.get('bias')
        if bias not in (None, False, 'none', '') and not attrs.get('bias_broadcast', True):
            pointer = ctx.ptr(bias) if isinstance(bias, str) else bias
            matmul_attrs['bias'] = f.gep(pointer, f.mul_i(row_start, f.const(co)), INT_TYPES['i32'])
        if prepared_columns is not None:
            lower_prepared(ctx, scratch, output, count, prepared_columns)
        else:
            lower_matmul(ctx, scratch, rhs, output, count, k, co, matmul_attrs)

    full_rows = m // tile_rows * tile_rows
    f.for_loop(0, full_rows, tile_rows,
               lambda row: block(row, tile_rows), retain_loop=True)
    if full_rows < m:
        block(f.const(full_rows), m - full_rows)

    finish_pool()
