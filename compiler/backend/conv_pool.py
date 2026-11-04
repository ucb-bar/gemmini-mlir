"""Private convolution/requant/maxpool composition in bounded ACC stripes.

This is an optional adapter for an atomic CONV2D maxpool epilogue. It retains
raw producer results in a private halo stripe and applies one proved readout
when the hardware pools that stripe. No full i8 intermediate is published.
Overlapping halo products can be recomputed; cycle selection remains external.
"""
from __future__ import annotations

from math import isfinite, prod

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.spatial_producer import SpatialProducerPlan, emit_spatial_producer
from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from .conv_control import ConvolutionControlAdapter as FnBuilderControlAdapter

from . import isa


def lower_conv_pool(ctx, ifm, rhs, dst, attributes, *, require_full_block=False):
    """Return False without emission when this atomic source chain is unsupported.

    The caller must pass the original CONV2D epilogue, whose intermediate has
    no separate source identity or observer. Separate graph operations require
    an independently proved source matcher before invoking this adapter.
    """
    attrs = dict(attributes or {})
    ep = tuple(attrs.get('epilogue', ()))
    if not ep or ep[-1] != 'maxpool' or attrs.get('external_observers'):
        return False
    try:
        ins, weights, out = ctx.spec(ifm), ctx.spec(rhs), ctx.spec(dst)
        shape = tuple(map(int, ins['shape']))
        kh, kw, ci, co = map(int, attrs['kernel'])
        if (len(shape) != 4 or shape[-1] != ci or attrs.get('layout', 'nhwc').lower() != 'nhwc'
                or ins['dtype'] != 'i8' or weights['dtype'] != 'i8' or out['dtype'] != 'i8'
                or attrs.get('output_dtype', 'i8') != 'i8'
                or prod(weights['shape']) != kh * kw * ci * co):
            return False
        padding = tuple(map(int, attrs.get('padding', (0, 0, 0, 0))))
        if len(padding) == 2:
            padding = (*padding, *padding)
        window = Window2D(shape, (kh, kw), tuple(attrs.get('stride', (1, 1))),
                          tuple(attrs.get('dilation', (1, 1))), padding)
        ph, pw = map(int, attrs['pool_size'])
        sh, sw = map(int, attrs['pool_stride'])
        if ph != pw or not 1 <= ph <= 3 or not 1 <= sw <= 3 or sh < 1:
            return False
        pool_padding = tuple(map(int, attrs.get('pool_padding', (0, 0, 0, 0))))
        if any(pool_padding) and attrs.get('pool_pad_value') != -128:
            return False
        if tuple(attrs['pool_in_dims']) != window.output_spatial:
            return False
        pool = SpatialMaxReadout(window.output_spatial, (ph, pw), (sh, sw),
                                pool_padding, -128 if any(pool_padding) else None)
        if prod(out['shape']) != shape[0] * prod(pool.output_hw) * co:
            return False
        # Uniform finite nonnegative scaling is monotone through signed max.
        # The library additionally proves every rounding/saturation boundary.
        scales = attrs.get('acc_scale_by_column')
        if scales is not None and len(scales) != co:
            return False
        column_readout = None
        if scales is not None and len(set(map(float, scales))) != 1:
            from .pool_fusion_columns import ColumnSpatialReadout
            column_readout = ColumnSpatialReadout.prepare(attrs, co, automatic=require_full_block)
        scale = float(scales[0] if scales else attrs.get('acc_scale', 1.0))
        if not isfinite(scale) or scale < 0 or attrs.get('requant_shift') is not None:
            return False
        if any(stage not in ('bias_add', 'acc_scale', 'relu', 'maxpool', 'requant') for stage in ep):
            return False
        bias = attrs.get('bias')
        if 'bias_add' in ep and not bias:
            return False
        if bias and (ctx.spec(bias)['dtype'] != 'i32' or prod(ctx.spec(bias)['shape']) != co):
            return False
        if dst in (ifm, rhs, bias):
            return False
        buffers = getattr(ctx, 'buffers', {})
        output_pointer = buffers.get(dst)
        if output_pointer is not None and any(
                output_pointer is buffers.get(name) for name in (ifm, rhs, bias)):
            return False
        q = RoundedReadout((scale,), bool(attrs.get('relu') or 'relu' in ep))
        chain = SpatialReadoutChain(q, RoundedReadout((1.0,)), q)
        column_tile = 1 if column_readout is not None else min(isa.DIM, co)
        # Eight-bit pool geometry fields and physical ACC capacity bound the
        # stripe. No image extent determines a private allocation or code size.
        max_width = min(255, isa.ACC_ROWS // ph)
        stripe = min(pool.output_hw[1], (max_width - pw) // sw + 1, 255)
        if stripe < 1:
            return False
        p = CountedConvolutionPlan(window, co, isa.DIM,
            min(isa.DIM, window.output_spatial[1]), column_tile, min(isa.DIM, ci),
            isa.SPAD_ROWS, isa.ACC_ROWS, 0, isa.SPAD_ROWS // 2, 0, bias=bool(bias))
        source = SpatialProducerPlan(p, pool,
            (column_readout.chains if column_readout is not None else
             (chain,) * ((co + p.column_tile - 1) // p.column_tile)), stripe)
        block_plan = None
        epochs = None
        resident = None
        halo = None
        if sh == sw and pool.output_hw[1] <= 255:
            from .conv_pool_blocks import SpatialBlockPlan
            local_width = (stripe - 1) * sw + pw
            max_height = min(255, isa.ACC_ROWS // local_width)
            block_height = min(pool.output_hw[0], (max_height - ph) // sh + 1, 255)
            if block_height >= 1:
                block_plan = SpatialBlockPlan(source, block_height)
                from .pool_fusion_padding import MinimumSpatialHalo
                halo = MinimumSpatialHalo.prepare(block_plan)
                from .conv_pool_blocks import plan_spatial_epochs
                try:
                    from .pool_fusion_resident import ResidentSpatialRHS
                    resident = ResidentSpatialRHS.prepare(block_plan)
                    epochs = plan_spatial_epochs(block_plan, complete_reduction=resident is not None)
                except ValueError:
                    # Unknown ordering keeps the complete synchronous route.
                    epochs = None
                    resident = None
    except (KeyError, TypeError, ValueError, OverflowError):
        return False

    # Auto selects a shared physical resource class, never source/model IDs.
    # Both measured representatives fit one spatial union and one primitive N
    # tile with complete-K retained RHS. Capacity-bound multiblock schedules
    # remain explicitly available through on, without extrapolated timing.
    if require_full_block and (
            block_plan is None or resident is None or epochs is None
            or co > p.primitive
            or block_plan.output_tile_height != pool.output_hw[0]
            or source.output_tile_width != pool.output_hw[1]):
        return False

    fb = ctx.fb
    control = FnBuilderControlAdapter(fb)
    ctx.fence()
    halo_name = None
    if halo is not None:
        halo_name = '_pool_raw_minimum_' + str(len(ctx.tensors))
        while halo_name in ctx.tensors:
            halo_name += '_'
        ctx.constant(halo_name, [isa.DIM], 'i32', [halo.raw_value] * isa.DIM)
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(co, channel=1))
    if resident is not None:
        ctx.emit(*isa.config_ld(window.stride[1] * shape[-1], channel=0))
    if bias or halo is not None:
        ctx.emit(*isa.config_ld(0, channel=2, shrink=False))

    def lower(kind, f, *, complete=True):
        if kind == 'load_input':
            if resident is None:
                ctx.emit(*isa.config_ld(f['row_stride_elements'], fb=fb))
            pointer = 0 if f['zero'] else ctx.ptr(ifm, f['first_element'])
            ctx.emit(*isa.mvin(pointer, f['local_row'], f['rows'], f['columns'], fb=fb))
        elif kind == 'load_weight':
            offset = control.op('add', control.op('mul', f['reduction_row'], co), f['output_column'])
            ctx.emit(*isa.mvin(ctx.ptr(rhs, offset), f['local_row'], f['rows'], f['columns'], channel=1, fb=fb))
        elif kind == 'initialize_bias':
            ctx.emit(*isa.mvin(ctx.ptr(bias, f['output_column']), isa.acc_addr(f['accumulator_row'], fb=fb),
                              f['rows'], f['columns'], channel=2, fb=fb))
        elif kind == 'initialize_pool_halo':
            if halo_name is None:
                raise ValueError('unproved raw pooling halo initialization')
            ctx.emit(*isa.mvin(ctx.ptr(halo_name), isa.acc_addr(f['accumulator_row'], fb=fb),
                              f['rows'], f['columns'], channel=2, fb=fb))
        elif kind == 'compute':
            def product(accumulate):
                acc = isa.acc_addr(f['accumulator_row'], accumulate=accumulate, fb=fb)
                ctx.emit(*isa.preload(f['rhs_row'], acc, weight_rows=f['depth'], weight_cols=f['columns'],
                                     c_rows=f['rows'], c_cols=f['columns'], fb=fb))
                ctx.emit(*isa.compute_preloaded(f['lhs_row'], a_rows=f['rows'], a_cols=f['depth'],
                                               b_rows=f['depth'], b_cols=f['columns'], fb=fb))
            if bias:
                product(True)
            else:
                control.branch(f['first_reduction'], lambda: product(False), lambda: product(True))
        elif kind == 'finish_raw_tile':
            pass  # Keep the completed raw result in its original ACC rows.
        elif kind == 'store_spatial_max':
            pool_fields = dict(pool_stride=sw, pool_size=ph,
                pool_out_dim=f.get('output_row_stride', f['output_pixels']),
                porows=f.get('output_rows', 1), pocols=f['output_pixels'],
                orows=f['local_height'], ocols=f['local_width'])
            if column_readout is None:
                ctx.emit(*isa.config_st(co, scale=scale, activation=int(q.relu), **pool_fields))
            else:
                column_readout.emit_config(ctx, f['output_column'], stride=co, **pool_fields)
            offset = control.op('add', control.op('mul', f['output_row'], co), f['output_column'])
            ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(f['accumulator_row'], fb=fb),
                               0, f['columns'], fb=fb))
        else:
            raise ValueError('unknown spatial producer effect ' + kind)
        if complete:
            ctx.fence()

    if block_plan is None:
        receipt = emit_spatial_producer(source, control, lower, synchronous_completion=True)
    else:
        from .conv_pool_blocks import emit_spatial_blocks
        if epochs is None:
            receipt = emit_spatial_blocks(block_plan, control, lower, synchronous_completion=True, halo=halo)
        else:
            receipt = emit_spatial_blocks(block_plan, control,
                lambda kind, fields: lower(kind, fields, complete=False),
                synchronous_completion=False, epochs=epochs,
                complete=lambda kind, fields: ctx.fence(), resident=resident, halo=halo)
    ctx.fence()
    if column_readout is not None:
        receipt = dict(receipt, column_readout=column_readout.receipt())
    ctx.metadata.setdefault('convolution_pool_compositions', []).append(dict(receipt,
        selection=('resident_rhs_accumulator_blocks' if resident else
            'accumulator_blocks' if block_plan else 'accumulator_stripes'),
        full_intermediate_eliminated=True,
        automatic_admission=(dict(
            predicate='one full pooled spatial block; one primitive N tile; complete-K resident RHS and checked epochs',
            primitive=p.primitive, output_channels=co,
            output_tile_height=block_plan.output_tile_height,
            output_tile_width=source.output_tile_width,
            private_accumulator_rows=block_plan.local_height * block_plan.local_width,
            retained_rhs_rows=resident.panel.rhs_rows,
            timing_scope='paired complete calls on two representatives; larger multiblock route unselected')
            if require_full_block else None),
        catalog_route='producer_pool', target_encoder='backend.isa', timing='UNKNOWN'))
    return True
