"""Counted NHWC convolution using checked affine window transfers.

Interior spatial tiles transfer a strided rectangle in one DMA. Border rows
load either original input or the accelerator zero source, with synchronous
completion before any local storage is reused. Host code only forms addresses.
"""
from __future__ import annotations

from math import prod
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import (
    CountedConvolutionPlan, Window2D, emit_counted_convolution,
)
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from . import isa
from .dense_readout import store_tile


def lower_conv(ctx, ifm: str, rhs: str, dst: str, attributes: dict, *, physical_views=None):
    attrs = dict(attributes or {})
    if any('pool' in str(stage).lower() for stage in attrs.get('epilogue', ())):
        raise ValueError('convolution pooling epilogue needs spatial pooling lowering')
    shape = tuple(map(int, ctx.spec(ifm)['shape']))
    if len(shape) != 4 or attrs.get('layout', 'nhwc').lower() != 'nhwc':
        raise ValueError('convolution requires explicit rank-four NHWC input')
    if ctx.spec(ifm)['dtype'] != 'i8' or ctx.spec(rhs)['dtype'] != 'i8':
        raise ValueError('convolution operands must be signed i8')
    kernel = tuple(map(int, attrs['kernel']))
    if len(kernel) != 4 or kernel[2] != shape[-1]:
        raise ValueError('convolution kernel/input channel geometry differs')
    kh, kw, ci, co = kernel
    rhs_pitch = output_pitch = co
    if physical_views is not None:
        from .conv_channel_phases import ConvPhysicalViews
        if (type(physical_views) is not ConvPhysicalViews
                or physical_views.logical_channels != co
                or physical_views.channel_base < 0
                or physical_views.channel_base + co > physical_views.source_channels
                or physical_views.rhs_row_stride != physical_views.source_channels
                or physical_views.output_row_stride != physical_views.source_channels):
            raise ValueError('convolution physical views must preserve the original channel pitches')
        rhs_pitch = physical_views.rhs_row_stride
        output_pitch = physical_views.output_row_stride
    padding = tuple(map(int, attrs.get('padding', (0, 0, 0, 0))))
    if len(padding) == 2:
        padding = (*padding, *padding)
    window = Window2D(shape, (kh, kw), tuple(attrs.get('stride', (1, 1))),
                      tuple(attrs.get('dilation', (1, 1))), padding)
    oh, ow = window.output_spatial
    if prod(ctx.spec(rhs)['shape']) != kh * kw * ci * co:
        raise ValueError('convolution RHS must contain original flattened KH/KW/CI/CO weights')
    if prod(ctx.spec(dst)['shape']) != shape[0] * oh * ow * co:
        raise ValueError('convolution output shape differs from original source window')
    bias = attrs.get('bias')
    if 'bias_add' in attrs.get('epilogue', ()) and not bias:
        raise ValueError('convolution bias epilogue requires a named bias tensor')
    if bias and (ctx.spec(bias)['dtype'] != 'i32' or prod(ctx.spec(bias)['shape']) != co):
        raise ValueError('convolution bias must contain one i32 per output channel')
    scales = attrs.get('acc_scale_by_column')
    if scales is not None and len(scales) != co:
        raise ValueError('convolution column scales differ from output channels')
    # Mixed scale columns each own lane zero, since the readout address has no
    # arbitrary column origin. Uniform columns retain the whole primitive tile.
    mixed_scales = scales is not None and len(set(map(float, scales))) > 1
    configured = knob('CONV_COLUMN_READOUT', default='recompute',
        values=('recompute', 'raw_reload'), root=Path(__file__).resolve().parents[1])
    column_route = attrs.get('conv_column_readout', configured)
    if column_route not in ('recompute', 'raw_reload'):
        raise ValueError('CONV_COLUMN_READOUT must be recompute or raw_reload')
    if mixed_scales and column_route == 'raw_reload':
        from .conv_column import lower_conv_column_readout
        return lower_conv_column_readout(ctx, ifm, rhs, dst, attrs)
    tile_n = 1 if mixed_scales else min(isa.DIM, co)
    # A one-fragment reduction still has the source's additive identity. Make
    # that identity an explicit device initializer, then accumulate the actual
    # product. The plan owns the seed just as it owns a broadcast bias, including
    # separate retained output tiles and every spatial tile lifetime.
    zero_seed = not bias and kh * kw == 1 and ci <= isa.DIM
    seeded = bool(bias) or zero_seed
    plan = CountedConvolutionPlan(window, co, isa.DIM, min(isa.DIM, ow),
        tile_n, min(isa.DIM, ci), isa.SPAD_ROWS, isa.ACC_ROWS,
        0, isa.SPAD_ROWS // 2, 0, bias=seeded)
    fb = ctx.fb
    control = FnBuilderControlAdapter(fb)
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(rhs_pitch, channel=1))
    if seeded:
        ctx.emit(*isa.config_ld(0, channel=2, shrink=False))

    def emit(kind, f, *, complete=True):
        if kind == 'load_input':
            # A null DRAM source is the selected accelerator's zero fill. No
            # subsequent write overwrites this row before its consuming compute.
            ctx.emit(*isa.config_ld(f['row_stride_elements'], channel=0, fb=fb))
            address = 0 if f['zero'] else ctx.ptr(ifm, f['first_element'])
            ctx.emit(*isa.mvin(address, f['local_row'], f['rows'], f['columns'], fb=fb))
            if complete:
                ctx.fence()
        elif kind == 'load_weight':
            offset = control.op('add', control.op('mul', f['reduction_row'], rhs_pitch), f['output_column'])
            ctx.emit(*isa.mvin(ctx.ptr(rhs, offset), f['local_row'],
                f['rows'], f['columns'], channel=1, fb=fb))
            if complete:
                ctx.fence()
        elif kind == 'initialize_bias':
            pointer = 0 if zero_seed else ctx.ptr(bias, f['output_column'])
            ctx.emit(*isa.mvin(pointer,
                isa.acc_addr(f['accumulator_row'], fb=fb), f['rows'],
                f['columns'], channel=2, fb=fb))
            if complete:
                ctx.fence()
        elif kind == 'compute':
            def product(accumulate):
                # Emit the actual overwrite/continuation modes at distinct
                # command sites. Dynamic source coordinates still select the
                # branch, while the PRELOAD accumulator-mode bits are literal.
                acc = isa.acc_addr(f['accumulator_row'], accumulate=accumulate, fb=fb)
                ctx.emit(*isa.preload(f['rhs_row'], acc, weight_rows=f['depth'],
                    weight_cols=f['columns'], c_rows=f['rows'], c_cols=f['columns'], fb=fb))
                ctx.emit(*isa.compute_preloaded(f['lhs_row'], a_rows=f['rows'],
                    a_cols=f['depth'], b_rows=f['depth'], b_cols=f['columns'], fb=fb))
            if seeded:
                product(True)
            else:
                control.branch(f['first_reduction'], lambda: product(False), lambda: product(True))
            if complete:
                ctx.fence()
        elif kind == 'store':
            store_tile(ctx, dst, f['accumulator_row'], f['rows'], f['columns'],
                       output_pitch, f['output_row'], f['output_column'], attrs)
        else:
            raise ValueError('unknown convolution effect ' + kind)

    selection = knob('CONV_INPUT_RETENTION', default='off',
        values=('off', 'on', 'resident', 'compute_epochs', 'reduction_epochs'), root=Path(__file__).resolve().parents[1])
    receipt = _emit_plan(ctx, plan, control, emit, selection, ifm, rhs, dst)
    if zero_seed and hasattr(ctx, 'metadata'):
        ctx.metadata.setdefault('convolution_zero_initializers', []).append(dict(
            source_sha256=plan.identity, initializer='device_null_mvin_i32',
            lifetime='one exact zero seed per scheduled accumulator tile',
            first_product='accumulate_into_initialized_zero'))
    ctx.fence()
    return receipt


def _emit_plan(ctx, plan, control, emit, selection, ifm, rhs, dst):
    """Apply one reusable lifetime alternative to the original convolution.

    Both paths retain every terminal callback fence. Input retention needs no
    asynchronous ordering admission: only the actual disjoint private SPAD/ACC
    assignment and immutable external operands. Resident weights are prepared
    once per invocation and complete before any product reads them.
    """
    if selection == 'off':
        return emit_counted_convolution(plan, control, emit, synchronous_completion=True)
    from merlin.compile.scheduling.window_transfer import ConvolutionInputRetentionLaw
    from merlin.compile.scheduling.convolution_exploration import (
        ConvolutionExplorationSource, ConvolutionScheduleChoice,
        ConvolutionRhsResidencyLaw, plan_convolution_schedule,
        plan_convolution_rhs_residency, emit_convolution_schedule,
    )
    intervals = tuple((column, min(plan.column_tile, plan.output_channels-column))
                      for column in range(0, plan.output_channels, plan.column_tile))
    # This consumer statically names original N intervals while K and spatial
    # axes stay counted. Bound that expansion to keep target text independent
    # of the large spatial/reduction workload. Wider domains use the incumbent.
    if len(intervals) > 32:
        receipt = emit_counted_convolution(plan, control, emit, synchronous_completion=True)
        application = dict(selection=selection, applied_selection='off',
            source_sha256=plan.identity, input_retention=False, rhs_resident=False,
            static_column_intervals=len(intervals),
            alternative_declined='more than 32 static original N intervals',
            completion='original synchronous terminal fences retained', timing='UNKNOWN')
        # The compiler's caller consumes metadata rather than this return value.
        # Preserve a declined route there so a requested epoch trial cannot be
        # mistaken for an applied epoch program on wide or mixed-scale outputs.
        if hasattr(ctx, 'metadata'):
            ctx.metadata.setdefault('convolution_alternatives', []).append(application)
        return dict(receipt, requested_alternative=selection,
                    alternative_declined=application['alternative_declined'],
                    applied_alternative=application)
    if ctx.buffer(dst) is ctx.buffer(ifm) or ctx.buffer(dst) is ctx.buffer(rhs):
        raise ValueError('convolution retention requires disjoint original output storage')
    source = ConvolutionExplorationSource(plan, intervals, plan.identity)
    hardware = 'gemmini-fire-aligned-b8ff28edc096392c'
    numeric = 'signed-i8-products-i32-source-ordered-readout'
    qualification = 'synchronous-disjoint-lhs-rhs-acc-private-rows-v1'
    retention = ConvolutionInputRetentionLaw(hardware, numeric, source.identity,
        qualification, max_live_tiles=min(len(intervals),
            (plan.accumulator_rows-plan.accumulator_row)//plan.row_tile),
        input_preserved=True, independent_outputs=True,
        immutable_mapped_disjoint_buffers=True)
    issue_law = epoch_law = epoch_report = None
    pairs = 1
    if selection in ('compute_epochs', 'reduction_epochs'):
        from .conv_epochs import epoch_admissions
        issue_law, epoch_law, pairs, epoch_report = epoch_admissions(
            source, hardware, numeric, qualification,
            complete_reduction=selection=='reduction_epochs')
    schedule = plan_convolution_schedule(source,
        ConvolutionScheduleChoice(live_tiles=None, load_issue=issue_law is not None,
                                  compute_pairs=pairs),
        retention_law=retention, issue_law=issue_law, epoch_law=epoch_law)
    owner = None
    if selection in ('resident', 'compute_epochs', 'reduction_epochs'):
        rhs_law = ConvolutionRhsResidencyLaw(hardware, numeric, source.identity,
            qualification, weight_transfer_only=True,
            immutable_mapped_disjoint_buffers=True, other_effects_preserve_rhs=True,
            exclusive_private_rows=True, entry_quiescence_and_complete_preparation=True)
        base = -(-(plan.lhs_row + plan.row_tile) // plan.channel_tile) * plan.channel_tile
        owner = plan_convolution_rhs_residency(schedule, operand_rows=plan.operand_rows,
            law=rhs_law, rhs_base=base)
    effect = (lambda kind, fields: emit(kind, fields, complete=False)) if epoch_law else emit
    receipt = emit_convolution_schedule(schedule, control, effect,
        complete=lambda kind, fields: ctx.fence(), synchronous_completion=True, rhs=owner)
    application = dict(selection=selection, applied_selection=selection,
        source_sha256=source.identity,
        schedule_sha256=schedule.identity, input_retention=True,
        live_column_tiles=max(map(len, schedule.column_blocks)),
        rhs_resident=owner is not None,
        rhs_residency_sha256=owner.identity if owner is not None else None,
        completion=('library-owned runtime compute epochs; original store/full-call drains'
                    if epoch_law else 'original synchronous terminal fences retained'),
        compute_epochs=epoch_report,
        static_column_intervals=len(intervals), timing='UNKNOWN')
    if hasattr(ctx, 'metadata'):
        ctx.metadata.setdefault('convolution_alternatives', []).append(application)
    return dict(receipt, applied_alternative=application)
