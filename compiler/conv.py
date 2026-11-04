"""Counted convolution over window row runs, with retained input and RHS panels.

Source tensors keep their logical row-major ABI. The library controls all window
indices, border predicates, channel tails and invocation-private panel lifetimes.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from merlin.compile.scheduling.window_transfer import (
    Window2D, CountedConvolutionPlan, ConvolutionInputRetentionLaw,
)
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource, ConvolutionScheduleChoice,
    ConvolutionRhsResidencyLaw, plan_convolution_schedule,
    plan_convolution_rhs_residency, emit_convolution_schedule,
)


@lru_cache(maxsize=1)
def _geometry():
    facts = json.loads((Path(__file__).resolve().parent / 'rtl_facts/facts.json').read_text())['facts']
    mesh = next(a for a in facts['arrays'] if a['name'] == 'mesh')
    mem = {a['name']: a for a in facts['memories']}
    edge = min(mesh['rows'], mesh['cols'])
    return edge, mem['scratchpad']['bytes'] // edge, mem['accumulator']['bytes'] // (edge * 4)


def _source_name(cb, name):
    aliases = {c['operands']['dst']: c['operands']['src'] for c in cb['commands']
               if c['opcode'] == 'RES_PACK'}
    seen = set()
    while name in aliases:
        if name in seen:
            raise ValueError('cyclic resident convolution alias')
        seen.add(name)
        name = aliases[name]
    return name


def _column_intervals(attrs, cout, edge):
    scales = attrs.get('acc_scale_by_column')
    if scales is not None:
        if len(scales) != cout:
            raise ValueError('convolution scale vector differs from output channels')
        scales = [float(v) for v in scales]
    else:
        scales = [float(attrs.get('acc_scale', 1.0))] * cout
    result = []
    col = 0
    while col < cout:
        end = col + 1
        while end < min(cout, col + edge) and scales[end] == scales[col]:
            end += 1
        result.append((col, end-col))
        col = end
    return tuple(result), scales


def lower_conv(builder, command_buffer, command, pointers):
    """Lower one CONV2D into ordinary Gemmini commands; return schedule evidence.

    ``pointers`` maps logical source names (and optionally resident aliases) to
    Builder pointer values. Bias, scale, activation and narrowing are completed
    by accumulator transfer/readout. No contraction arithmetic runs on the CPU.
    """
    try:
        from . import target
    except ImportError:
        import target
    b, cb = builder, command_buffer
    attrs, operands = command.get('attributes', {}), command['operands']
    ifm_name = _source_name(cb, operands['ifm'])
    weight_name = _source_name(cb, operands['weight'])
    output_name = operands.get('dst', operands.get('out'))
    tensors = cb['tensors']
    shape = tuple(tensors[ifm_name]['shape'])
    layout = attrs.get('layout', 'nhwc').lower()
    if layout != 'nhwc':
        raise ValueError('source convolution layout must be NHWC')
    if len(shape) != 4 or tensors[ifm_name]['dtype'] != 'i8' or tensors[weight_name]['dtype'] != 'i8':
        raise ValueError('convolution requires rank-four NHWC i8 input and i8 weights')
    kernel = attrs.get('kernel')
    ci = shape[-1]
    kh, kw = (int(kernel[0]), int(kernel[1])) if kernel else (int(attrs['kh']), int(attrs['kw']))
    wshape = tensors[weight_name]['shape']
    cout = int(kernel[3]) if kernel and len(kernel) == 4 else int(wshape[-1])
    if list(wshape) != [kh*kw*ci, cout]:
        raise ValueError('convolution RHS must be logical [KH*KW*CI,CO]')
    window = Window2D(shape, (kh, kw), tuple(attrs.get('stride', (1, 1))),
                      tuple(attrs.get('dilation', (1, 1))), tuple(attrs.get('padding', (0, 0, 0, 0))))
    epilogue = list(attrs.get('epilogue', ()))
    if 'maxpool' in epilogue:
        if epilogue[-1] != 'maxpool' or epilogue.count('maxpool') != 1:
            raise ValueError('convolution pooling must be the final single epilogue stage')
        if attrs.get('output_dtype', tensors[output_name]['dtype']) != 'i8':
            raise ValueError('native pooling requires signed-i8 epilogue values')
        batch = shape[0]
        oh, ow = window.output_spatial
        if tuple(attrs.get('pool_in_dims', ())) != (oh, ow):
            raise ValueError('pool input dimensions differ from convolution output geometry')
        temporary_name = '__conv_before_pool_' + output_name
        temporary = b.alloca(batch*oh*ow*cout, 'i8')
        inner_cb = dict(cb, tensors=dict(tensors))
        inner_cb['tensors'][temporary_name] = {'shape': [batch*oh*ow, cout], 'dtype': 'i8', 'role': 'intermediate'}
        inner_operands = dict(operands, dst=temporary_name)
        inner_attrs = dict(attrs, epilogue=epilogue[:-1])
        inner_command = dict(command, operands=inner_operands, attributes=inner_attrs)
        inner_pointers = dict(pointers, **{temporary_name: temporary})
        convolution_evidence = lower_conv(b, inner_cb, inner_command, inner_pointers)
        pooling_evidence = lower_maxpool(b, temporary, pointers[output_name], cout, attrs, batches=batch)
        return {'convolution': convolution_evidence, 'pooling': pooling_evidence}
    allowed = {'bias', 'bias_add', 'acc_scale', 'relu', 'clamp', 'saturate', 'cast', 'to_i8'}
    unknown = set(epilogue) - allowed
    if unknown:
        raise ValueError('unsupported convolution epilogue stages: ' + repr(sorted(unknown)))
    dtype = attrs.get('output_dtype', tensors[output_name]['dtype'])
    if dtype not in ('i8', 'i32'):
        raise ValueError('convolution supports exact i32 or narrowing i8 output')
    if dtype == 'i32' and any(stage in epilogue for stage in ('acc_scale', 'relu')):
        raise ValueError('full-width accumulator readout cannot absorb scale or activation')
    bias = any(stage in epilogue for stage in ('bias', 'bias_add'))
    if bias:
        bias_name = attrs.get('bias', operands.get('bias'))
        if not bias_name:
            raise ValueError('convolution bias epilogue lacks a tensor')
    edge, operand_rows, accumulator_rows = _geometry()
    row_tile = min(edge, window.output_spatial[1])
    plan = CountedConvolutionPlan(window, cout, edge, row_tile, edge, min(edge, ci),
                                  operand_rows, accumulator_rows, 0, edge, 0, bias=bias)
    intervals, scales = _column_intervals(attrs, cout, edge)
    source_id = hashlib.sha256(json.dumps(command, sort_keys=True).encode()).hexdigest()
    source = ConvolutionExplorationSource(plan, intervals, source_id)
    identities = ('gemmini-ws-i8-fenced-v1', 'signed-i8-product-modular-i32-v1', source_id,
                  'private-disjoint-row-windows-synchronous-v1')
    retention = ConvolutionInputRetentionLaw(*identities, max_live_tiles=len(intervals),
        input_preserved=True, independent_outputs=True, immutable_mapped_disjoint_buffers=True)
    schedule = plan_convolution_schedule(source, ConvolutionScheduleChoice(live_tiles=None),
                                         retention_law=retention)
    rhs_law = ConvolutionRhsResidencyLaw(*identities, True, True, True, True, True)
    rhs = plan_convolution_rhs_residency(schedule, operand_rows=operand_rows, law=rhs_law)
    from merlin.compile.scheduling.convolution_row_cache import (
        plan_convolution_row_cache, emit_convolution_row_cache)
    from merlin.compile.tuning_knobs import knob
    tuning_root = Path(__file__).resolve().parent
    if knob('conv_operand_cache', default=True, values=(False, True), root=tuning_root):
        row_cache, row_cache_refusal = plan_convolution_row_cache(schedule, rhs_law,
            spatial_group=knob('conv_spatial_accumulation', default=False, values=(False, True), root=tuning_root))
    else:
        row_cache, row_cache_refusal = None, 'rolling operand cache disabled by declared setting'
    epoch_law, epoch_refusal = None, None
    if (row_cache is not None and (row_cache.rhs is not None or row_cache.rhs_groups) and
        knob('conv_resident_compute_epochs', default=True, values=(False, True), root=tuning_root)):
        from merlin.compile.scheduling.convolution_ordering_facts import selected_epoch_law, Unavailable
        root = Path(__file__).resolve().parent
        try:
            selection = json.loads((root / 'rtl_facts/command_ordering_selection.json').read_text())
            pairs = kh*kw*((ci+plan.channel_tile-1)//plan.channel_tile)*max(map(len, schedule.column_blocks))*row_cache.spatial_live
            if pairs > 1:
                epoch_law, _ = selected_epoch_law(selection,
                    (root / 'rtl_facts/facts.json').read_bytes(), source_id, pairs)
        except (OSError, KeyError, ValueError) as error:
            epoch_refusal = str(error)
    ifm, weights, out = pointers[ifm_name], pointers[weight_name], pointers[output_name]
    # Explicit quiescence precedes invocation-private weight preparation.
    b.fence()
    target.cfg_ex(b, dataflow=1)
    target.cfg_ld(b, cout, stream=1)
    itemsize = 4 if dtype == 'i32' else 1

    def address(base, element, width=1):
        return b.ptr(base, b.op('mul', element, width))

    def lower(kind, f):
        if kind == 'load_input':
            target.cfg_ld(b, f['row_stride_elements'], stream=0,
                          block_stride=f.get('block_stride', edge))
            ptr = 0 if f['zero'] else address(ifm, f['first_element'])
            target.mvin(b, ptr, f['local_row'], rows=f['rows'], cols=f['columns'], stream=0)
            b.fence()
        elif kind == 'load_weight':
            index = b.op('add', b.op('mul', f['reduction_row'], cout), f['output_column'])
            target.mvin(b, address(weights, index), f['local_row'], rows=f['rows'], cols=f['columns'], stream=1)
            if not f.get('disjoint_preparation', False):
                b.fence()
        elif kind == 'initialize_bias':
            # A zero DMA stride broadcasts the source bias row into each live ACC row.
            target.cfg_ld(b, 0, stream=2, shrunk=False)
            target.mvin(b, address(pointers[bias_name], f['output_column'], 4),
                         b.op('add', f['accumulator_row'], 0x80000000),
                         rows=f['rows'], cols=f['columns'], stream=2)
            b.fence()
        elif kind == 'compute':
            def emit(accumulate):
                caddr = b.op('add', f['accumulator_row'], 0xc0000000 if accumulate else 0x80000000)
                target.preload(b, f['rhs_row'], caddr, brows=f['depth'], bcols=f['columns'],
                               crows=f['rows'], ccols=f['columns'])
                target.compute(b, f['lhs_row'], arows=f['rows'], acols=f['depth'], flip=True)
                if not f.get('ordered_compute_epoch', False):
                    b.fence()
            if bias:
                emit(True)
            else:
                b.branch(f['first_reduction'], lambda: emit(False), lambda: emit(True))
        elif kind == 'store':
            col = f['output_column']
            if not isinstance(col, int):
                raise ValueError('convolution readout boundary must be static')
            target.cfg_st(b, cout*itemsize, scale=scales[col] if 'acc_scale' in epilogue else 1.0,
                          activation=int('relu' in epilogue))
            index = b.op('add', b.op('mul', f['output_row'], cout), col)
            flag = 0xa0000000 if dtype == 'i32' else 0x80000000
            target.mvout(b, address(out, index, itemsize), b.op('add', f['accumulator_row'], flag),
                         rows=f['rows'], cols=f['columns'])
            if not f.get('disjoint_readout_batch', False):
                b.fence()
        else:
            raise ValueError('unknown convolution effect: ' + kind)

    if row_cache is not None:
        evidence = emit_convolution_row_cache(row_cache, b, lower,
            complete=lambda *_: b.fence(), synchronous_transport=True, epoch_law=epoch_law,
            coalesced_preparation=knob('conv_coalesced_rhs_preparation', default=True,
                values=(False, True), root=tuning_root),
            persistent_zero_bands=knob('conv_persistent_zero_bands', default=True,
                values=(False, True), root=tuning_root),
            zero_row_elision=knob('conv_zero_row_elision', default=True,
                values=(False, True), root=tuning_root),
            readout_batch=(epoch_law is not None and
                len({target.f32_bits(v) for v in scales}) == 1 and
                knob('conv_readout_batch', default=True, values=(False, True), root=tuning_root)))
        evidence['epoch_refusal'] = epoch_refusal
    else:
        evidence = emit_convolution_schedule(schedule, b, lower, complete=lambda *_: b.fence(),
                                             synchronous_completion=True, rhs=rhs)
        evidence['row_cache_refusal'] = row_cache_refusal
    b.fence()
    return evidence


def lower_maxpool(builder, source, output, channels, attrs, *, batches=1):
    """Native signed-i8 spatial max, staged as complete window-height strips.

    ``source`` already contains the source epilogue's rounded/clamped i8 values.
    Required attributes are pool_in_dims, pool_size, pool_stride; pool_padding
    defaults to zeros. Padding, when present, must declare pool_pad_value=-128.
    No payload maximum or requantization runs on the host.
    """
    from merlin.compile.logical_abi import pool_params
    from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
    from merlin.compile.scheduling.spatial_max_stripes import plan_striped_spatial_max
    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
    try:
        from . import target
    except ImportError:
        import target
    p = pool_params(attrs, op='native maxpool')
    spatial = SpatialMaxReadout(p['pool_in_dims'], p['pool_size'], p['pool_stride'],
                                p['pool_padding'], p['pad_value'])
    kh, kw = spatial.window
    sh, sw = spatial.stride
    if kh != kw or sh != sw or not 0 < kh < 4 or not 0 < sh < 4:
        raise ValueError('native pooling needs equal window axes and equal stride axes, each 1..3')
    edge, _, acc_rows = _geometry()
    stripe = plan_striped_spatial_max(spatial, batches=batches, channels=channels,
        channel_block=edge, transfer_rows=edge, capacity_rows=acc_rows)
    height, width = stripe.stripe_hw
    oh, ow = spatial.output_hw
    if max(height, width, ow) > 255:
        raise ValueError('native pooling strip exceeds eight-bit configuration dimensions')
    b = builder
    b.fence()
    target.cfg_ld(b, channels, stream=2, shrunk=True)
    # Each command reads one complete strip and writes one pooled output row.
    # Clamped source indices duplicate border values, preserving minimum-valued
    # padding without relying on a hardware padding numerical convention.
    target.cfg_st(b, channels, scale=1.0, activation=0, pool_size=kh, pool_stride=sh,
                  pool_out_dim=ow, porows=1, pocols=ow, orows=height, ocols=width)

    def lower(kind, f):
        if kind == 'stage_spatial_rows':
            offset = b.op('add', b.op('mul', f['source_row'], channels), f['first_column'])
            target.mvin(b, b.ptr(source, offset), b.op('add', f['local_row'], 0x80000000),
                        rows=f['rows'], cols=f['columns'], stream=2)
            b.fence()
        elif kind == 'read_spatial_strip':
            offset = b.op('add', b.op('mul', f['output_row'], channels), f['first_column'])
            # In pooled mode the controller derives the spatial traversal from
            # CONFIG_ST dimensions; the ordinary transfer row count is zero.
            target.mvout(b, b.ptr(output, offset), 0x80000000, rows=0, cols=f['columns'])
            b.fence()
        else:
            raise ValueError('unknown spatial strip effect: ' + kind)
    evidence = stripe.revalidate()
    emit_index_program(stripe.kernel, FnBuilderIndexAdapter(b), lower)
    b.fence()
    target.cfg_st(b, channels)
    return evidence
