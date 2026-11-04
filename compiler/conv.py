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
    Window2D, CountedConvolutionPlan, ConvolutionInputRetentionLaw, ConvolutionResidentInputRows, ConvolutionReadoutTransferLaw,
)
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource, ConvolutionScheduleChoice,
    ConvolutionRhsResidencyLaw, plan_convolution_schedule,
    plan_convolution_rhs_residency, emit_convolution_schedule,
    plan_convolution_storage_alternative, ConvolutionRhsPhaseLaw,
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
    # A complete row version remains resident for the entire source consumer.
    channel_tile = min(edge, ci)
    row_pitch = ((shape[2] + window.padding[1] + window.padding[3] + edge-1)//edge)*edge
    from merlin.compile.tuning_knobs import knob
    output_row_limit = knob('conv_output_rows', default=1, values=(1, 2, 4), root=Path(__file__).resolve().parent)
    oh, ow = window.output_spatial
    spatial_blocks_per_row = (ow+row_tile-1)//row_tile
    channel_blocks = (cout+edge-1)//edge
    rows_by_acc = max(1, accumulator_rows//(edge*channel_blocks*spatial_blocks_per_row))
    output_rows = min(output_row_limit, oh, rows_by_acc)
    slots = (kh-1)*window.dilation[0]+1 + (output_rows-1)*window.stride[0]
    ring_end = slots*row_pitch*((ci+channel_tile-1)//channel_tile)
    rolling = window.stride[1] == 1 and row_pitch < 65536 and ring_end+channel_tile <= operand_rows
    rhs_base = ((ring_end+channel_tile-1)//channel_tile)*channel_tile if rolling else edge
    plan = CountedConvolutionPlan(window, cout, edge, row_tile, edge, channel_tile,
                                  operand_rows, accumulator_rows, 0, rhs_base, 0, bias=bias)
    intervals, scales = _column_intervals(attrs, cout, edge)
    source_id = hashlib.sha256(json.dumps(command, sort_keys=True).encode()).hexdigest()
    source = ConvolutionExplorationSource(plan, intervals, source_id)
    identities = ('gemmini-ws-i8-fenced-v1', 'signed-i8-product-modular-i32-v1', source_id,
                  'private-disjoint-row-windows-synchronous-v1')
    retention = ConvolutionInputRetentionLaw(*identities, max_live_tiles=len(intervals),
        input_preserved=True, independent_outputs=True, immutable_mapped_disjoint_buffers=True)
    from conv_laws import selected_laws
    pair_bound = max(2, kh*kw*((ci+edge-1)//edge)*len(intervals)*16)
    stage_channels = channel_tile
    # Bound software issue work independently of physical ready/valid capacity.
    stage_bound = slots*((shape[2]+edge-1)//edge + (window.padding[1]+edge-1)//edge
                         +(window.padding[3]+edge-1)//edge + 2)*((ci+stage_channels-1)//stage_channels)
    issue, epoch, ordering = selected_laws(source_id, max(stage_bound, row_tile+1+max(len(intervals), accumulator_rows//edge)*int(bias)), pair_bound)
    issued = issue is not None and epoch is not None
    if not issued:
        from conv_seed import lower_conv as seed_lower_conv
        return seed_lower_conv(builder, command_buffer, command, pointers)
    if issued:
        identities = (issue.hardware_identity, issue.numerical_policy_identity, source_id,
                      issue.qualification_identity)
        retention = ConvolutionInputRetentionLaw(*identities, max_live_tiles=len(intervals),
            input_preserved=True, independent_outputs=True, immutable_mapped_disjoint_buffers=True,
            independent_output_rows=True, no_intermediate_observers=True)
    schedule = plan_convolution_schedule(source,
        ConvolutionScheduleChoice(live_tiles=None, load_issue=issued, compute_pairs=pair_bound if issued else 1),
        retention_law=retention, issue_law=issue if issued else None, epoch_law=epoch if issued else None)
    rhs_law = ConvolutionRhsResidencyLaw(*identities, True, True, True, True, True)
    rhs = plan_convolution_rhs_residency(schedule, operand_rows=operand_rows, law=rhs_law)
    input_rows = ConvolutionResidentInputRows(window, source.identity, channel_tile,
        0, row_pitch, slots, 0, ring_end, edge, halo_value=0,
        stage_channels=stage_channels) if rolling and issued else None
    from conv_laws import selected_readout_law
    readout_law = selected_readout_law(issue, epoch, source_id, max(len(intervals), accumulator_rows//edge))
    readout_transfer = None
    if readout_law is not None and dtype == 'i8':
        states = tuple((col, width, hashlib.sha256(json.dumps(dict(
            dtype=dtype, scale=target.f32_bits(scales[col] if 'acc_scale' in epilogue else 1.0),
            relu='relu' in epilogue), sort_keys=True).encode()).hexdigest()) for col, width in intervals)
        readout_transfer = ConvolutionReadoutTransferLaw(*identities, edge, edge, edge, states,
            scaled_i8=True, exact_block_column_map=True, original_numerical_state=True)
        # This law binds source exits to the original plan, not its provenance ID.
        from dataclasses import replace
        readout_transfer = replace(readout_transfer, source_identity=plan.identity)
    from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
    joint_storage = None
    output_group = None
    if input_rows is not None:
        phase_law = ConvolutionRhsPhaseLaw(source.identity,
            'independent-original-output-columns-private-full-phase-completion-v1',
            independent_output_columns=True, immutable_inputs=True,
            no_intermediate_observers=True, full_completion_before_reprepare=True)
        facts = json.loads((Path(__file__).resolve().parent/'rtl_facts/facts.json').read_text())['facts']
        spad = next(m for m in facts['memories'] if m['name'] == 'scratchpad')
        bank_rows = spad['depth']
        bank_base = ((input_rows.end+bank_rows-1)//bank_rows)*bank_rows
        bank_base = ((bank_base+channel_tile-1)//channel_tile)*channel_tile
        rhs_rows = kh*kw*((ci+channel_tile-1)//channel_tile)*len(intervals)*channel_tile
        assigned_rhs_base = bank_base if (knob('conv_bank_separation', default=False, values=(False, True),
            root=Path(__file__).resolve().parent) and bank_base+rhs_rows <= operand_rows) else None
        joint_storage = plan_convolution_storage_alternative(schedule, input_rows,
            operand_rows=operand_rows, rhs_law=rhs_law, rhs_phase_law=phase_law,
            max_row_blocks=16, max_output_rows=output_rows, rhs_base=assigned_rhs_base)
        rhs = joint_storage.rhs
        if readout_law is not None:
            output_group = joint_storage.output_group
    groups = output_group if isinstance(output_group, tuple) else (output_group,)
    grouped_transport = any(g is not None and g.output_rows > 1 for g in groups)
    ifm, weights, out = pointers[ifm_name], pointers[weight_name], pointers[output_name]
    # Explicit quiescence precedes invocation-private weight preparation.
    b.fence()
    target.cfg_ex(b, dataflow=1)
    target.cfg_ld(b, cout, stream=1)
    itemsize = 4 if dtype == 'i32' else 1
    # This emitter's complete effect vocabulary has no intervening write to
    # these configuration states. The selected controllers capture settings
    # by stream at the in-order command head (source-bound in conv_laws.py).
    if input_rows is not None:
        target.cfg_ld(b, ci, stream=0, block_stride=row_pitch)
    if bias:
        target.cfg_ld(b, 0, stream=2, shrunk=False)
    fixed_store = len({target.f32_bits(scales[col] if 'acc_scale' in epilogue else 1.0)
                       for col, width in intervals}) == 1
    if fixed_store:
        target.cfg_st(b, cout*itemsize, scale=scales[0] if 'acc_scale' in epilogue else 1.0,
                      activation=int('relu' in epilogue))

    def address(base, element, width=1):
        return b.ptr(base, b.op('mul', element, width))

    def lower(kind, f):
        if kind in ('stage_input_rows', 'fill_input_rows', 'load_input', 'load_weight', 'initialize_bias', 'store'):
            if not (type(f['columns']) is int and 0 < f['columns'] <= edge <= 127
                    and type(f['rows']) is int and 0 < f['rows'] <= edge):
                raise ValueError('convolution move is outside proven primitive target operand bounds')
        if kind in ('stage_input_rows', 'fill_input_rows'):
            if f['block_stride_rows'] != row_pitch:
                raise ValueError('rolling transfer changed fixed input block stride')
            ptr = address(ifm, f['first_element']) if kind == 'stage_input_rows' else 0
            target.mvin(b, ptr, f['local_row'], rows=f['rows'], cols=f['columns'], stream=0)
        elif kind == 'load_input':
            target.cfg_ld(b, f['row_stride_elements'], stream=0)
            ptr = 0 if f['zero'] else address(ifm, f['first_element'])
            target.mvin(b, ptr, f['local_row'], rows=f['rows'], cols=f['columns'], stream=0)
            if not issued: b.fence()
        elif kind == 'load_weight':
            index = b.op('add', b.op('mul', f['reduction_row'], cout), f['output_column'])
            target.mvin(b, address(weights, index), f['local_row'], rows=f['rows'], cols=f['columns'], stream=1)
            if not issued: b.fence()
        elif kind == 'initialize_bias':
            # A zero DMA stride broadcasts the source bias row into each live ACC row.
            target.mvin(b, address(pointers[bias_name], f['output_column'], 4),
                         b.op('add', f['accumulator_row'], 0x80000000),
                         rows=f['rows'], cols=f['columns'], stream=2)
            if not issued: b.fence()
        elif kind == 'compute':
            def emit(accumulate):
                caddr = b.op('add', f['accumulator_row'], 0xc0000000 if accumulate else 0x80000000)
                target.preload(b, f['rhs_row'], caddr, brows=f['depth'], bcols=f['columns'],
                               crows=f['rows'], ccols=f['columns'])
                target.compute(b, f['lhs_row'], arows=f['rows'], acols=f['depth'], flip=True)
                if not issued: b.fence()
            if bias:
                emit(True)
            else:
                b.branch(f['first_reduction'], lambda: emit(False), lambda: emit(True))
        elif kind == 'store':
            col = f['output_column']
            if not isinstance(col, int):
                raise ValueError('convolution readout boundary must be static')
            if not fixed_store:
                target.cfg_st(b, cout*itemsize, scale=scales[col] if 'acc_scale' in epilogue else 1.0,
                              activation=int('relu' in epilogue))
            index = b.op('add', b.op('mul', f['output_row'], cout), col)
            flag = 0xa0000000 if dtype == 'i32' else 0x80000000
            target.mvout(b, address(out, index, itemsize), b.op('add', f['accumulator_row'], flag),
                         rows=f['rows'], cols=f['columns'])
            if not issued: b.fence()
        else:
            raise ValueError('unknown convolution effect: ' + kind)

    evidence = emit_convolution_schedule(schedule, b, lower, complete=lambda *_: b.fence(),
                                         synchronous_completion=True, rhs=rhs, input_rows=input_rows,
                                         readout_law=readout_law, readout_transfer_law=readout_transfer,
                                         output_group=output_group, array_law=None, ready_readout=output_group is not None,
                                         grouped_transport=grouped_transport)
    b.fence()
    return dict(evidence, ordering_admission=ordering,
                joint_storage=None if joint_storage is None else joint_storage.summary())


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
        if kind in ('stage_input_rows', 'fill_input_rows', 'load_input', 'load_weight', 'initialize_bias', 'store'):
            if not (type(f['columns']) is int and 0 < f['columns'] <= edge <= 127
                    and type(f['rows']) is int and 0 < f['rows'] <= edge):
                raise ValueError('convolution move is outside proven primitive target operand bounds')
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
