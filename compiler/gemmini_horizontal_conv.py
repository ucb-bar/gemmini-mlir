"""Optional original-order horizontal panels with synchronous private ownership.

Merlin owns the complete-K panel construction and the padded input transporter.
The ordinary convolution terminal owns all arithmetic and readout instructions.
The default uses one M block and only fresh ARRAY captures. Separately
qualified bounded M alternatives reuse the selected WS ARRAY across each run.
"""
from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path

from merlin.compile.scheduling.horizontal_convolution import (
    plan_horizontal_convolution_panels, emit_horizontal_convolution_panels,
)
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.llvmlower.convolution_packing import emit_nhwc_window_panel


def _selected_transfer_law(root, edge):
    """Bind stream-zero multiblock extents to the selected immutable facts."""
    from gemmini_pointwise_tiles import PointwiseTileResources
    facts = (root/'data/rtl_facts.json').read_bytes()
    pin = json.loads((root/'data/COMMAND_ORDERING_SELECTION.json').read_text())
    record = pin['ordering']
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    if identity != pin['ordering_sha256']:
        raise ValueError('horizontal selected ordering identity differs')
    resources = PointwiseTileResources.from_selected(facts, pin)
    if resources.block != edge or resources.max_rows < edge:
        raise ValueError('horizontal transfer block differs from compute primitive')
    required = {'MVIN.spad_addr': 2, 'MVIN2.spad_addr': 1,
                'MVIN3.spad_addr': 14, 'PRELOAD.weight_spad': 6,
                'PRELOAD.c_addr': 6, 'COMPUTE_PRELOADED.a_spad': 4,
                'MVOUT.acc_addr': 3}
    for key, funct in required.items():
        access = record.get('accesses', {}).get(key, {})
        if (access.get('funct') != funct
                or record.get('extents', {}).get(key, {}).get('status') != 'derived'):
            raise ValueError('horizontal selected instruction or extent unknown: '+key)
    if record.get('completion_order', {}).get('execute', {}).get('order') != 'in_order':
        raise ValueError('horizontal fresh preload/compute execute order unknown')
    return resources, dict(facts_sha256=resources.facts_sha256,
        ordering_sha256=identity, stream=0, block=edge,
        local_row_law='local_row + (column // block) * block_stride_rows + row',
        local_lane_law='column % block', array_capture='fresh COMPUTE_PRELOADED only',
        completion='real fence after every terminal effect and at library lifetime boundaries')


def _selected_compute_epoch(plan, selection, compute_pairs):
    """Reuse the ordinary selected dependency/progress law for bounded pairs.

    Inputs, immutable RHS and bias already complete before any pair. Only
    adjacent PRELOAD/COMPUTE pairs are grouped with original K order and private
    accumulator cells. ARRAY stays need the separate admission below. Store
    and row-reuse drains remain.
    The software batch never exceeds selected physical pending pair capacity.
    """
    from dataclasses import asdict, replace
    from gemmini_conv import _epoch_schedule
    from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
    if type(compute_pairs) is not int or not 1 < compute_pairs < 2**31:
        raise ValueError('horizontal compute epoch requires a bounded integer greater than one')
    source = ConvolutionExplorationSource(plan.source, plan.column_intervals,
                                         selection['source_identity'])
    schedule = _epoch_schedule(source, compute_pairs)
    if schedule is None or schedule.epoch_law is None:
        return None
    original = schedule.epoch_law
    # This adapter deliberately makes no above-capacity software issue grant.
    if original.pending_compute_pairs is None or compute_pairs > original.pending_compute_pairs:
        return None
    program_identity = hashlib.sha256(repr(plan).encode()).hexdigest()
    epoch = replace(original, source_identity=program_identity,
        qualification_identity=original.qualification_identity +
            (';horizontal-complete-private-panels-all-fresh-original-K-store-drains-v1'
             if plan.row_blocks == 1 else
             ';horizontal-complete-private-panels-adjacent-M-original-K-store-drains-v1'))
    epoch.require(compute_pairs)
    return epoch, dict(selected=True, compute_pairs=compute_pairs,
        maximum_pending_pairs=original.pending_compute_pairs,
        ordinary_selected_source_identity=source.identity,
        horizontal_program_identity=program_identity, epoch_law=asdict(epoch),
        retained_completions=['each RHS load', 'each horizontal input load',
            'each bias load', 'horizontal_rhs_ready', 'horizontal_group_ready',
            'before first store after pending compute', 'each store',
            'horizontal_group_last_use', 'whole invocation exit'],
        removed_completions=('compute-pair terminal fence only within at most compute_pairs adjacent fresh pairs'
            if plan.row_blocks == 1 else
            'compute-pair terminal fence only within bounded adjacent fresh/stay pairs; every two-M run stays inside the epoch'),
        pass_order=['original-window', 'horizontal-private-panel-plan',
            'selected-compute-epoch-admission', 'Merlin-padded-input-transport',
            'horizontal-panel-emission', 'original-terminal-with-bounded-compute-drains'],
        array_stay=plan.row_blocks > 1, target_execution_verified=False, timing=None)


def _selected_horizontal_array(plan, selection, requested_max_row_blocks=2):
    """Bind the existing WS fresh/stay admission to this exact private M run.

    This is the selected endpoint check used by gemmini_conv_array, with the
    horizontal library's complete immutable RHS and disjoint bounded M cells as
    source/storage owner. No rolling-input or asynchronous readout grant is used.
    """
    from dataclasses import asdict
    from gemmini_conv_array import _names, ISA_SOURCE
    from merlin.compile.scheduling.dense_array_retention import DenseArrayRetentionAdmission
    plan.__post_init__()
    root = Path(__file__).resolve().parent
    facts = (root/'data/rtl_facts.json').read_bytes()
    pin = json.loads((root/'data/COMMAND_ORDERING_SELECTION.json').read_text())
    order = pin['ordering']
    identity = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
    epoch = selection['compute_epoch']['epoch_law']
    if (pin['base_facts']['sha256'] != hashlib.sha256(facts).hexdigest()
            or identity != pin['ordering_sha256'] or identity != epoch['hardware_identity']):
        return None
    if not any(n.get('4') == 'COMPUTE_AND_FLIP_CMD'
               and n.get('5') == 'COMPUTE_AND_STAY_CMD'
               and n.get('6') == 'PRELOAD_CMD' for n in _names(json.loads(facts))):
        return None
    for name, funct in (('COMPUTE_PRELOADED.a_spad', 4), ('COMPUTE_ACCUMULATE.a_spad', 5)):
        access = order.get('accesses', {}).get(name, {})
        if (access.get('funct') != funct or access.get('queue') != 'execute'
                or order.get('extents', {}).get(name, {}).get('status') != 'derived'):
            return None
    if order.get('completion_order', {}).get('execute', {}).get('order') != 'in_order':
        return None
    d = plan.source.primitive
    if (requested_max_row_blocks not in (2, 4, 8) or plan.row_blocks > requested_max_row_blocks
            or plan.input_block_stride != plan.row_blocks*d
            or plan.lhs_rows != plan.reduction_tiles*plan.input_block_stride
            or plan.rhs_base < plan.lhs_rows
            or plan.accumulator_rows != plan.row_blocks*len(plan.column_intervals)*d):
        return None
    law = DenseArrayRetentionAdmission(epoch['hardware_identity'], epoch['numerical_policy_identity'],
        'selected-ws-funct4-captures-B-funct5-retains-B-output-descriptor-only-v1')
    diagnostic = dict(selected=True, mechanism='merlin.compile.scheduling.dense_array_retention.DenseArrayRetentionAdmission',
        array_law=asdict(law), isa_source=ISA_SOURCE,
        selected_ordering_sha256=identity, horizontal_program_identity=selection['program_identity'],
        requested_max_row_blocks=requested_max_row_blocks, row_blocks=plan.row_blocks,
        source_contract='complete original immutable RHS; each K/N followed by adjacent original M feeds; no intermediate observers',
        array_capture='funct4 captures exact K/N RHS; adjacent funct5 retains it; PRELOAD garbage B updates only output descriptor',
        lhs_cells='slot*row_blocks*primitive + mi*primitive + row; disjoint until group last-use fence',
        accumulator_cells='(ni*row_blocks + mi)*primitive + row; lane=original column within interval',
        pair_retirement='even bounded compute epoch; no configuration, load, store or completion between fresh and stay',
        readiness='all immutable RHS, horizontal input and bias loads really complete before first feed',
        completion='real fence before every store; each store fence; group last-use before private row reuse; invocation exit fence',
        original_reduction_order=['kernel_y','kernel_x','input_channel'],
        pass_order=['original-window','horizontal-private-panel-plan','selected-compute-epoch-admission',
            'selected-ARRAY-retention-admission','Merlin-source-transport','horizontal-adjacent-M-emission',
            'original-terminal-with-bounded-compute-drains'],
        target_execution_verified=False, timing=None)
    if requested_max_row_blocks == 4:
        diagnostic['pair_retirement'] = ('compute budget multiple of four; drain only between complete bounded M runs; '
            'no configuration, load, store or completion within fresh/stay run')
        diagnostic['capacity_fallback'] = 'shared capacity bound rounded down to 4, 2 or 1; original outputs unchanged'
    if requested_max_row_blocks == 8:
        diagnostic['pair_retirement'] = ('compute budget exactly selected eight; reserve complete bounded M run before fresh capture; '
            'no configuration, load, store or completion within fresh/stay run')
        diagnostic['capacity_fallback'] = 'shared capacity bound rounded down to 8, 4, 2 or 1; original outputs unchanged'
    return diagnostic


def _validate_column_intervals(column_intervals, co, edge, readout):
    if type(column_intervals) is not tuple or not column_intervals:
        raise ValueError('horizontal explicit intervals require a nonempty immutable tuple')
    end = 0
    for item in column_intervals:
        if (type(item) is not tuple or len(item) != 2
                or any(type(x) is not int for x in item)
                or item[0] != end or not 0 < item[1] <= edge
                or item[0] + item[1] > co):
            raise ValueError('horizontal explicit intervals must partition original columns in order')
        if (readout is not None
                and any(readout.column_scale_bits[c] != readout.column_scale_bits[item[0]]
                        for c in range(item[0], item[0]+item[1]))):
            raise ValueError('horizontal explicit interval crosses original source scale bits')
        end += item[1]
    if end != co:
        raise ValueError('horizontal explicit intervals must cover every original column')
    return column_intervals


def plan_horizontal_conv(*, input_shape, weight_shape, attrs=None, bias=False,
                         output_dtype='i32', max_row_blocks=1, compute_pairs=1,
                         source_destination_disjoint=False, split_input_blocks=False,
                         column_intervals=None):
    """Pure admission; return (checked plan, diagnostics), or None before mutation.

    The route owns the full accelerator private epoch between entry/exit fences.
    Its fresh BSS source copy cannot alias external buffers or other scratch sites.
    No empirical profitability or missing asynchronous target law is inferred.
    Optional column_intervals is an immutable ordered full column partition;
    each atomic interval must fit one primitive and retain uniform source F32 bits.
    """
    from gemmini_conv import _geometry, _resources
    if type(split_input_blocks) is not bool:
        raise ValueError('horizontal split_input_blocks requires an explicit bool')
    if type(source_destination_disjoint) is not bool:
        raise ValueError('horizontal source disjointness must be an explicit bool')
    if type(compute_pairs) is not int or compute_pairs < 1:
        raise ValueError('horizontal compute_pairs requires a positive integer')
    attrs = dict(attrs or {})
    if type(max_row_blocks) is not int or max_row_blocks not in (1, 2, 4, 8):
        return None
    # Keep each exact fresh/stay run inside one selected compute epoch. An
    # odd or synchronous batch would insert a completion within that run.
    if max_row_blocks > 1 and (compute_pairs < max_row_blocks or compute_pairs % max_row_blocks):
        return None
    unknown = set(attrs.get('epilogue', ())) - {'bias_add', 'acc_scale', 'relu'}
    if (unknown or any(k in attrs for k in ('requant_shift', 'pool', 'pool_size', 'maxpool'))
            or ('bias_add' in attrs.get('epilogue', ()) and not bias)
            or str(output_dtype) not in ('i8', 'int8', 'i32', 'int32')):
        return None
    window, co = _geometry(input_shape, weight_shape, attrs)
    edge, operand_rows, accumulator_rows = _resources()
    kh, kw = window.kernel
    ci = window.input_shape[-1]
    # Structural benefit: preserve K while reducing primitive padding versus
    # the ordinary per-kernel-x channel tiles. This is not a cycle prediction.
    if (window.output_spatial[1] < edge or window.dilation[1] != 1
            or kw <= 1 or ci >= edge
            or (kw*ci+edge-1)//edge >= kw*((ci+edge-1)//edge)):
        return None
    resources, law = _selected_transfer_law(Path(__file__).resolve().parent, edge)
    if kw*ci > min(resources.max_columns, resources.max_request_bytes):
        return None
    readout = None
    if str(output_dtype) in ('i8', 'int8'):
        scales = attrs.get('acc_scale_by_column', attrs.get('acc_scale',
            attrs.get('output_scale', attrs.get('scale', 1.0))))
        readout = DenseTiledIntegerReadout.from_source_scales(scales, columns=co,
            tile_columns=1, relu=bool(attrs.get('relu', False) or 'relu' in attrs.get('epilogue', ())))
    if column_intervals is None:
        intervals = []
        column = 0
        while column < co:
            end = min(column+edge, co)
            if readout is not None:
                end = column+1
                while end < min(column+edge, co) and readout.column_scale_bits[end] == readout.column_scale_bits[column]:
                    end += 1
            intervals.append((column, end-column))
            column = end
    else:
        # A phased source already owns atomic numerical intervals. Validate
        # that exact immutable partition independently instead of greedily
        # repacking it in the phase's shifted coordinate system.
        _validate_column_intervals(column_intervals, co, edge, readout)
        intervals = column_intervals
    source = CountedConvolutionPlan(window, co, edge, edge, edge, min(ci, edge),
        operand_rows, accumulator_rows, 0, edge, 0, bool(bias))
    plan = plan_horizontal_convolution_panels(source, max_row_blocks=max_row_blocks,
        maximum_transfer_columns=min(resources.max_columns, resources.max_request_bytes),
        private_rows_owned=True, column_intervals=tuple(intervals))
    if max_row_blocks == 4 and plan is not None and plan.row_blocks == 3:
        # Use a divisor of the requested issue budget for capacity fallback.
        # A short final run may still have three live M feeds; emission reserves
        # a complete four-feed budget before starting any such run.
        plan = plan_horizontal_convolution_panels(source, max_row_blocks=2,
            maximum_transfer_columns=min(resources.max_columns, resources.max_request_bytes),
            private_rows_owned=True, column_intervals=tuple(intervals))
    if max_row_blocks == 8 and plan is not None and plan.row_blocks not in (1, 2, 4, 8):
        # Only divisors of the selected eight-pair budget form steady runs.
        # The shared planner still owns the complete-K capacity proof.
        fallback = 4 if plan.row_blocks >= 4 else 2
        plan = plan_horizontal_convolution_panels(source, max_row_blocks=fallback,
            maximum_transfer_columns=min(resources.max_columns, resources.max_request_bytes),
            private_rows_owned=True, column_intervals=tuple(intervals))
    if plan is None or plan.input_block_stride > resources.max_block_stride:
        return None
    padded_bytes = prod(plan.padded_window.input_shape)
    if padded_bytes >= 1 << 40:
        return None
    source_identity = hashlib.sha256(repr((tuple(input_shape),tuple(weight_shape),attrs,
        str(output_dtype),bool(bias))).encode()).hexdigest()
    selection = dict(selected=True, source_identity=source_identity,
        program_identity=hashlib.sha256(repr(plan).encode()).hexdigest(),
        effect_identity=hashlib.sha256(repr((plan,law)).encode()).hexdigest(),
        transfer_law=law, private_storage=dict(operand_rows=[0,plan.operand_rows],
            accumulator_rows=[0,plan.accumulator_rows], padded_input_bytes=padded_bytes,
            source_copy='fresh internal scratch_buffer BSS site'),
        source_reduction_order=['kernel_y','kernel_x','input_channel'],
        original_reduction_tiles=kh*kw*((ci+edge-1)//edge),
        horizontal_reduction_tiles=plan.reduction_tiles,
        physical_mapping_verified=False, target_execution_verified=False, timing=None)
    if compute_pairs > 1:
        admitted = _selected_compute_epoch(plan, selection, compute_pairs)
        if admitted is None:
            return None
        _, diagnostic = admitted
        selection['compute_epoch'] = diagnostic
        selection['effect_identity'] = hashlib.sha256(repr((plan,law,diagnostic)).encode()).hexdigest()
    if max_row_blocks > 1:
        array = (_selected_horizontal_array(plan, selection) if max_row_blocks == 2 else
                 _selected_horizontal_array(plan, selection, requested_max_row_blocks=max_row_blocks))
        if array is None:
            return None
        selection['array_retention'] = array
        if max_row_blocks >= 4:
            selection['compute_epoch']['removed_completions'] = ('compute-pair terminal fence only within bounded complete M runs; '
                'reserve row_blocks pending-pair capacity before every fresh capture; short final runs drain between captures')
            selection['effect_identity'] = hashlib.sha256(repr((plan,law,selection['compute_epoch'])).encode()).hexdigest()
        selection['effect_identity'] = hashlib.sha256(repr((selection['effect_identity'],array)).encode()).hexdigest()
    if not any(window.padding) and source_destination_disjoint:
        from dataclasses import asdict
        from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
        effects = BorrowedBufferEffects(
            'horizontal-exact-input-SSA-readonly-DMA-noescape-no-release-exit-fence-v1',
            no_retained_aliases=True, no_releases=True, complete_before_return=True)
        # Exact original NHWC coordinates: no copy, cast, reshape or view
        # canonicalization occurs. The LLVM input SSA is bound unchanged below.
        # Disjoint output is supplied by the original tensor-allocation owner;
        # pointer inequality is never treated as that witness.
        selection['input_identity'] = dict(selected=True,
            qualification_identity=effects.qualification_identity,
            original_shape=window.input_shape,
            original_layout='nhwc', original_dtype='signed_i8',
            source_destination_disjoint=True, original_padding=window.padding,
            coordinates='same original NHWC element offset and source pointer SSA',
            source_mutation=False, retained_aliases=False, releases=False,
            completion='entry fence; load readiness; library last-use and exit fences',
            borrowed_effects=asdict(effects), host_copy_bytes=0,
            tensor_preparation_identity='TensorReadView not applicable to LLVM pointers; exact same SSA binding',
            target_execution_verified=False)
        selection['private_storage'] = dict(selection['private_storage'],
            padded_input_bytes=0, borrowed_input_live_bytes=padded_bytes,
            source_copy='none; exact immutable original input SSA borrowed until exit fence')
        selection['effect_identity'] = hashlib.sha256(repr((plan,selection['transfer_law'],
            selection.get('compute_epoch'),selection['input_identity'])).encode()).hexdigest()
        if 'array_retention' in selection:
            selection['effect_identity'] = hashlib.sha256(repr((selection['effect_identity'],
                selection['array_retention'])).encode()).hexdigest()
    if split_input_blocks:
        from dataclasses import asdict
        from merlin.compile.scheduling.tile_load_coalescing import TileLoadCoalescing
        # Request primitive-width rectangles from the exported exact-address
        # mechanism. Its row bound covers every smaller guarded gather run;
        # local_base=0 makes returned addresses offsets from the actual run.
        slices = TileLoadCoalescing(rows=edge, columns=plan.horizontal_columns,
            tile=edge, max_columns=edge, local_base=0,
            block_stride=plan.input_block_stride, capacity_rows=plan.lhs_rows)
        commands = tuple(slices.commands())
        if any(width > resources.max_columns or width > resources.max_request_bytes
               for _, width, _ in commands):
            return None
        selection['split_input_blocks'] = dict(selected=True,
            mechanism='merlin.compile.scheduling.tile_load_coalescing.TileLoadCoalescing.commands',
            plan=asdict(slices), commands=commands,
            selected_max_columns=resources.max_columns,
            selected_max_rows=resources.max_rows,
            selected_max_request_bytes=resources.max_request_bytes,
            source_offset='first_element + slice_column',
            local_row='local_row + slice_local_row',
            row_stride='unchanged original gather source_row_stride',
            completion='real fence after each nonoverlapping primitive-column slice',
            physical_packet_mapping_verified=False, timing=None)
        selection['effect_identity'] = hashlib.sha256(repr((selection['effect_identity'],
            selection['split_input_blocks'])).encode()).hexdigest()
    return plan, selection


def try_emit_horizontal_conv(fn, input_ptr, weight_ptr, dst_ptr, *, input_shape,
                             weight_shape, attrs=None, bias=None, output_dtype='i32',
                             max_row_blocks=1, compute_pairs=1,
                             source_destination_disjoint=False, split_input_blocks=False,
                             input_transport='padded', readout_epochs=False,
                             load_epochs=False):
    """Emit the optional whole ordinary convolution; refusal leaves fn untouched."""
    if (input_transport not in ('padded', 'boundary', 'device') or type(readout_epochs) is not bool
            or type(load_epochs) is not bool):
        raise ValueError('horizontal transport/readout option is invalid')
    prepared = plan_horizontal_conv(input_shape=input_shape, weight_shape=weight_shape,
        attrs=attrs, bias=bias is not None, output_dtype=output_dtype,
        max_row_blocks=max_row_blocks, compute_pairs=compute_pairs,
        source_destination_disjoint=source_destination_disjoint,
        split_input_blocks=split_input_blocks)
    if prepared is None:
        return None
    return _emit_horizontal_plan(fn, input_ptr, weight_ptr, dst_ptr,
        input_shape=input_shape, weight_shape=weight_shape, attrs=attrs, bias=bias,
        output_dtype=output_dtype, max_row_blocks=max_row_blocks, compute_pairs=compute_pairs,
        source_destination_disjoint=source_destination_disjoint, split_input_blocks=split_input_blocks,
        input_transport=input_transport, readout_epochs=readout_epochs, load_epochs=load_epochs,
        prepared=prepared)


def _emit_horizontal_plan(fn, input_ptr, weight_ptr, dst_ptr, *, input_shape,
                          weight_shape, attrs, bias, output_dtype, max_row_blocks,
                          compute_pairs, source_destination_disjoint, split_input_blocks,
                          input_transport, readout_epochs, load_epochs, prepared,
                          prepared_input=None, original_column_start=0, rhs_ready_boundary=None):
    """Consume a checked shared plan through the single original terminal.

    A phased caller binds an exact strided original-column view and one common
    immutable input preparation. Local ARRAY checks precede original-coordinate
    projection, so local cells and external tensor pitches stay separate.
    """
    plan, selection = prepared
    if type(original_column_start) is not int or original_column_start < 0:
        raise ValueError('horizontal original column origin must be nonnegative')
    if original_column_start + plan.source.output_channels > weight_shape[-1]:
        raise ValueError('horizontal column view exceeds original source width')
    if prepared_input is not None:
        from gemmini_horizontal_phases import HorizontalPreparedInput
        if type(prepared_input) is not HorizontalPreparedInput:
            raise ValueError('horizontal common input requires its typed original preparation')
        prepared_input.revalidate(fn, input_ptr, plan)
    if selection.get('input_identity', {}).get('selected') and input_ptr is dst_ptr:
        return None  # A contradictory alias witness must fail before mutation.
    input_admission = None
    device_span_transport = None
    if input_transport == 'device' and prepared_input is None and not selection.get('input_identity', {}).get('selected'):
        from gemmini_horizontal_transport import make_device_span_transport
        # Resource admission is pure and precedes any fence or allocation.
        device_span_transport = make_device_span_transport(fn)
    if input_transport == 'boundary':
        from gemmini_horizontal_input import plan_horizontal_input
        if input_ptr is not dst_ptr:
            input_admission = plan_horizontal_input(plan,
                source_destination_disjoint=source_destination_disjoint,
                private_rows_owned=True, split_input_blocks=split_input_blocks)
        # Keep the original horizontal plan when borrowing cannot be proved.
        # An optional input optimization must not replace it with another path.
        if input_admission is None:
            selection['boundary_input_refusal'] = dict(requested=True,
                selected=False, fallback='original padded horizontal input',
                reason='requires original disjoint source owner and selected exact rectangles')
    readout_admission = None
    if readout_epochs:
        from gemmini_horizontal_readout import admit_horizontal_readout
        root = Path(__file__).resolve().parent / 'data'
        readout_admission = admit_horizontal_readout(plan, selection,
            facts_bytes=(root/'rtl_facts.json').read_bytes(),
            ordering_selection=json.loads((root/'COMMAND_ORDERING_SELECTION.json').read_text()))
        if readout_admission is None:
            return None
    load_admission = None
    if load_epochs:
        from gemmini_horizontal_loads import admit_horizontal_loads
        root = Path(__file__).resolve().parent / 'data'
        load_admission = admit_horizontal_loads(plan, selection,
            facts_bytes=(root/'rtl_facts.json').read_bytes(),
            ordering_selection=json.loads((root/'COMMAND_ORDERING_SELECTION.json').read_text()),
            # Device preparation completes the same private padded bytes and
            # pitch before terminal configuration. Epochs see that buffer,
            # never the preparation's ACC/DMA effects.
            input_transport='padded' if input_transport == 'device' else input_transport,
            split_input_blocks=split_input_blocks)
        if load_admission is None:
            selection['load_epoch_refusal'] = dict(requested=True, selected=False,
                fallback='original horizontal load completions',
                reason='requires selected exact padded unsplit private load epoch law')
    from gemmini_conv import make_conv_lowerer
    from gemmini_support import scratch_buffer
    import gemmini_encode as enc
    attrs = dict(attrs or {})
    w = plan.source.window
    size = prod(plan.padded_window.input_shape)
    enc.fence(fn)
    input_loader = None
    if prepared_input is not None:
        padded = prepared_input.pointer
        transport = prepared_input.receipt()
    elif input_admission is not None:
        from gemmini_horizontal_input import make_horizontal_input_loader
        input_loader, transport = make_horizontal_input_loader(fn, input_ptr, input_admission,
            split_input_blocks=split_input_blocks)
        padded = None
        selection['boundary_input'] = input_admission.record()
        selection['private_storage'] = dict(selection['private_storage'],
            padded_input_bytes=0, boundary_panel_bytes=input_admission.boundary_bytes,
            borrowed_input_live_bytes=prod(w.input_shape),
            source_copy='bounded boundary runs only; immutable interior DMA')
        selection['effect_identity'] = hashlib.sha256(repr((selection['effect_identity'],
            selection['boundary_input'])).encode()).hexdigest()
    elif selection.get('input_identity', {}).get('selected'):
        padded = input_ptr  # Exact original SSA, not an inferred view alias.
        transport = dict(kind='borrowed_input_identity', host_copy_bytes=0,
            original_shape=w.input_shape, input_layout='nhwc',
            source_destination_disjoint=True,
            borrowed_effects=selection['input_identity']['borrowed_effects'],
            storage_completion='original immutable source live through final device fence')
    else:
        padded = scratch_buffer(fn, size, 'horizontal_conv_padded_input')
        transport_window = Window2D(w.input_shape, (1,1), (1,1), (1,1), w.padding)
        transport = emit_nhwc_window_panel(fn, transport_window, input_ptr, padded,
            fn.const(0), prod(plan.padded_window.input_shape[:-1]),
            source_destination_disjoint=True, maximum_panel_bytes=size,
            inline_memory_spans=True, span_transport=device_span_transport)
        if device_span_transport is not None:
            transport.update(kind='device_accumulator_window_spans', host_copy_bytes=0,
                signed_i8_exact=True, zero_source=0,
                storage_completion='each span drains before ACC reuse; preparation precedes terminal configuration')
    # Terminal geometry stays original: only input transfer origins use padded.
    terminal = make_conv_lowerer(fn,input_ptr,weight_ptr,input_shape=input_shape,
        weight_shape=weight_shape,attrs=attrs,bias=bias,output_dtype=output_dtype,
        dst_ptr=dst_ptr,synchronous_completion=compute_pairs == 1)
    pending_pairs = 0
    previous_capture = None
    capture_feeds = 0
    def drain_compute():
        nonlocal pending_pairs
        if pending_pairs:
            enc.fence(fn)
            pending_pairs = 0
    def lower(kind, fields, *, complete_after_effect=True):
        nonlocal pending_pairs, previous_capture, capture_feeds
        if compute_pairs > 1 and kind != 'compute':
            drain_compute()
        if kind == 'load_horizontal_input':
            if input_loader is not None:
                input_loader(fields)
            elif split_input_blocks:
                control = FnBuilderControlAdapter(fn)
                for column, columns, local_offset in selection['split_input_blocks']['commands']:
                    enc.config_load(fn, fields['row_stride_elements'], stream=0,
                                    block_stride=fields['block_stride_rows'])
                    first = control.op('add', fields['first_element'], column)
                    local = control.op('add', fields['local_row'], local_offset)
                    enc.mvin(fn, enc.pointer_offset(fn,padded,first),
                        local,fields['rows'],columns,stream=0)
                    enc.fence(fn)
            else:
                enc.config_load(fn, fields['row_stride_elements'], stream=0,
                                block_stride=fields['block_stride_rows'])
                enc.mvin(fn, enc.pointer_offset(fn,padded,fields['first_element']),
                    fields['local_row'],fields['rows'],fields['columns'],stream=0)
                if complete_after_effect:
                    enc.fence(fn)
        else:
            if kind == 'compute':
                if fields['fresh_weights'] is not True:
                    if not selection.get('array_retention', {}).get('selected'):
                        raise ValueError('horizontal all-fresh owner received ARRAY stay')
                    fresh = previous_capture
                    if (fresh is None or capture_feeds >= plan.row_blocks
                            or (max_row_blocks == 2 and pending_pairs % 2 != 1)
                            or any(fields[k] != fresh[k] for k in
                                ('rhs_row','depth','columns','output_column','first_reduction','bias'))
                            or fields['lhs_row'] != fresh['lhs_row'] + capture_feeds * plan.source.primitive
                            or fields['accumulator_row'] != fresh['accumulator_row'] + capture_feeds * plan.source.primitive):
                        raise ValueError('horizontal ARRAY stay differs from the exact adjacent original M run')
                    capture_feeds += 1
                else:
                    if max_row_blocks >= 4 and pending_pairs + plan.row_blocks > compute_pairs:
                        drain_compute()
                    previous_capture = fields
                    capture_feeds = 1
            if kind == 'compute' and type(fields['first_reduction']) is bool:
                from xdsl.dialects.builtin import i1
                fields = dict(fields, first_reduction=fn.const(int(fields['first_reduction']), i1))
            terminal_fields = fields
            if original_column_start and 'output_column' in fields:
                control = FnBuilderControlAdapter(fn)
                column = fields['output_column']
                terminal_fields = dict(fields, output_column=(column + original_column_start
                    if type(column) is int else control.op('add', column, original_column_start)))
            terminal(kind,terminal_fields)
            if compute_pairs > 1:
                if kind == 'compute':
                    pending_pairs += 1
                    if pending_pairs == compute_pairs:
                        drain_compute()
                elif complete_after_effect:
                    # The asynchronous terminal is used only to suppress its
                    # compute fence. Restore every original other effect drain.
                    enc.fence(fn)
    def complete(kind, fields):
        nonlocal pending_pairs
        enc.fence(fn)
        pending_pairs = 0
        if kind == 'horizontal_rhs_ready' and rhs_ready_boundary is not None:
            rhs_ready_boundary(fn.blk, fn.blk.last_op)
    if load_admission is not None:
        from gemmini_horizontal_loads import emit_horizontal_load_epochs
        receipt = emit_horizontal_load_epochs(load_admission,
            FnBuilderControlAdapter(fn), lower, complete, readout_admission=readout_admission)
        selection['load_epoch'] = load_admission.receipt()
        selection['effect_identity'] = hashlib.sha256(repr((selection['effect_identity'],
            selection['load_epoch'])).encode()).hexdigest()
        if readout_admission is not None:
            selection['readout_epoch'] = readout_admission.receipt()
            selection['effect_identity'] = hashlib.sha256(repr((selection['effect_identity'],
                selection['readout_epoch'])).encode()).hexdigest()
    elif readout_admission is None:
        receipt = emit_horizontal_convolution_panels(plan, FnBuilderControlAdapter(fn),
            lower, complete)
    else:
        from gemmini_horizontal_readout import emit_horizontal_readout_epochs
        receipt = emit_horizontal_readout_epochs(readout_admission,
            FnBuilderControlAdapter(fn), lower, complete)
        selection['readout_epoch'] = readout_admission.receipt()
        selection['effect_identity'] = hashlib.sha256(repr((selection['effect_identity'],
            selection['readout_epoch'])).encode()).hexdigest()
    enc.fence(fn)
    return dict(receipt, horizontal_selection=selection, input_transport=transport)
