"""Counted NHWC convolution through Merlin's checked window scheduling API.

The terminal callback consumes the library's real transfer/compute/store effects;
CPU code computes addresses and loop control only. Every effect completes before
private rows can be reused. Packed RHS is [kernel_y,kernel_x,input_channel,output_channel].
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource, plan_convolution_schedule, emit_convolution_schedule,
)
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout
from merlin.compile.tuning_knobs import knob

try:
    from . import gemmini_encode as enc
    from .gemmini_conv_policy import apply_retention, prepare_rolling_input, select_regional_schedule
except ImportError:
    import gemmini_encode as enc
    from gemmini_conv_policy import apply_retention, prepare_rolling_input, select_regional_schedule


def _geometry(input_shape, weight_shape, attrs):
    layout = str(attrs.get('layout', 'nhwc')).lower()
    if layout not in ('nhwc', 'row_major_contiguous'):
        raise ValueError('convolution requires the declared NHWC input layout')
    shape = tuple(map(int, input_shape))
    if len(shape) != 4:
        raise ValueError('convolution input must have rank four')
    weights = tuple(map(int, weight_shape))
    kernel = tuple(map(int, attrs.get('kernel', weights if len(weights) == 4 else ())))
    if len(kernel) not in (2, 4):
        raise ValueError('packed convolution weights require an explicit original kernel')
    kh, kw = kernel[:2]
    ci = shape[-1]
    co = weights[-1]
    if len(kernel) == 4 and kernel[2:] != (ci, co):
        raise ValueError('kernel input/output channels differ from the original tensors')
    if weights not in ((kh*kw*ci, co), (kh, kw, ci, co)):
        raise ValueError('convolution weights must have the declared packed_conv_rhs layout')
    window = Window2D(shape, (kh, kw), tuple(map(int, attrs.get('stride', (1, 1)))),
        tuple(map(int, attrs.get('dilation', (1, 1)))),
        tuple(map(int, attrs.get('padding', (0, 0, 0, 0)))))
    return window, co


def _resources():
    root = Path(__file__).resolve().parent
    path = root/'data/rtl_facts.json'
    if not path.is_file():
        path = root.parent/'rtl_facts/facts.json'
    facts = json.loads(path.read_text())['facts']
    edge = min(facts['arrays'][0]['rows'], facts['arrays'][0]['cols'])
    memories = {m['name']: m for m in facts['memories']}
    sp, ac = memories['scratchpad'], memories['accumulator']
    return edge, sp['banks']*sp['depth'], ac['banks']*ac['depth']


def _epoch_schedule(source, pairs):
    """Bind the opt-in issued schedule to the selected public ordering record.

    A missing premise preserves the synchronous schedule. Cross-queue overlap
    records are checked for each actual load stream and actual operand access;
    the load->compute->reload consumer chain orders out-of-order DMA writes.
    """
    from merlin.compile.scheduling.window_transfer import (
        ConvolutionInputRetentionLaw, ConvolutionLoadIssueLaw, ConvolutionComputeEpochLaw,
    )
    from merlin.compile.scheduling.convolution_exploration import ConvolutionScheduleChoice
    root = Path(__file__).resolve().parent
    pin_path = root/'data/COMMAND_ORDERING_SELECTION.json'
    facts_path = root/'data/rtl_facts.json'
    if not pin_path.is_file():
        pin_path = root.parent/'COMMAND_ORDERING_SELECTION.json'
    if not facts_path.is_file():
        facts_path = root.parent/'rtl_facts/facts.json'
    if not pin_path.is_file():
        return None
    pin = json.loads(pin_path.read_text())
    facts_bytes = facts_path.read_bytes()
    if pin['base_facts']['sha256'] != hashlib.sha256(facts_bytes).hexdigest():
        return None
    record = pin['ordering']
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    if digest != pin['ordering_sha256']:
        return None
    accesses = record.get('accesses', {})
    loads = ['MVIN.spad_addr', 'MVIN2.spad_addr']
    if source.plan.bias:
        loads += ['MVIN3.spad_addr']
    readers = ['COMPUTE_PRELOADED.a_spad', 'PRELOAD.weight_spad']
    writes = 'PRELOAD.c_addr'
    readout = 'MVOUT.acc_addr'
    required = loads+readers+[writes, readout]
    if any((record.get('extents', {}).get(k, {}).get('status') != 'derived') for k in required):
        return None
    cross = {(x['older'],x['younger'],x['space']):x for x in record.get('cross_queue', ())}
    def waits(a,b,space):
        return cross.get((a,b,space), {}).get('overlap') == 'completion'
    if not all(waits(l,r,'scratchpad') and waits(r,l,'scratchpad') for l in loads for r in readers):
        return None
    if not (waits(writes,readout,'accumulator') and waits(readout,writes,'accumulator')):
        return None
    if source.plan.bias and not all(waits(l,writes,'accumulator') for l in loads):
        return None
    if any(accesses[k]['queue'] != 'execute' for k in readers+[writes]):
        return None
    same = record.get('same_queue', {})
    if same.get('execute', {}).get('release') not in ('issue','completion'):
        return None
    if record.get('completion_order', {}).get('execute', {}).get('order') != 'in_order':
        return None
    codes = {int(c):q for x in record.get('funct_queue', ()) for c,q in x.get('by_code', {}).items()}
    if codes.get(0) != 'execute' or codes.get(1) != 'load' or codes.get(2) != 'store':
        return None
    if same.get('load', {}).get('release') not in ('issue','completion'):
        return None
    classes = json.loads(facts_bytes)['facts']['timing_structure']['command_classes']['classes']
    def entries(queue):
        e=classes[queue]['entries']
        return e['value'] if e['status']=='derived' else 0
    if entries('load') < 1 or entries('execute') < 2:
        return None
    ids=(digest, 'gemmini-signed-i8-i32-original-readout', source.identity,
         'public-ordering-overlap-completion-and-private-consumer-chain')
    retention=ConvolutionInputRetentionLaw(*ids, 1, True, True, True)
    issue=ConvolutionLoadIssueLaw(*ids, source.plan.row_tile+1+int(source.plan.bias),
        True, True, True, True, True, True)
    epoch=ConvolutionComputeEpochLaw(*ids, pairs,
        True, True, True, True, True, True, True, True, True, True,
        pending_compute_pairs=entries('execute')//2)
    return plan_convolution_schedule(source, ConvolutionScheduleChoice(1, True, pairs),
        retention_law=retention, issue_law=issue, epoch_law=epoch)


def make_conv_lowerer(fn, input_ptr, weight_ptr, *, input_shape, weight_shape,
                      attrs=None, bias=None, output_dtype='i32', dst_ptr=None,
                      synchronous_completion=True, a_stride=1):
    """Build the shared terminal effect callback for checked spatial producers.

    ``finish_raw_tile`` retains the actual private accumulator cells and performs
    the requested real completion; the composition owner supplies their readout.
    The callback never creates a producer schedule or materializes an intermediate.
    """
    attrs = dict(attrs or {})
    window, co = _geometry(input_shape, weight_shape, attrs)
    edge, _, _ = _resources()
    dtype = str(output_dtype)
    if dtype not in ('i8', 'int8', 'i32', 'int32'):
        raise ValueError('convolution output must be i8 or i32')
    narrow = dtype in ('i8', 'int8')
    activation = int(bool(attrs.get('relu', False) or 'relu' in attrs.get('epilogue', ())))
    scales = attrs.get('acc_scale_by_column', attrs.get('acc_scale',
        attrs.get('output_scale', attrs.get('scale', 1.0))))
    readout = DenseTiledIntegerReadout.from_source_scales(scales, columns=co,
        tile_columns=1, relu=bool(activation)) if narrow else None
    builder = FnBuilderControlAdapter(fn)
    scale_bits_ptr = None
    width = 1 if narrow else 4
    enc.fence(fn)
    enc.config_ex(fn, dataflow='ws', a_stride=a_stride)
    enc.config_store(fn, co*width, activation=activation)

    def add(a, b):
        return builder.op('add', a, b)

    def mul(a, b):
        return builder.op('mul', a, b)

    def lower(kind, f):
        nonlocal scale_bits_ptr
        if kind in ('stage_input_rows', 'fill_input_rows'):
            zero = kind == 'fill_input_rows'
            if zero and f['value'] != 0:
                raise ValueError('only original integer-zero padding is admitted')
            enc.config_load(fn, 0 if zero else f['row_stride_elements'], stream=0,
                            block_stride=f['block_stride_rows'])
            ptr = 0 if zero else enc.pointer_offset(fn, input_ptr, f['first_element'])
            enc.mvin(fn, ptr, f['local_row'], f['rows'], f['columns'], stream=0)
        elif kind == 'load_input':
            enc.config_load(fn, f['row_stride_elements'], stream=0)
            # The selected Gemmini zero-source transfer uses address zero; the
            # window planner selects it instead of issuing a conflicting fill.
            ptr = 0 if f['zero'] else enc.pointer_offset(fn, input_ptr, f['first_element'])
            enc.mvin(fn, ptr, f['local_row'], f['rows'], f['columns'], stream=0)
        elif kind == 'load_weight':
            enc.config_load(fn, co, stream=1)
            offset = add(mul(f['reduction_row'], co), f['output_column'])
            enc.mvin(fn, enc.pointer_offset(fn, weight_ptr, offset),
                     f['local_row'], f['rows'], f['columns'], stream=1)
        elif kind == 'initialize_bias':
            enc.config_load(fn, 0, stream=2)
            ptr = enc.pointer_offset(fn, bias, mul(f['output_column'], 4))
            enc.mvin(fn, ptr, add(enc.ACC, f['accumulator_row']),
                     f['rows'], f['columns'], stream=2)
        elif kind == 'compute':
            # Accumulator continuation is independent of the fresh-weight
            # compute opcode. A checked spatial M group may retain its RHS
            # across outputs; ordinary reductions capture their current panel.
            first = f['first_reduction']
            if f['bias']:
                flags = enc.ACC_ADD
            else:
                flags = builder.select(first, enc.ACC, enc.ACC_ADD)
            acc = add(flags, f['accumulator_row'])
            fresh=f.get('fresh_weights',True)
            if type(fresh) is not bool:
                raise ValueError('convolution array persistence requires a static checked first/remaining M partition')
            enc.preload(fn, f['rhs_row'] if fresh else enc.GARBAGE, acc, f['depth'], f['columns'],
                        f['rows'], f['columns'])
            enc.compute(fn, f['lhs_row'], enc.GARBAGE, f['rows'], f['depth'],
                        edge, edge, accumulate=not fresh)
        elif kind == 'finish_raw_tile':
            pass
        elif kind == 'store':
            if readout is not None:
                if type(f['output_column']) is int:
                    bits = readout.column_scale_bits[f['output_column']]
                    scale = struct.unpack('<f', struct.pack('<I', bits))[0]
                    enc.config_store(fn, co, scale=scale, activation=activation)
                else:
                    if f['columns'] != 1 or len(set(readout.column_scale_bits)) <= 1:
                        raise ValueError('counted scaled convolution requires a checked singleton column')
                    from gemmini_column_readout import config_column_readout, source_scale_bits_buffer
                    if scale_bits_ptr is None:
                        scale_bits_ptr = source_scale_bits_buffer(fn, readout.column_scale_bits)
                    config_column_readout(fn, co, scale_bits_ptr, f['output_column'],
                        relu=bool(activation))
            if dst_ptr is None:
                raise ValueError('a convolution store requires its original destination pointer')
            offset = mul(add(mul(f['output_row'], co), f['output_column']), width)
            ptr = enc.pointer_offset(fn, dst_ptr, offset)
            local = add(enc.ACC if narrow else enc.ACC_FULL, f['accumulator_row'])
            enc.mvout(fn, ptr, local, f['rows'], f['columns'])
        else:
            raise ValueError('unimplemented convolution schedule effect '+kind)
        if synchronous_completion:
            enc.fence(fn)

    return lower


def _emit_conv(fn, input_ptr, weight_ptr, dst_ptr, *, input_shape, weight_shape,
              attrs=None, bias=None, output_dtype='i32', _prepared_columns=False,
              _static_column_readout=False):
    """Emit a whole original convolution, returning its checked application receipt.

    Bias, if present, is a pointer to the original contiguous i32 CO vector.
    Scaled i8 output retains the source scalar or per-column F32 scales and
    ReLU; raw i32 uses the full accumulator readout selector. Declared tuning
    knobs select bounded lifetime alternatives; the default is synchronous.
    """
    attrs = dict(attrs or {})
    unknown = set(attrs.get('epilogue', ())) - {'bias_add', 'acc_scale', 'relu'}
    if unknown or any(k in attrs for k in ('requant_shift', 'pool', 'pool_size', 'maxpool')):
        raise ValueError('unsupported convolution readout epilogue: '+repr(sorted(unknown)))
    if 'bias_add' in attrs.get('epilogue', ()) and bias is None:
        raise ValueError('convolution source bias epilogue requires its original i32 bias pointer')
    window, co = _geometry(input_shape, weight_shape, attrs)
    edge, operand_rows, accumulator_rows = _resources()
    row_tile = min(edge, window.output_spatial[1])
    channel_tile = min(edge, window.input_shape[-1])
    # Preserve a real partial-sum continuation for a nontrivial single-tap
    # reduction even when its channels fit one array tile. Both partitions
    # consume original disjoint channel ranges; the second adds to live ACC.
    if window.kernel == (1,1) and bias is None and 1 < window.input_shape[-1] <= edge:
        channel_tile = (window.input_shape[-1]+1)//2
    p = CountedConvolutionPlan(window, co, edge, row_tile, edge,
        channel_tile, operand_rows, accumulator_rows,
        0, edge, 0, bias is not None)
    dtype = str(output_dtype)
    if dtype not in ('i8', 'int8', 'i32', 'int32'):
        raise ValueError('convolution output must be i8 or i32')
    narrow = dtype in ('i8', 'int8')
    activation = int(bool(attrs.get('relu', False) or 'relu' in attrs.get('epilogue', ())))
    scales = attrs.get('acc_scale_by_column', attrs.get('acc_scale',
        attrs.get('output_scale', attrs.get('scale', 1.0))))
    readout = DenseTiledIntegerReadout.from_source_scales(scales, columns=co,
        tile_columns=1, relu=bool(activation)) if narrow else None
    intervals = []
    c = 0
    while c < co:
        end = min(c+edge, co)
        if readout is not None:
            end = c+1
            while end < min(c+edge, co) and readout.column_scale_bits[end] == readout.column_scale_bits[c]:
                end += 1
        intervals.append((c, end-c))
        c = end
    identity = hashlib.sha256(repr((tuple(input_shape), tuple(weight_shape), attrs,
                                   output_dtype, bias is not None)).encode()).hexdigest()
    source = ConvolutionExplorationSource(p, tuple(intervals), identity)
    retention_mode = knob('conv_retention', default='off',
        values=('off', 'input', 'input_rhs', 'rolling_rhs'), root=Path(__file__).resolve().parent)
    pairs = knob('conv_epoch_pairs', default=1, values=(1, 4, 8, 16),
                 root=Path(__file__).resolve().parent)
    policy = knob('conv_policy', default='off', values=('off','regional'),
                  root=Path(__file__).resolve().parent)
    selection = dict(policy='off',selected=retention_mode+str(pairs))
    if policy == 'regional':
        (retention_mode,pairs),selection = select_regional_schedule(source,
            uniform_readout=readout is None or len(set(readout.column_scale_bits))==1,
            resource_identity=hashlib.sha256((Path(__file__).resolve().parent/'data/rtl_facts.json').read_bytes()).hexdigest())
    pointwise_mode=knob('conv_grouped_pointwise',default='array',values=('off','resident','array'),
        root=Path(__file__).resolve().parent)
    if _prepared_columns and knob('conv_retention',default='off',
            values=('off','input','input_rhs','rolling_rhs'),root=Path(__file__).resolve().parent)=='off':
        from gemmini_pointwise_policy import select_pointwise
        retention_mode,pairs,selection=select_pointwise(source,retention_mode,pairs,selection,mode=pointwise_mode)
        pointwise_pairs=knob('conv_pointwise_compute_pairs',default=8,values=(8,32,128),
            root=Path(__file__).resolve().parent)
        if selection.get('pointwise_experiment',{}).get('admitted'):
            # This bounds compiler-issued work only. _epoch_schedule still
            # derives pending pairs from the selected physical execute queue;
            # shared group consumers retain all last-use/readout drains.
            pairs=pointwise_pairs
            selection=dict(selection,pointwise_experiment=dict(
                selection['pointwise_experiment'],logical_compute_pairs=pairs))
    input_rows = None
    if retention_mode == 'rolling_rhs':
        tile_mode=knob('conv_pointwise_tiles',default='off',values=('off','stage64','stage64_balanced'),
            root=Path(__file__).resolve().parent)
        if tile_mode!='off' and _prepared_columns and selection.get('pointwise_experiment',{}).get('admitted'):
            from gemmini_pointwise_tiles import PointwiseTileResources, plan_pointwise_tiles
            root = Path(__file__).resolve().parent
            resources = PointwiseTileResources.from_selected(
                (root/'data/rtl_facts.json').read_bytes(),
                json.loads((root/'data/COMMAND_ORDERING_SELECTION.json').read_text()))
            source,input_rows,selection = plan_pointwise_tiles(source,resources,selection,
                prepared_columns=_prepared_columns,stage_channels=64,
                balanced_rows=tile_mode=='stage64_balanced')
        if input_rows is None:
            source, input_rows = prepare_rolling_input(source)
    schedule = plan_convolution_schedule(source)
    if pairs > 1:
        schedule = _epoch_schedule(source, pairs) or schedule
    schedule, rhs = apply_retention(schedule,
        'input_rhs' if retention_mode == 'rolling_rhs' else retention_mode,
        input_rows=input_rows, phased=input_rows is not None)
    if input_rows is not None and schedule.choice.load_issue:
        # Staging transfers have their own unchanged ready/last-use drains.
        # Bound the actual initial row transport, independently of queue slots.
        from dataclasses import replace
        issue = replace(schedule.issue_law,
            load_issue_budget=max(schedule.issue_law.load_issue_budget,input_rows.initial_load_commands))
        from merlin.compile.scheduling.convolution_exploration import ConvolutionScheduleChoice
        schedule = plan_convolution_schedule(source,schedule.choice,
            retention_law=schedule.retention_law,issue_law=issue,epoch_law=schedule.epoch_law)
        # Rebind the residency owner to the actual transformed schedule identity.
        schedule, rhs = apply_retention(schedule, 'input_rhs',
            input_rows=input_rows, phased=True)
    array_kwargs={}
    array_selection=dict(selected=False,reason='off')
    array_mode=knob('conv_array_retention',default='off',values=('off','on','auto','multirow'),root=Path(__file__).resolve().parent)
    from gemmini_pointwise_policy import select_array_mode
    if array_mode == 'auto':
        array_mode=select_array_mode(source,selection,array_mode)
    auto_admission=None
    wide_auto=False
    if array_mode=='auto':
        from gemmini_conv_array import admit_regional_multirow,admit_regional_wide_rows
        admitted,auto_admission=admit_regional_multirow(source,policy=policy,
            retention_mode=retention_mode)
        if not admitted:
            wide_auto,wide_admission=admit_regional_wide_rows(source,policy=policy,
                retention_mode=retention_mode)
            auto_admission={**auto_admission,'wide_region':wide_admission}
            admitted=wide_auto
        array_selection=dict(selected=False,reason='outside regional multirow geometry')
    if array_mode in ('on','multirow') or (array_mode=='auto' and admitted):
        from gemmini_conv_array import plan_array_retention
        if selection.get('pointwise_experiment',{}).get('admitted'):
            from gemmini_pointwise_policy import plan_pointwise_array_retention as plan_array_retention
        original_schedule,original_rhs=schedule,rhs
        schedule,rhs,array_kwargs,array_selection=plan_array_retention(schedule,rhs,input_rows,
            multirow=array_mode=='multirow' or wide_auto)
        if wide_auto:
            from merlin.compile.scheduling.convolution_exploration import ConvolutionRhsPhases
            group=array_kwargs.get('output_group')
            if (not array_selection.get('selected') or isinstance(rhs,ConvolutionRhsPhases)
                    or group is None or getattr(group,'output_rows',0)<2):
                schedule,rhs=original_schedule,original_rhs
                array_kwargs={}
                array_selection=dict(selected=False,
                    reason='wide auto requires complete unphased RHS and checked multiple source rows')
        input_rows=array_kwargs.pop('input_rows',input_rows)
    array_selection['requested_mode']=array_mode
    if auto_admission is not None:
        array_selection['auto_admission']=auto_admission
    counted_columns=bool(not _static_column_readout and readout is not None
        and len(set(readout.column_scale_bits)) > 1
        and all(width == 1 for _, width in intervals))
    column_mode=knob('conv_column_reuse',default=4,values=('off',4,8,16),
        root=Path(__file__).resolve().parent)
    column_selection=dict(requested=column_mode,selected=False)
    if (column_mode != 'off' and counted_columns and rhs is None and input_rows is None
            and not array_kwargs and not schedule.choice.load_issue
            and schedule.choice.compute_pairs == 1 and schedule.issue_law is None
            and schedule.epoch_law is None):
        from gemmini_column_reuse import plan_column_reuse
        owner,_=apply_retention(schedule,'input')
        if column_mode <= owner.retention_law.max_live_tiles:
            schedule=plan_column_reuse(schedule,law=owner.retention_law,live_tiles=column_mode)
            column_selection.update(selected=True,live_tiles=column_mode,
                source_identity=source.identity,schedule_identity=schedule.identity)
    spatial_epoch_mode=knob('conv_spatial_epochs',default='off',
        values=('off','pairs32','readout8','combined32'),root=Path(__file__).resolve().parent)
    spatial_epoch_selection=dict(requested=spatial_epoch_mode,selected=False,reason='off')
    if spatial_epoch_mode!='off':
        from gemmini_conv_spatial_epochs import plan_spatial_epochs
        root=Path(__file__).resolve().parent
        schedule,rhs,array_kwargs,spatial_epoch_selection=plan_spatial_epochs(
            schedule,rhs,input_rows,array_kwargs,
            facts_bytes=(root/'data/rtl_facts.json').read_bytes(),
            ordering_selection=json.loads((root/'data/COMMAND_ORDERING_SELECTION.json').read_text()),
            compute_pairs=8 if spatial_epoch_mode=='readout8' else 32,
            readout_epoch=spatial_epoch_mode in ('readout8','combined32'))
        spatial_epoch_selection['requested']=spatial_epoch_mode
    builder = FnBuilderControlAdapter(fn)
    lower = make_conv_lowerer(fn, input_ptr, weight_ptr, input_shape=input_shape,
        weight_shape=weight_shape, attrs=attrs, bias=bias, output_dtype=output_dtype,
        dst_ptr=dst_ptr, synchronous_completion=not schedule.choice.load_issue,
        a_stride=1 if input_rows is None else input_rows.row_step)

    from gemmini_activity import observe_control
    builder, lower, complete, activity = observe_control(
        builder, lower, lambda kind, fields: enc.fence(fn))
    receipt = emit_convolution_schedule(schedule, builder, lower,
        complete=complete,
        synchronous_completion=not schedule.choice.load_issue, rhs=rhs, input_rows=input_rows,
        counted_columns=counted_columns, **array_kwargs)
    schedule.verify_current()
    receipt['schedule_sha256'] = schedule.identity
    receipt['requested_retention'] = retention_mode
    receipt['policy_selection'] = selection
    receipt['array_retention_selection']=array_selection
    receipt['column_reuse_selection']=column_selection
    receipt['spatial_epoch_selection']=spatial_epoch_selection
    receipt['activity'] = activity.report(schedule.identity, output_dtype=output_dtype)
    enc.fence(fn)
    return receipt


def emit_conv(fn, input_ptr, weight_ptr, dst_ptr, *, input_shape, weight_shape,
              attrs=None, bias=None, output_dtype='i32'):
    """Select exact column layout preparation around the original emitter.

    Admission uses the source scale bits and original hardware intervals;
    all arithmetic and completion remain owned by the supplied emitter.
    """
    if knob('conv_column_direct',default='on',values=('off','on'),root=Path(__file__).resolve().parent)=='on':
        from gemmini_column_direct import try_emit_direct_conv
        direct = try_emit_direct_conv(fn,input_ptr,weight_ptr,dst_ptr,
            input_shape=input_shape,weight_shape=weight_shape,attrs=attrs,bias=bias,
            output_dtype=output_dtype,ordinary_emitter=_emit_conv,primitive=_resources()[0])
        if direct is not None:
            return direct
    mode = knob('conv_column_grouping', default='on', values=('off', 'on'),
                root=Path(__file__).resolve().parent)
    if mode == 'on':
        try:
            from .gemmini_column_grouping import emit_grouped_conv
        except ImportError:
            from gemmini_column_grouping import emit_grouped_conv
        return emit_grouped_conv(fn, input_ptr, weight_ptr, dst_ptr,
            input_shape=input_shape, weight_shape=weight_shape, attrs=attrs,
            bias=bias, output_dtype=output_dtype, ordinary_emitter=_emit_conv,
            prepared_emitter=lambda *a,**k:_emit_conv(*a,**k,_prepared_columns=True))
    return _emit_conv(fn, input_ptr, weight_ptr, dst_ptr,
        input_shape=input_shape, weight_shape=weight_shape, attrs=attrs,
        bias=bias, output_dtype=output_dtype)
