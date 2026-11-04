"""Compose a full-width raw producer with exact scalar column readouts.

The original bias stays in the producer's i32 accumulator. The private raw
matrix crosses no source numerical boundary; original scaling and activation
occur once in the checked readout. Explicit refusal precedes any target issue.
"""
from copy import deepcopy
from math import prod
from math import gcd
from pathlib import Path

from merlin.compile.tuning_knobs import knob

DECLARATION = {
    'values': ['off', 'raw_then_readout', 'raw_then_grouped_readout', 'supported_geometry'],
    'mechanism': 'full-width raw i32 contraction followed by checked per-column hardware readout',
    'note': 'Experimental nonpool positive-scale composition; off incumbent. Complete-call timing includes raw spill/reload and original narrowing.',
}


def selected():
    return knob('column_producer_schedule', default='off',
                values=('off', 'raw_then_readout', 'raw_then_grouped_readout', 'supported_geometry'), root=Path(__file__).parent)


def prepare(attrs, m, n, *, family='dense'):
    """Return a fully admitted exact readout before allocating/emitting anything."""
    mode = selected()
    if mode == 'off' or attrs.get('output_dtype', 'i32') != 'i8':
        return None
    scales = attrs.get('acc_scale_by_column')
    stages = attrs.get('epilogue', [])
    if scales is None or 'maxpool' in stages or n <= 1:
        return None
    if ('acc_scale' not in stages or
            set(stages) - {'bias', 'bias_add', 'acc_scale', 'relu'}):
        return None
    if mode == 'supported_geometry' and family == 'convolution':
        # The ordinary convolution emitter already preserves a full channel
        # tile when scale changes align to that tile. Paired complete timings
        # favor the composition only for its single-column partition.
        from isa import DIM
        partition = DIM
        for col in range(1, n):
            if float(scales[col]) != float(scales[col-1]):
                partition = gcd(partition, col)
        if partition != 1:
            return None
    from column_readout import plan_column_readout
    try:
        return plan_column_readout(m, n, scales, activation='relu' if 'relu' in stages else (),
                                   group_runs=mode in ('raw_then_grouped_readout','supported_geometry'))
    except ValueError:
        return None


def raw_attributes(attrs):
    raw = deepcopy(attrs)
    raw['output_dtype'] = 'i32'
    raw['epilogue'] = [stage for stage in attrs.get('epilogue', []) if stage in ('bias', 'bias_add')]
    raw.pop('acc_scale_by_column', None)
    raw.pop('acc_scale', None)
    return raw


def record(builder, producer, readout, family):
    result = {'mechanism': 'full_width_raw_then_column_readout', 'family': family,
              'source_numerical_boundary': 'bias in i32 producer; one original f32 scale and narrowing in hardware readout',
              'raw_producer': producer, 'column_readout': readout,
              'performance_status': 'experimental complete-call comparison required'}
    if not hasattr(builder, 'column_producer_receipts'):
        builder.column_producer_receipts = []
    builder.column_producer_receipts.append(result)
    return result


def lower_convolution(builder, command, tensors, pointers, *, residents=None):
    from convolution import lower_conv
    from pointwise_convolution import emit as emit_pointwise
    pointwise = emit_pointwise(builder, command, tensors, pointers, residents=residents)
    if pointwise is not None:
        if not hasattr(builder, 'pointwise_convolution_receipts'):
            builder.pointwise_convolution_receipts = []
        builder.pointwise_convolution_receipts.append(pointwise)
        return pointwise
    attrs = dict(command.get('attributes') or {})
    dst = command['operands']['dst']
    shape = tensors[dst]['shape']
    n = shape[-1]
    m = prod(shape[:-1])
    pool_selection = _select_pool(command, tensors)
    pool_options = pool_selection.get('options', {}) if pool_selection else {}
    if pool_selection:
        if not hasattr(builder, 'conv_pool_selection_receipts'):
            builder.conv_pool_selection_receipts = []
        builder.conv_pool_selection_receipts.append(pool_selection)
    pooled = _materialized_pool(builder, command, tensors, pointers, residents=residents,
                                mode=pool_options.get('conv_pool_materialization'))
    if pooled is not None:
        return pooled
    readout = prepare(attrs, m, n, family='convolution')
    if readout is None:
        return lower_conv(builder, command, tensors, pointers, residents=residents,
                          schedule_options=pool_options)
    raw_name = '__private_column_raw'
    while raw_name in tensors:
        raw_name += '_'
    local_tensors = dict(tensors)
    local_pointers = dict(pointers)
    local_tensors[raw_name] = {'shape': list(shape), 'dtype': 'i32', 'role': 'temporary'}
    raw = builder.global_buffer(m*n, 'i32')
    local_pointers[raw_name] = raw
    raw_command = deepcopy(command)
    raw_command['operands']['dst'] = raw_name
    raw_command['attributes'] = raw_attributes(attrs)
    producer = lower_conv(builder, raw_command, local_tensors, local_pointers, residents=residents)
    narrowed = readout.emit(builder, raw, pointers[dst])
    return record(builder, producer, narrowed, 'convolution')


def _select_pool(command, tensors):
    from conv_pool_selection import selected, select
    attrs = dict(command.get('attributes') or {})
    if selected() == 'off' or 'maxpool' not in attrs.get('epilogue', []):
        return None
    from convolution import _name, _capacities
    from merlin.compile.scheduling.window_transfer import Window2D, CountedConvolutionPlan
    operands = command['operands']
    src = _name(operands,'ifm','input','A','lhs','src')
    shape = tuple(tensors[src]['shape'])
    window = Window2D(shape,tuple(attrs['kernel'])[:2],tuple(attrs.get('stride',(1,1))),
        tuple(attrs.get('dilation',(1,1))),tuple(attrs.get('padding',(0,0,0,0))))
    edge, operand_rows, accumulator_rows = _capacities()
    channels = tensors[operands['dst']]['shape'][-1]
    column_tile = edge
    scales = attrs.get('acc_scale_by_column')
    if scales:
        for column in range(1,channels):
            if float(scales[column]) != float(scales[column-1]):
                column_tile = gcd(column_tile,column)
    plan = CountedConvolutionPlan(window,channels,edge,min(edge,window.output_spatial[1]),
        column_tile,min(edge,shape[-1]),operand_rows,accumulator_rows,0,edge,0,
        bool(attrs.get('bias',operands.get('bias'))))
    # This adapter owns allocation of exactly this complete private BSS image
    # if the selected route admits it. No source tensor aliases that storage.
    workspace = shape[0]*prod(window.output_spatial)*channels
    return select(plan,attrs,workspace_limit_bytes=workspace)


def _materialized_pool(builder, command, tensors, pointers, *, residents=None, mode=None):
    """Preserve quantized convolution, then consume checked spatial maximum."""
    from conv_pool_policy import materialization_selected, plan_quantized_once, emit_quantized_once
    attrs = dict(command.get('attributes') or {})
    if (materialization_selected() if mode is None else mode) == 'off' or 'maxpool' not in attrs.get('epilogue', []):
        return None
    operands = command['operands']
    dst = operands['dst']
    if tensors[dst]['dtype'] != 'i8' or attrs.get('output_dtype','i8') != 'i8':
        return None
    from convolution import _name, lower_conv
    from merlin.compile.scheduling.window_transfer import Window2D
    src = _name(operands,'ifm','input','A','lhs','src')
    source_shape = tuple(tensors[src]['shape'])
    window = Window2D(source_shape,tuple(attrs['kernel'])[:2],tuple(attrs.get('stride',(1,1))),
                      tuple(attrs.get('dilation',(1,1))),tuple(attrs.get('padding',(0,0,0,0))))
    oh, ow = window.output_spatial
    size, stride = tuple(attrs['pool_size']), tuple(attrs['pool_stride'])
    if len(size)!=2 or size[0]!=size[1] or len(stride)!=2 or stride[0]!=stride[1]:
        return None
    channels = tensors[dst]['shape'][-1]
    try:
        plan = plan_quantized_once(batches=source_shape[0],input_hw=(oh,ow),channels=channels,
            pool_size=size[0],pool_stride=stride[0],padding=tuple(attrs.get('pool_padding',(0,0,0,0))),
            pad_value=int(attrs.get('pool_pad_value',-128)),relu='relu' in attrs.get('epilogue',[]))
    except ValueError:
        return None
    if prod(tensors[dst]['shape']) != source_shape[0]*prod(plan.source.output_hw)*channels:
        raise ValueError('materialized pooling output differs from the original source extent')
    name = '__private_prepool_i8'
    while name in tensors:
        name += '_'
    workspace_bytes = source_shape[0]*oh*ow*channels
    workspace = builder.global_buffer(workspace_bytes,'i8')
    local_tensors, local_pointers = dict(tensors), dict(pointers)
    local_tensors[name] = {'shape':[source_shape[0],oh,ow,channels],'dtype':'i8','role':'temporary'}
    local_pointers[name] = workspace
    producer = deepcopy(command)
    producer['operands']['dst'] = name
    producer['attributes']['epilogue'] = [x for x in attrs['epilogue'] if x!='maxpool']
    for key in tuple(producer['attributes']):
        if key.startswith('pool_'):
            del producer['attributes'][key]
    receipt = emit_quantized_once(builder,plan,intermediate=workspace,output=pointers[dst],
        workspace_bytes=workspace_bytes,
        emit_producer=lambda _: lower_conv(builder,producer,local_tensors,local_pointers,residents=residents))
    if not hasattr(builder,'conv_pool_materialization_receipts'):
        builder.conv_pool_materialization_receipts = []
    builder.conv_pool_materialization_receipts.append(receipt)
    return receipt
