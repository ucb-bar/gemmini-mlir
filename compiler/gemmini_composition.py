"""Optional source-bound producer/maxpool composition over bounded ACC stripes.

The default backend remains its complete-producer spill/readout route. This
entry consumes the shared spatial producer's actual counted program and the
ordinary convolution terminal callback. No complete intermediate is created.
Source observations must be supplied explicitly; observed intermediates refuse.
"""
from __future__ import annotations

from pathlib import Path
import math

from merlin.compile.composition_catalog import composition_capabilities
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, emit_convolution_tile
from merlin.compile.scheduling.spatial_producer import SpatialProducerPlan, emit_spatial_producer
from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter, emit_index_program
from merlin.compile.scheduling.spatial_max_stripes import plan_striped_spatial_max
import gemmini_encode as E
from gemmini_conv import _geometry, _resources, make_conv_lowerer
from gemmini_pool import pool_geometry
from gemmini_readout import source_scales, _relu, config_pool_store
from gemmini_pool_pipeline import plan_producer_pool_epochs, emit_epoch_producer_tile
from merlin.compile.tuning_knobs import knob


class CompositionUnavailable(ValueError):
    """The source requires the incumbent complete-producer route."""


def _catalog():
    root = Path(__file__).resolve().parent / 'compiler_library'
    modules = []
    for path in (root / 'merlin').rglob('*.py'):
        parts = list(path.relative_to(root).with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        modules.append('.'.join(parts))
    catalog = composition_capabilities(selected_modules=tuple(set(modules)))
    row = next(r for r in catalog['capabilities'] if r['name'] == 'bounded_spatial_packing_and_rolling')
    # This sibling single-readout realization uses the same source/halo contract,
    # but retains raw ACC stripes instead of requiring packed host workspace.
    return {'catalog_row': row['name'], 'access': row['access'],
            'realization': 'spatial_producer.single_readout',
            'catalog_source_availability': row['source_availability']}


def plan_producer_pool(*, kind, input_shape, weight_shape, attrs,
                       bias_present=False, external_observers=(), stripe_outputs=None):
    """Admit one original conv/dense -> scaled/ReLU/maxpool source chain.

    Recomputing pure halo products preserves its original ordered reduction and
    both source readout boundaries. No source parameter is retuned. For dense,
    reinterpret the row-major matrix as a 1x1 NHWC convolution; no packing or
    tensor copy occurs. Scalar readout scales are currently required.
    """
    attrs = dict(attrs or {})
    external_observers = tuple(external_observers)
    if external_observers:
        raise CompositionUnavailable('an original intermediate output has observers; retain its incumbent realization')
    if attrs.get('output_dtype', 'i8') not in ('i8', 'int8'):
        raise CompositionUnavailable('producer pool requires signed-i8 source output')
    epilogue = attrs.get('epilogue', []) or []
    if isinstance(epilogue, str):
        epilogue = [epilogue]
    if 'maxpool' not in epilogue:
        raise CompositionUnavailable('source has no maxpool epilogue')
    if set(epilogue) - {'bias_add', 'acc_scale', 'relu', 'maxpool'}:
        raise CompositionUnavailable('source epilogue contains an unsupported stage')
    if ('bias_add' in epilogue) != bool(bias_present):
        raise CompositionUnavailable('source bias stage and original bias binding differ')
    if attrs.get('requant_shift', attrs.get('shift', 0)):
        raise CompositionUnavailable('half-up source requantization requires its original readout')
    scale = source_scales(attrs)
    if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise CompositionUnavailable('bounded producer pool currently needs a positive scalar source scale')
    geometry_attrs = {k: attrs[k] for k in ('kernel', 'stride', 'padding', 'dilation', 'layout') if k in attrs}
    if kind in ('dense', 'MATMUL', 'MATMUL_RESIDENT'):
        if len(input_shape) != 2 or len(weight_shape) != 2 or input_shape[1] != weight_shape[0]:
            raise CompositionUnavailable('dense producer needs original rank-two compatible inputs')
        if 'pool_in_dims' not in attrs:
            raise CompositionUnavailable('dense producer requires original pool_in_dims')
        h, w = tuple(attrs['pool_in_dims'])
        if input_shape[0] % (h * w):
            raise CompositionUnavailable('dense rows do not cover complete original spatial images')
        batches, ci = input_shape[0] // (h * w), input_shape[1]
        actual_input_shape = (batches, h, w, ci)
        geometry_attrs = {'kernel': (1, 1, ci, weight_shape[-1]),
                          'stride': (1, 1), 'padding': (0, 0, 0, 0),
                          'dilation': (1, 1), 'layout': 'nhwc'}
    elif kind in ('conv', 'CONV2D'):
        actual_input_shape = tuple(input_shape)
    else:
        raise CompositionUnavailable('composition supports original dense and NHWC convolution producers')
    window, channels = _geometry(actual_input_shape, weight_shape, geometry_attrs)
    h, w = window.output_spatial
    pool_attrs = {**attrs, 'pool_in_dims': [h, w]}
    # Convolution dilation belongs to its producer, never to the pool window.
    pool_attrs.pop('dilation', None)
    spatial = pool_geometry((window.input_shape[0], h, w, channels), pool_attrs)['source']
    if spatial.window[0] != spatial.window[1] or spatial.stride[0] != spatial.stride[1]:
        raise CompositionUnavailable('hardware pool config needs square window and symmetric stride')
    if spatial.stride[0] not in (1, 2, 3):
        raise CompositionUnavailable('hardware pool stride exceeds the selected two-bit ISA field')
    edge, operand_rows, accumulator_rows = _resources()
    row_tile = min(edge, w)
    producer = CountedConvolutionPlan(window, channels, edge, row_tile, edge,
        min(edge, window.input_shape[-1]), operand_rows, accumulator_rows,
        0, edge, 0, bias_present)
    kh, kw = spatial.window
    max_input_width = min(255, accumulator_rows // kh)
    widest = min(spatial.output_hw[1], 255,
                  max(0, (max_input_width - kw) // spatial.stride[1] + 1))
    if widest < 1:
        raise CompositionUnavailable('bounded spatial stripe does not fit selected accumulator rows')
    if stripe_outputs is not None:
        if type(stripe_outputs) is not int or not 1 <= stripe_outputs <= widest:
            raise CompositionUnavailable('requested stripe exceeds actual capacity or source output width')
        widest = stripe_outputs
    q1 = RoundedReadout((float(scale),), _relu(attrs))
    identity_q2 = RoundedReadout((1.0,), False)
    # The exact source quantizer is preserved; Q2 is identity over signed i8.
    chain = SpatialReadoutChain(q1, identity_q2, q1)
    blocks = (channels + edge - 1) // edge
    plan = SpatialProducerPlan(producer, spatial, (chain,) * blocks, widest,
                               external_observers=external_observers,
                               readout_mode='single_readout')
    return plan, geometry_attrs, actual_input_shape


def emit_producer_pool(fn, input_ptr, weight_ptr, dst_ptr, *, kind,
                       input_shape, weight_shape, attrs, bias=None,
                       external_observers=(), stripe_outputs=None, geometry='halo'):
    """Emit the optional bounded catalog composition, returning checked receipts.

    The caller selects this alternative explicitly. Admission failures occur
    before target instructions and let the caller keep the incumbent. Original
    output pointers are supplied by the caller and are written exactly once.
    """
    if geometry in ('readout_cache', 'strip_epochs'):
        from gemmini_pool_store import emit_pool_store
        return emit_pool_store(fn,input_ptr,weight_ptr,dst_ptr,kind=kind,
            input_shape=input_shape,weight_shape=weight_shape,attrs=attrs,bias=bias,
            external_observers=external_observers,
            readout_schedule='epochs' if geometry == 'strip_epochs' else 'strips')
    plan, geometry_attrs, actual_input_shape = plan_producer_pool(
        kind=kind, input_shape=input_shape, weight_shape=weight_shape, attrs=attrs,
        bias_present=bias is not None, external_observers=external_observers,
        stripe_outputs=stripe_outputs)
    if geometry not in ('halo', 'striped'):
        raise CompositionUnavailable('unknown optional producer geometry')
    if geometry == 'striped':
        return _emit_striped_producer_pool(fn, input_ptr, weight_ptr, dst_ptr,
            plan=plan, geometry_attrs=geometry_attrs, input_shape=actual_input_shape,
            weight_shape=weight_shape, bias=bias, stripe_outputs=stripe_outputs)
    catalog = _catalog()
    lower_product = make_conv_lowerer(fn, input_ptr, weight_ptr,
        input_shape=actual_input_shape, weight_shape=weight_shape,
        attrs=geometry_attrs, bias=bias, output_dtype='i32',
        synchronous_completion=True)
    control = FnBuilderControlAdapter(fn)
    channels = plan.producer.output_channels
    scale = plan.readouts[0].replacement.scales[0]
    activation = int(plan.readouts[0].replacement.relu)

    def lower(kind, fields):
        if kind != 'store_spatial_max':
            lower_product(kind, fields)
            return
        config_pool_store(fn, channels,
            pool_stride=fields['stride_width'], pool_size=fields['window_width'],
            pool_out_dim=fields['output_pixels'],
            porows=1, pocols=fields['output_pixels'],
            orows=fields['local_height'], ocols=fields['local_width'],
            scale=scale, activation=activation)
        offset = control.op('add', control.op('mul', fields['output_row'], channels),
                            fields['output_column'])
        dst = E.pointer_offset(fn, dst_ptr, offset)
        address = control.op('add', E.ACC, fields['accumulator_row'])
        E.mvout(fn, dst, address, 0, fields['columns'])
        E.fence(fn)

    receipt = emit_spatial_producer(plan, control, lower, synchronous_completion=True)
    E.fence(fn)
    return {**receipt, **catalog, 'optional_route': True,
            'original_output_retained': True, 'host_payload_arithmetic': False,
            'host_intermediate_bytes': 0, 'device_execution_verified': False}


def _striped_plans(plan, *, stripe_outputs=None):
    """Use the library's canonical border/interior partition without packing.

    The complete horizontal strip is mandatory for this library realization.
    Its temporary rows, and the hardware's eight-bit local width, bound its
    admission. The older narrower halo realization remains independently usable.
    """
    p, s = plan.producer, plan.spatial
    width = (s.output_hw[1] - 1) * s.stride[1] + s.window[1]
    if stripe_outputs is not None and stripe_outputs != s.output_hw[1]:
        raise CompositionUnavailable('striped geometry requires a complete horizontal output strip')
    if width > 255 or s.output_hw[1] > 255:
        raise CompositionUnavailable('complete strip exceeds hardware pool geometry fields')
    plans = []
    for column in range(0, p.output_channels, p.column_tile):
        columns = min(p.column_tile, p.output_channels - column)
        try:
            stripe = plan_striped_spatial_max(s, batches=p.window.input_shape[0],
                channels=columns, channel_block=columns, transfer_rows=p.row_tile,
                capacity_rows=p.accumulator_rows - p.accumulator_row)
        except ValueError as exc:
            raise CompositionUnavailable(str(exc)) from exc
        plans.append((column, columns, stripe))
    return plans


def _emit_striped_producer_program(plan, control, lower_product, store, *, stripe_outputs=None,
                                   compute_schedule=None, complete=None):
    """Consume capacity-derived vertical epochs of the original strip source.

    Each stage row denotes one or more contiguous original producer pixels.
    Only clamped border slots are produced separately. Original convolution
    source order, bias and raw accumulator cells retain the same terminal owner.
    """
    plans = _striped_plans(plan, stripe_outputs=stripe_outputs)
    p = plan.producer
    receipts = []
    for column, columns, stripe in plans:
        epoch = plan_producer_pool_epochs(stripe)
        compute_receipts=[]
        def lower(opcode, fields):
            if opcode == 'stage_spatial_rows':
                acc=control.op('add',p.accumulator_row,fields['local_row'])
                if compute_schedule is not None:
                    compute_receipts.append(emit_epoch_producer_tile(compute_schedule,
                        control,lower_product,complete,output_row=fields['source_row'],
                        output_column=column,rows=fields['rows'],columns=columns,
                        accumulator_row=acc))
                    return
                def effect(kind, values):
                    lower_product('finish_raw_tile' if kind == 'store' else kind, values)
                emit_convolution_tile(p, control, effect,
                    output_row=fields['source_row'], output_column=column,
                    rows=fields['rows'], columns=columns,
                    accumulator_row=acc,
                    synchronous_completion=True)
            elif opcode == 'read_spatial_strip':
                store(fields['output_row'], column, columns,
                      control.op('add',p.accumulator_row,fields['local_row']))
            else:
                raise ValueError('unknown checked producer strip effect ' + opcode)
        emission = emit_index_program(epoch.kernel, control, lower)
        receipts.append({**epoch.revalidate(), **emission,
                         'original_output_column': column, 'columns': columns,
                         'compute_fragments':compute_receipts})
    return receipts


def _emit_striped_producer_pool(fn, input_ptr, weight_ptr, dst_ptr, *, plan,
        geometry_attrs, input_shape, weight_shape, bias, stripe_outputs):
    # Admission, including every channel tail, precedes any target command.
    plans = _striped_plans(plan, stripe_outputs=stripe_outputs)
    p, s = plan.producer, plan.spatial
    control = FnBuilderControlAdapter(fn)
    pairs=knob('pool_producer_epoch_pairs',default=1,values=(1,8),root=Path(__file__).resolve().parent)
    compute_schedule=None
    if pairs>1:
        from gemmini_conv import _epoch_schedule
        from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
        source=ConvolutionExplorationSource(p,tuple((c,min(p.column_tile,p.output_channels-c))
            for c in range(0,p.output_channels,p.column_tile)),plan.identity)
        compute_schedule=_epoch_schedule(source,pairs)
    lower_product = make_conv_lowerer(fn, input_ptr, weight_ptr,
        input_shape=input_shape, weight_shape=weight_shape, attrs=geometry_attrs,
        bias=bias, output_dtype='i32', synchronous_completion=compute_schedule is None)
    # Product lowering modifies load/execute configuration only. This store
    # configuration therefore remains invariant across every channel and row.
    height, width = plans[0][2].stripe_hw
    q = plan.readouts[0].replacement
    config_pool_store(fn, p.output_channels,
        pool_stride=s.stride[1], pool_size=s.window[1],
        pool_out_dim=s.output_hw[1], porows=1, pocols=s.output_hw[1],
        orows=height, ocols=width, scale=q.scales[0], activation=int(q.relu))

    def store(row, column, columns, accumulator_row):
        offset = control.op('add', control.op('mul', row, p.output_channels), column)
        E.mvout(fn, E.pointer_offset(fn, dst_ptr, offset),
                control.op('add',E.ACC,accumulator_row),0,columns)
        E.fence(fn)

    receipts = _emit_striped_producer_program(plan, control, lower_product, store,
        stripe_outputs=stripe_outputs,compute_schedule=compute_schedule,
        complete=lambda kind,fields:E.fence(fn))
    E.fence(fn)
    return {**_catalog(), 'realization': 'spatial_max_stripes.raw_producer_epochs',
        'producer_geometry': 'striped', 'source_plan_sha256': plan.identity,
        'strip_receipts': receipts, 'retained_readout_chains': [str(c) for c in plan.readouts],
        'private_accumulator_rows': max(r['temporary_rows'] for r in receipts),
        'original_output_retained': True, 'optional_route': True,
        'host_payload_arithmetic': False, 'host_intermediate_bytes': 0,
        'device_execution_verified': False}
