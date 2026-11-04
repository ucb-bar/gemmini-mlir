"""Bounded spatial maximum using Gemmini accumulator readout pooling.

The shared spatial gather planner owns complete windows, border duplication,
channel tails and output ordering. Its actual counted program is consumed here.
Every window occupies at most nine private accumulator rows. Host instructions
compute indices only; data scaling, activation and maximum all run on device.
"""
from __future__ import annotations

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.spatial_gather import plan_gathered_spatial_max
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain, prove_spatial_readout
import gemmini_encode as E
from gemmini_readout import config_pool_store, column_scale_runs, _relu


def _pair(value, name):
    if isinstance(value, int):
        value = (value, value)
    value = tuple(value)
    if len(value) != 2 or any(type(x) is not int or x <= 0 for x in value):
        raise ValueError(f'{name} requires two positive integers')
    return value


def pool_geometry(shape, attrs):
    """Read source NHWC or flattened spatial shape, preserving floor/ceil mode."""
    shape, attrs = tuple(shape), dict(attrs or {})
    if any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('pool requires a static nonempty source shape')
    if 'pool_in_dims' in attrs:
        h, w = _pair(attrs['pool_in_dims'], 'pool_in_dims')
        c = shape[-1]
        total = 1
        for x in shape:
            total *= x
        if total % (h * w * c):
            raise ValueError('pool_in_dims does not cover the source tensor')
        batches = total // (h * w * c)
    elif len(shape) == 4:
        if attrs.get('layout', 'nhwc').lower() != 'nhwc':
            raise ValueError('pool source must have NHWC layout')
        batches, h, w, c = shape
    elif len(shape) == 3:
        h, w, c = shape
        batches = 1
    else:
        raise ValueError('flattened pool source requires pool_in_dims')
    window = _pair(attrs.get('pool_size', attrs.get('kernel_size', attrs.get('kernel', (2, 2)))), 'pool_size')
    stride = _pair(attrs.get('pool_stride', attrs.get('stride', window)), 'pool_stride')
    if window[0] != window[1] or window[0] not in (1, 2, 3):
        raise ValueError('selected Gemmini pooling needs a square window of size 1, 2 or 3')
    dilation = _pair(attrs.get('pool_dilation', attrs.get('dilation', (1, 1))), 'pool_dilation')
    if dilation != (1, 1):
        raise ValueError('dilated pooling is not implemented by this target adapter')
    padding = attrs.get('pool_padding', attrs.get('padding', (0, 0, 0, 0)))
    if isinstance(padding, int):
        padding = (padding,) * 4
    padding = tuple(padding)
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    if len(padding) != 4 or any(type(x) is not int or x < 0 for x in padding):
        raise ValueError('pool padding must be top,left,bottom,right nonnegative integers')
    if attrs.get('ceil_mode', attrs.get('pool_ceil_mode', False)):
        # Convert ceil output extents to an equivalent floor-window source.
        pt, pl, pb, pr = padding
        kh, kw = window
        sh, sw = stride
        oh = (h + pt + pb - kh + sh - 1) // sh + 1
        ow = (w + pl + pr - kw + sw - 1) // sw + 1
        if (oh - 1) * sh >= h + pt:
            oh -= 1
        if (ow - 1) * sw >= w + pl:
            ow -= 1
        pb = max(pb, (oh - 1) * sh + kh - h - pt)
        pr = max(pr, (ow - 1) * sw + kw - w - pl)
        padding = (pt, pl, pb, pr)
    pad = attrs.get('pool_pad_value', attrs.get('pad_value', -128))
    if pad in ('-inf', float('-inf')):
        pad = -128
    if any(padding) and pad != -128:
        if pad == 0 and _relu(attrs):
            # ReLU values are nonnegative. Border duplication and zero padding
            # then have identical maxima for every nonempty source window.
            pad = -128
        else:
            raise ValueError('pool padding requires minimum signed-i8 value; zero padding is admitted only after ReLU')
    spatial = SpatialMaxReadout((h, w), window, stride, padding, int(pad))
    return {'batches': batches, 'channels': c, 'source': spatial,
            'input_shape': (batches, h, w, c),
            'output_shape': (batches, *spatial.output_hw, c)}


def emit_pool(fn, src, dst, *, shape, attrs, dtype='i8'):
    """Pool a contiguous host source through bounded accelerator windows.

    `dtype='i32'` accepts raw producer accumulators (bias already added), and
    applies the original acc_scale/ReLU exactly once before maximum. `i8`
    accepts an already materialized source; pass only the pool's own readout
    parameters so a producer quantizer is never repeated. Outputs are signed
    i8. Every padding window must contain at least one source cell.
    """
    attrs = dict(attrs or {})
    if dtype not in ('i8', 'i32'):
        raise ValueError('pool supports i8 source or raw i32 producer accumulators')
    if attrs.get('output_dtype', 'i8') not in ('i8', 'int8'):
        raise ValueError('Gemmini pooling outputs require signed i8')
    if attrs.get('requant_shift', attrs.get('shift', 0)):
        raise ValueError('pool readout cannot replace half-up requantization by half-even scaling')
    epilogue = attrs.get('epilogue', []) or []
    if isinstance(epilogue, str):
        epilogue = [epilogue]
    unknown = set(epilogue) - {'maxpool', 'relu', 'acc_scale', 'bias_add'}
    if unknown:
        raise ValueError(f'unsupported pool readout epilogue {sorted(unknown)}')
    geometry = pool_geometry(shape, attrs)
    source = geometry['source']
    batches, channels = geometry['batches'], geometry['channels']
    runs = column_scale_runs(attrs, channels)
    plans = []
    # Build all source/capacity admissions before emitting any partial program.
    for first, columns, scale in runs:
        plan = plan_gathered_spatial_max(source, batches=batches,
                                       channels=columns, channel_block=columns,
                                       max_window_elements=9)
        scalar = RoundedReadout((scale,), _relu(attrs))
        proof = prove_spatial_readout(SpatialReadoutChain(None, scalar, scalar))
        plans.append((first, columns, scale, plan, proof))
    control = FnBuilderIndexAdapter(fn)
    width = 1 if dtype == 'i8' else 4
    E.fence(fn)
    E.config_load(fn, channels * width, shrunk=dtype == 'i8')
    window = source.window[0]
    config_pool_store(fn, channels, pool_stride=1, pool_size=window,
                      pool_out_dim=1, porows=1, pocols=1,
                      orows=window, ocols=window,
                      scale=plans[0][2], activation=int(_relu(attrs)))

    def add(a, b):
        return control.op('add', a, b)

    def mul(a, b):
        return control.op('mul', a, b)

    for first, columns, scale, plan, proof in plans:
        config_pool_store(fn, channels, pool_stride=1, pool_size=window,
                          pool_out_dim=1, porows=1, pocols=1,
                          orows=window, ocols=window,
                          scale=scale, activation=int(_relu(attrs)))

        def lower(opcode, fields):
            column = add(first, fields['first_column'])
            if opcode == 'gather_window_row':
                offset = mul(add(mul(fields['source_row'], channels), column), width)
                ptr = E.pointer_offset(fn, src, offset)
                local = add(E.ACC, fields['window_row'])
                E.mvin(fn, ptr, local, 1, fields['columns'])
                E.fence(fn)
            elif opcode == 'store_window_max':
                offset = add(mul(fields['output_row'], channels), column)
                ptr = E.pointer_offset(fn, dst, offset)
                # rows=0 selects the configured spatial pooling traversal.
                E.mvout(fn, ptr, E.ACC, 0, fields['columns'])
                E.fence(fn)
            else:
                raise ValueError(f'unknown shared spatial pool operation {opcode}')

        emit_index_program(plan.kernel, control, lower)
    E.fence(fn)
    return {'route': 'gathered_accumulator_pool_readout',
            'input_dtype': dtype, 'output_dtype': 'i8',
            'input_shape': geometry['input_shape'],
            'output_shape': geometry['output_shape'],
            'temporary_rows': window * window,
            'scale_runs': len(plans),
            'source_plans': [plan.kernel.digest() for _, _, _, plan, _ in plans],
            'readout_proofs': [proof for _, _, _, _, proof in plans],
            'device_execution_verified': False}


def emit_accumulator_pool(fn, dst, local, *, input_hw, channels, attrs,
                          output_stride=None):
    """Read one resident spatial channel tile directly with hardware pooling.

    Rows must contain contiguous flattened HxW positions with each pixel's
    channel tile starting at lane zero. This fast adapter requires no padding;
    use emit_pool's bounded gathering route for exact padded sources.
    Caller completes this operation before reusing producer accumulator rows.
    """
    attrs = dict(attrs or {})
    geometry = pool_geometry((input_hw[0] * input_hw[1], channels),
                             {**attrs, 'pool_in_dims': list(input_hw)})
    source = geometry['source']
    if any(source.padding):
        raise ValueError('resident pool readout does not substitute hardware zero for source padding')
    if source.stride[0] != source.stride[1] or source.stride[0] not in (1, 2, 3):
        raise ValueError('resident pool stride must be symmetric and fit the ISA')
    if input_hw[0] * input_hw[1] > 1024 or max(input_hw) > 255:
        raise ValueError('resident pool exceeds selected accumulator rows or ISA spatial fields')
    if not 1 <= channels <= 16:
        raise ValueError('resident pool needs one lane-zero channel tile')
    runs = column_scale_runs(attrs, channels)
    if len(runs) != 1:
        raise ValueError('resident pool needs equal source scales across its channel tile')
    oh, ow = source.output_hw
    config_pool_store(fn, output_stride or channels,
                      pool_stride=source.stride[0], pool_size=source.window[0],
                      pool_out_dim=ow, porows=oh, pocols=ow,
                      orows=input_hw[0], ocols=input_hw[1],
                      scale=runs[0][2], activation=int(_relu(attrs)))
    address = fn.or_i(fn.and_i(E._i64(fn, local), fn.const((1 << 29) - 1)), fn.const(E.ACC))
    E.mvout(fn, dst, address, 0, channels)
    return geometry['output_shape']
