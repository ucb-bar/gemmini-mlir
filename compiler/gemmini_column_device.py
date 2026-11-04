"""Original-column unary device epilogue after an exact raw-i32 producer.

The producer owns source K order and bias. A counted shared index program then
loads each original column into accumulator lane zero in at most sixteen-row
chunks. Full-width MVIN is an exact copy; narrowing MVOUT owns scale/RNE/clamp.
No CPU instruction reads or writes tensor payload. Source scale words are encoded
as constant command configuration, with source-index selection where needed.
"""
from dataclasses import dataclass
from math import prod
from pathlib import Path
import hashlib
import json
import struct

from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter, emit_index_effect_program
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select
import gemmini_encode as E
from gemmini_readout import _relu, source_scales


@dataclass(frozen=True)
class DeviceColumnPlan:
    rows: int
    columns: int
    row_tile: int
    readout: DenseTiledIntegerReadout
    kernel: Kernel
    evidence: dict


def selected_device_evidence():
    """Admit the actual full-width load path and immutable selected declarations."""
    import yaml
    from gemmini_residual_schedule import selected_epoch_evidence
    evidence = selected_epoch_evidence()
    root = Path(__file__).resolve().parent / 'data'
    contract = yaml.safe_load((root/'target_contract.yaml').read_text())
    ep = contract['output_epilogue']
    preload = ep['bias_preload']
    if (preload.get('status') != 'derived' or preload.get('dtype') != 'i32'
            or preload.get('bits') != 32 or preload.get('rounding') != 'none'
            or preload.get('range') != [-2147483648, 2147483647]
            or preload.get('lanes', {}).get('lanes') != E.DIM):
        raise ValueError('original-column epilogue lacks exact signed-i32 accumulator MVIN')
    if evidence['resources']['accumulator_rows'] < E.DIM:
        raise ValueError('original-column epilogue lacks sixteen private accumulator rows')
    evidence = dict(evidence, full_width_load=preload,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return evidence


def plan_device_columns(rows, columns, attrs, *, external_buffers_disjoint=False,
                        synchronous_raw_completion=False):
    """Admit all offsets, source scales, capacity and completion before emission.

    The disjointness premise refers to a new invocation-private raw buffer,
    separate from every external tensor. Raw producer completion is a real full
    fence supplied by the caller and repeated at this adapter's boundary.
    """
    if external_buffers_disjoint is not True or synchronous_raw_completion is not True:
        raise ValueError('device columns require private raw storage and synchronous raw producer')
    if any(type(x) is not int or x <= 0 for x in (rows, columns)):
        raise ValueError('device columns require static positive original extents')
    if rows*columns*4 >= 1 << 40 or columns*4 >= 1 << 32:
        raise ValueError('device columns exceed scratch or DMA stride bounds')
    attrs = dict(attrs or {})
    if set(attrs.get('epilogue', ())) - {'bias_add', 'acc_scale', 'relu'}:
        raise ValueError('device columns require only original bias/scale/relu epilogue')
    if any(k in attrs for k in ('requant_shift', 'shift', 'pool', 'pool_size', 'maxpool')):
        raise ValueError('device columns cannot replace another source quantizer or pool')
    readout = DenseTiledIntegerReadout.from_source_scales(source_scales(attrs),
        columns=columns, tile_columns=1, relu=_relu(attrs))
    evidence = selected_device_evidence()
    tile = evidence['resources']['edge']
    chunks = (rows+tile-1)//tile
    row = Var('chunk')*tile
    height = select('chunk', chunks-1, (rows-1)%tile+1, tile)
    args = (('row', row), ('column', Var('column')), ('rows', height),
            ('columns', Const(1)), ('local_row', Const(0)))
    body = (Call('reload_i32', args), Call('complete', ()),
            Call('readout_i8', args), Call('complete', ()))
    kernel = Kernel('original_column_device_readout', (
        TensorArg('raw', (rows,columns), 'i32', 'read'),
        TensorArg('output', (rows,columns), 'i8', 'write')),
        (Loop('column', columns, (Call('configure_column', (('column', Var('column')),)),
             Call('complete', ()), Loop('chunk', chunks, body))),))
    from merlin.sched.ir import check_structure
    errors = check_structure(kernel)
    if errors:
        raise ValueError('invalid original-column device program: '+repr(errors))
    return DeviceColumnPlan(rows, columns, tile, readout, kernel, evidence)


def lower_device_column_effect(fn, raw, dst, plan, scales, control, kind, fields):
    """Terminal encoding of actual schedule operands; also used by local replay."""
    if kind == 'configure_column':
        from gemmini_column_constants import source_configuration_word
        from gemmini_readout import _store_config
        bits = source_configuration_word(control, fields['column'], plan.readout.column_scale_bits)
        _store_config(fn, plan.columns, bits, plan.readout.relu)
        return
    offset = control.op('add', control.op('mul', fields['row'], plan.columns), fields['column'])
    local = control.op('add', E.ACC, fields['local_row'])
    if kind == 'reload_i32':
        pointer = E.pointer_offset(fn, raw, control.op('mul', offset, 4))
        E.mvin(fn, pointer, local, fields['rows'], fields['columns'], stream=0)
    elif kind == 'readout_i8':
        E.mvout(fn, E.pointer_offset(fn, dst, offset), local, fields['rows'], fields['columns'])
    else:
        raise ValueError('unknown original-column device effect '+kind)


def emit_device_columns(fn, raw, dst, plan):
    """Consume every actual counted effect, with drains before row reuse."""
    # Recheck selected identities before issuing any source configuration.
    if selected_device_evidence() != plan.evidence:
        raise ValueError('device column evidence changed after admission')
    scales = None
    control = FnBuilderControlAdapter(fn)
    E.fence(fn)
    E.config_load(fn, plan.columns*4, stream=0, scale=1.0, shrunk=False)
    E.fence(fn)
    def lower(kind, fields):
        return lower_device_column_effect(fn, raw, dst, plan, scales, control, kind, fields)
    receipt = emit_index_effect_program(plan.kernel, control, lower,
        complete_epoch=lambda: E.fence(fn), completion_instruction='complete')
    E.fence(fn)
    return dict(receipt, route='original_column_device_readout', rows=plan.rows,
        columns=plan.columns, row_tile=plan.row_tile, source_readout_identity=plan.readout.identity,
        raw_bytes=plan.rows*plan.columns*4, host_payload_loads=0, host_payload_stores=0,
        device_reload_pairs=plan.columns*((plan.rows+plan.row_tile-1)//plan.row_tile),
        original_columns=True, full_width_reload=True, source_scale_bits=True,
        selected_evidence=plan.evidence, device_execution_verified=False)


def try_emit_device_columns(fn, input_ptr, weight_ptr, dst_ptr, *, input_shape,
        weight_shape, attrs=None, bias=None, output_dtype='i32', raw_emitter,
        external_buffers_disjoint=False, synchronous_raw_completion=False,
        readout_schedule='serial'):
    """Pre-admit a whole convolution and invoke the caller's exact raw producer.

    Returns None for a source without mixed column readout. The callback receives
    the ordinary convolution signature with only bias/window source attributes
    and output_dtype=i32. Caller must validate its own raw producer before this
    call; this adapter never catches a post-mutation callback failure.
    """
    from gemmini_conv import _geometry
    from gemmini_support import scratch_buffer
    attrs = dict(attrs or {})
    if str(output_dtype) not in ('i8', 'int8'):
        return None
    if not callable(raw_emitter):
        raise ValueError('device columns require a pre-admitted ordinary raw emitter')
    # Keep the incumbent for uniform, already-grouped or nonreducing source
    # intervals. This alternative repairs exactly the host preparation region.
    from gemmini_column_grouping import select_column_grouping
    grouping, _ = select_column_grouping(attrs=attrs, columns=weight_shape[-1],
        output_dtype=output_dtype, max_columns=E.DIM)
    if grouping is None:
        return None
    window, columns = _geometry(input_shape, weight_shape, attrs)
    rows = window.input_shape[0]*prod(window.output_spatial)
    plan = plan_device_columns(rows, columns, attrs,
        external_buffers_disjoint=external_buffers_disjoint,
        synchronous_raw_completion=synchronous_raw_completion)
    if len(set(plan.readout.column_scale_bits)) <= 1:
        return None
    if 'bias_add' in attrs.get('epilogue', ()) and bias is None:
        raise ValueError('device columns require the original declared bias pointer')
    if readout_schedule == 'classes8':
        from gemmini_column_device_classes import plan_device_columns_classes, emit_device_columns_classes
        class_plan = plan_device_columns_classes(plan, window_tiles=8)
        if class_plan is not None:
            emit_readout = lambda: emit_device_columns_classes(fn, raw, dst_ptr, plan,
                window_tiles=8, class_plan=class_plan)
        else:
            from gemmini_column_device_blocked import plan_device_columns_blocked, emit_device_columns_blocked
            blocked_plan = plan_device_columns_blocked(plan, window_tiles=8)
            emit_readout = lambda: emit_device_columns_blocked(fn, raw, dst_ptr, plan,
                window_tiles=8, blocked_plan=blocked_plan)
    elif readout_schedule in ('blocked8', 'blocked16'):
        from gemmini_column_device_blocked import plan_device_columns_blocked, emit_device_columns_blocked
        window_tiles = 8 if readout_schedule == 'blocked8' else 16
        blocked_plan = plan_device_columns_blocked(plan, window_tiles=window_tiles)
        emit_readout = lambda: emit_device_columns_blocked(fn, raw, dst_ptr, plan,
            window_tiles=window_tiles, blocked_plan=blocked_plan)
    elif readout_schedule == 'epochs64':
        from gemmini_column_device_epochs import plan_device_columns_epochs, emit_device_columns_epochs
        epoch_plan = plan_device_columns_epochs(plan, epoch_tiles=64)
        emit_readout = lambda: emit_device_columns_epochs(fn, raw, dst_ptr, plan,
            epoch_tiles=64, epoch_plan=epoch_plan)
    elif readout_schedule == 'serial':
        emit_readout = lambda: emit_device_columns(fn, raw, dst_ptr, plan)
    else:
        raise ValueError('unknown original-column readout schedule')
    raw_attrs = {k:v for k,v in attrs.items() if k not in (
        'acc_scale_by_column','acc_scale','output_scale','scale','relu','activation','epilogue')}
    raw_attrs['epilogue'] = ['bias_add'] if bias is not None else []
    raw = scratch_buffer(fn, rows*columns*4, 'column_device_raw_i32')
    E.fence(fn)
    product = raw_emitter(fn,input_ptr,weight_ptr,raw,input_shape=input_shape,
        weight_shape=weight_shape,attrs=raw_attrs,bias=bias,output_dtype='i32')
    E.fence(fn)
    readout = emit_readout()
    return dict(product,column_device=readout)
