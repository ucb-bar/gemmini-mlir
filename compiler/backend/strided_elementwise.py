"""Source-bound rank-two elementwise DMA with distinct physical input pitches.

The compact movement/loaded-sum epoch consumers describe contiguous logical
sources. This adapter retains their incumbent synchronous terminal effects:
source row*pitch+column, output row*columns+column, and a real completion fence
before dependent accumulation/readout and every accumulator-slot reuse.
"""
from dataclasses import dataclass
import math
import struct

from . import isa


@dataclass(frozen=True)
class StridedInputs:
    shape: tuple
    pitches: tuple
    views: tuple


def plan_strided_inputs(ctx, sources, dst, shape, *, attributes=None):
    """Read-only admission; return None for the existing contiguous route."""
    shape = tuple(shape)
    columns = shape[-1]
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    destination = aliases.get(dst)
    if isinstance(destination, dict) and destination.get('access') == 'read':
        raise ValueError('elementwise destination cannot be a read-only source alias')
    row_stride = getattr(ctx, 'row_stride', lambda name, default: default)
    if row_stride(dst, columns) != columns:
        raise ValueError('elementwise destination must be contiguous')
    pitches = tuple(row_stride(name, columns) for name in sources)
    for name, pitch in zip(sources, pitches, strict=True):
        alias = aliases.get(name)
        if (type(pitch) is not int or pitch < columns
            or isinstance(alias, dict) and (type(alias.get('row_stride', columns)) is not int
                or alias.get('row_stride', columns) != pitch)):
            raise ValueError('elementwise source pitch differs from its declared row view')
    if all(pitch == columns for pitch in pitches):
        return None
    if len(shape) != 2:
        raise ValueError('pitched elementwise inputs require rank-two source views')
    attributes = attributes or {}
    if (attributes.get('layout') not in (None, 'row_major', 'contiguous')
        or attributes.get('output_dtype', ctx.spec(dst)['dtype']) != ctx.spec(dst)['dtype']):
        raise ValueError('pitched elementwise attributes disagree with its typed storage layout')
    if dst in aliases or dst in sources:
        raise ValueError('pitched elementwise output requires a distinct contiguous destination')
    if any(ctx.spec(name).get('layout') not in (None, 'row_major', 'contiguous')
           for name in (*sources, dst)):
        raise ValueError('pitched elementwise has no source proof for this layout')
    rows = shape[0]
    views = []
    for name, pitch in zip(sources, pitches, strict=True):
        if name not in ctx.buffers:
            raise ValueError('pitched elementwise source must already be initialized')
        alias = aliases.get(name)
        if alias is None:
            if pitch != columns:
                raise ValueError('physical source pitch needs a registered read-only view')
            root = name
            view = None
        else:
            if (not isinstance(alias, dict) or alias.get('access') != 'read'
                or alias.get('source') in aliases or alias.get('source') == dst
                or tuple(alias.get('shape', ())) != shape
                or alias.get('dtype') != ctx.spec(name)['dtype']):
                raise ValueError('pitched elementwise needs a direct typed read-only source view')
            root = alias['source']
            offset = alias.get('element_offset', 0)
            span = (rows - 1) * pitch + columns
            if (type(offset) is not int or offset < 0
                or ctx.spec(root)['dtype'] != alias['dtype']
                or offset + span > math.prod(ctx.spec(root)['shape'])
                or pitch != columns and alias.get('element_span') != span):
                raise ValueError('pitched elementwise view changed its exact physical source span')
            view = dict(alias)
        if root == dst or dst in ctx.buffers and ctx.buffers[name] is ctx.buffers[dst]:
            raise ValueError('pitched elementwise cannot overwrite an input source')
        views.append(view)
    return StridedInputs(shape, pitches, tuple(views))


def _tiles(ctx, shape, lower):
    rows, columns = shape
    def row_tile(row, height):
        complete = columns // isa.DIM * isa.DIM
        if complete:
            ctx.fb.for_loop(0, complete, isa.DIM,
                lambda col: lower(row, col, height, isa.DIM), retain_loop=True)
        if complete < columns:
            lower(row, ctx.fb.const(complete), height, columns - complete)
    complete = rows // isa.DIM * isa.DIM
    if complete:
        ctx.fb.for_loop(0, complete, isa.DIM,
            lambda row: row_tile(row, isa.DIM), retain_loop=True)
    if complete < rows:
        row_tile(ctx.fb.const(complete), rows - complete)


def _offset(ctx, row, col, pitch):
    return ctx.fb.add_i(ctx.fb.mul_i(row, ctx.fb.const(pitch)), col)


def _finite_f32(scale):
    bits = isa.f32_bits(scale)
    if not math.isfinite(struct.unpack('<f', struct.pack('<I', bits))[0]):
        raise ValueError('pitched elementwise scale must remain finite in selected f32')
    return bits


def _receipt(plan, kind, configs):
    return dict(route=kind, shape=list(plan.shape),
        source_row_strides_elements=list(plan.pitches), source_views=list(plan.views),
        destination_row_stride_elements=plan.shape[1],
        source_element_address='view_base + row * source_row_stride + column',
        destination_element_address='destination_base + row * logical_columns + column',
        configuration_words=[list(command) for command in configs],
        completion='full device drain after each load, accumulating load, and store; private ACC row zero')


def emit_strided_movement(ctx, src, dst, plan, scale, activation):
    """Emit the incumbent typed load/readout effects at checked physical rows."""
    inp, out = ctx.spec(src)['dtype'], ctx.spec(dst)['dtype']
    scale_bits = _finite_f32(scale)
    configs = (isa.config_ld(plan.pitches[0] * (1 if inp == 'i8' else 4), shrink=inp == 'i8'),
        isa.config_st(plan.shape[1] * (1 if out == 'i8' else 4),
                      scale_bits=scale_bits, activation=activation))
    # Every shape, alias, numeric and complete configuration check precedes
    # scratch allocation or device emission.
    if dst not in ctx.buffers:
        ctx.buffer(dst)
    ctx.fence()
    for command in configs:
        ctx.emit(*command)
    def tile(row, col, rows, columns):
        ctx.emit(*isa.mvin(ctx.ptr(src, _offset(ctx, row, col, plan.pitches[0])),
            isa.acc_addr(0), rows, columns))
        ctx.fence()
        ctx.emit(*isa.mvout(ctx.ptr(dst, _offset(ctx, row, col, plan.shape[1])),
            isa.acc_addr(0, full=out == 'i32'), rows, columns))
        ctx.fence()
    _tiles(ctx, plan.shape, tile)
    ctx.fence()
    return dict(_receipt(plan, 'strided_movement', configs), input_dtype=inp,
        output_dtype=out, readout_scale_bits=scale_bits, activation=activation)


def emit_strided_residual(ctx, lhs, rhs, dst, plan, proof, activation, original_contract):
    """Preserve the original exhaustive residual proof and ordered DMA sum."""
    bits = tuple(_finite_f32(proof[key]) for key in
        ('lhs_load_f32', 'rhs_load_f32', 'readout_scale_f32'))
    configs = (isa.config_ld(plan.pitches[0], channel=0, scale_bits=bits[0], shrink=True),
        isa.config_ld(plan.pitches[1], channel=1, scale_bits=bits[1], shrink=True),
        isa.config_st(plan.shape[1], scale_bits=bits[2], activation=activation))
    if dst not in ctx.buffers:
        ctx.buffer(dst)
    ctx.fence()
    for command in configs:
        ctx.emit(*command)
    def tile(row, col, rows, columns):
        for i, name in enumerate((lhs, rhs)):
            ctx.emit(*isa.mvin(ctx.ptr(name, _offset(ctx, row, col, plan.pitches[i])),
                isa.acc_addr(0, accumulate=bool(i)), rows, columns, channel=i))
            ctx.fence()
        ctx.emit(*isa.mvout(ctx.ptr(dst, _offset(ctx, row, col, plan.shape[1])),
            isa.acc_addr(0), rows, columns))
        ctx.fence()
    _tiles(ctx, plan.shape, tile)
    ctx.fence()
    return dict(_receipt(plan, 'strided_residual', configs), numeric_proof=dict(proof),
        original_contract=dict(original_contract), activation=activation)
