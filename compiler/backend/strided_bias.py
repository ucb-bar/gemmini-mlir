"""Exact modular i32 column bias addition over a checked pitched matrix view."""
import math

from . import isa
from .strided_elementwise import _tiles, _offset


def _bias_view(ctx, bias, dst, columns):
    """Admit an initialized contiguous vector or a direct readonly offset view."""
    if bias not in ctx.buffers or bias == dst:
        raise ValueError('pitched bias needs an initialized, separate column vector')
    if ctx.spec(bias).get('layout') not in (None, 'row_major', 'contiguous'):
        raise ValueError('pitched bias column vector must be contiguous')
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    alias = aliases.get(bias)
    if alias is not None:
        if (not isinstance(alias, dict) or alias.get('access') != 'read'
            or alias.get('source') in aliases or alias.get('source') == dst
            or tuple(alias.get('shape', ())) != (columns,) or alias.get('dtype') != 'i32'
            or 'row_stride' in alias):
            raise ValueError('pitched bias needs a typed readonly contiguous column view')
        offset = alias.get('element_offset', 0)
        if (type(offset) is not int or offset < 0
            or type(alias.get('element_span', columns)) is not int
            or alias.get('element_span', columns) != columns
            or ctx.spec(alias['source'])['dtype'] != 'i32'
            or offset + columns > math.prod(ctx.spec(alias['source'])['shape'])):
            raise ValueError('bias column view differs from its exact physical source span')
    if dst in ctx.buffers and ctx.buffers[bias] is ctx.buffers[dst]:
        raise ValueError('pitched bias cannot overwrite its column source')
    return dict(alias) if alias is not None else None


def emit_strided_bias(ctx, src, bias, dst, plan):
    """Preserve full-width identity loads, i32 wrap accumulation and raw store.

    Matrix and vector view checks plus all complete configuration encodings
    precede private output allocation and every terminal device instruction.
    """
    columns = plan.shape[1]
    bias_view = _bias_view(ctx, bias, dst, columns)
    configs = (isa.config_ld(plan.pitches[0] * 4, channel=0, shrink=False),
        isa.config_ld(0, channel=1, shrink=False), isa.config_st(columns * 4))
    if dst not in ctx.buffers:
        ctx.buffer(dst)
    ctx.fence()
    for command in configs:
        ctx.emit(*command)
    def tile(row, col, rows, cols):
        ctx.emit(*isa.mvin(ctx.ptr(src, _offset(ctx, row, col, plan.pitches[0])),
            isa.acc_addr(0), rows, cols, channel=0))
        ctx.fence()
        ctx.emit(*isa.mvin(ctx.ptr(bias, col), isa.acc_addr(0, accumulate=True),
            rows, cols, channel=1))
        ctx.fence()
        ctx.emit(*isa.mvout(ctx.ptr(dst, _offset(ctx, row, col, columns)),
            isa.acc_addr(0, full=True), rows, cols))
        ctx.fence()
    _tiles(ctx, plan.shape, tile)
    ctx.fence()
    return dict(route='strided_i32_bias', shape=list(plan.shape),
        matrix_source_row_stride_elements=plan.pitches[0], matrix_source_view=plan.views[0],
        bias_source_view=bias_view, bias_source_row_stride_bytes=0,
        destination_row_stride_elements=columns,
        scalar_contract='output[row,column] = signed_i32_wrap(source[row,column] + bias[column])',
        matrix_element_address='view_base + row * matrix_source_row_stride + column',
        bias_element_address='bias_view_base + column; broadcast source row stride zero',
        destination_element_address='destination_base + row * logical_columns + column',
        input_dtype='i32', bias_dtype='i32', output_dtype='i32',
        configuration_words=[list(command) for command in configs],
        completion='full device drain after initialization, accumulation and raw store; private ACC row zero')
