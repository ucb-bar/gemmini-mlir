"""Hardware accumulator readout, preserving source binary32 coefficients.

A local address selects an accumulator ROW, never an individual column lane.
Producers must place each equal-scale column interval at lane zero. This module
refuses mixed-scale intervals rather than pretending local+column selects a lane.
All output strides are in BYTES; row/column offsets address the original dst.
"""
from __future__ import annotations
import math
import struct
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i32, i64
import gemmini_encode as E


def _value(fn, x):
    return E._i64(fn, x)


def _add(fn, a, b):
    return fn.add_i(_value(fn, a), _value(fn, b))


def _mul(fn, a, b):
    return fn.mul_i(_value(fn, a), _value(fn, b))


def _relu(attrs):
    activation = attrs.get('activation', 0)
    if activation not in (0, 1, 'none', 'relu', 'ReLU', None):
        raise ValueError('only source-declared ReLU is supported by this readout')
    return bool(attrs.get('relu', False) or activation in (1, 'relu', 'ReLU')
                or 'relu' in attrs.get('epilogue', ()))


def source_scales(attrs):
    """Return the source scalar or original per-column coefficient vector."""
    return attrs.get('acc_scale_by_column', attrs.get('acc_scale',
        attrs.get('output_scale', attrs.get('scale', 1.0))))


def _scale_bits(scale):
    value = float(scale)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('hardware readout requires positive finite source f32 scale')
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    converted = struct.unpack('<f', struct.pack('<I', bits))[0]
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError('source scale is not positive finite binary32')
    return bits


def column_scale_runs(attrs, columns, *, max_columns=16):
    """Return (original column, length, exact f32 scale) producer intervals.

    Each interval must be COMPUTED starting at accumulator lane zero; this is a
    producer tiling request, not a way to split an already packed accumulator.
    """
    if columns <= 0 or max_columns <= 0 or max_columns > E.DIM:
        raise ValueError('invalid source columns or physical readout tile width')
    scales = source_scales(attrs)
    if isinstance(scales, (int, float)):
        bits = (_scale_bits(scales),) * columns
    else:
        if len(scales) != columns:
            raise ValueError('source needs exactly one f32 coefficient per output column')
        bits = tuple(_scale_bits(x) for x in scales)
    runs = []
    first = 0
    while first < columns:
        stop = first + 1
        while stop < min(first + max_columns, columns) and bits[stop] == bits[first]:
            stop += 1
        runs.append((first, stop-first, struct.unpack('<f', struct.pack('<I', bits[first]))[0]))
        first = stop
    return tuple(runs)


def _store_config(fn, stride, bits, relu):
    # The scale is already binary32 bits. Runtime loads carry source parameters,
    # never accumulator values or CPU tensor element arithmetic.
    rs2 = E._pack(fn, (stride, 0, 32), (bits, 32, 32))
    return E.insn(fn, 0, 2 | (int(relu) << 2), rs2)


def emit_readout(fn, dst, local, rows, cols, *, output_stride, attrs,
                 output_dtype='i32', scale_ptr=None, column_offset=0, row_offset=0):
    """Issue one readout from lane zero; caller owns completion and row reuse.

    `local` is an accumulator row index or flagged accumulator address. `dst` is
    the whole tensor base. `output_stride` is bytes per output row. Offsets are
    source row/column indices. Dynamic rows/cols are supported for scalar scale;
    source per-column vectors require a static equal-scale interval. `scale_ptr`
    points to an original f32 coefficient vector; its selected coefficient must
    apply to the entire produced tile (use cols=1 for arbitrary runtime vectors).
    """
    attrs = dict(attrs or {})
    if output_dtype not in ('i32', 'int32', 'i8', 'int8'):
        raise ValueError('hardware readout supports signed i32 and i8 outputs')
    if attrs.get('requant_shift', attrs.get('shift', 0)):
        raise ValueError('source half-up requant shift is not hardware half-even scaling')
    narrow = output_dtype in ('i8', 'int8')
    relu = _relu(attrs)
    scales = source_scales(attrs)
    if not narrow:
        if scale_ptr is not None or relu or (isinstance(scales, (list, tuple))
                or float(scales) != 1.0):
            raise ValueError('raw i32 readout cannot apply an epilogue')
        E.config_store(fn, output_stride)
    elif scale_ptr is not None:
        if cols != 1:
            raise ValueError('runtime per-column scales need one produced lane-zero column')
        ptr = E.pointer_offset(fn, scale_ptr, _mul(fn, column_offset, 4))
        raw = fn.add(llvm.LoadOp(ptr, i32)).results[0]
        bits = fn.add(llvm.ZExtOp(raw, i64)).results[0]
        _store_config(fn, output_stride, bits, relu)
    else:
        if isinstance(scales, (list, tuple)):
            if not isinstance(column_offset, int) or not isinstance(cols, int):
                raise ValueError('source vector scales require a static producer column interval')
            selected = scales[column_offset:column_offset+cols]
            if len(selected) != cols or cols <= 0:
                raise ValueError('readout interval exceeds the source scale vector')
            bits_set = {_scale_bits(x) for x in selected}
            if len(bits_set) != 1:
                raise ValueError('mixed source scales require separate lane-zero producer tiles')
            bits = bits_set.pop()
        else:
            bits = _scale_bits(scales)
        _store_config(fn, output_stride, bits, relu)
    offset = _add(fn, _mul(fn, row_offset, output_stride),
                  _mul(fn, column_offset, 1 if narrow else 4))
    ptr = E.pointer_offset(fn, dst, offset)
    # Clear any previous ADD/FULL selector before selecting this output format.
    row = fn.and_i(_value(fn, local), fn.const((1 << 29)-1))
    address = fn.or_i(row, fn.const(E.ACC if narrow else E.ACC_FULL))
    E.mvout(fn, ptr, address, rows, cols)


def config_pool_store(fn, stride_bytes, *, pool_stride, pool_size, pool_out_dim,
                      porows, pocols, orows, ocols, upad=0, lpad=0,
                      scale=1.0, activation=0):
    """Encode selected max-pool readout fields from ConfigMvoutRs1/Rs2.

    The caller owns a qualified spatial producer layout, padding contract and
    mvout address/dimensions. Padding values and row-group layout cannot be
    inferred from a packed dense tile.
    """
    if pool_stride not in (1, 2, 3) or pool_size not in (1, 2, 3):
        raise ValueError('pool stride and size must fit the nonzero two-bit fields')
    if activation not in (0, 1):
        raise ValueError('pool readout supports none or ReLU activation')
    rs1 = E._pack(fn, (2,0,2), (activation,2,2), (pool_stride,4,2),
        (pool_size,6,2), (upad,8,2), (lpad,10,2), (pool_out_dim,24,8),
        (porows,32,8), (pocols,40,8), (orows,48,8), (ocols,56,8))
    rs2 = E._pack(fn, (stride_bytes,0,32), (_scale_bits(scale),32,32))
    return E.insn(fn, 0, rs1, rs2)
