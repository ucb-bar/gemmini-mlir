"""Gemmini DMA and accumulator lowering for tensor maps.

All arithmetic on tensor values happens in the accelerator. Scalar LLVM
arithmetic below computes tile extents, addresses and command register words.
"""

import math
import struct


_ACC = 1 << 31
_ACC_ADD = _ACC | (1 << 30)
_ACC_FULL = _ACC | (1 << 29)


def _fp32(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def _rounded_fp32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def _load_config(b, stride, scale=1.0, state=0, dtype='i8'):
    # Accumulator DMA uses shrink to identify an input-width source matrix.
    word = (_fp32(scale) << 32) | (16 << 16) | (1 << 8) | (state << 3) | 1
    if dtype == 'i8':
        word |= 1 << 2
    b.rocc(0, word, stride)


def _store_config(b, stride, scale=1.0, relu=False):
    b.rocc(0, 2 | (4 if relu else 0), (_fp32(scale) << 32) | stride)


def _matrix_word(b, address, rows, cols):
    return address | (cols << 32) | (rows << 48)


def _tile_loops(b, shape, callback):
    """Visit runtime tiles with literal full-block and remainder dimensions."""
    n = int(shape[-1])
    m = math.prod(int(x) for x in shape[:-1])
    full_rows = (m // 16) * 16
    full_cols = (n // 16) * 16

    def row_tile(row, rows):
        def col_tile(col):
            index = b.op('add', b.op('mul', row, n), col)
            callback(index, col, rows, 16)

        if full_cols:
            b.loop(0, full_cols, 16, col_tile)
        if full_cols != n:
            index = b.op('add', b.op('mul', row, n), full_cols)
            callback(index, full_cols, rows, n - full_cols)

    if full_rows:
        b.loop(0, full_rows, 16, lambda row: row_tile(row, 16))
    if full_rows != m:
        row_tile(full_rows, m - full_rows)


def _address(b, ptr, index, dtype):
    return b.cast('ptrtoint', b.gep(ptr, index, dtype), 'ptr', 'i64')


def movement(b, src, out, shape, src_dtype, out_dtype, attrs, tensors, ptrs):
    """Transfer a matrix through the accumulator, widening i8 when requested."""
    n = int(shape[-1])
    _load_config(b, n * (4 if src_dtype == 'i32' else 1), dtype=src_dtype)
    _store_config(b, n * (4 if out_dtype == 'i32' else 1))

    def tile(index, col, rows, cols):
        b.rocc(2, _address(b, src, index, src_dtype), _matrix_word(b, _ACC, rows, cols))
        read_address = _ACC_FULL if out_dtype == 'i32' else _ACC
        b.rocc(3, _address(b, out, index, out_dtype), _matrix_word(b, read_address, rows, cols))

    _tile_loops(b, shape, tile)
    b.fence()


def residual_add(b, lhs, rhs, out, shape, attrs, tensors, ptrs):
    """Form a high-precision weighted sum using i8 systolic products.

    The two coefficients are represented with ten fractional bits and split
    into signed i8 chunks. Every product accumulates in i32, so input DMA
    does not round or saturate individual scaled operands.
    """
    out_dtype = attrs.get('output_dtype', 'i8')
    lhs_scale = _rounded_fp32(attrs.get('lhs_scale', 1.0))
    rhs_scale = _rounded_fp32(attrs.get('rhs_scale', 1.0))
    relu = 'relu' in attrs.get('epilogue', [])
    if out_dtype == 'i32':
        if lhs_scale != 1.0 or rhs_scale != 1.0 or attrs.get('acc_scale', 1.0) != 1.0 or relu:
            raise ValueError('i32 residual readout requires unit gains and no activation')
        n = int(shape[-1])
        _load_config(b, n, 1.0, state=0)
        _load_config(b, n, 1.0, state=1)
        _store_config(b, n * 4)

        def direct_tile(index, col, rows, cols):
            b.rocc(2, _address(b, lhs, index, 'i8'), _matrix_word(b, _ACC, rows, cols))
            b.rocc(1, _address(b, rhs, index, 'i8'), _matrix_word(b, _ACC_ADD, rows, cols))
            b.rocc(3, _address(b, out, index, 'i32'), _matrix_word(b, _ACC_FULL, rows, cols))

        _tile_loops(b, shape, direct_tile)
        b.fence()
        return
    if out_dtype != 'i8':
        raise ValueError('residual_add supports i8 or unit-gain i32 outputs')
    if attrs.get('bound_lsb', 1) == 0 and (lhs_scale != 1.0 or rhs_scale != 1.0):
        raise ValueError('fractional residual scales require a nonzero error bound')

    denominator = 1024
    zero = b.alloc(1, 'i32')

    def chunk_table(scale):
        coefficient = round(scale * denominator)
        magnitude = abs(coefficient)
        sign = -1 if coefficient < 0 else 1
        count, tail = divmod(magnitude, 127)
        chunks = [127 * sign] * count
        if tail:
            chunks.append(tail * sign)
        if not chunks:
            chunks = [0]
        symbol = '@residual_coeff_' + b.name('g')
        values = ', '.join('i8 ' + str(x) for x in chunks)
        b.globals.append(f'{symbol} = private constant [{len(chunks)} x i8] [{values}], align 64')
        return symbol, len(chunks)

    lhs_table, lhs_count = chunk_table(lhs_scale)
    rhs_table, rhs_count = chunk_table(rhs_scale)
    one = _fp32(1.0)
    # Weight-stationary execution; local A and C advance one SRAM row.
    b.rocc(0, 4 | (1 << 16) | (one << 32), 1 << 48)
    # Source operands and coefficient tables enter ordinary i8 SRAM rows.
    _load_config(b, 1, state=0, dtype='sram_i8')
    _load_config(b, 0, state=1, dtype='sram_i8')
    _load_config(b, 0, state=2, dtype='i32')
    _store_config(b, 1, float(attrs.get('acc_scale', 1.0)) / denominator, relu)

    def tile(index, col, rows, cols):
        b.rocc(14, _address(b, zero, 0, 'i32'), _matrix_word(b, _ACC, rows, 1))

        def operand(source, table, count):
            if not count:
                return
            b.rocc(2, _address(b, source, index, 'i8'), _matrix_word(b, 0, rows, 1))

            def multiply(chunk):
                b.rocc(1, _address(b, table, chunk, 'i8'), _matrix_word(b, 16, 1, 1))
                b.rocc(6, _matrix_word(b, 16, 1, 1), _matrix_word(b, _ACC_ADD, rows, 1))
                b.rocc(4, _matrix_word(b, 0, rows, 1), _matrix_word(b, 0xffffffff, rows, 1))
                b.fence()

            b.loop(0, count, 1, multiply)

        operand(lhs, lhs_table, lhs_count)
        operand(rhs, rhs_table, rhs_count)
        b.rocc(3, _address(b, out, index, 'i8'), _matrix_word(b, _ACC, rows, 1))
        b.fence()

    _tile_loops(b, [math.prod(int(x) for x in shape), 1], tile)
    b.fence()


def bias_add(b, src, bias, out, shape, attrs, tensors, ptrs):
    """Broadcast an i32 bias row while loading into the accumulator."""
    n = int(shape[-1])
    out_dtype = attrs.get('output_dtype', 'i32')
    _load_config(b, n * 4, state=0, dtype='i32')
    # A zero DRAM row stride broadcasts the same bias row to every local row.
    _load_config(b, 0, state=1, dtype='i32')
    relu = 'relu' in attrs.get('epilogue', [])
    _store_config(b, n * (4 if out_dtype == 'i32' else 1), attrs.get('acc_scale', 1.0), relu)

    def tile(index, col, rows, cols):
        b.rocc(2, _address(b, src, index, 'i32'), _matrix_word(b, _ACC, rows, cols))
        b.rocc(1, _address(b, bias, col, 'i32'), _matrix_word(b, _ACC_ADD, rows, cols))
        read_address = _ACC_FULL if out_dtype == 'i32' else _ACC
        b.rocc(3, _address(b, out, index, out_dtype), _matrix_word(b, read_address, rows, cols))

    _tile_loops(b, shape, tile)
    b.fence()
