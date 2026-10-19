"""Runtime-bounded Gemmini weight-stationary contraction lowering.

The RoCC register fields follow the selected RTL-fact bundle layouts.  CPU
loops only prepare layouts and scatter readouts; products and epilogues use
Gemmini's integer accumulator and scaling datapaths.
"""
import struct


DIM = 16
ACC = 0x80000000
ACC_ADD = 0xC0000000
ACC_FULL = 0xA0000000
GARBAGE = 0xFFFFFFFF
ONE = 0x3F800000


def float_bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def _tiles(b, total, callback):
    full = (total // DIM) * DIM
    if full:
        b.loop(0, full, DIM, lambda iv: callback(iv, DIM))
    if full != total:
        callback(full, total - full)


def _matrix(b, address, rows, cols):
    if isinstance(address, int) and isinstance(rows, int) and isinstance(cols, int):
        return address | (cols << 32) | (rows << 48)
    col_bits = b.op('shl', cols, 32)
    row_bits = b.op('shl', rows, 48)
    return b.op('or', b.op('or', address, col_bits), row_bits)


def _address(b, pointer):
    return b.cast('ptrtoint', pointer, 'ptr', 'i64')


def _mvin_config(b, state, dram_stride):
    b.rocc(0, (ONE << 32) | (DIM << 16) | (1 << 8) | (state << 3) | 1,
           dram_stride)


def _mvout_config(b, stride, scale, relu):
    bits = float_bits(scale)
    b.rocc(0, 2 | (4 if relu else 0), (bits << 32) | stride)



def pack_resident_rhs(b, w, k, n):
    """Load an admitted complete RHS once into disjoint SRAM tile rows.

    Return None when the complete padded layout exceeds the selected 256 KiB
    SRAM. The caller retains one such state until eviction; ordinary matmul
    staging uses rows zero and sixteen and cannot overwrite this region.
    """
    kt = (k + DIM - 1) // DIM
    nt = (n + DIM - 1) // DIM
    base = 256
    if k <= 0 or n <= 0 or base + kt * nt * DIM > 16384:
        return None
    state = {'k': k, 'n': n, 'base': base, 'k_tiles': kt,
             'n_tiles': nt, 'source': w}
    _mvin_config(b, 1, n)
    b.fence()
    for nj in range(0, n, DIM):
        nr = min(DIM, n - nj)
        for kk in range(0, k, DIM):
            kr = min(DIM, k - kk)
            local = base + ((nj // DIM) * kt + kk // DIM) * DIM
            src = b.gep(w, kk * n + nj)
            b.rocc(1, _address(b, src), _matrix(b, local, kr, nr))
    b.fence()
    return state


def resident_matmul(b, a, state, out, m, attrs, tensors, ptrs):
    """Contract with a complete RHS retained by pack_resident_rhs."""
    if state is None:
        raise ValueError('resident_matmul requires an admitted resident state')
    return matmul(b, a, state['source'], out, m, state['k'], state['n'],
                  attrs, tensors, ptrs, _resident=state)

def matmul(b, a, w, out, m, k, n, attrs, tensors, ptrs,
           transpose_rhs=False, _resident=None):
    """Emit (m,k) @ (k,n), with bounded tile loops and hardware readout.

    With transpose_rhs, w stores (n,k). Bias is a named i32 vector. The
    output pointer is contiguous row-major with the dtype requested by attrs.
    """
    epilogue = attrs.get('epilogue', [])
    if 'maxpool' in epilogue:
        raise ValueError('maxpool requires a pooling-aware output lowering')
    dtype = attrs.get('output_dtype', 'i32')
    if dtype not in ('i8', 'i32'):
        raise ValueError('Gemmini contraction output must be i8 or i32')
    relu = 'relu' in epilogue
    bias_name = attrs.get('bias')
    bias_ptr = ptrs[bias_name] if bias_name else None
    zero = b.alloc(DIM, 'i32') if bias_ptr is None and _resident is None else None
    packed_rhs = b.alloc(DIM * DIM, 'i8') if transpose_rhs else None
    scales = attrs.get('acc_scale_by_column')
    scale_table = None
    readout = None
    scale_values = []
    native_segments = None
    if scales:
        if len(scales) != n:
            raise ValueError('acc_scale_by_column length must match contraction width')
        scale_values = list(dict.fromkeys(float(x) for x in scales))
        if len(scale_values) == 1:
            attrs = dict(attrs, acc_scale=scale_values[0])
            scales = None
        elif _resident is None and dtype == 'i8':
            # Source scales define contiguous column groups. Each group is a
            # native contraction whose DRAM row strides retain the full N.
            # This avoids reading scaled output payloads through the CPU.
            proposed_segments = []
            begin = 0
            while begin < n:
                scale = float(scales[begin])
                end = begin + 1
                while end < n and float_bits(scales[end]) == float_bits(scale):
                    end += 1
                for segment in range(begin, end, DIM):
                    proposed_segments.append((segment, min(DIM, end - segment), scale))
                begin = end
            tile_count = (n + DIM - 1) // DIM
            if len(proposed_segments) <= 64 and len(proposed_segments) <= 2 * tile_count:
                native_segments = proposed_segments
        if scales and native_segments is None:
            scale_table = '@scales_' + b.name('g')
            initializers = ', '.join('i32 ' + str(float_bits(x)) for x in scales)
            b.globals.append(f'{scale_table} = private constant [{n} x i32] [{initializers}], align 64')
            readout = b.alloc(DIM * DIM, 'i8')
    # a_stride/c_stride are local SRAM row strides, independent of DMA strides.
    b.rocc(0, 4 | (1 << 16) | (ONE << 32), 1 << 48)
    _mvin_config(b, 0, k)
    if _resident is None:
        _mvin_config(b, 1, DIM if transpose_rhs else n)
    _mvin_config(b, 2, 0)

    def rows(mi, mr):
        def columns(nj, nr, column_scale=None):
            seed = b.gep(bias_ptr, nj, 'i32') if bias_ptr else zero
            b.fence()
            if _resident is None or bias_ptr is not None:
                b.rocc(14, _address(b, seed), _matrix(b, ACC, mr, nr))

            def reduction(kk, kr):
                aidx = b.op('add', b.op('mul', mi, k), kk)
                ap = b.gep(a, aidx)
                if _resident is not None:
                    rhs_local = _resident['base'] + ((nj // DIM) * _resident['k_tiles'] + kk // DIM) * DIM
                elif transpose_rhs:
                    def pack_row(ri):
                        def pack_col(ci):
                            source_row = b.op('add', nj, ci)
                            source_col = b.op('add', kk, ri)
                            source_idx = b.op('add', b.op('mul', source_row, k), source_col)
                            val = b.load(b.gep(w, source_idx), 'i8')
                            destination = b.op('add', b.op('mul', ri, DIM), ci)
                            b.store(val, b.gep(packed_rhs, destination), 'i8')
                        b.loop(0, nr, 1, pack_col)
                    b.loop(0, kr, 1, pack_row)
                    wp = packed_rhs
                    b.fence()
                else:
                    widx = b.op('add', b.op('mul', kk, n), nj)
                    wp = b.gep(w, widx)
                b.rocc(2, _address(b, ap), _matrix(b, 0, mr, kr))
                if _resident is None:
                    b.rocc(1, _address(b, wp), _matrix(b, DIM, kr, nr))
                    rhs_local = DIM
                destination = ACC if _resident is not None and bias_ptr is None and kk == 0 else ACC_ADD
                b.rocc(6, _matrix(b, rhs_local, kr, nr), _matrix(b, destination, mr, nr))
                b.rocc(4, _matrix(b, 0, mr, kr), _matrix(b, GARBAGE, mr, nr))
                # Also makes reusing a packed RHS staging allocation safe.
                b.fence()
            if _resident is None:
                _tiles(b, k, reduction)
            else:
                for kk in range(0, k, DIM):
                    reduction(kk, min(DIM, k - kk))
            output_idx = b.op('add', b.op('mul', mi, n), nj)
            if dtype == 'i32':
                _mvout_config(b, n * 4, 1.0, False)
                dst = b.gep(out, output_idx, 'i32')
                b.rocc(3, _address(b, dst), _matrix(b, ACC_FULL, mr, nr))
                b.fence()
            elif native_segments is not None:
                _mvout_config(b, n, column_scale, relu)
                dst = b.gep(out, output_idx)
                b.rocc(3, _address(b, dst), _matrix(b, ACC, mr, nr))
                b.fence()
            elif not scales:
                _mvout_config(b, n, attrs.get('acc_scale', 1.0), relu)
                dst = b.gep(out, output_idx)
                b.rocc(3, _address(b, dst), _matrix(b, ACC, mr, nr))
                b.fence()
            else:
                # Keep each scale command literal so instruction audits can
                # prove the readout mode while data layout loops stay bounded.
                for scale in scale_values:
                    _mvout_config(b, DIM, scale, relu)
                    b.rocc(3, _address(b, readout), _matrix(b, ACC, mr, nr))
                    b.fence()
                    def scatter_col(ci):
                        global_col = b.op('add', nj, ci)
                        expected = b.load(b.gep(scale_table, global_col, 'i32'), 'i32')
                        matches = b.cmp('eq', expected, float_bits(scale), 'i32')
                        def copy_rows():
                            def scatter_row(ri):
                                value_idx = b.op('add', b.op('mul', ri, DIM), ci)
                                val = b.load(b.gep(readout, value_idx), 'i8')
                                dest_idx = b.op('add', b.op('mul', b.op('add', mi, ri), n), global_col)
                                b.store(val, b.gep(out, dest_idx), 'i8')
                            b.loop(0, mr, 1, scatter_row)
                        b.branch_if(matches, copy_rows)
                    b.loop(0, nr, 1, scatter_col)
        if native_segments is not None:
            for nj, nr, scale in native_segments:
                columns(nj, nr, scale)
        elif _resident is None:
            _tiles(b, n, columns)
        else:
            for nj in range(0, n, DIM):
                columns(nj, min(DIM, n - nj))
    if _resident is not None and ((m + DIM - 1) // DIM) * _resident['k_tiles'] * _resident['n_tiles'] <= 64:
        for mi in range(0, m, DIM):
            rows(mi, min(DIM, m - mi))
    else:
        _tiles(b, m, rows)
    b.fence()
