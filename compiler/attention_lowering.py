"""DMA/mesh attention QK products without a host transpose.

The public register facts expose transpose bits, but do not establish all
legal dataflow/transpose combinations. This lowering needs no such claim:
load one contiguous key row as a K-by-one matrix with a one-byte DMA stride,
then use the same ordinary WS contraction path as non-transposed matmul.
An optional preparation schedule transposes K through DMA into private DRAM
before calling the shared full-column matmul schedule.
Only pointer/index/loop arithmetic executes on the CPU; tensor values never do.
"""
from math import prod
from pathlib import Path

from merlin.compile.tuning_knobs import knob

from isa_encoder import (ACC_ADDR, ACC_ACCUMULATE, ACC_FULL, GARBAGE_ADDR,
                         config_ex, config_load, config_store, float_bits)
from matmul_lowering import extent, index2, packed


def _prefix_layout(lhs_shape, rhs_shape, attrs):
    """Return broadcast strides, preserving explicitly declared grouped heads."""
    left, right = tuple(lhs_shape[:-2]), tuple(rhs_shape[:-2])
    rank = max(len(left), len(right))
    left = (1,) * (rank-len(left)) + left
    right = (1,) * (rank-len(right)) + right
    output = []
    grouped = None
    for axis, (q, k) in enumerate(zip(left, right)):
        if q == k or q == 1 or k == 1:
            output.append(max(q, k))
        elif (axis == rank-1 and attrs.get('grouped_query_heads') == q
              and attrs.get('kv_heads') == k and q % k == 0):
            output.append(q)
            grouped = (axis, q // k)
        else:
            raise ValueError('attention batch/head dimensions need equal or broadcast extents')
    strides = lambda shape: [prod(shape[i+1:]) for i in range(rank)]
    return tuple(output), left, right, strides(left), strides(right), grouped


def _emit_prepared_qk(e, lhs, rhs, output, M, D, N, geometry, attrs,
                      compute_provenance):
    """Transpose unique K matrices with DMA, then contract full B columns.

    Scratch rows 0..15 are transient; semantic resident storage begins at 32.
    Each contiguous K[n,d:d+kk] becomes kk rows of one scratch column. The
    ordinary DRAM mvout uses destination row pitch N to write B[d:d+kk,n].
    STORE_SPAD (funct23) is not the DRAM destination form used here.
    """
    from matmul_lowering import emit_matmul
    prefix, lp, rp, lstrides, rstrides, grouped = geometry
    batches = prod(rp)
    prepared = e.scratch(batches * D * N, 'i8')
    b, val = e.b, e.value
    # Finish preceding producers before reading K or recycling local rows.
    e.fence()
    e.insn('config', *config_load(1, state_id=1))
    e.insn('config', *config_store(N))
    def prepare_batch(key_batch):
        source = e.ptr(rhs, b.mul_i(key_batch, val(N * D)))
        target = e.ptr(prepared, b.mul_i(key_batch, val(D * N)))
        def column(n):
            def depth(d):
                kk = extent(e, D, d)
                kp = e.ptr(source, index2(e, n, D, d))
                bp = e.ptr(target, index2(e, d, N, n))
                e.insn('mvin2', e.address(kp), packed(e, 0, kk, 1))
                e.insn('mvout', e.address(bp), packed(e, 0, kk, 1))
                # Complete the scratch readout before the next DMA overwrites
                # these rows, and publish the prepared bytes before matmul.
                e.fence()
            e.for_loop(0, D, 16, depth)
        e.for_loop(0, N, 1, column)
    e.for_loop(0, batches, 1, prepare_batch)

    dtype = attrs.get('output_dtype', 'i32')
    def batch(flat):
        left_index, right_index = val(0), val(0)
        for axis, size in enumerate(prefix):
            coord = b.urem_i(b.udiv_i(flat, val(prod(prefix[axis+1:]))), val(size))
            if lp[axis] != 1:
                left_index = b.add_i(left_index, b.mul_i(coord, val(lstrides[axis])))
            if rp[axis] != 1:
                rcoord = b.udiv_i(coord, val(grouped[1])) if grouped and grouped[0] == axis else coord
                right_index = b.add_i(right_index, b.mul_i(rcoord, val(rstrides[axis])))
        qptr = e.ptr(lhs, b.mul_i(left_index, val(M * D)))
        bptr = e.ptr(prepared, b.mul_i(right_index, val(D * N)))
        outptr = e.ptr(output, b.mul_i(flat, val(M * N)), dtype)
        # The shared contraction restores every load/store/execute setting it
        # uses. Grouped query heads select the same already prepared K matrix.
        emit_matmul(e, qptr, bptr, outptr, M, D, N, attrs=attrs,
                    compute_provenance=compute_provenance)
    if prefix:
        e.for_loop(0, prod(prefix), 1, batch)
    else:
        batch(val(0))
    e.fence()


def emit_attention_qk(e, lhs, rhs, output, lhs_shape, rhs_shape, attrs=None, *,
                      output_shape=None, compute_provenance=None):
    """Emit Q[...,M,D] @ K[...,N,D]^T directly from logical i8 buffers.

    The result keeps exact i32 accumulation until an explicitly requested i8
    accumulator scale/readout. Leading axes broadcast by source shape. Grouped
    heads require attrs grouped_query_heads and kv_heads matching the final
    leading dimensions; each query head then selects floor(qhead/group_size).
    This does not fold independent heads into the reduction dimension.
    """
    attrs = attrs or {}
    compute_provenance = compute_provenance or {}
    ls, rs = tuple(map(int, lhs_shape)), tuple(map(int, rhs_shape))
    if min(len(ls), len(rs)) < 2 or min(*ls, *rs) <= 0:
        raise ValueError('attention QK requires positive static matrices')
    M, D = ls[-2:]
    N, kd = rs[-2:]
    if D != kd:
        raise ValueError('attention QK reduction dimensions differ')
    prefix, lp, rp, lstrides, rstrides, grouped = _prefix_layout(ls, rs, attrs)
    expected = prefix + (M, N)
    if output_shape is not None and tuple(output_shape) != expected:
        raise ValueError('attention QK output axes differ from source geometry')
    dtype = attrs.get('output_dtype', 'i32')
    stages = set(attrs.get('epilogue', []))
    if dtype not in ('i32', 'i8') or stages - {'acc_scale', 'relu'}:
        raise ValueError('unsupported attention QK output epilogue')
    if dtype == 'i32' and stages:
        raise ValueError('full-width QK readout cannot apply a narrowing epilogue')
    scale = attrs.get('acc_scale', 1.0)
    scales = attrs.get('acc_scale_by_column')
    if not isinstance(scale, (int, float)) or scale < 0 and 'relu' in stages:
        raise ValueError('unsupported QK scalar readout ordering')
    if scales is not None:
        if dtype != 'i8' or 'acc_scale' not in stages or len(scales) != N:
            raise ValueError('QK column scales need an i8 scale epilogue and N entries')
        if 'relu' in stages and any(x < 0 for x in scales):
            raise ValueError('negative gain and post-scale relu cannot fuse')
    preparation = knob('qk_preparation', default='direct',
                       values=('direct', 'device_transpose'), root=Path(__file__).parent)
    # Bound aggregate private storage as well as each matrix. Large source
    # layouts retain the direct DMA/mesh implementation without a host copy.
    if preparation == 'device_transpose' and prod(rp) * D * N <= 64 * 1024 * 1024:
        return _emit_prepared_qk(e, lhs, rhs, output, M, D, N,
                                (prefix, lp, rp, lstrides, rstrides, grouped),
                                attrs, compute_provenance)
    if scales is not None:
        table = e.constant_array([float_bits(x) for x in scales], 'i32')
    b = e.b
    val = e.value
    e.insn('config', *config_ex(dataflow=1))
    e.insn('config', *config_load(D, state_id=0))
    # Source K[n,d] is contiguous in d. Read it as D rows, one column.
    e.insn('config', *config_load(1, state_id=1))
    e.insn('config', *config_load(0, state_id=2))
    e.insn('config', *config_store(N * (4 if dtype == 'i32' else 1),
                                  scale=scale, activation=int('relu' in stages)))

    def batch(flat):
        left_index, right_index = val(0), val(0)
        for axis, size in enumerate(prefix):
            coord = b.urem_i(b.udiv_i(flat, val(prod(prefix[axis+1:]))), val(size))
            if lp[axis] != 1:
                left_index = b.add_i(left_index, b.mul_i(coord, val(lstrides[axis])))
            if rp[axis] != 1:
                rcoord = b.udiv_i(coord, val(grouped[1])) if grouped and grouped[0] == axis else coord
                right_index = b.add_i(right_index, b.mul_i(rcoord, val(rstrides[axis])))
        qptr = e.ptr(lhs, b.mul_i(left_index, val(M*D)))
        kptr = e.ptr(rhs, b.mul_i(right_index, val(N*D)))
        outptr = e.ptr(output, b.mul_i(flat, val(M*N)), dtype)
        def rows(m):
            mm = extent(e, M, m)
            def column(n):
                if scales is not None:
                    bits = b.and_i(b.load_i64(table, n, 'i32'), val(0xFFFFFFFF))
                    rs2 = b.or_i(val(N), b.shl_i(bits, 32))
                    e.insn('config', 2 | int('relu' in stages) << 2, rs2)
                e.insn('mvin3', 0, packed(e, ACC_ADDR, mm, 1))
                def reduction(d):
                    kk = extent(e, D, d)
                    # Fully initialize physical padding before a ragged load.
                    if D % 16:
                        e.insn('mvin', 0, packed(e, 0, 16, 16))
                    ap = e.ptr(qptr, index2(e, m, D, d))
                    bp = e.ptr(kptr, index2(e, n, D, d))
                    e.insn('mvin', e.address(ap), packed(e, 0, mm, kk))
                    e.insn('mvin2', e.address(bp), packed(e, 16, kk, 1))
                    e.insn('preload', packed(e, 16, kk, 1),
                           packed(e, ACC_ADDR | ACC_ACCUMULATE, mm, 1), **compute_provenance)
                    e.insn('compute', packed(e, 0, mm, kk),
                           packed(e, GARBAGE_ADDR, 16, 16), **compute_provenance)
                e.for_loop(0, D, 16, reduction)
                addr = ACC_ADDR | (ACC_FULL if dtype == 'i32' else 0)
                cp = e.ptr(outptr, index2(e, m, N, n), dtype)
                e.insn('mvout', e.address(cp), packed(e, addr, mm, 1))
            e.for_loop(0, N, 1, column)
        e.for_loop(0, M, 16, rows)
    if prefix:
        e.for_loop(0, prod(prefix), 1, batch)
    else:
        batch(val(0))
    e.fence()
