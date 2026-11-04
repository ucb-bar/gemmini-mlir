"""Proved static tensor movement through ordinary Gemmini DMA commands.

Only addresses and loop controls execute on the scalar core. Payloads retain their
integer bit patterns in scratchpad, or in full-width accumulator transfers.
"""
from dataclasses import dataclass
from math import prod
from pathlib import Path

from merlin.compile.tuning_knobs import knob

from xdsl.dialects.builtin import TensorType
from xdsl.ir.affine import AffineDimExpr
from merlin.targetgen.contract import linalg_iface as li
from host_lowering import TensorValue
from isa_encoder import ACC_ADDR, ACC_FULL, config_load, config_store, pack_matrix


@dataclass(frozen=True)
class MovementPiece:
    source: object
    shape: tuple
    # For each source axis, the corresponding piece axis, offset and stride.
    axes: tuple
    offsets: tuple
    strides: tuple
    destination_offsets: tuple


@dataclass(frozen=True)
class MovementPlan:
    shape: tuple
    dtype: str
    pieces: tuple


def _tensor(value):
    typ = value.type
    if not isinstance(typ, TensorType):
        return None
    shape = tuple(typ.get_shape())
    dtype = str(typ.element_type)
    if not shape or any(x <= 0 for x in shape) or dtype not in ('i8', 'i32'):
        return None
    # Encoded tensors require a physical-layout adapter.
    if str(typ.encoding) != 'none':
        return None
    return shape, dtype


def plan_graph_movement(op):
    """Return an exact, immutable coordinate mapping, or decline without effects."""
    if len(op.results) != 1 or (result := _tensor(op.results[0])) is None:
        return None
    shape, dtype = result
    rank = len(shape)
    zero = (0,) * rank
    one = (1,) * rank
    identity = tuple(range(rank))
    kind = li._op_name(op)
    if kind == 'tensor.concat':
        axis = int(op.dim.value.data)
        if not 0 <= axis < rank:
            return None
        pieces = []
        offset = 0
        for source in op.operands:
            info = _tensor(source)
            if info is None or info[1] != dtype or len(info[0]) != rank:
                return None
            srcshape = info[0]
            if any(srcshape[i] != shape[i] for i in range(rank) if i != axis):
                return None
            dest = list(zero)
            dest[axis] = offset
            pieces.append(MovementPiece(source, srcshape, identity, zero, one, tuple(dest)))
            offset += srcshape[axis]
        return MovementPlan(shape, dtype, tuple(pieces)) if pieces and offset == shape[axis] else None
    if kind == 'tensor.extract_slice':
        if len(op.operands) != 1 or _tensor(op.operands[0]) is None:
            return None
        srcshape, srcdtype = _tensor(op.operands[0])
        offsets = tuple(op.static_offsets.get_values())
        sizes = tuple(op.static_sizes.get_values())
        strides = tuple(op.static_strides.get_values())
        if srcdtype != dtype or len(srcshape) != rank or sizes != shape:
            return None
        if len(offsets) != rank or len(strides) != rank:
            return None
        if any(o < 0 or s <= 0 or o + (n - 1) * s >= extent
               for o, s, n, extent in zip(offsets, strides, shape, srcshape)):
            return None
        return MovementPlan(shape, dtype, (MovementPiece(op.operands[0], shape, identity, offsets, strides, zero),))
    if kind not in ('linalg.generic', 'linalg.transpose'):
        return None
    ins, outs = li._ins_outs(op)
    if len(ins) != 1 or len(outs) != 1 or _tensor(ins[0]) is None or _tensor(outs[0]) != result:
        return None
    srcshape, srcdtype = _tensor(ins[0])
    if srcdtype != dtype or len(srcshape) != rank:
        return None
    # Named transpose has an implicit direct-yield body in some parser versions.
    if op.regions:
        if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
            return None
        block = op.regions[0].blocks[0]
        body = list(block.ops)
        if len(body) != 1 or li._op_name(body[0]) != 'linalg.yield' or len(body[0].operands) != 1:
            return None
        if not block.args or body[0].operands[0] is not block.args[0]:
            return None
    if kind == 'linalg.transpose':
        permutation = tuple(op.permutation.get_values())
        if sorted(permutation) != list(identity):
            return None
        axes = tuple(permutation.index(i) for i in identity)
    else:
        if len(op.iterator_types) != rank or any(getattr(it.data, 'value', None) != 'parallel' for it in op.iterator_types):
            return None
        maps = [attr.data for attr in op.indexing_maps]
        if len(maps) != 2:
            return None
        permutations = []
        for amap in maps:
            if amap.num_dims != rank or amap.num_symbols or len(amap.results) != rank:
                return None
            if any(not isinstance(expr, AffineDimExpr) for expr in amap.results):
                return None
            permutation = tuple(expr.position for expr in amap.results)
            if sorted(permutation) != list(identity):
                return None
            permutations.append(permutation)
        axes = tuple(permutations[1].index(i) for i in permutations[0])
    if any(srcshape[i] != shape[axes[i]] for i in identity):
        return None
    return MovementPlan(shape, dtype, (MovementPiece(ins[0], shape, axes, zero, one, zero),))


def _emit_movement_panel(e, host, piece, src, dst, contiguous, local, readout):
    """Batch independent vectors without touching resident weights at SP row32+."""
    b = e.b
    last = piece.shape[-1]
    columns = (last + 15) // 16
    total = prod(piece.shape[:-1]) * columns
    # A gathered vector consumes sixteen rows, a contiguous vector one row.
    # i8 uses only SP rows0..31; i32 uses the 1024-row accumulator domain.
    rows_per_vector = 1 if contiguous else 16
    capacity = min(64, (32 if src.dtype == 'i8' else 1024) // rows_per_vector)
    def packed(address, rows, cols):
        return b.or_i(e.value(address), b.or_i(b.shl_i(e.value(rows), 48),
                                             b.shl_i(e.value(cols), 32)))
    def panel(first, count):
        def transfer(slot, load):
            ordinal = b.add_i(e.value(first), slot)
            row = b.udiv_i(ordinal, e.value(columns))
            col = b.mul_i(b.urem_i(ordinal, e.value(columns)), e.value(16))
            width = b.smin(b.sub_i(e.value(last), col), e.value(16))
            coords = (host.coords(row, piece.shape[:-1]) if len(piece.shape) > 1 else []) + [col]
            if load:
                coords = [b.add_i(e.value(offset), b.mul_i(coords[axis], e.value(stride)))
                          for axis, offset, stride in zip(piece.axes, piece.offsets, piece.strides)]
                tensor = src
            else:
                coords = [b.add_i(c, e.value(offset)) for c, offset in zip(coords, piece.destination_offsets)]
                tensor = dst
            address = e.address(e.ptr(tensor.ptr, host.index(coords, tensor.shape), tensor.dtype))
            scratch = b.add_i(e.value(local if load else readout),
                              b.mul_i(slot, e.value(rows_per_vector)))
            rows, cols = (1, width) if contiguous else (width, 1)
            e.insn('mvin' if load else 'mvout', address, packed(scratch, rows, cols))
        e.for_loop(0, count, 1, lambda slot: transfer(slot, True))
        e.fence()  # Every panel load completes before the independent store queue.
        e.for_loop(0, count, 1, lambda slot: transfer(slot, False))
        e.fence()  # Retire every store before panel rows or DRAM storage are reused.
    full = total // capacity * capacity
    if full:
        e.for_loop(0, full, capacity, lambda first: panel(first, capacity))
    if full != total:
        panel(e.value(full), total - full)


def try_emit_graph_movement(e, op, host):
    """Emit an admitted movement, retaining the ordinary host tensor value ABI."""
    plan = plan_graph_movement(op)
    if plan is None:
        return False
    sources = [host.values.get(piece.source) for piece in plan.pieces]
    if any(not isinstance(src, TensorValue) for src in sources):
        return False
    for piece, src in zip(plan.pieces, sources):
        if (src.shape, src.dtype) != _tensor(piece.source):
            return False
    bound = host.bound_outputs.get(op.results[0])
    if bound is not None and any(bound is src.ptr for src in sources):
        return False
    dst = host.result(op.results[0])
    if any(dst.ptr is src.ptr for src in sources):
        raise ValueError('device movement requires disjoint source and result storage')
    b = e.b
    nbytes = 1 if plan.dtype == 'i8' else 4
    local = 0 if plan.dtype == 'i8' else ACC_ADDR
    readout = local if plan.dtype == 'i8' else local | ACC_FULL
    order = knob('movement_order', default='tile', values=('tile', 'panel'), root=Path(__file__).parent)
    e.fence()  # Complete any preceding DRAM producer before loading its values.
    for piece, src in zip(plan.pieces, sources):
        src_axis = piece.axes.index(len(piece.shape) - 1)
        source_step = piece.strides[src_axis] * prod(src.shape[src_axis + 1:])
        contiguous = source_step == 1
        e.insn('config', *config_load(nbytes if contiguous else source_step * nbytes))
        e.insn('config', *config_store(nbytes))
        if order == 'panel':
            _emit_movement_panel(e, host, piece, src, dst, contiguous, local, readout)
            continue
        last = piece.shape[-1]
        def row_body(row):
            outer = host.coords(row, piece.shape[:-1]) if len(piece.shape) > 1 else []
            def block_body(col, width):
                coords = outer + [col]
                source_coords = [b.add_i(e.value(offset), b.mul_i(coords[axis], e.value(stride)))
                                 for axis, offset, stride in zip(piece.axes, piece.offsets, piece.strides)]
                target_coords = [b.add_i(c, e.value(offset)) for c, offset in zip(coords, piece.destination_offsets)]
                address = e.address(e.ptr(src.ptr, host.index(source_coords, src.shape), src.dtype))
                target = e.address(e.ptr(dst.ptr, host.index(target_coords, dst.shape), dst.dtype))
                rows, cols = (1, width) if contiguous else (width, 1)
                e.insn('mvin', address, pack_matrix(local, rows, cols))
                e.fence()  # The independent load and store queues share these rows.
                e.insn('mvout', target, pack_matrix(readout, rows, cols))
                e.fence()  # Retire stores before reusing local rows or pooled DRAM.
            full = last // 16 * 16
            if full:
                e.for_loop(0, full, 16, lambda col: block_body(col, 16))
            if full != last:
                block_body(e.value(full), last - full)
        e.for_loop(0, prod(piece.shape[:-1]), 1, row_body)
    host.preparation_optimizations.append({'kind': 'device_tensor_movement', 'operation': li._op_name(op),
                                         'source_op_index': host.source_op_index, 'dtype': plan.dtype,
                                         'shape': list(plan.shape)})
    return True
