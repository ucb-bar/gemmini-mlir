"""Exact projected-permutation preparation for signed integer graph dots.

Only layout copies execute on the CPU; the established dense emitter owns the
contraction. One source reduction dimension preserves its original order.
"""
from __future__ import annotations
from math import prod


def recognize(op):
    """Return a checked layout dict, or None for an unproved source domain."""
    from xdsl.dialects import builtin
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.ir import Block
    from xdsl.ir.affine import AffineDimExpr as D, AffineMap
    from merlin.transforms.dense_ir import recognize_signed_integer_matmul
    if type(op) is not GenericOp or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    values = tuple(op.operands) + tuple(op.results)
    if any(not isinstance(v.type, builtin.TensorType) or not isinstance(v.type.encoding, builtin.NoneAttr)
           or any(d <= 0 for d in v.type.get_shape()) for v in values):
        return None
    if [str(v.type.element_type) for v in values] != ['i8', 'i8', 'i32', 'i32'] or values[2].type != values[3].type:
        return None
    if set(op.properties) - {'indexing_maps', 'iterator_types', 'operandSegmentSizes'}:
        return None
    iters = tuple(x.data for x in op.iterator_types)
    reductions = tuple(i for i, t in enumerate(iters) if t == IteratorType.REDUCTION)
    parallels = tuple(i for i, t in enumerate(iters) if t == IteratorType.PARALLEL)
    if len(reductions) != 1 or len(parallels) + 1 != len(iters):
        return None
    maps = tuple(x.data for x in op.indexing_maps)
    if len(maps) != 3 or any(mp.num_dims != len(iters) or mp.num_symbols or
            any(not isinstance(x, D) for x in mp.results) for mp in maps):
        return None
    axes = tuple(tuple(x.position for x in mp.results) for mp in maps)
    if any(len(set(ax)) != len(ax) for ax in axes):
        return None
    a, b, c = axes
    kd = reductions[0]
    if set(c) != set(parallels) or kd not in a or kd not in b or set(a) | set(b) != set(c) | {kd}:
        return None
    extents = {}
    for value, ax in zip(values, axes):
        if len(ax) != len(value.type.get_shape()):
            return None
        for dim, extent in zip(ax, value.type.get_shape()):
            if dim in extents and extents[dim] != extent:
                return None
            extents[dim] = extent
    batch = tuple(x for x in c if x in a and x in b)
    ma = tuple(x for x in c if x in a and x not in b)
    na = tuple(x for x in c if x in b and x not in a)
    ca = tuple(x for x in batch if x in a) + ma + (kd,)
    cb = tuple(x for x in batch if x in b) + (kd,) + na
    cc = batch + ma + na
    if set(ca) != set(a) or set(cb) != set(b) or set(cc) != set(c):
        return None
    m, k, n = prod(extents[x] for x in ma), extents[kd], prod(extents[x] for x in na)
    fake = Block(arg_types=[builtin.TensorType(builtin.i8, (m, k)), builtin.TensorType(builtin.i8, (k, n)), builtin.TensorType(builtin.i32, (m, n))])
    shell = GenericOp(inputs=fake.args[:2], outputs=fake.args[2:], body=op.regions[0].clone(),
        indexing_maps=builtin.ArrayAttr(builtin.AffineMapAttr(mp) for mp in (
            AffineMap(3, 0, (D(0), D(2))), AffineMap(3, 0, (D(2), D(1))), AffineMap(3, 0, (D(0), D(1))))),
        iterator_types=builtin.ArrayAttr(IteratorTypeAttr(t) for t in (IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION)),
        result_types=[fake.args[2].type])
    shell.attributes = dict(op.attributes)
    try:
        recognize_signed_integer_matmul(shell)
    except ValueError:
        return None
    return dict(m=m, k=k, n=n, batch_shape=tuple(extents[x] for x in batch),
                lhs_batch_axes=tuple(range(len(batch))), rhs_batch_axes=tuple(range(len(batch))),
                out_batch_axes=tuple(range(len(batch))), permutation_preparation=True,
                source_axes=axes, canonical_axes=(ca, cb, cc), extents=extents)


def _copy(fn, source, destination, source_axes, destination_axes, extents, dtype):
    """Copy a bounded permutation using the common host integer memory API."""
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    route = knob('graph_layout_copy', default='linear', values=('linear', 'nested'),
                 root=Path(__file__).resolve().parent)
    if route == 'nested':
        from gemmini_layout_copy import copy_layout
        return copy_layout(fn, source, destination, source_axes, destination_axes,
                           extents, dtype)
    if set(source_axes) != set(destination_axes):
        raise ValueError('layout preparation requires a permutation')
    shape = tuple(extents[x] for x in destination_axes)
    def body(iv):
        coordinates = {}
        for i, axis in enumerate(destination_axes):
            stride = prod(shape[i + 1:])
            value = fn.udiv_i(iv, fn.const(stride)) if stride != 1 else iv
            coordinates[axis] = fn.urem_i(value, fn.const(extents[axis]))
        index = fn.const(0)
        for axis in source_axes:
            index = fn.add_i(fn.mul_i(index, fn.const(extents[axis])), coordinates[axis])
        fn.store_i64(fn.load_i64(source, index, dtype), destination, iv, dtype)
    fn.for_loop(0, prod(shape), 1, body, retain_loop=True)


def emit(fn, lhs, rhs, dst, geometry, layout, *, output_dtype, initial=None):
    """Prepare layouts, call the existing dot emitter, and restore source order.

    Source buffers remain immutable. Private storage is initialized on every
    call and uses the backend's disjoint BSS allocator instead of the stack.
    """
    from gemmini_support import scratch_buffer
    from gemmini_graph import _emit_dot
    from gemmini_encode import fence
    if not layout.get('permutation_preparation') or geometry != (layout['m'], layout['k'], layout['n']):
        raise ValueError('layout preparation requires its checked geometry')
    axes, canonical, extents = layout['source_axes'], layout['canonical_axes'], layout['extents']
    def prepare(pointer, number, dtype):
        if axes[number] == canonical[number]:
            return pointer
        storage = scratch_buffer(fn, prod(extents[x] for x in axes[number]) * (4 if dtype == 'i32' else 1), 'dot_layout')
        _copy(fn, pointer, storage, axes[number], canonical[number], extents, dtype)
        return storage
    ap, bp = prepare(lhs, 0, 'i8'), prepare(rhs, 1, 'i8')
    seed = None if initial is None else prepare(initial, 2, 'i32')
    output_permuted = axes[2] != canonical[2]
    target = scratch_buffer(fn, prod(extents[x] for x in axes[2]) * (4 if output_dtype == 'i32' else 1), 'dot_result_layout') if output_permuted else dst
    ordinary = {key: value for key, value in layout.items() if key != 'permutation_preparation'}
    _emit_dot(fn, ap, bp, target, geometry, ordinary, output_dtype=output_dtype, initial=seed)
    if output_permuted:
        fence(fn)
        _copy(fn, target, dst, canonical[2], axes[2], extents, output_dtype)
