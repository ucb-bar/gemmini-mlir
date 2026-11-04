"""Bounded, division-free dense permutation copies through ordinary FnBuilder.

The caller owns an immutable source and a disjoint destination allocation of at
least ``prod(extents[axis]) * itemsize`` bytes. Distinct pointer SSA values do
not establish that ownership; this helper does not infer runtime nonaliasing.
"""
from __future__ import annotations


def copy_layout(fn, source, destination, source_axes, destination_axes, extents, dtype):
    """Emit a synchronous exact i8/i32 permutation; reject before IR mutation.

    Axis names are nonnegative integer dimension identifiers. Each order must
    contain exactly the same distinct axes. Extra entries in ``extents`` are
    permitted because graph callers share one extent map between operands.
    Rank zero copies its one scalar. All address spans fit signed i64 bytes.
    No allocation, device work, data conversion or runtime division is emitted.
    """
    source_axes, destination_axes = tuple(source_axes), tuple(destination_axes)
    if (any(type(axis) is not int or axis < 0 for axis in source_axes + destination_axes)
            or len(set(source_axes)) != len(source_axes)
            or len(set(destination_axes)) != len(destination_axes)
            or set(source_axes) != set(destination_axes)):
        raise ValueError('layout copy requires complete distinct axis permutations')
    if dtype not in ('i8', 'i32'):
        raise ValueError('layout copy requires i8 or i32')
    sizes = {}
    for axis in source_axes:
        try:
            extent = extents[axis]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError('layout copy requires every static extent') from error
        if type(extent) is not int or extent <= 0:
            raise ValueError('layout copy requires positive static integer extents')
        sizes[axis] = extent
    count = 1
    for extent in sizes.values():
        count *= extent
    if count * (4 if dtype == 'i32' else 1) >= 1 << 63:
        raise ValueError('layout copy exceeds signed i64 byte address bound')
    if source is destination:
        raise ValueError('layout copy requires a distinct destination allocation')

    def strides(axes):
        result, stride = {}, 1
        for axis in reversed(axes):
            result[axis] = stride
            stride *= sizes[axis]
        return result

    src_strides, dst_strides = strides(source_axes), strides(destination_axes)
    # Unit axes contribute zero to both addresses. Omitting their loops also
    # bounds recursion by the signed byte span, even for high-rank unit shapes.
    loop_axes = tuple(axis for axis in destination_axes if sizes[axis] != 1)

    def walk(level, src_offset, dst_offset):
        if level == len(loop_axes):
            fn.store_i64(fn.load_i64(source, src_offset, dtype), destination, dst_offset, dtype)
            return
        axis = loop_axes[level]

        def body(coordinate):
            def advance(offset, stride):
                term = coordinate if stride == 1 else fn.mul_i(coordinate, fn.const(stride))
                return fn.add_i(offset, term)
            walk(level + 1, advance(src_offset, src_strides[axis]),
                 advance(dst_offset, dst_strides[axis]))

        fn.for_loop(0, sizes[axis], 1, body, retain_loop=True)

    walk(0, fn.const(0), fn.const(0))
