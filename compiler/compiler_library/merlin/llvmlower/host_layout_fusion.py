"""Emit an exact static affine copy in destination row-major order.

This is an address construction adapter, not a source rewrite or placement
decision. The caller must own an immutable input and disjoint destination and
establish that the original copy belongs on the host. Scalar values are passed
through unchanged, including floating payload bits; no arithmetic is performed.
"""
from math import prod


def emit_affine_copy(builder, shape, source_strides, *, source_offset,
                     source_elements, load, store):
    """Use nested loops rather than recovering coordinates by div/rem.

    ``load(index)`` and ``store(index, value)`` retain the caller's shared typed
    accesses. All bounds are checked before IR construction. Empty and scalar
    tensors are admitted without division or speculative input reads.
    """
    shape, strides = tuple(shape), tuple(source_strides)
    numbers = (*shape, *strides, source_offset, source_elements)
    if (len(shape) != len(strides) or any(type(n) is not int for n in numbers)
            or any(n < 0 or n >= 1 << 63 for n in numbers) or prod(shape) >= 1 << 63):
        raise ValueError('affine copy requires bounded nonnegative static dimensions')
    if prod(shape) and (source_offset + sum((n-1)*s for n,s in zip(shape,strides))
                        >= source_elements):
        raise ValueError('affine copy source address exceeds tensor storage')
    if not prod(shape):
        return
    destination_strides = tuple(prod(shape[i+1:]) for i in range(len(shape)))

    def add_index(base, index, stride):
        if not stride:
            return base
        term = index if stride == 1 else builder.mul_i(index, builder.const(stride))
        return builder.add_i(base, term)

    def axis(position, src, dst):
        if position == len(shape):
            store(dst, load(src))
            return
        def body(index):
            axis(position+1, add_index(src,index,strides[position]),
                 add_index(dst,index,destination_strides[position]))
        if shape[position] == 1:
            axis(position+1,src,dst)
        else:
            builder.for_loop(0,shape[position],1,body,retain_loop=True)

    axis(0,builder.const(source_offset),builder.const(0))
