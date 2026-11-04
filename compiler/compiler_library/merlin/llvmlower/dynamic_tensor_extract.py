"""Read a ranked tensor element through the shared host memory ABI.

Indices are already lowered source index scalars (the host uses i64 for index).
As in tensor.extract, callers must supply in-bounds indices; this lowering adds
neither wrapping, clamping nor a new out-of-bounds behavior. No tensor arithmetic
or numeric conversion is introduced beyond the shared dtype-specific load.
"""
from math import prod

from .host_tensor_codegen import is_int, row_pitch


def load_element(fp, ptr, shape, dtype, indices, *, row_alignment=1):
    shape = tuple(shape)
    indices = tuple(indices)
    if len(indices) != len(shape) or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError('dynamic tensor extract requires a static ranked shape and one index per dimension')
    # The last dimension is contiguous; preceding dimensions span ABI-padded
    # rows. This also handles rank-zero tensors (offset zero).
    pitches = ([prod(shape[i+1:-1]) * row_pitch(shape[-1], alignment=row_alignment)
                for i in range(len(shape)-1)] + [1]) if shape else []
    fn = fp.fb
    offset = fn.const(0)
    for index, pitch in zip(indices, pitches):
        term = index if pitch == 1 else fn.mul_i(index, fn.const(pitch))
        offset = fn.add_i(offset, term)
    return fn.load_i64(ptr, offset, dtype) if is_int(dtype) else fp.load(ptr, offset, dtype)
