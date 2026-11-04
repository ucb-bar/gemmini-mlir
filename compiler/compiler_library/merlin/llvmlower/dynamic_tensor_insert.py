"""Complete immutable packed tensor update using the shared host memory API.

Runtime indices have the original tensor.insert in-bounds precondition. This
lowering introduces neither clipping nor wrapping. The caller proves distinct
complete destination storage before emitting any operations.
"""
from math import prod
from .host_tensor_codegen import is_int


def insert_element(fp, source, destination, shape, dtype, indices, value):
    shape, indices = tuple(shape), tuple(indices)
    sizes = {'i8': 1, 'i32': 4, 'i64': 8, 'f32': 4}
    if (dtype not in sizes or len(shape) != len(indices)
            or any(type(d) is not int or d <= 0 for d in shape)
            or prod(shape) * sizes.get(dtype, 0) >= 1 << 63
            or source is destination):
        raise ValueError('insert requires bounded packed shape and distinct complete storage')
    fn = fp.fb
    def copy(index):
        if is_int(dtype):
            fn.store_i64(fn.load_i64(source, index, dtype), destination, index, dtype)
        else:
            fp.store(fp.load(source, index, dtype), destination, index, dtype)
    fn.for_loop(0, prod(shape), 1, copy, retain_loop=True)
    offset = fn.const(0)
    for axis, index in enumerate(indices):
        pitch = prod(shape[axis + 1:])
        term = index if pitch == 1 else fn.mul_i(index, fn.const(pitch))
        offset = fn.add_i(offset, term)
    if is_int(dtype):
        fn.store_i64(value, destination, offset, dtype)
    else:
        fp.store(value, destination, offset, dtype)
