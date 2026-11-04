"""Original binary32 scale storage for counted hardware column readout.

Loads contain source parameter bits only. Accumulator values remain on-device;
scale, RNE and saturation stay in the selected Gemmini store configuration.
"""
from xdsl.dialects import builtin, llvm
from merlin.llvmlower.host_builder import PTR
from gemmini_readout import _scale_bits, _store_config


def source_scale_bits_buffer(fn, column_scale_bits):
    """Materialize exact checked source coefficients as an immutable i32 vector."""
    bits = tuple(column_scale_bits)
    if not bits or any(type(x) is not int or not 0 <= x < 1 << 32 for x in bits):
        raise ValueError('source column scales need a nonempty binary32 bit vector')
    # Validate positive finite binary32 without changing or re-rounding its bits.
    import struct
    if any(_scale_bits(struct.unpack('<f', struct.pack('<I', x))[0]) != x for x in bits):
        raise ValueError('source column scales are not positive finite binary32')
    globals_ = getattr(fn, '_gemmini_globals', None)
    if globals_ is None:
        globals_ = fn._gemmini_globals = []
    symbol = f'gemmini_column_scale_{len(globals_)}'
    value = builtin.DenseIntOrFPElementsAttr.from_list(builtin.TensorType(builtin.i32, (len(bits),)), list(bits))
    globals_.append(llvm.GlobalOp(llvm.LLVMArrayType(len(bits), builtin.i32), symbol,
        'internal', constant=True, value=value, alignment=64))
    # The first dynamic store may be nested in one spatial-tail loop while a
    # later store lives in another. Hoist this invariant address so a lazily
    # cached source table pointer dominates every readout in the invocation.
    address = llvm.AddressOfOp(symbol, PTR)
    terminator = fn.entry.last_op
    if terminator is not None and terminator.has_trait(llvm.IsTerminator):
        fn.entry.insert_op_before(address, terminator)
    else:
        fn.entry.add_op(address)
    return address.results[0]


def config_column_readout(fn, stride, source_bits_ptr, column, *, relu=False):
    """Configure scalar hardware readout from the original selected column bits.

    The producer must prove one output column starting at accumulator lane zero.
    This is an encoding helper; the shared schedule owns range and completion.
    """
    if type(relu) is not bool:
        raise ValueError('readout activation requires a checked Boolean')
    index = fn.const(column) if type(column) is int else column
    pointer = fn.gep(source_bits_ptr, index, builtin.i32)
    raw = fn.add(llvm.LoadOp(pointer, builtin.i32)).results[0]
    bits = fn.add(llvm.ZExtOp(raw, builtin.i64)).results[0]
    return _store_config(fn, stride, bits, relu)
