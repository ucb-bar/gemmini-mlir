"""Optional constant operand hoisting for the existing terminal Encoder.

This deliberately retains every CONFIG command, in its original effect callback,
with the original second operand, command order, data effects and completion.
Only integer bundle construction moves before the counted loops. No persistent
configuration purity, queue capture, or cross-invocation state is assumed.
"""
from contextlib import contextmanager
import os


@contextmanager
def encoded_configuration_operands(enc, *, scale=1.0, relu=False):
    """Scope the same Encoder's setters to invocation-local constant operands.

    The caller enters after pointer binding and before actual schedule emission.
    Unrecognized dynamic/nonstandard setters use the original implementation.
    Disabled mode does not construct any LLVM values or replace any methods.
    """
    mode = os.environ.get('DENSE_CONFIG_CLEANUP', 'off')
    if mode not in ('off', 'operands'):
        raise ValueError('DENSE_CONFIG_CLEANUP must be off or operands')
    receipt = dict(mode=mode, removed_native_commands=0, timing='UNKNOWN')
    if mode == 'off':
        yield receipt
        return
    from encoding import float_bits
    # Exact terminal Encoder bundle definitions, with the actual public helper;
    # leave RS2 stride arithmetic and its SSA dependencies at each original use.
    loads = {stream: enc.flag(enc.bundle('ConfigMvinRs1', shrink=0,
              state_id=stream, pixel_repeats=1, stride=16,
              scale=float_bits(1.0)), 1) for stream in range(3)}
    store_rs1 = enc.bundle('ConfigMvoutRs1', cmd_type=2, activation=int(relu))
    original_load, original_store = enc.configure_load, enc.configure_store
    had_load = 'configure_load' in enc.__dict__
    had_store = 'configure_store' in enc.__dict__
    counts = dict(load_setters=0, store_setters=0, fallback_setters=0)

    def load(stride, stream=0, scale=1.0, shrunk=False, block_stride=16):
        if (type(stream) is int and stream in loads and type(scale) in (int,float)
                and float_bits(scale) == float_bits(1.0)
                and type(shrunk) is bool and not shrunk
                and type(block_stride) is int and block_stride == 16):
            enc.issue('CONFIG_CMD', loads[stream], stride)
            counts['load_setters'] += 1
        else:
            counts['fallback_setters'] += 1
            original_load(stride, stream=stream, scale=scale, shrunk=shrunk,
                          block_stride=block_stride)

    def store(stride, dtype='i32', scale=1.0, relu=False, **pool):
        # RS2 remains the exact original scale/stride bundle at the original use.
        # This avoids introducing a new constant scale/readout binding contract.
        if type(relu) is bool and relu == expected_relu and not pool:
            enc.issue('CONFIG_CMD', store_rs1,
                enc.bundle('ConfigMvoutRs2', stride=stride, acc_scale=float_bits(scale)))
            counts['store_setters'] += 1
        else:
            counts['fallback_setters'] += 1
            original_store(stride, dtype=dtype, scale=scale, relu=relu, **pool)

    expected_relu = relu
    enc.configure_load, enc.configure_store = load, store
    try:
        yield receipt
    finally:
        if had_load:
            enc.configure_load = original_load
        else:
            del enc.configure_load
        if had_store:
            enc.configure_store = original_store
        else:
            del enc.configure_store
        receipt.update(static_setter_sites=counts,
            scope='same terminal Encoder; original native commands and completion retained')
