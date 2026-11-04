"""Terminal Gemmini command packing for the selected FireSim hardware.

All command helpers return ``(funct, rs1, rs2)`` for ``ctx.emit(*command)``.
DRAM operands pass through unchanged. Local operands are row addresses, never
byte offsets. ``acc_addr(row, accumulate=False, full=False)`` attaches accumulator
metadata; ``GARBAGE`` is the no-input/no-write sentinel. Shapes are (rows, cols).

``config_ex(dataflow=1, activation=0, a_stride=1, c_stride=1, ...)``
``config_ld(stride, channel=0, scale=1., shrink=False, block_stride=16, ...)``
``config_st(stride, scale=1., activation=0, **pool_fields)``
``mvin(dram, addr, rows=16, cols=16, channel=0, fb=None)``
``mvout(dram, addr, rows=16, cols=16, fb=None)``
``preload(weight, c, weight_rows=16, weight_cols=16, c_rows=16, c_cols=16, fb=None)``
``compute_preloaded(a, b=GARBAGE, a_rows=16, a_cols=16, b_rows=16, b_cols=16, fb=None)``
``compute_accumulated(...)`` uses the same signature. ``flush(skip=0)`` packs TLB
flush. ``fence(ctx)`` calls the lowering context's real completion fence; fences
are RISC-V synchronization instructions and have no Gemmini funct number.

Supply ``fb=ctx.fb`` for SSA-valued addresses, shapes or configuration fields.
``pack_matrix(addr, rows, cols, fb=None)`` and ``pack_fields`` are also public.
``encode(name, **operands)`` dispatches the named helper; no hardware loop
commands are admitted.

Grounding: rtl_facts/facts.json SHA256
b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af;
register_bundle_layouts, funct_decode_table, arrays and memory_address_spaces.
COMMAND_ORDERING_SELECTION.json gives accumulator select and sentinel. Address
metadata below follows the public Gemmini LocalAddr representation: select,
accumulate and full-width readout occupy the top three bits of a 32-bit address.
Only lower physical row bits address memory. Actual field widths left symbolic
by base facts are not asserted here: pack_fields uses their declared slots.
"""
from __future__ import annotations

import struct

DIM = 16
SP_ROWS = SPAD_ROWS = SCRATCHPAD_ROWS = 4 * 4096
ACC_ROWS = ACCUMULATOR_ROWS = 2 * 512
CUSTOM_OPCODE = 0x7B
FUNCT3 = 3
ACC_SELECT = 1 << 31
ACC_ACCUMULATE = 1 << 30
ACC_FULL = 1 << 29
GARBAGE = 0xFFFFFFFF
F32_ONE = 0x3F800000
FUNCT = {'config': 0, 'mvin2': 1, 'mvin': 2, 'mvout': 3,
         'compute_preloaded': 4, 'compute_accumulated': 5,
         'preload': 6, 'flush': 7, 'mvin3': 14}


def f32_bits(value):
    """IEEE binary32 value to its unsigned 32-bit register representation."""
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def _wide(value, fb):
    if isinstance(value, int):
        # FnBuilder constants are signed i64, while packed registers are words.
        value &= (1 << 64) - 1
        return fb.const(value - (1 << 64) if value >= 1 << 63 else value)
    from xdsl.dialects.builtin import i64
    if value.type != i64:
        from xdsl.dialects import llvm
        return fb.add(llvm.ZExtOp(value, i64)).results[0]
    return value


def pack_fields(fields, fb=None):
    """Pack iterable (value, offset, slot_width), accepting i64 SSA with fb.

    Constant fields are checked for overflow. SSA fields are masked to their
    declared slot; the schedule remains responsible for dynamic range legality.
    """
    static = 0
    dynamic = []
    occupied = 0
    for value, offset, width in fields:
        if offset < 0 or width <= 0 or offset + width > 64:
            raise ValueError('invalid register field slot')
        mask = (1 << width) - 1
        slot = mask << offset
        if slot & occupied:
            raise ValueError('overlapping register field slots')
        occupied |= slot
        if isinstance(value, int):
            if not 0 <= value <= mask:
                raise ValueError(f'field value {value} exceeds {width}-bit slot')
            static |= value << offset
        else:
            dynamic.append((value, offset, mask))
    if not dynamic:
        return static
    if fb is None:
        raise ValueError('SSA register packing requires fb=FnBuilder')
    result = _wide(static, fb)
    for value, offset, mask in dynamic:
        part = fb.and_i(_wide(value, fb), _wide(mask, fb))
        if offset:
            part = fb.shl_i(part, offset)
        result = fb.or_i(result, part)
    return result


def pack_matrix(addr, rows=DIM, cols=DIM, fb=None):
    return pack_fields(((addr, 0, 32), (cols, 32, 16), (rows, 48, 16)), fb)


def acc_addr(row, accumulate=False, full=False, fb=None):
    """Select an accumulator row; full chooses unscaled i32 store readout."""
    if isinstance(row, int) and not 0 <= row < ACC_ROWS:
        raise ValueError('accumulator row outside selected physical memory')
    return pack_fields(((row, 0, 29), (int(full) if isinstance(full, bool) else full, 29, 1),
                        (int(accumulate) if isinstance(accumulate, bool) else accumulate, 30, 1),
                        (1, 31, 1)), fb)


def config_ex(dataflow=1, activation=0, a_stride=1, c_stride=1,
              scale=1.0, in_shift=0, relu6_shift=0, a_transpose=False,
              b_transpose=False, set_only_strides=False, scale_bits=None, fb=None):
    rs1 = pack_fields(((0, 0, 2), (dataflow, 2, 1), (activation, 3, 2),
                       (int(set_only_strides), 7, 1), (int(a_transpose), 8, 1),
                       (int(b_transpose), 9, 1), (a_stride, 16, 16),
                       (f32_bits(scale) if scale_bits is None else scale_bits, 32, 32)), fb)
    rs2 = pack_fields(((in_shift, 0, 32), (relu6_shift, 32, 16), (c_stride, 48, 16)), fb)
    return (0, rs1, rs2)


def config_ld(stride, channel=0, scale=1.0, shrink=False, block_stride=DIM,
              pixel_repeats=1, scale_bits=None, fb=None):
    """Configure load state: shrink=True reads i8 into an accumulator row.

    For scratchpad i8 loads shrink is irrelevant. For full-width i32 bias loads
    use shrink=False. Host stride is in bytes; block_stride is local rows.
    """
    if channel not in (0, 1, 2):
        raise ValueError('Gemmini has three selected load channels')
    rs1 = pack_fields(((1, 0, 2), (int(shrink), 2, 1), (channel, 3, 2),
                       (pixel_repeats, 8, 8), (block_stride, 16, 16),
                       (f32_bits(scale) if scale_bits is None else scale_bits, 32, 32)), fb)
    return (0, rs1, stride)


def config_st(stride, scale=1.0, activation=0, pool_stride=0, pool_size=0,
              upad=0, lpad=0, pool_out_dim=0, porows=0, pocols=0,
              orows=0, ocols=0, scale_bits=None, fb=None):
    rs1 = pack_fields(((2, 0, 2), (activation, 2, 2), (pool_stride, 4, 2),
                       (pool_size, 6, 2), (upad, 8, 2), (lpad, 10, 2),
                       (pool_out_dim, 24, 8), (porows, 32, 8), (pocols, 40, 8),
                       (orows, 48, 8), (ocols, 56, 8)), fb)
    rs2 = pack_fields(((stride, 0, 32),
                       (f32_bits(scale) if scale_bits is None else scale_bits, 32, 32)), fb)
    return (0, rs1, rs2)


def mvin(dram, addr, rows=DIM, cols=DIM, channel=0, fb=None):
    if channel not in (0, 1, 2):
        raise ValueError('Gemmini has three selected load channels')
    return ((2, 1, 14)[channel], dram, pack_matrix(addr, rows, cols, fb))


def mvin2(dram, addr, rows=DIM, cols=DIM, fb=None):
    return mvin(dram, addr, rows, cols, channel=1, fb=fb)


def mvin3(dram, addr, rows=DIM, cols=DIM, fb=None):
    return mvin(dram, addr, rows, cols, channel=2, fb=fb)


def mvout(dram, addr, rows=DIM, cols=DIM, fb=None):
    return (3, dram, pack_matrix(addr, rows, cols, fb))


def preload(weight, c, weight_rows=DIM, weight_cols=DIM,
            c_rows=DIM, c_cols=DIM, fb=None):
    return (6, pack_matrix(weight, weight_rows, weight_cols, fb),
            pack_matrix(c, c_rows, c_cols, fb))


def compute_preloaded(a, b=GARBAGE, a_rows=DIM, a_cols=DIM,
                      b_rows=DIM, b_cols=DIM, fb=None):
    return (4, pack_matrix(a, a_rows, a_cols, fb),
            pack_matrix(b, b_rows, b_cols, fb))


def compute_accumulated(a, b=GARBAGE, a_rows=DIM, a_cols=DIM,
                        b_rows=DIM, b_cols=DIM, fb=None):
    return (5, pack_matrix(a, a_rows, a_cols, fb),
            pack_matrix(b, b_rows, b_cols, fb))


def flush(skip=0):
    if skip not in (0, 1):
        raise ValueError('flush skip must be a single bit')
    return (7, skip, 0)


def fence(ctx):
    """Emit the context's real RISC-V accelerator completion fence."""
    return ctx.fence()


def encode(name, operands=None, **kwargs):
    """Dispatch a terminal helper; hardware loop instructions are excluded."""
    names = {'config_ex': config_ex, 'config_ld': config_ld, 'config_st': config_st,
             'mvin': mvin, 'mvin2': mvin2, 'mvin3': mvin3, 'mvout': mvout,
             'preload': preload, 'compute_preloaded': compute_preloaded,
             'compute_accumulated': compute_accumulated, 'flush': flush}
    try:
        action = names[name.lower()]
    except KeyError:
        raise ValueError(f'unsupported terminal command {name!r}') from None
    return action(**({**(operands or {}), **kwargs}))


# Common spelling aliases for lowerings.
config_mvin = config_ld
config_mvout = config_st
compute_accumulate = compute_accumulated
