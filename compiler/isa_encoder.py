"""Terminal encoding for the selected Gemmini custom-3 command ISA.

Field positions and instruction numbers are the public rtl_facts/facts.json
register_bundle_layouts and funct_decode_table. Hardware-loop commands are
intentionally outside this terminal emitter's vocabulary.
"""
import struct

OPCODE = 0x7B
FUNCT3 = 3
DIM = 16
GARBAGE_ADDR = 0xFFFFFFFF
ACC_ADDR = 1 << 31
ACC_ACCUMULATE = 1 << 30
ACC_FULL = 1 << 29
FUNCTS = {
    'config': 0, 'mvin2': 1, 'mvin': 2, 'mvout': 3,
    'compute': 4, 'compute_preloaded': 4, 'compute_accumulated': 5,
    'compute_and_flip': 4, 'compute_and_stay': 5,
    'preload': 6, 'flush': 7, 'mvin3': 14, 'mvout_spad': 23,
}
_ALIASES = {'CONFIG_CMD': 'config', 'LOAD_CMD': 'mvin', 'LOAD2_CMD': 'mvin2',
            'LOAD3_CMD': 'mvin3', 'STORE_CMD': 'mvout', 'PRELOAD_CMD': 'preload',
            'COMPUTE_AND_FLIP_CMD': 'compute_and_flip',
            'COMPUTE_AND_STAY_CMD': 'compute_and_stay', 'FLUSH_CMD': 'flush'}


def funct(op):
    """Resolve a supported semantic instruction or public RTL instruction name."""
    if isinstance(op, int):
        if op not in FUNCTS.values():
            raise ValueError(f'unsupported instruction funct {op}')
        return op
    return FUNCTS[_ALIASES.get(op, op)]


def insn_asm(op):
    return f'.insn r 0x{OPCODE:02x}, {FUNCT3}, {funct(op)}, x0, $0, $1'


def encode(op, rs1, rs2):
    """Return terminal instruction and masked XLEN operand constants."""
    return insn_asm(op), int(rs1) & ((1 << 64) - 1), int(rs2) & ((1 << 64) - 1)


def float_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def pack_matrix(addr, rows=DIM, cols=DIM):
    if not 0 <= rows <= 0xFFFF or not 0 <= cols <= 0xFFFF:
        raise ValueError('matrix command extent does not fit its 16-bit field')
    return (int(addr) & 0xFFFFFFFF) | (int(cols) << 32) | (int(rows) << 48)


def config_ex(*, dataflow=1, activation=0, a_stride=1, c_stride=1,
              a_transpose=False, b_transpose=False, scale=1.0, shift=0):
    rs1 = ((int(dataflow) & 1) << 2) | ((int(activation) & 3) << 3)
    rs1 |= int(bool(a_transpose)) << 8 | int(bool(b_transpose)) << 9
    rs1 |= (int(a_stride) & 0xFFFF) << 16 | float_bits(scale) << 32
    rs2 = (int(shift) & 0xFFFFFFFF) | (int(c_stride) & 0xFFFF) << 48
    return rs1, rs2


def config_load(stride, *, state_id=0, scale=1.0, shrink=False,
                block_stride=DIM, pixel_repeats=1):
    # CONFIG_LD subtype is 1. The DRAM byte stride occupies rs2.
    rs1 = 1 | int(bool(shrink)) << 2 | (int(state_id) & 3) << 3
    rs1 |= (int(pixel_repeats) & 0xFF) << 8
    rs1 |= (int(block_stride) & 0xFFFF) << 16 | float_bits(scale) << 32
    return rs1, int(stride)


def config_store(stride, *, scale=1.0, activation=0):
    return 2 | ((int(activation) & 3) << 2), (int(stride) & 0xFFFFFFFF) | float_bits(scale) << 32
