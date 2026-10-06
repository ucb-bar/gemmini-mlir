"""Observed RV64GC host opcode classes for this target's functional census.

Encodings are cross-checked with the pinned Spike installed encoding.h. Classes
are instruction multiplicities. They provide no latency or CPU/array-overlap
claim. Wider or unrecognized encodings remain explicit unknown buckets.
"""

import struct

from merlin.targetgen.elf_lanes import executable_sections

from .no_fsm_audit import _instruction_bytes

CLASSES = (
    "load_integer",
    "load_fp",
    "store_integer",
    "store_fp",
    "branch_conditional",
    "branch_direct",
    "branch_indirect",
    "integer",
    "integer_mul",
    "integer_div_rem",
    "fp_add_sub",
    "fp_mul",
    "fp_fma",
    "fp_convert",
    "fp_div",
    "fp_sqrt",
    "fp_compare",
    "fp_sign",
    "fp_minmax",
    "fp_move_class",
    "fp_other",
    "custom",
    "csr",
    "system",
    "fence",
    "atomic",
    "unknown_encoding",
    "outside_elf_unknown",
)


def opcode_index(blob: bytes) -> dict[int, str]:
    """Index executable PCs once for source-bound histogram scope joins."""
    return {pc: role for pc, (role, _) in instruction_index(blob).items()}


def instruction_index(blob: bytes) -> dict[int, tuple[str, int]]:
    """Index the observed class and encoded byte width for each executable PC."""
    result = {}
    for _, offset, size, address in executable_sections(blob):
        pos = 0
        while pos < size:
            pc = address + pos
            if pos + 2 > size:
                raise ValueError("CPU opcode index has truncated code")
            width = _instruction_bytes(struct.unpack_from("<H", blob, offset + pos)[0])
            if pc in result or pos + width > size:
                raise ValueError("CPU opcode index has overlapping or truncated code")
            word = int.from_bytes(blob[offset + pos : offset + pos + width], "little")
            result[pc] = (classify(word, width), width)
            pos += width
    return result


def classify(word: int, width: int) -> str:
    """Classify one instruction under RV64GC, without disassembly aliases."""
    if width == 2:
        quadrant, funct = word & 3, word >> 13 & 7
        if quadrant == 0:
            return {
                0: "integer",
                1: "load_fp",
                2: "load_integer",
                3: "load_integer",
                5: "store_fp",
                6: "store_integer",
                7: "store_integer",
            }.get(funct, "unknown_encoding")
        if quadrant == 1:
            if funct <= 4:
                return "integer"
            return {
                5: "branch_direct",
                6: "branch_conditional",
                7: "branch_conditional",
            }[funct]
        if quadrant == 2:
            if funct == 4:
                rs2, rd = word >> 2 & 31, word >> 7 & 31
                if rs2:
                    return "integer"
                if rd:
                    return "branch_indirect"
                return "system" if word & (1 << 12) else "unknown_encoding"
            return {
                0: "integer",
                1: "load_fp",
                2: "load_integer",
                3: "load_integer",
                5: "store_fp",
                6: "store_integer",
                7: "store_integer",
            }.get(funct, "unknown_encoding")
        return "unknown_encoding"
    if width != 4:
        return "unknown_encoding"
    opcode, funct3, funct7 = word & 127, word >> 12 & 7, word >> 25 & 127
    if opcode in (0x33, 0x3B):
        if funct7 == 1:
            return "integer_mul" if funct3 < 4 else "integer_div_rem"
        return "integer"
    if opcode in (0x13, 0x1B, 0x17, 0x37):
        return "integer"
    if opcode == 0x73:
        return "csr" if funct3 else "system"
    if opcode in (0x43, 0x47, 0x4B, 0x4F):
        return "fp_fma"
    if opcode == 0x53:
        group = funct7 & ~1  # RV64GC single/double formats only.
        return {
            0x00: "fp_add_sub",
            0x04: "fp_add_sub",
            0x08: "fp_mul",
            0x0C: "fp_div",
            0x2C: "fp_sqrt",
            0x20: "fp_convert",
            0x60: "fp_convert",
            0x68: "fp_convert",
            0x50: "fp_compare",
            0x10: "fp_sign",
            0x14: "fp_minmax",
            0x70: "fp_move_class",
            0x78: "fp_move_class",
        }.get(group, "fp_other")
    if opcode in (0x0B, 0x2B, 0x5B, 0x7B):
        return "custom"
    return {
        0x03: "load_integer",
        0x07: "load_fp",
        0x23: "store_integer",
        0x27: "store_fp",
        0x63: "branch_conditional",
        0x6F: "branch_direct",
        0x67: "branch_indirect",
        0x0F: "fence",
        0x2F: "atomic",
    }.get(opcode, "unknown_encoding")
