"""Final-ELF guard for the golden's strict zero-FSM rule."""

import struct
import unittest

from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.tables import isa, rtl_facts


def elf_with_instructions(words: list[int]) -> bytes:
    names = b"\0.text\0.shstrtab\0"
    code = b"".join(struct.pack("<I", w) for w in words)
    text_offset = 64
    names_offset = text_offset + len(code)
    shoff = (names_offset + len(names) + 7) & ~7
    data = bytearray(shoff + 3 * 64)
    data[:16] = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into("<HHI", data, 16, 2, 243, 1)
    struct.pack_into("<Q", data, 40, shoff)
    struct.pack_into("<HHHHHH", data, 52, 64, 0, 0, 64, 3, 2)
    data[text_offset:text_offset + len(code)] = code
    data[names_offset:names_offset + len(names)] = names
    section = struct.Struct("<IIQQQQIIQQ")
    section.pack_into(data, shoff + 64, 1, 1, 4, 0x80000000,
                      text_offset, len(code), 0, 0, 2, 0)
    section.pack_into(data, shoff + 128, 7, 3, 0, 0,
                      names_offset, len(names), 0, 0, 1, 0)
    return bytes(data)


def rocc(funct: int, funct3: int = 3) -> int:
    return (funct << 25) | (funct3 << 12) | rtl_facts.CUSTOM_OPCODE


class NoFsmAuditTests(unittest.TestCase):
    def test_primitive_elf_passes(self):
        result = audit_elf(elf_with_instructions([rocc(0), rocc(2), rocc(6)]))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["forbidden"], [])
        self.assertEqual(result["unknown"], [])

    def test_every_loop_family_fails(self):
        for funct in (8, 9, 13, 15, 21, 24):
            with self.subTest(funct=funct):
                result = audit_elf(elf_with_instructions([rocc(funct)]))
                self.assertEqual(result["status"], "fail")
                self.assertEqual(result["forbidden"][0]["funct"], funct)
                with self.assertRaisesRegex(ValueError, "hardware loop FSM"):
                    isa.assert_legal(funct)

    def test_unknown_or_wrong_funct3_fails(self):
        self.assertTrue(audit_elf(elf_with_instructions([rocc(25)]))["unknown"])
        self.assertTrue(audit_elf(elf_with_instructions([rocc(2, funct3=0)]))["unknown"])


if __name__ == "__main__":
    unittest.main()
