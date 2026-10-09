"""Audit every executable instruction in a final RISC-V ELF for Gemmini loop commands.

The archive used as the performance reference contains unused LOOP_WS code.  A source
grep, a command-buffer inspection, or a dynamic trace would miss that distinction.
This audit walks the executable ELF sections after linking and fails on *any* loop
funct, including a loop configuration command.  Decoder names and the custom opcode
come from this target's recorded RTL facts; an unrecognised Gemmini funct also fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path

from .tables import rtl_facts as facts

_ELF64_SECTION = struct.Struct("<IIQQQQIIQQ")
_SHF_EXECINSTR = 4
_SHT_NOBITS = 8
_EM_RISCV = 243


class AuditError(ValueError):
    """An ELF cannot establish the zero-FSM claim."""


def _instruction_bytes(first: int) -> int:
    """RISC-V variable-length encoding, refusing lengths we cannot traverse."""
    if first & 0b11 != 0b11:
        return 2
    if first & 0b11111 != 0b11111:
        return 4
    if first & 0b111111 != 0b111111:
        return 6
    if first & 0b1111111 != 0b1111111:
        return 8
    raise AuditError("executable section contains an unsupported RISC-V instruction length")


def _section_names(data: bytes, headers: list[tuple[int, ...]], index: int) -> bytes:
    if index >= len(headers):
        raise AuditError("ELF section-name table index is outside the section table")
    row = headers[index]
    offset, size = row[4], row[5]
    if offset + size > len(data):
        raise AuditError("ELF section-name table extends past the file")
    return data[offset:offset + size]


def audit_elf(data: bytes) -> dict:
    """Return a content-bound verdict; malformed or incomplete ELFs raise AuditError."""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise AuditError("input is not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise AuditError("this target requires a little-endian ELF64 image")
    if struct.unpack_from("<H", data, 18)[0] != _EM_RISCV:
        raise AuditError("ELF machine is not RISC-V")
    shoff = struct.unpack_from("<Q", data, 40)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 58)
    if not shoff or not shnum or shentsize != _ELF64_SECTION.size:
        raise AuditError("ELF has no ordinary section-header table")
    if shoff + shentsize * shnum > len(data):
        raise AuditError("ELF section-header table extends past the file")
    headers = [_ELF64_SECTION.unpack_from(data, shoff + i * shentsize)
               for i in range(shnum)]
    names = _section_names(data, headers, shstrndx)
    counts: dict[int, int] = {}
    forbidden: list[dict] = []
    unknown: list[dict] = []
    scanned = 0
    for index, row in enumerate(headers):
        name_offset, kind, flags, address, offset, size = row[:6]
        if not flags & _SHF_EXECINSTR or kind == _SHT_NOBITS or not size:
            continue
        if offset + size > len(data) or size % 2 or address % 2:
            raise AuditError(f"executable section {index} has invalid bounds or alignment")
        if name_offset >= len(names):
            raise AuditError(f"executable section {index} has an invalid name")
        end = names.find(b"\0", name_offset)
        if end < 0:
            raise AuditError(f"executable section {index} has an unterminated name")
        section = names[name_offset:end].decode("utf-8", errors="replace")
        scanned += 1
        pos = 0
        while pos < size:
            if pos + 2 > size:
                raise AuditError(f"truncated instruction in {section}")
            first = struct.unpack_from("<H", data, offset + pos)[0]
            width = _instruction_bytes(first)
            if pos + width > size:
                raise AuditError(f"truncated {width}-byte instruction in {section}")
            if width == 4:
                word = struct.unpack_from("<I", data, offset + pos)[0]
                if word & 0x7f == facts.CUSTOM_OPCODE:
                    funct = (word >> 25) & 0x7f
                    counts[funct] = counts.get(funct, 0) + 1
                    name = facts.FUNCT_NAMES.get(funct)
                    hit = {"section": section, "address": address + pos,
                           "funct": funct, "funct3": (word >> 12) & 7, "name": name}
                    if name and name.startswith("LOOP_"):
                        forbidden.append(hit)
                    elif (funct not in facts.LEGAL_FUNCTS or name is None or
                          hit["funct3"] != facts.FUNCT3):
                        unknown.append(hit)
            pos += width
    if not scanned:
        raise AuditError("ELF has no nonempty executable section")
    return {
        "schema": "gemmini_nofsm_elf_audit_v1",
        "elf_sha256": hashlib.sha256(data).hexdigest(),
        "executable_sections": scanned,
        "custom_funct_counts": {str(k): counts[k] for k in sorted(counts)},
        "forbidden": forbidden,
        "unknown": unknown,
        "status": "fail" if forbidden or unknown else "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path, help="final linked RISC-V ELF, or a zip archive")
    parser.add_argument("--zip-member", help="ELF member within a zip archive")
    args = parser.parse_args()
    try:
        if args.zip_member:
            with zipfile.ZipFile(args.elf) as archive:
                data = archive.read(args.zip_member)
        else:
            data = args.elf.read_bytes()
        result = audit_elf(data)
    except (AuditError, OSError, KeyError, zipfile.BadZipFile) as exc:
        result = {"schema": "gemmini_nofsm_elf_audit_v1", "status": "fail",
                  "reason": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
