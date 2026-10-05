"""Profile the instruction schedule in Jack's archived ResNet ELF/disassembly.

The ZIP has no source.  This script reads only copies of its ELF and disassembly,
counts static instructions by symbol, and distinguishes a static instruction count
from a dynamic execution count or a cycle measurement.  Four-byte ``fx_conv``
symbols are tail jumps to a shared implementation; their tiny bodies must not be
mistaken for cheap convolutions.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from .tables import rtl_facts as facts

_SYM = re.compile(r"^\s*\d+:\s+([0-9a-f]+)\s+(\d+)\s+FUNC\s+\w+\s+\w+\s+\d+\s+(\S+)")
_INSN = re.compile(r"^([0-9a-f]+):\s+(.+)$")
_CUSTOM = re.compile(r"\.insn\s+4,\s+0x([0-9a-fA-F]{8})")
_BRANCHES = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bgez", "bltz", "blez", "bgtz"}


def profile(elf: Path, disassembly: Path) -> dict:
    symbol_text = subprocess.run(["readelf", "-sW", str(elf)], check=True,
                                 capture_output=True, text=True).stdout
    symbols = []
    for line in symbol_text.splitlines():
        match = _SYM.match(line)
        if match and int(match[2]):
            address, size, name = int(match[1], 16), int(match[2]), match[3]
            symbols.append((address, address + size, name))
    symbols.sort(key=lambda row: (row[0], -(row[1] - row[0]), row[2]))
    rows = {name: {"name": name, "address": start, "bytes": end - start,
                   "instructions": 0, "branches": 0, "calls": 0,
                   "custom_functs": Counter(), "mnemonics": Counter()}
            for start, end, name in symbols}
    # Symbol ranges overlap for aliases.  The largest enclosing function is the
    # implementation, while a 4-byte alias retains its own metadata below.
    active: list[tuple[int, int, str]] = []
    pos = 0
    for line in disassembly.read_text().splitlines():
        insn = _INSN.match(line)
        if not insn:
            continue
        address = int(insn[1], 16)
        while pos < len(symbols) and symbols[pos][0] <= address:
            active.append(symbols[pos])
            pos += 1
        active = [s for s in active if address < s[1]]
        if not active:
            continue
        _, _, name = max(active, key=lambda s: (s[1] - s[0], s[2]))
        row = rows[name]
        row["instructions"] += 1
        body = insn[2].strip()
        mnemonic = body.split()[0]
        row["mnemonics"][mnemonic] += 1
        if mnemonic in _BRANCHES:
            row["branches"] += 1
        if mnemonic in ("jal", "jalr", "call", "tail"):
            row["calls"] += 1
        custom = _CUSTOM.search(body)
        if custom:
            word = int(custom[1], 16)
            if (word & 0x7f) == facts.CUSTOM_OPCODE:
                row["custom_functs"][(word >> 25) & 0x7f] += 1
    selected = []
    for name, row in rows.items():
        if name.startswith(("fx_conv_", "fx_res_conv_", "fx_fc_", "fx_gap")) or name == "main":
            row["custom_functs"] = {
                facts.FUNCT_NAMES.get(f, str(f)): count
                for f, count in sorted(row["custom_functs"].items())}
            row["mnemonics"] = dict(row["mnemonics"].most_common(12))
            row["tail_jump_alias"] = row["bytes"] <= 4 and row["calls"] == 0
            selected.append(row)
    selected.sort(key=lambda row: (row["address"], row["name"]))
    library_fsm = []
    for name, row in rows.items():
        if name.startswith("sp_tiled_matmul"):
            hits = {facts.FUNCT_NAMES.get(f, str(f)): count
                    for f, count in row["custom_functs"].items()
                    if facts.FUNCT_NAMES.get(f, "").startswith("LOOP_")}
            if hits:
                library_fsm.append({"name": name, "loop_functs": hits})
    return {"schema": "resnet_q1013_static_schedule_v1",
            "caveat": "static ELF/disassembly counts; no dynamic execution or cycle attribution",
            "functions": selected, "library_fsm": library_fsm}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("disassembly", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(profile(args.elf, args.disassembly), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
