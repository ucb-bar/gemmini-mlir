"""Source-bound RV64GC FP def/use distances, with no hardware cycle prices.

Distances count emitted instructions inside straight-line blocks. Dynamic sink
multiplicities come from the exact ELF's Spike histogram; the histogram supplies
no chronological trace. Loop-carried, cross-block, integer-address, memory alias,
exception-state and resource dependencies remain unknown. These features may
distinguish schedules that have equal opcode counts; they do not approve a
reordering or predict its execution time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

from merlin.targetgen.elf_lanes import executable_sections
from merlin.perf import depgraph
from merlin.perf.deps.liveness import Access, Effects, Instruction

from .cpu_opcode_census import classify
from .executed_features import _symbol_ranges, parse_pc_histogram
from .no_fsm_audit import _instruction_bytes, audit_elf


def encoding_forms(header):
    definitions = {name: int(value, 16) for name, value in re.findall(
        r"^#define ((?:MATCH|MASK)_[A-Z0-9_]+) (0x[0-9a-f]+)$", header, re.MULTILINE,
    )}
    names = {"FLW", "FLD", "FSW", "FSD", "C_FLD", "C_FLDSP", "C_FSD", "C_FSDSP",
             "FMV_X_W", "FMV_X_D", "FMV_W_X", "FMV_D_X", "FCVT_S_D", "FCVT_D_S"}
    for fmt in ("S", "D"):
        names.update(op + "_" + fmt for op in (
            "FADD", "FSUB", "FMUL", "FDIV", "FSQRT", "FMADD", "FMSUB", "FNMSUB", "FNMADD",
            "FSGNJ", "FSGNJN", "FSGNJX", "FMIN", "FMAX", "FEQ", "FLT", "FLE", "FCLASS",
        ))
        for integer in ("W", "WU", "L", "LU"):
            names.update(("FCVT_" + fmt + "_" + integer, "FCVT_" + integer + "_" + fmt))
    if any("MATCH_" + n not in definitions or "MASK_" + n not in definitions for n in names):
        raise ValueError("installed ISA declaration lacks required RV64GC forms")
    return tuple((n, definitions["MATCH_" + n], definitions["MASK_" + n]) for n in sorted(names))


def fp_effects(word, width, forms):
    matches = [name for name, match, mask in forms
               if (name.startswith("C_") == (width == 2)) and word & mask == match]
    if len(matches) > 1:
        raise ValueError("FP operand form is ambiguous")
    if not matches:
        role = classify(word, width)
        return (), (), role != "unknown_encoding" and not (role.startswith("fp_") or role.endswith("_fp"))
    name = matches[0]
    rd, rs1, rs2, rs3 = (word >> shift & 31 for shift in (7, 15, 20, 27))
    if name == "C_FLD":
        return (8 + (word >> 2 & 7),), (), True
    if name == "C_FLDSP":
        return (rd,), (), True
    if name == "C_FSD":
        return (), (8 + (word >> 2 & 7),), True
    if name == "C_FSDSP":
        return (), (word >> 2 & 31,), True
    if name in {"FLW", "FLD", "FMV_W_X", "FMV_D_X"}:
        return (rd,), (), True
    if name in {"FSW", "FSD"}:
        return (), (rs2,), True
    if name in {"FMV_X_W", "FMV_X_D"} or name.startswith("FCLASS_"):
        return (), (rs1,), True
    if name.startswith("FCVT_"):
        _, dest, source = name.split("_")
        return ((rd,) if dest in {"S", "D"} else ()), ((rs1,) if source in {"S", "D"} else ()), True
    op = name.split("_")[0]
    if op in {"FEQ", "FLT", "FLE"}:
        return (), (rs1, rs2), True
    if op in {"FMADD", "FMSUB", "FNMSUB", "FNMADD"}:
        return (rd,), (rs1, rs2, rs3), True
    return (rd,), ((rs1,) if op == "FSQRT" else (rs1, rs2)), True


def direct_target(pc, word, width):
    """Immediate layouts match the pinned Spike decode.h RV64GC definitions."""
    if width == 4:
        op = word & 127
        if op == 0x63:
            imm = ((word >> 8 & 15) << 1) | ((word >> 25 & 63) << 5) | ((word >> 7 & 1) << 11) | ((word >> 31) << 12)
            return pc + (imm - (1 << 13) if imm & (1 << 12) else imm)
        if op == 0x6F:
            imm = ((word >> 21 & 1023) << 1) | ((word >> 20 & 1) << 11) | ((word >> 12 & 255) << 12) | ((word >> 31) << 20)
            return pc + (imm - (1 << 21) if imm & (1 << 20) else imm)
    if width == 2 and word & 3 == 1:
        funct = word >> 13 & 7
        if funct == 5:
            imm = ((word >> 3 & 7) << 1) | ((word >> 11 & 1) << 4) | ((word >> 2 & 1) << 5) | ((word >> 7 & 1) << 6) | ((word >> 6 & 1) << 7) | ((word >> 9 & 3) << 8) | ((word >> 8 & 1) << 10) | ((word >> 12 & 1) << 11)
            return pc + (imm - (1 << 12) if imm & (1 << 11) else imm)
        if funct in {6, 7}:
            imm = ((word >> 3 & 3) << 1) | ((word >> 10 & 3) << 3) | ((word >> 2 & 1) << 5) | ((word >> 5 & 3) << 6) | ((word >> 12 & 1) << 8)
            return pc + (imm - (1 << 9) if imm & (1 << 8) else imm)
    return None


def census(elf, histogram, symbols, encoding_header, decode_header):
    elf, histogram, encoding_header, decode_header = map(Path, (elf, histogram, encoding_header, decode_header))
    blob = elf.read_bytes()
    if audit_elf(blob)["status"] != "pass":
        raise ValueError("final executable instruction policy failed")
    forms = encoding_forms(encoding_header.read_text())
    hist = parse_pc_histogram(histogram.read_text())
    ranges = _symbol_ranges(elf, symbols, "readelf")
    instructions = {}
    for _, offset, size, address in executable_sections(blob):
        pos = 0
        while pos < size:
            width = _instruction_bytes(struct.unpack_from("<H", blob, offset + pos)[0])
            if pos + width > size or address + pos in instructions:
                raise ValueError("executable instruction extent is invalid")
            instructions[address + pos] = (int.from_bytes(blob[offset + pos:offset + pos + width], "little"), width)
            pos += width
    reports = {}
    for region in ranges:
        pcs = sorted(p for p in instructions if region["start"] <= p < region["end"])
        if not pcs or pcs[0] != region["start"] or pcs[-1] + instructions[pcs[-1]][1] != region["end"]:
            raise ValueError("source function range splits an instruction")
        if any(region["start"] <= p < region["end"] and p not in instructions for p in hist):
            raise ValueError("executed PC is not an instruction start")
        heads = {region["start"]}
        for pc in pcs:
            word, width = instructions[pc]
            target = direct_target(pc, word, width)
            if target is not None and region["start"] <= target < region["end"]:
                if target not in instructions:
                    raise ValueError("direct branch target splits an instruction")
                heads.add(target)
            if classify(word, width).startswith("branch_"):
                heads.add(pc + width)
            if not fp_effects(word, width, forms)[2]:
                heads.update((pc, pc + width))
        blocks = []
        for pc in pcs:
            if not blocks or pc in heads:
                blocks.append([])
            blocks[-1].append(pc)
        distances, unbound, unknown = defaultdict(Counter), Counter(), Counter()
        examples = []
        for block in blocks:
            nodes, effects = [], []
            for index, pc in enumerate(block):
                word, width = instructions[pc]
                role = classify(word, width)
                defs, uses, complete = fp_effects(word, width, forms)
                nodes.append(Instruction(index, role))
                effects.append(Effects(tuple(Access("fpr", r) for r in defs),
                                       tuple(Access("fpr", r) for r in sorted(set(uses))),
                                       ("partial FP register census; other architectural effects unknown",)))
                if not complete and hist.get(pc, 0):
                    unknown[role] += hist[pc]
            # Delegate all register dependency construction to existing Merlin.
            # Node costs are disabled; only RAW topology is consumed. Neither
            # makespan nor critical_path is called or exported as a cycle price.
            dag = depgraph.build_dag(
                nodes, effects,
                issue=depgraph.IssueModel(0, 0, "UNPRICED", "graph topology only; no timing interpretation"),
                stall_mnemonic="<no target stall declaration>",
            )
            bound = set()
            for edge in dag.edges:
                if edge.kind != depgraph.RAW:
                    continue
                producer_pc, pc = block[edge.src], block[edge.dst]
                weight = hist.get(pc, 0)
                bound.add((edge.dst, edge.value.slot))
                if hist.get(producer_pc, 0) < weight:
                    unbound["unobserved_block_arrival"] += weight
                    continue
                distance = edge.dst - edge.src
                distances[nodes[edge.src].mnemonic][distance] += weight
                if weight and len(examples) < 12:
                    examples.append({"producer_pc": producer_pc, "consumer_pc": pc, "fp_register": edge.value.slot,
                                     "instruction_distance": distance, "sink_multiplicity": weight})
            for index, effect in enumerate(effects):
                for value in effect.uses:
                    if (index, value.slot) not in bound:
                        unbound[nodes[index].mnemonic] += hist.get(block[index], 0)
        reports[region["symbol"]] = {
            "scope": region, "executed_instructions": sum(hist.get(pc, 0) for pc in pcs),
            "sink_weighted_intrablock_distances_by_producer_class": {
                role: {str(k): v for k, v in sorted(counts.items()) if v}
                for role, counts in sorted(distances.items())
            },
            "unbound_fp_reads_by_consumer_class": dict(unbound),
            "undecoded_executed_instructions": dict(unknown), "examples": examples,
        }
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {"schema": "rv64gc_fp_emitted_distance_census_v1", "functions": reports,
            "pins": [{"path": str(p), "sha256": digest(p)} for p in (elf, histogram, encoding_header, decode_header)],
            "feature_kind": "Emitted within-block FP register def/use distances weighted by observed sink multiplicity",
            "hardware_cycles": "UNKNOWN", "chronological_execution": "UNKNOWN",
            "dependency_builder": {"module": "merlin.perf.depgraph", "sha256": digest(Path(depgraph.__file__)),
                                   "scope": "RAW topology only, all costs disabled and non-FP effects unknown"},
            "limitations": ["No loop-carried/cross-block edges", "No memory/address/exception/resource dependencies",
                            "Indirect target paths unobserved; features do not prove a whole execution DAG",
                            "No latency, initiation interval, reordering legality or optimization ranking licensed"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--histogram", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--encoding-header", type=Path, required=True)
    parser.add_argument("--decode-header", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = census(args.elf, args.histogram, args.symbols, args.encoding_header, args.decode_header)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("CPU_FP_DISTANCES_COMPLETE", list(result["functions"]))
