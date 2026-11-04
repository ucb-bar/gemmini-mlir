"""Explicit CPU packets for adjacent independently proved bounded RNE results.

Every lane has the complete original typed clamp/fraction/parity/sign proof.
Only consecutive result instructions in one basic block can form a packet.
All raw inputs must be defined earlier in that same block: no arithmetic is
moved across memory, calls, terminators, or a dependency on another result.
Original chains remain available to other users and ordinary LLVM DCE.

As with scalar CPU legalization, the selected host policy treats floating
exception flags as unobserved. Strict/constrained FP refuses. Empty policy
preserves all bytes; no model, accelerator, or function name selects a route.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .late_quant_rne import _functions, _identity, _instruction, _match, _tokens


def rewrite(source: str, *, host_isa: str | None = None, width: int = 2):
    report = {
        "schema": "bounded_rne_basic_block_v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "routes": [],
    }
    if host_isa is None:
        return source, report
    if host_isa not in ("rv64gc", "portable"):
        raise ValueError("explicit supported CPU ISA policy required")
    if type(width) is not int or not 2 <= width <= 4:
        raise ValueError("packet width must be an integer from two through four")
    try:
        tokens = _tokens(source)
        if any(
            t.text == "strictfp"
            or (t.text.startswith("@") and _identity(t.text).startswith("llvm.experimental.constrained."))
            for t in tokens
        ):
            report["refusal"] = "strict or constrained FP"
            return source, report
        bodies = _functions(tokens)
    except ValueError as error:
        report["refusal"] = str(error)
        return source, report
    used = {_identity(t.text) for t in tokens if t.text.startswith("%")}
    edits = []
    serial = 0
    for body in bodies:
        definitions = {}
        locations = {}
        block = 0
        candidates = []
        for index, token in enumerate(body):
            if index + 1 < len(body) and body[index + 1].text == ":":
                block += 1
            if not token.text.startswith("%") or index + 1 >= len(body) or body[index + 1].text != "=":
                continue
            locations[_identity(token.text)] = (token.start, block)
            op = _instruction(body, index)
            if op is not None:
                definitions[_identity(op.result)] = op
                candidates.append((op, block, index))
        proved = [
            (op, block, index, proof)
            for op, block, index in candidates
            if (proof := _match(op, definitions)) is not None
        ]
        cursor = 0
        while cursor < len(proved):
            group = [proved[cursor]]
            while cursor + len(group) < len(proved) and len(group) < width:
                previous = group[-1]
                following = proved[cursor + len(group)]
                # Token adjacency proves there is no intervening statement,
                # metadata, label, unknown call or memory operation to cross.
                end_index = previous[2]
                while end_index < len(body) and body[end_index].end <= previous[0].end:
                    end_index += 1
                if (
                    end_index != following[2]
                    or previous[1] != following[1]
                    or previous[3]["bounds"] != following[3]["bounds"]
                    or previous[3]["integer_dtype"] != following[3]["integer_dtype"]
                ):
                    break
                group.append(following)
            cursor += len(group)
            if len(group) < 2:
                continue
            insertion = group[0][0].start
            block = group[0][1]
            if any(
                (location := locations.get(_identity(row[3]["raw_input"]))) is None
                or location[1] != block
                or location[0] >= insertion
                for row in group
            ):
                continue
            indent = source[source.rfind("\n", 0, insertion) + 1 : insertion]
            if indent.strip():
                continue
            lanes = len(group)

            def names(number):
                prefix = f"merlin.rne.packet.{number}"
                return [
                    prefix + ".asm",
                    *[prefix + f".wide{i}" for i in range(lanes)],
                    *[prefix + f".float{i}" for i in range(lanes)],
                ]

            while any(name in used for name in names(serial)):
                serial += 1
            prefix = f"merlin.rne.packet.{serial}"
            used.update(names(serial))
            serial += 1
            aggregate = "{ " + ", ".join(["i32"] * lanes) + " }"
            lower, upper = group[0][3]["bounds"]
            if host_isa == "rv64gc":
                instructions = [f"fmax.s ft{i}, ${lanes + i}, ${2 * lanes}" for i in range(lanes)]
                instructions += [f"fmin.s ft{i}, ft{i}, ${2 * lanes + 1}" for i in range(lanes)]
                instructions += [f"fcvt.w.s ${i}, ft{i}, rne" for i in range(lanes)]
                constraints = ",".join(["=r"] * lanes + ["f"] * (lanes + 2) + [f"~{{ft{i}}}" for i in range(lanes)])
                operands = ", ".join(
                    [*(f"float {row[3]['raw_input']}" for row in group), f"float {lower:.6e}", f"float {upper:.6e}"]
                )
                lines = [
                    f'%{prefix}.asm = call {aggregate} asm "'
                    + r"\0A".join(instructions)
                    + f'", "{constraints}"({operands})'
                ]
                lines += [f"%{prefix}.wide{i} = extractvalue {aggregate} %{prefix}.asm, {i}" for i in range(lanes)]
                for i, row in enumerate(group):
                    op, _, _, proof = row
                    replacement = f"{op.result} = trunc i32 %{prefix}.wide{i} to {proof['integer_dtype']}"
                    if i == 0:
                        replacement = ("\n" + indent).join(lines + [replacement])
                    edits.append((op.start, op.end, replacement))
            else:
                for i, row in enumerate(group):
                    op, _, _, proof = row
                    replacement = (
                        f"%{prefix}.float{i} = call float @llvm.roundeven.f32(float {proof['clamped_input']})"
                        f"\n{indent}{op.result} = fptosi float %{prefix}.float{i} to {proof['integer_dtype']}"
                    )
                    edits.append((op.start, op.end, replacement))
            report["routes"].append(
                {
                    "lanes": lanes,
                    "integer_dtype": group[0][3]["integer_dtype"],
                    "bounds": [lower, upper],
                    "raw_inputs": [r[3]["raw_input"] for r in group],
                    "selection": "adjacent typed RNE results; independent earlier same-block inputs",
                    "proofs": [row[3] for row in group],
                }
            )
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    if (
        host_isa == "portable"
        and edits
        and not any(t.text.startswith("@") and _identity(t.text) == "llvm.roundeven.f32" for t in tokens)
    ):
        source += "\ndeclare float @llvm.roundeven.f32(float)\n"
    report["rewritten_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return source, report


def merlin_host_llvm_transform(
    llvm_bin: str | Path, *, host_isa: str, width: int = 2, temporary_prefix: str = "merlin.rne"
):
    """Verify packets and remaining scalar proofs before normal object identity."""
    from .late_quant_rne import rewrite as scalar_rewrite

    llvm_bin = Path(llvm_bin)
    if host_isa != "rv64gc":
        raise ValueError("explicit RV64GC CPU build policy required")

    def transform(source, work):
        source, work = Path(source), Path(work)
        work.mkdir(parents=True, exist_ok=True)
        assembler = llvm_bin / "llvm-as"

        def verify(path, name):
            subprocess.run([str(assembler), str(path), "-o", str(work / name)], check=True)

        verify(source, "source.bc")
        original = source.read_text()
        target, packet = rewrite(original, host_isa=host_isa, width=width)
        target, scalar = scalar_rewrite(
            target, host_isa=host_isa, combine_clamp=True, temporary_prefix=temporary_prefix
        )
        if not packet["routes"]:
            raise ValueError("no independently proved adjacent RNE packet")
        selected = work / "model.ll"
        selected.write_text(target)
        verify(selected, "model.bc")
        native, native_packet = rewrite(original, host_isa="portable", width=width)
        native, native_scalar = scalar_rewrite(native, host_isa="portable", temporary_prefix=temporary_prefix)
        native_path = work / "model.native.ll"
        native_path.write_text(native)
        verify(native_path, "model.native.bc")
        report = {
            "schema": "bounded_rne_basic_block_build_v1",
            "packet": packet,
            "scalar": scalar,
            "native_packet": native_packet,
            "native_scalar": native_scalar,
            "native_oracle_sha256": hashlib.sha256(native.encode()).hexdigest(),
            "llvm_as_sha256": hashlib.sha256(assembler.read_bytes()).hexdigest(),
            "scope": "pre-object explicit CPU legalization; normal Merlin identity/link owner",
        }
        (work / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        return selected

    return transform
