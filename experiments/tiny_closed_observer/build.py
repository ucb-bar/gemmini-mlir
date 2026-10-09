"""Reproduce an authenticated source observer experiment, not a model selector."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path: Path, expected: str) -> Path:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Qualification input changed: {path}")
    return path


def reclose(record: dict, overrides: dict[str, Path]) -> None:
    for name, expected in record["pins"].items():
        checked(overrides.get(name, Path(name)), expected)


def source_outputs(files: dict[str, Path], source_receipt: dict) -> dict:
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import (
        source_continuation_outline,
        source_expression_interval_llvm,
    )
    from merlin.llvmlower import source_expression_interval as interval
    from merlin.llvmlower.late_quant_rne import _functions, _tokens

    for module in (
        interval,
        source_expression_interval_llvm,
        source_continuation_outline,
    ):
        path = Path(inspect.getsourcefile(module)).resolve()
        expected = [
            row
            for row in source_receipt["actual_core_ast_identity"]
            if Path(row["path"]).name == path.name
        ]
        if len(expected) != 1:
            raise ValueError(f"Generic helper is not qualified: {path}")
        checked(path, expected[0]["sha256"])
    binding = json.loads(files["source_binding"].read_text())
    effects = interval.IntervalEffectContract(**binding["effects"])
    proofs, refused = interval.find_closed_scalar_i8_observers(
        parse_mlir_text(files["typed_source"].read_text()), effects=effects
    )
    if refused or len(proofs) != len(binding["whole_helper_guards"]):
        raise ValueError("Typed observer closure differs from the accepted source")
    if len({p.expression.canonical_sha256 for p in proofs}) != 1:
        raise ValueError("This experiment requires one shared source expression")
    table = interval.build_source_interval_table(
        proofs[0].expression,
        effects=effects,
        leading_bits=binding["leading_bits"],
        max_table_bytes=binding["readonly_bytes"],
    )
    if (
        table.data != files["table_bytes"].read_bytes()
        or table.sha256 != binding["source_table_sha256"]
    ):
        raise ValueError("Source-derived interval table differs")
    table_llvm = interval.emit_immutable_bytes_llvm(
        table.data, symbol="source_interval_table", alignment=64
    )
    lookup = interval.emit_source_interval_i8_lookup(
        table_name="source_interval_table",
        activation_name="source_activation",
        quantizer_name="source_quantize",
        lookup_name="source_lookup_activation",
        leading_bits=table.leading_bits,
    )
    activation, continuation = source_continuation_outline.outline_source_continuations(
        files["activation_input"].read_text(),
        bindings=(
            source_continuation_outline.SourceContinuationBinding(
                "source_activation", proofs[0].expression
            ),
        ),
        expected_source_sha256=digest(files["activation_input"]),
        effects=effects,
    )
    for actual, role in (
        (table_llvm, "expected_table_llvm"),
        (lookup, "expected_lookup_c"),
        (activation, "expected_activation_llvm"),
    ):
        if actual != files[role].read_text():
            raise ValueError(f"Regenerated source differs: {role}")
    if continuation != binding["continuation"]:
        raise ValueError("Source continuation proof differs")
    source = files["float_source"].read_text()
    changed, report = source_expression_interval_llvm.rewrite_source_interval_i8_lookup(
        source,
        proofs=proofs,
        table=table,
        lookup_symbol="source_lookup_activation",
        effects=effects,
    )
    if report != binding["source_bindings"]:
        raise ValueError("Typed integer observer routing differs")
    tokens, new_tokens = _tokens(source), _tokens(changed)
    bodies, new_bodies = _functions(tokens), _functions(new_tokens)
    indices = {row["function_body_index"] for row in report["routes"]}
    for guard in binding["whole_helper_guards"]:
        found = 0
        for index in indices:
            body = bodies[index]
            start = max(
                t.start
                for t in tokens
                if t.text == "define" and t.start < body[0].start
            )
            stop = next(
                t.end for t in tokens if t.text == "}" and t.start > body[-1].end
            )
            old = source[start:stop]
            if not old.startswith(
                "define internal void @" + guard["original_symbol"] + "("
            ):
                continue
            body = new_bodies[index]
            start = max(
                t.start
                for t in new_tokens
                if t.text == "define" and t.start < body[0].start
            )
            stop = next(
                t.end for t in new_tokens if t.text == "}" and t.start > body[-1].end
            )
            new = changed[start:stop]
            if (
                hashlib.sha256(old.encode()).hexdigest()
                != guard["original_body_sha256"]
                or hashlib.sha256(new.encode()).hexdigest()
                != guard["candidate_body_sha256"]
            ):
                raise ValueError("Original/candidate helper body differs")
            found += 1
        if found != 1:
            raise ValueError("Original helper identity is ambiguous")
    return {
        "table.ll": table_llvm,
        "lookup.c": lookup,
        "source_activation.ll": activation,
        "integer_routes.ll": changed,
        "proof.json": json.dumps(
            {"source_bindings": report, "continuation": continuation}, indent=2
        )
        + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--emit-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(
            "Output must not already exist; historical artifacts are immutable"
        )
    inputs_path = args.inputs.resolve()
    inputs = json.loads(inputs_path.read_text())
    if inputs["schema"] != "tiny_closed_observer_reproduction_inputs_v1":
        raise ValueError("Wrong explicit-input experiment schema")
    files = {
        role: checked(Path(row["path"]), row["sha256"])
        for role, row in inputs["files"].items()
    }
    overrides = {
        name: (inputs_path.parent / replacement).resolve()
        for name, replacement in inputs["path_overrides"].items()
    }
    records = {
        role: json.loads(files[role].read_text())
        for role in ("qualification", "source_reemission", "preparation", "whole_build")
    }
    for record in records.values():
        reclose(record, overrides)
    for row in inputs["source_snapshots"]:
        checked(inputs_path.parent / row["packaged"], row["sha256"])
    core = Path(inputs["core"]).resolve()
    sys.path.insert(0, str(core / "src"))
    sources = source_outputs(files, records["source_reemission"])
    build = records["whole_build"]
    if (
        not build["baseline_byte_exact"]
        or build["only_changed_linked_object"] != "model.o"
    ):
        raise ValueError("Controlled final-link closure is not applicable")
    reference = json.loads(files["reference_validation"].read_text())
    if (
        digest(files["golden"]) != reference["torch_golden_sha256"]
        or digest(files["original_output"]) != reference["reference_sha256"]
        or not reference["all_original_compiled_words_exact"]
    ):
        raise ValueError("Original numerical validation is not bound")
    output.mkdir(parents=True)
    for name, contents in sources.items():
        (output / name).write_text(contents)
    commands = []
    if not args.emit_only:
        from mlir_oot.no_fsm_audit import audit_elf

        for arm, source_role, link_role, expected_hash in (
            (
                "control",
                "control_llvm",
                "baseline_reproduction_argv",
                build["baseline_elf_sha256"],
            ),
            (
                "candidate",
                "candidate_llvm",
                "candidate_link_argv",
                build["candidate_elf_sha256"],
            ),
        ):
            destination = output / arm
            destination.mkdir()
            original_source = files[source_role]
            expected_object = original_source.parent.parent / "model.o"
            matching = [
                command
                for command in records["preparation"]["commands"]
                if "-c" in command and command[-1] == str(expected_object)
            ]
            if len(matching) != 1:
                raise ValueError("Target model compilation is ambiguous")
            command = matching[0]
            if digest(Path(command[0])) != digest(files["clang"]):
                raise ValueError("Target compiler differs from the accepted engine")
            command = [
                str(files["clang"]),
                *command[1:-1],
                str(destination / "model.o"),
            ]
            subprocess.run(command, check=True)
            commands.append(command)
            if digest(destination / "model.o") != digest(expected_object):
                raise ValueError("Recompiled model object differs")
            original = build[link_role]
            if digest(Path(original[0])) != digest(files["gcc"]):
                raise ValueError("Controlled linker differs")
            linked_objects = [
                value for value in original if Path(value).name == "model.o"
            ]
            if len(linked_objects) != 1 or digest(Path(linked_objects[0])) != digest(
                expected_object
            ):
                raise ValueError(
                    "Selected model object does not occur once in final link"
                )
            command = [
                str(destination / "model.o") if value == linked_objects[0] else value
                for value in original
            ]
            command[0], command[-1] = str(files["gcc"]), str(destination / "model.elf")
            subprocess.run(command, check=True)
            commands.append(command)
            elf = destination / "model.elf"
            if (
                digest(elf) != expected_hash
                or audit_elf(elf.read_bytes())["status"] != "pass"
            ):
                raise ValueError(
                    "Final executable identity or instruction policy differs"
                )
    result = {
        "schema": "tiny_closed_observer_reproduction_v1",
        "scope": inputs["scope"],
        "source_policy_reemitted": True,
        "source_routes": len(
            json.loads(sources["proof.json"])["source_bindings"]["routes"]
        ),
        "compiled_control_candidate_identity": not args.emit_only,
        "fresh_whole_execution": False,
        "prior_whole_numeric_gate_bound_by_exact_implementation_identity": not args.emit_only,
        "hardware_measurement": "NONE",
        "inputs": str(inputs_path),
        "inputs_sha256": digest(inputs_path),
        "explicit_historical_path_overrides": inputs["path_overrides"],
        "commands": commands,
        "pins": {
            str(path): digest(path) for path in output.rglob("*") if path.is_file()
        },
    }
    (output / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("source_routes", "compiled_control_candidate_identity")
            }
        )
    )


if __name__ == "__main__":
    main()
