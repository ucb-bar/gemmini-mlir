#!/usr/bin/env python3
"""Four-entrypoint fresh10 xDSL compiler driver."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(ROOT / "compiler_library"))
sys.path.insert(0, str(ROOT))


def write_command_buffer(source, command_buffer, artifact, output):
    """Publish source programs under their exact plan and emitted whole ABI.

    Canonical interface kernels retain the existing builder path. A source
    program instead owns computation in its emitted module and nested device
    regions; its top-level logical command list may correctly be empty.
    """
    params = command_buffer.get("params", {})
    abi = command_buffer.get("kernel_abi", {})
    source_program = (
        "global_program_plan" in params
        or (command_buffer.get("commands") == [] and abi.get("kind") == "whole_program")
    )
    if not source_program:
        from merlin.targetgen.oot_starterkit.cmdbuf import CommandBufferBuilder
        builder = CommandBufferBuilder("gemmini", schema_path=ROOT / "backend/data/command_buffer.schema.json")
        builder._cb = command_buffer
        builder.write(output)
        return

    if abi.get("kind") != "whole_program" or command_buffer.get("declined"):
        raise ValueError("source program publication requires the emitted whole-program ABI without decline")
    from jsonschema import Draft202012Validator
    from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan
    schema = json.loads((ROOT / "backend/data/command_buffer.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(command_buffer)
    check = validate_mixed_program_plan(
        source, command_buffer, artifact,
        schema_path=ROOT / "backend/data/mixed_program_plan.schema.json",
        lowered_abi={"entry_symbol": "kernel_entry", "completion_symbol": "kernel_complete", "completion_asm": "fence"},
    )
    if not check["ok"]:
        raise ValueError("source program publication preflight: " + "; ".join(check["findings"]))
    # Validate and serialize completely before opening the destination. Keep
    # tensor declaration order, which is part of the logical pointer ABI.
    serialized = json.dumps(command_buffer, indent=2)
    Path(output).write_text(serialized)


def main(argv):
    paths = [Path(arg) for arg in argv if not arg.startswith("-")]
    if len(paths) != 1:
        raise ValueError("expected one MLIR input file")
    text = paths[0].read_text()
    if "--verify-diagnostics" in argv:
        if 'merlin_iface.' in text:
            from backend.frontend import parse_source, infer_tensors
            infer_tensors(parse_source(text))
        else:
            from merlin.frontends.linalg_mlir import parse_mlir_text
            parse_mlir_text(text).verify()
        return
    out = next((x.split("=",1)[1] for x in argv if x.startswith("--emit-command-buffer=")), None)
    from backend.program import compile_program
    cb, artifact = compile_program(text)
    if out:
        write_command_buffer(text, cb, artifact, out)
    if "--emit-target-artifact" in argv or not out:
        sys.stdout.write(artifact)



if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:
        print(f"compiler: {type(exc).__name__}: {exc}", file=sys.stderr)
        if "--debug" in sys.argv:
            raise
        sys.exit(1)
