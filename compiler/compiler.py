#!/usr/bin/env python3
"""Four-entrypoint fresh10 xDSL compiler driver."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "compiler_library"))
sys.path.insert(0, str(ROOT))


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
    from backend.python_dependencies import ensure_selected_solver
    dependency = ensure_selected_solver()
    from backend.program import compile_program
    cb, artifact = compile_program(text)
    cb.setdefault('metadata', {})['compiler_solver'] = dependency
    if out:
        from merlin.targetgen.oot_starterkit.cmdbuf import CommandBufferBuilder
        builder = CommandBufferBuilder("gemmini", schema_path=ROOT / "backend/data/command_buffer.schema.json")
        builder._cb = cb
        builder.write(out)
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
