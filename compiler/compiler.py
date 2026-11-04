#!/usr/bin/env python3
"""Four-entrypoint fresh10 xDSL compiler driver."""
from pathlib import Path
import json
import os
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
    # Receipts remain in emitted metadata. Keep the CLI's diagnostic prefix
    # available for the actual error when an admission report is not requested.
    os.environ.setdefault('MERLIN_SCHEDULING_ADMISSION_REPORT', '0')
    from backend.runtime_bootstrap import prepare_z3
    prepare_z3()
    from backend.program import compile_program
    cb, artifact = compile_program(text)
    if out:
        from merlin.targetgen.oot_starterkit.cmdbuf import CommandBufferBuilder
        complete_program = cb.get('kernel_abi', {}).get('kind') == 'whole_program'
        if complete_program and not cb.get('commands'):
            from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan
            check = validate_mixed_program_plan(text, cb, artifact,
                schema_path=ROOT / 'backend/data/mixed_program_plan.schema.json',
                lowered_abi={'entry_symbol':'kernel_entry',
                    'completion_symbol':'kernel_complete','completion_asm':'fence'})
            if not check['ok']:
                raise ValueError('complete-program serialization preflight: ' + '; '.join(check['findings']))
        builder = CommandBufferBuilder("gemmini", schema_path=ROOT / "backend/data/command_buffer.schema.json",
            allow_empty_commands=complete_program)
        builder._cb = cb
        builder.write(out)
    if "--emit-target-artifact" in argv or not out:
        sys.stdout.write(artifact)



if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:
        print(f"compiler: {type(exc).__name__}: {exc}", file=sys.stderr)
        from backend.runtime_bootstrap import missing_z3_error, dependency_diagnostic
        if missing_z3_error(exc):
            print('compiler dependency context: ' + json.dumps(dependency_diagnostic(), sort_keys=True), file=sys.stderr)
        if "--debug" in sys.argv:
            raise
        sys.exit(1)
