#!/usr/bin/env python3
"""Four executable compiler entrypoints over the selected public grammar."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'compiler_library'))
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-diagnostics', action='store_true')
    ap.add_argument('--convert-iface-to-gemmini', action='store_true')
    ap.add_argument('--convert-gemmini-to-llvm', action='store_true')
    ap.add_argument('--emit-command-buffer')
    ap.add_argument('--emit-target-artifact', action='store_true')
    ap.add_argument('input_mlir')
    opts = ap.parse_args()
    source = Path(opts.input_mlir).read_bytes().decode('utf-8')
    # Original tensor SSA can contain bounded dynamic intermediates.  Keep it
    # intact for the graph/control lowering instead of first forcing every
    # intermediate through the static interface command-buffer parser.
    if 'merlin_iface.' not in source:
        from graph import lower_graph
        cb, lowered = lower_graph(source)
        if opts.verify_diagnostics:
            return 0
        if opts.emit_command_buffer:
            Path(opts.emit_command_buffer).write_text(json.dumps(cb, indent=2) + '\n')
        if not opts.emit_command_buffer or opts.emit_target_artifact:
            sys.stdout.write(lowered)
        return 0
    from frontend import parse_source
    cb = parse_source(source)
    if opts.verify_diagnostics:
        return 0
    # The selected target declares no half-up requantization route. Keep this
    # semantic refusal explicit and effect-free, preserving declared outputs.
    if any('requant' in c.get('attributes', {}).get('epilogue', []) for c in cb['commands']):
        cb['commands'] = []
        cb['declined'] = {'op': 'requant', 'reason': 'selected target readout applies half-even scaling; no admitted half-up requantization stage'}
        lowered = 'builtin.module {}\n'
    elif not opts.emit_command_buffer or opts.emit_target_artifact:
        from backend import compile_buffer
        lowered = compile_buffer(cb)
    if opts.emit_command_buffer:
        Path(opts.emit_command_buffer).write_text(json.dumps(cb, indent=2) + '\n')
    if not opts.emit_command_buffer or opts.emit_target_artifact:
        sys.stdout.write(lowered)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'gemmini compiler: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
