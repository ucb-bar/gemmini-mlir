"""Process-local selection of the shared stock LLVM host lowering transport."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from merlin.llvmlower.lower import lower_model


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    source, work = Path(request['source']), Path(request['work'])
    request['options']['features'] = frozenset(request['options'].get('features') or ())
    source_text = source.read_text()
    # Stock native transport rejects all Python-runner feature names, including
    # source passes already consumed by lower_model. Run the same public exact
    # preparation pass before that seam and retain its realized rewrite record.
    from merlin.llvmlower.tensor_provenance_cse import FEATURE, rewrite_text
    prepared = None
    if FEATURE in request['options']['features']:
        import hashlib
        selected, rewrites = rewrite_text(source_text)
        prepared = dict(schema='merlin.tensor_preparation_cse.v1', rewrites=rewrites,
            source_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
            output_sha256=hashlib.sha256(selected.encode()).hexdigest(),
            binding_stage='caller_before_stock_lowering', hardware_cycles='not_measured')
        request['options']['features'] -= {FEATURE}
        source_text = selected
    from merlin.llvmlower.scalar_window_reduction import FEATURE as WINDOW_FEATURE, prepare_scalar_window_reductions
    window = None
    if WINDOW_FEATURE in request['options']['features']:
        source_text, window = prepare_scalar_window_reductions(source_text, enabled=True)
        request['options']['features'] -= {WINDOW_FEATURE}
    result = lower_model(source_text, work, targets=(), retain_llvm_dialect=False, **request['options'])
    if prepared is not None:
        (work / 'tensor-preparation-cse.json').write_text(json.dumps(prepared, indent=2))
        (work / 'model.tensor-preparations.mlir').write_text(source_text)
        result.stats['tensor_preparation_cse'] = prepared
    if window is not None:
        (work / 'scalar-window-reductions.json').write_text(json.dumps(window, indent=2))
        (work / 'model.scalar-windows.mlir').write_text(source_text)
        result.stats['scalar_window_reductions'] = window
    Path(request['result']).write_text(json.dumps({'ll_path': str(result.ll_path), 'stats': result.stats}, default=str))


if __name__ == '__main__':
    main()
