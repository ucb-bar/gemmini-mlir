"""Process-local selection of the shared stock LLVM host lowering transport."""
from __future__ import annotations
import json
from pathlib import Path
import sys

# The public driver adds these paths in its own process. A fresh interpreter
# does not inherit sys.path, and broker snapshots cannot use a developer's
# workspace PYTHONPATH. Resolve the same submitted library beside this worker.
_submission_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_submission_root / 'compiler_library'))
sys.path.insert(0, str(_submission_root))
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
    cse_source = source_text
    from backend.host_preparation import prepare
    source_text, layout_preparation = prepare(source_text, work,
        request.get('host_preparation', 'baseline'))
    result = lower_model(source_text, work, targets=(), retain_llvm_dialect=False, **request['options'])
    if layout_preparation is not None:
        import hashlib
        recipe = work / 'lowering_recipe.json'
        layout_preparation['lowering_recipe'] = dict(path=str(recipe),
            sha256=hashlib.sha256(recipe.read_bytes()).hexdigest())
        layout_preparation['returned_llvm_ir'] = dict(path=str(result.ll_path),
            sha256=hashlib.sha256(result.ll_path.read_bytes()).hexdigest())
        (work / 'host-preparation.json').write_text(json.dumps(layout_preparation, indent=2))
        result.stats['adapter_host_preparation'] = layout_preparation
    if prepared is not None:
        (work / 'tensor-preparation-cse.json').write_text(json.dumps(prepared, indent=2))
        (work / 'model.tensor-preparations.mlir').write_text(cse_source)
        result.stats['tensor_preparation_cse'] = prepared
    Path(request['result']).write_text(json.dumps({'ll_path': str(result.ll_path), 'stats': result.stats}, default=str))


if __name__ == '__main__':
    main()
