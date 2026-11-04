"""Share proven immutable tensor preparations before device-region discovery.

Merlin owns structural equivalence, scalar exactness and provenance retention.
Running its existing pass here exposes preparations that would otherwise be
hidden inside distinct opaque device calls before ordinary host preparation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def prepare(source: str, work: Path):
    from merlin.llvmlower import tensor_provenance_cse

    selected, rewrites = tensor_provenance_cse.rewrite_text(source)
    work = Path(work)
    work.mkdir(exist_ok=True)
    (work / 'input.mlir').write_text(source)
    (work / 'selected.mlir').write_text(selected)
    report = dict(schema='submission.source_tensor_preparations.v1',
        binding_stage='before_device_region_discovery', rewrites=rewrites,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        selected_source_sha256=hashlib.sha256(selected.encode()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(tensor_provenance_cse.__file__).read_bytes()).hexdigest(),
        metadata='Merlin retains all merged source provenance',
        source_effects='immutable tensor value semantics; exact supported preparations only',
        numerical_execution='not_established', performance_cycles=None)
    (work / 'selection.json').write_text(json.dumps(report, indent=2))
    return selected, report
