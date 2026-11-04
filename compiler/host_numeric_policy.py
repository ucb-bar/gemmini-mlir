"""Stock transport for Merlin's independently checked bounded host rounding.

No arithmetic is implemented here. The shared automatic-host legalization proves
its bounded quantization rewrite; automatic_rounding recognizes and checks that
exact expression before selecting the ordinary LLVM roundeven intrinsic. Target
instruction selection remains LLVM's RV64GC responsibility, without inline asm.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import automatic_host, automatic_rounding, fusion_guard
from host_policy import run_stock_pipeline as _incumbent


def selection():
    from merlin.compile.tuning_knobs import knob
    return knob('host_numeric_optimization', default='off', values=('off', 'bounded_rne'),
                root=Path(__file__).resolve().parent)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def run_stock_pipeline(source, lowered, passes, *, tool, invoke, print_module, report=None):
    """Drop-in stock host pipeline, preserving the existing fusion selection.

    Split at the last actual elementwise fusion stage, still before buffering.
    Original source and attribution survive; callers add task ownership to the
    resulting LLVM module just as on the incumbent route.
    """
    selected = selection()
    if selected == 'off':
        return _incumbent(source, lowered, passes, tool=tool, invoke=invoke,
                         print_module=print_module, report=report)
    locations = [i for i, stage in enumerate(passes) if stage == fusion_guard.FUSE_PASS]
    if not locations:
        raise ValueError('bounded host rounding requires an actual elementwise fusion stage')
    split = locations[-1] + 1
    source = Path(source)
    lowered = Path(lowered)
    original = source.read_bytes()
    fused = source.parent / 'numeric-fused.mlir'
    fusion_report = {}
    _incumbent(source, fused, passes[:split], tool=tool, invoke=invoke,
              print_module=print_module, report=fusion_report)
    before = parse_mlir_text(fused.read_text())
    # Read the same source matcher used by the library legalization. Report
    # refusals from actual scalar SSA; sample scales grant no range premise.
    from merlin.llvmlower.quant_round_chain import _match_chain
    refusals = Counter()
    eligible_chains = 0
    for operation in before.walk():
        if operation.name != 'arith.fptosi':
            continue
        admission = _match_chain(operation)
        if isinstance(admission, str):
            refusals[admission] += 1
        else:
            eligible_chains += 1
    prepared, preparation = automatic_host.optimize_host_quantization(
        before, numerical_policy='source-f32-nearest-ties-even-preserve-stage-order')
    # This library matcher independently reconstructs and verifies the complete
    # bounded expression. It does not relax FP flags or commute source scaling.
    compacted = automatic_rounding.rewrite_rounding(prepared)
    # An expanded chain without its compact readout is not the selected recipe.
    # Keep the original fused program if a library code budget or guard refuses.
    realized = compacted
    if compacted < preparation['rewrites']:
        prepared = before
        realized = 0
    prepared.verify()
    chosen = source.parent / 'numeric-selected.mlir'
    chosen.write_text(print_module(prepared))
    invoke([tool, chosen, '--pass-pipeline=builtin.module(' + ','.join(passes[split:]) + ')',
            '--mlir-print-op-generic', '-o', lowered], stage='bounded host rounding lowering')
    if source.read_bytes() != original:
        raise ValueError('host numerical optimization modified original source')
    if report is not None:
        # Retain the fusion report in its original shape for current consumers.
        report.update(fusion_report)
        if fusion_report:
            report['fusion_pipeline_output_sha256'] = fusion_report['lowered_sha256']
            report['lowered_sha256'] = _sha(lowered.read_bytes())
        report['numeric_optimization'] = {
            'schema': 'merlin.stock_bounded_host_rounding.v1',
            'selection': selected,
            'mechanisms': ['merlin.llvmlower.automatic_host.optimize_host_quantization',
                           'merlin.llvmlower.automatic_rounding.rewrite_rounding'],
            'source_sha256': _sha(original), 'fused_sha256': _sha(fused.read_bytes()),
            'selected_sha256': _sha(chosen.read_bytes()),
            'lowered_sha256': _sha(lowered.read_bytes()),
            'preparation': preparation, 'compacted_bounded_rounds': compacted,
            'realized_bounded_rounds': realized,
            'chain_admission': {'eligible': eligible_chains, 'refusals': dict(refusals)},
            'library_sha256': {Path(module.__file__).name: _sha(Path(module.__file__).read_bytes())
                               for module in (automatic_host, automatic_rounding)},
            'target': 'rv64gc scalar; ordinary LLVM instruction selection',
            'runtime_cycles': 'not_measured',
        }
