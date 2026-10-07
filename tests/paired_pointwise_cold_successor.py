"""Retain the already-qualified source continuation placement in a fresh pair.

The first compound was numerically exact but inherited the older calibration's
inline source evaluator. This successor restores the current source-derived
continuation policy; it preserves the first artifacts and changes no table,
source operation, input, coefficient, quantizer or device implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import paired_pointwise_compound as compound
import paired_pointwise_prepare as preparation
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_continuation_outline import (
    SourceContinuationBinding,
    outline_source_continuations,
)
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    find_closed_scalar_i8_observers,
)


def main():
    root = (
        preparation.ROOT
        / "out/artifacts/probes/paired-pointwise-cold-successor-20261007"
    )
    root.mkdir(parents=True, exist_ok=False)
    assets = root / "assets"
    assets.mkdir()
    effects = IntervalEffectContract(True, True, True, True, True)
    proofs, refused = find_closed_scalar_i8_observers(
        parse_mlir_text((preparation.SOURCE / "source.mlir").read_text()),
        effects=effects,
    )
    assert len(proofs) == 1 and not refused
    original = preparation.ASSETS / "source_activation.ll"
    activation, continuation = outline_source_continuations(
        original.read_text(),
        bindings=(
            SourceContinuationBinding("source_activation", proofs[0].expression),
        ),
        expected_source_sha256=preparation.sha(original),
        effects=effects,
    )
    assert activation == original.read_text().replace(
        "alwaysinline", "noinline cold", 1
    )
    (assets / "source_activation.ll").write_text(activation)
    (assets / "source_quantize.ll").write_bytes(
        (preparation.ASSETS / "source_quantize.ll").read_bytes()
    )
    preparation.OUT = root / "consumer"
    preparation.ASSETS = assets
    preparation.main()
    compound.OUT = root / "compound"
    compound.CONSUMER = preparation.OUT
    compound.main()
    qualification = {
        "schema": "source_continuation_preserved_sibling_successor_v1",
        "status": "pass",
        "first_variant": str(
            compound.ROOT
            / "out/artifacts/probes/paired-pointwise-compound-20261007/qualification.json"
        ),
        "first_variant_qualification": "Numerical/source/guard PASS; retired22.48% worse. No hardware or promotion.",
        "change": "Existing typed-source continuation API restores noinline/cold source placement; all source scalar DAG/constants/rounding/table/certificates unchanged.",
        "continuation": continuation,
        "native": str(preparation.OUT / "qualification.json"),
        "target": str(compound.OUT / "qualification.json"),
        "pins": {
            str(p): preparation.sha(p)
            for p in [
                Path(__file__),
                Path(preparation.__file__),
                Path(compound.__file__),
                original,
                *root.rglob("*"),
            ]
            if p.is_file()
        },
        "stock_cycles": "UNKNOWN",
        "whole_performance": "UNKNOWN",
    }
    (root / "qualification.json").write_text(json.dumps(qualification, indent=2) + "\n")
    print("COLD_SOURCE_COMPOUND_SUCCESSOR_PASS", flush=True)


if __name__ == "__main__":
    main()
