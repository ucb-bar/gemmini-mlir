"""Seal two immutable early-zero screens and the single mixed-cost candidate."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rne_zero_product_builtin_capsule import CORE, HERE, OUT, save, sha


def main():
    archive = HERE / "docs/perf_records/rne_zero_product_screens_20261007"
    archive.mkdir(parents=True, exist_ok=False)
    word = OUT.parent / "rne-zero-product-observer-M8-20261007"
    word_q = json.loads((word / "qualification.json").read_text())
    builtin_q = json.loads((OUT / "qualification.json").read_text())
    assert word_q["status"] == builtin_q["status"] == "pass"
    pins = {}
    historical = []
    for name, case, receipt in (("word", word, word_q), ("builtin", OUT, builtin_q)):
        destination = archive / name
        destination.mkdir()
        for path, expected in receipt["pins"].items():
            if sha(path) != expected:
                assert (
                    name == "word"
                    and Path(path).name == "source_expression_interval.py"
                )
                historical.append(
                    {
                        "path": path,
                        "original_sha256": expected,
                        "current_sha256": sha(path),
                        "scope": "First word-only primary remains immutable. The later explicit builtin emission was added to the same private module after that run; all first generated C/IR/object/ELF/logs remain exact. This source pin is retained unresolved; no word-only promotion.",
                    }
                )
            else:
                pins[path] = expected
        for path in case.rglob("*"):
            if path.is_file():
                pins[str(path)] = sha(path)
        for filename in (
            "qualification.json",
            "lookup.c",
            "candidate.dump",
            "execution_features.json",
            "fp_dependencies.json",
        ):
            shutil.copyfile(case / filename, destination / filename)
        for scope in ("all_modes", "timing"):
            target = destination / scope
            target.mkdir()
            for filename in (
                "qualification.json",
                "nofsm_audit.json",
                "spike.stdout",
                "spike.stderr",
            ):
                shutil.copyfile(case / scope / filename, target / filename)
    shutil.copytree(OUT / "measured_core", archive / "builtin/measured_core")
    for path in [
        Path(__file__),
        HERE / "tests/rne_zero_product_observer_capsule.py",
        HERE / "tests/rne_zero_product_builtin_capsule.py",
        CORE / "src/merlin/llvmlower/source_expression_interval.py",
        CORE / "src/merlin/llvmlower/bounded_rne_word_cells.py",
        CORE / "merlin/tests/ir/test_bounded_rne_word_cells.py",
        CORE / "src/merlin/llvmlower/AGENT.md",
        *CORE.glob("out/artifacts/probes/zero-product-observer-20261007/*"),
    ]:
        if path.is_file():
            pins[str(path)] = sha(path)
    tests = (
        CORE
        / "out/artifacts/probes/zero-product-observer-20261007/source_tests_builtin.log"
    )
    assert "123 passed" in tests.read_text()
    structure = (
        CORE
        / "out/artifacts/probes/zero-product-observer-20261007/structure_builtin.log"
    )
    assert "All structure checks passed" in structure.read_text()
    for path in (tests, structure):
        shutil.copyfile(path, archive / path.name)
    classes = json.loads((OUT / "execution_features.json").read_text())
    assert classes["actual_entry_counts"]["cells_M8_rne"] == 2
    assert classes["actual_entry_counts"]["builtinzero_M8_rne"] == 3
    timing = json.loads((OUT / "timing/qualification.json").read_text())
    assert [int(row["id"]) for row in timing["rows"]] == [0, 1, 1, 0]
    for name, field in (("elf_path", "elf_sha256"),):
        assert sha(timing[name]) == timing[field]
    prelabel = {
        "schema": "zero_product_complete_M8_prospective_v1",
        "control": "Current zero-bin complete source M8 helper, the same generic mechanism used in whole2085",
        "candidate": "Only source-scale exact first-product zero threshold with explicit finite abs/max builtin materialization",
        "hypothesis": "Early zero admission removes63502 source finishing FMUL executions but adds abs/max operations, moves and branching. Different FP dependency topology and complete issue cost can outweigh instruction counts in either direction.",
        "prediction": {
            "winner": "UNKNOWN",
            "cycles": "UNKNOWN",
            "reason": "No sealed model prices this table/branch/abs-max/FP-mul/continuation mixed path. Parent stock2082 reversed count ordering; no fitted transfer or instruction conversion is justified.",
        },
        "unchanged": [
            "Both source scale scans",
            "Original source table512KiB and data",
            "Original216 cold source calls per helper",
            "Original finishing on failed certificate",
            "Unsupported rounding original source",
            "Full45056 stores/guards/commonaddresses and frame",
        ],
        "new_storage_bytes": 0,
        "physical_table_memory_and_cross_call_order": "UNKNOWN",
        "all22_and_whole_transfer": "UNKNOWN; only original firstM8 compiled here",
        "no_whole_forecast": True,
    }
    save(archive / "prospective_cost_declaration.json", prelabel)
    (archive / "README.md").write_text("""# Source-exact early zero admission

The original observer is RNE of two separately rounded binary32 products.
Merlin derives the largest finite first-product word whose original second
product enters the ties-even zero preimage, using exact rational arithmetic and
the retained typed source factor. One rounded product of max endpoint magnitude
and up magnitude bounds both original first products. Only a certified zero
omits the finishing operations; all other source products and continuations
remain. No up*scale reassociation, output approximation or input/golden selector.

The first integer-word materialization adds10.45% complete retired instructions.
It has no hardware label. A bounded orthogonal correction expresses the same
finite abs/max theorem via generic compiler builtins; OOT actual compiled RV64GC
code contains the qualified instructions, with no extra library/ABI imports.
The builtin version adds4.12% retired instructions while removing63502 finishing
FMULs. Cycle ranking remains UNKNOWN. Both complete actual M8 gates pass all
original45056 words, five rounding modes, seven sticky states, guards, immutable
inputs and executable noFSM.123 independent/source/native/refusal cases and
structural ownership gates pass. No all22 compiled or whole successor exists.

Source/table/closed-observer matching and explicit nontrapping, gradual, RNE,
unobserved deleted flags/signs are mandatory; unsupported modes retain source.
The builtin magnitude choice is explicit/default-off and requires actual target
emission closure. SourceNaN poison semantics are not replaced. Parent owns
stock admission after review; this packet submits nothing.

The first word-only primary has one historical source-module pin drift after
the separate builtin choice was added. It is retained explicitly unresolved;
all first generated code/binaries/logs remain pinned and no promotion uses it.
The builtin candidate has immutable exact core snapshots and complete pins.
Initial missing tool environment and unsupported source factor tests are
preserved diagnostics; final123 configured tests are the qualification.
""")
    for path in archive.rglob("*"):
        if path.is_file():
            pins[str(path)] = sha(path)
    counts = {
        i: sum(
            int(row["instructions"]) for row in timing["rows"] if int(row["id"]) == i
        )
        / 2
        for i in (0, 1)
    }
    record = {
        "schema": "source_exact_zero_product_complete_screens_v1",
        "status": "qualified_complete_M8_pending_cycle_price",
        "core_path": str(CORE),
        "core_commit": "03d8d101f",
        "core_parent": "c0f40f8f8d100b841c26fe8bbd6c09e79b6de17c",
        "typed_source_observation_policy": "Existing complete source i8 observer, source scale literal, separate rounded first/second products; exact zero only, explicit effects/unsupported-mode fallback",
        "control_instructions": counts[0],
        "builtin_instructions": counts[1],
        "builtin_retired_fraction_change": counts[1] / counts[0] - 1,
        "word_instructions": 1906769,
        "cycles": "UNKNOWN",
        "stock_or_whole_promotion": False,
        "all_mode_qualification": str(OUT / "all_modes/qualification.json"),
        "short_pair": {
            "elf": timing["elf_path"],
            "elf_sha256": timing["elf_sha256"],
            "qualification": str(OUT / "timing/qualification.json"),
            "prelabel": str(archive / "prospective_cost_declaration.json"),
            "protocol": "Unchanged INTEGER_RESULT_SOURCE_GATE, POINTERS,4ABBA ROW and PASS original45056i8",
        },
        "original_primary_receipts_unchanged": True,
        "historical_pin_drift": historical,
        "pins": pins,
        "token_usage_available": False,
    }
    save(archive / "receipt.json", record)
    print(
        "ZERO_PRODUCT_SCREEN_SEALED",
        len(pins),
        sha(archive / "receipt.json"),
        flush=True,
    )


if __name__ == "__main__":
    main()
