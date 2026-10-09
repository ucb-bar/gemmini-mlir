"""Seal qualified normal whole successor and independently priced M8 evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rne_zero_observer_whole_prepare import (
    CAPTURE,
    CORE,
    FINITE_CONTROL,
    HERE,
    LLVM,
    OUT,
    save,
    sha,
)

UPSTREAM = Path("/scratch/agustin/tmp/merlin-rounded-polynomial-main-20261007")
STOCK = Path(
    "/scratch/agustin/tmp/gemmini-current-profile-20261007/out/rne_zero_observer_stock/stock2082_terminal.json"
)


def main():
    archive = HERE / "docs/perf_records/rne_zero_observer_whole_2076_20261007"
    archive.mkdir(parents=True, exist_ok=False)
    native_path = OUT / "qualification.json"
    target_path = OUT / "whole/target_validation.json"
    upstream_path = OUT / "upstream_reemission.json"
    target = json.loads(target_path.read_text())
    native = json.loads(native_path.read_text())
    upstream = json.loads(upstream_path.read_text())
    stock = json.loads(STOCK.read_text())
    assert (
        target["status"] == "pass"
        and target["spike_full_output_match"]
        and target["torch_allclose"]
        and target["no_fsm"]
    )
    assert native["status"] == "pass" and len(native["context_events"]) == 616
    assert (
        upstream["status"] == "pass"
        and upstream["control_and_selected_lookup_C_byteidentical"]
    )
    assert stock["status"] == "verified_complete_original_M8_pair"
    assert (
        stock["observation"]["candidate_mean_cycles"]
        < stock["observation"]["control_mean_cycles"]
    )
    pins = {}

    def include(path):
        path = Path(path)
        assert path.is_file(), path
        pins[str(path)] = sha(path)

    for path in (
        native_path,
        target_path,
        upstream_path,
        OUT / "whole/build.json",
        STOCK,
    ):
        record = json.loads(path.read_text())
        include(path)
        for name, digest in record.get("pins", {}).items():
            assert sha(name) == digest, name
            pins[name] = digest
    for path in OUT.rglob("*"):
        if path.is_file():
            include(path)
    for path in [
        *HERE.glob("tests/rne_zero_observer*.py"),
        Path(__file__),
        HERE / "AGENTS.md",
        CORE / "src/merlin/llvmlower/bounded_rne_word_cells.py",
        CORE / "src/merlin/llvmlower/source_expression_interval.py",
        UPSTREAM / "out/rne_observer_qualification/qualification.json",
        LLVM / "clang",
        LLVM / "llvm-link",
        LLVM / "opt",
        FINITE_CONTROL / "native/host/output.npy",
        *[p for p in CAPTURE.glob("context_*/*.npy")],
        CAPTURE / "features_v2/contexts.json",
    ]:
        include(path)
    linked = json.loads((OUT / "whole/build.json").read_text())
    for argument in linked["candidate_link_argv"]:
        path = Path(argument)
        if path.is_file():
            include(path)
    for name, field in {
        "elf_path": "elf_sha256",
        "reference_path": "reference_sha256",
        "torch_golden_path": "torch_golden_sha256",
        "spike_console_path": "spike_console_sha256",
        "normal_lower_recipe_path": "normal_lower_recipe_sha256",
        "controlled_link_path": "controlled_link_sha256",
    }.items():
        assert sha(target[name]) == target[field]
        pins[target[name]] = target[field]
    for path in (
        native_path,
        target_path,
        upstream_path,
        OUT / "whole/build.json",
        STOCK,
    ):
        shutil.copyfile(
            path, archive / ("stock2082_terminal.json" if path == STOCK else path.name)
        )
    for name in ("whole_prepare.log", "whole_target.log", "upstream_reclosure.log"):
        include(HERE / name)
        shutil.copyfile(HERE / name, archive / name)
    (archive / "README.md").write_text("""# Exact zero-observer whole successor

Merlin derives the zero-bin preimage and typed all-use/effect/source contract.
The OOT experiment supplies the existing target mode guard and exact source
bindings. The normal host LLVM hook applies only the explicit zero-observer
alternative after the current finite scans; table, rounded source products,
original cold continuation, finishing for all other bins and non-RNE fallback
remain. No workload, provenance or golden value selects compiler legality.

Both native whole arms reproduce every original256000 word and pass the
unchanged Torch gate. All22 actual contexts preserve991232 original i8
observations across616 native rounding/sticky cases. The existing independent
target capsule preserves allfiveFRM/sevensticky cases. Deleted flags and signed
zero effects require the explicit source permission; non-RNE execution retains
the original source and exact observed flags. SourceNaN fptosi poison is not
redefined by this optimization.

The controlled link reproduces stock2076 byte-identically and substitutes
only the normal model.o. All11 nonmodel leaves, including155device bindings,
runtime, inputs, weights, main and startup are frozen. Final strict Spike
reproduces the full original digest, rank0/DONE and zeroFSM. The inherited
build marker is nonunique; exact ELF SHA identifies the candidate.

Stock2082 priced one complete M8 helper at6.1565% fewer cycles despite1.2994%
more instructions. The whole candidate likewise has more retired instructions;
its hardware cycles are UNKNOWN. Neither the helper result nor its zero density
predicts whole performance. Existing full-cell and one-sided alternatives
remain negative/refused. The13bit storage census remains uncompiled/unpriced.

Published main c0f40f8 independently re-emits both normal lookup bodies and
all22native/target scan wrappers byte-identically. This is a normal host hook
on frozen upstream LLVM plus controlled whole link, not a fresh recapture.
Root owns the queue and promotion decision; this packet submits nothing.
""")
    for path in archive.iterdir():
        include(path)
    control_instructions = 121135311
    candidate_instructions = target["functional_instructions_not_hardware_cycles"]
    packet = {
        "schema": "source_exact_zero_observer_normal_whole_review_v1",
        "status": "pass",
        "candidate_ELF_path": target["elf_path"],
        "candidate_ELF_sha256": target["elf_sha256"],
        "standard_reference_validation": str(target_path),
        "standard_reference_validation_sha256": sha(target_path),
        "control_job": 2076,
        "control_cycles": 380396343,
        "baseline2076_ELF_byteidentical": True,
        "only_changed_linked_object": "model.o",
        "unchanged_nonmodel_leaves": 11,
        "all155devicebindings": True,
        "original_tokens": 8,
        "original_layers": 22,
        "original_compiled_output_words": 256000,
        "original_source_all_words_exact": True,
        "original_Torch_gate": {"atol": 0.03125, "rtol": 0.02, "pass": True},
        "strict_final_ELF_noFSM": True,
        "native_original22context_i8_words": 991232,
        "native_mode_sticky_cases": 616,
        "source_producer_chains": 44,
        "source_helpers": 22,
        "upstream_generic_topic": upstream["upstream_head"],
        "upstream_actual_source_reemission": str(upstream_path),
        "control_retired_instructions": control_instructions,
        "candidate_retired_instructions": candidate_instructions,
        "retired_fraction_change": candidate_instructions / control_instructions - 1,
        "functional_retired_counts_are_not_hardware_cycles": True,
        "complete_M8_stock2082": {
            "receipt": str(STOCK),
            "receipt_sha256": sha(STOCK),
            "control_mean_cycles": stock["observation"]["control_mean_cycles"],
            "candidate_mean_cycles": stock["observation"]["candidate_mean_cycles"],
            "saving_fraction": stock["saving_fraction"],
        },
        "whole_hardware_cycles": "UNKNOWN",
        "prospective_model_rank": "UNKNOWN",
        "physical_memory_and_whole_composition_unpriced": True,
        "source_permissions": "RNE/gradual/nontrapping/deleted-fflags-and-signed-zero-unobserved; explicit original source fallback otherwise",
        "scope": "Normal source-bound host hook on immutable upstream LLVM and native companion, controlled final link; no fresh whole upstream recapture or device change.",
        "whole_submission": "ROOT review/sole queue owner required",
        "primary_capsule_and_stock_receipts_unchanged": True,
        "pins": pins,
        "token_usage_available": False,
    }
    save(archive / "receipt.json", packet)
    print(
        "SEALED_ZERO_OBSERVER_WHOLE",
        len(pins),
        sha(archive / "receipt.json"),
        flush=True,
    )


if __name__ == "__main__":
    main()
