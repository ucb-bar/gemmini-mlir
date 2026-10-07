"""Immutable complete-helper packet, with final source and named-storage closure."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import source_interval_hierarchy_census as census
import source_interval_hierarchy_packet as packet
import source_interval_hierarchy_target as target
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.bounded_rne_word_cells import prepare_bounded_rne_word_cells
from merlin.llvmlower.scaled_integer_finite_llvm import bind_finite_scale_helpers
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    emit_immutable_bytes_llvm,
    find_closed_scalar_i8_observers,
)
from merlin.llvmlower.source_interval_hierarchy import (
    build_source_interval_hierarchy,
    emit_source_interval_i8_hierarchy,
)
from merlin.perf.layer_bench import build_program
from merlin.runtime.backends import base as backends
from merlin.targetgen.runtime_build import derived_link_script

from mlir_oot.no_fsm_audit import audit_elf

OUT = census.HERE / "docs/perf_records/source_interval_hierarchy_M8_20261007"
RAW = (
    census.HERE
    / "out/artifacts/probes/source-interval-hierarchy-source-reclosure-20261007"
)


def fnv(data):
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return format(value, "x")


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    effects = IntervalEffectContract(True, True, True, True, True)
    proofs, refused = find_closed_scalar_i8_observers(
        parse_mlir_text(census.TYPED.read_text()), effects=effects
    )
    assert len(proofs) == 22 and not refused
    routes = json.loads((target.SOURCE / "source_binding.json").read_text())["routes"]
    helpers = bind_finite_scale_helpers(
        (target.SOURCE / "source.ll").read_text(),
        routes=routes,
        observers=proofs,
        effects=effects,
        immutable_inputs=True,
        fresh_disjoint_output=True,
    )
    keys = {(r["source_expression_sha256"], r["quant_factor_bits"]) for r in routes}
    selected = tuple(
        p
        for p in proofs
        if (p.expression.canonical_sha256, p.quant_factor_bits) in keys
    )
    hierarchy = build_source_interval_hierarchy(
        selected[0].expression,
        effects=effects,
        partition_bits=(13, 16),
        max_table_bytes=576 * 1024,
    )
    code = emit_source_interval_i8_hierarchy(
        hierarchy,
        table_names=("hierarchy_coarse_table", "table_b16"),
        activation_name="cells_source_activation",
        quantizer_name="cells_quantize",
        lookup_name="hierarchy_lookup_activation",
        observers=selected,
        finite_inputs=helpers,
        zero_observer_cells=tuple(
            prepare_bounded_rne_word_cells(p, effects=effects) for p in selected
        ),
    )
    (RAW / "regenerated_lookup.c").write_text(code)
    assert code == (target.OUT / "lookup.c").read_text()
    assert (
        emit_immutable_bytes_llvm(
            hierarchy.tables[0].data, symbol="hierarchy_coarse_table", alignment=64
        )
        == (target.OUT / "coarse_table.ll").read_text()
    )
    for table in hierarchy.tables:
        assert (
            table.data == (census.OUT / f"table_b{table.leading_bits}.bin").read_bytes()
        )
    qualified = json.loads((packet.OUT / "qualification.json").read_text())
    log = (packet.OUT / "spike.stdout").read_text()
    parsed = packet.parse_report(log)
    assert parsed["rows"] == qualified["rows"]
    owner = parsed["named_storage"][0]
    assert owner["fine_hash"] == fnv(hierarchy.tables[1].data)
    assert owner["coarse_hash"] == fnv(hierarchy.tables[0].data)
    for altered in [
        log.replace("id=1 sample=2", "id=0 sample=2"),
        log.replace("digest=b702909ca80dfe2a", "digest=0", 1),
        log.replace("fine_hash=" + owner["fine_hash"], "fine_hash=0", 1),
        log.replace("expected=" + owner["expected"], "expected=0", 1),
        log.replace("INTEGER_RESULT_PASS", "INCOMPLETE"),
    ]:
        try:
            packet.parse_report(altered)
        except ValueError:
            pass
        else:
            raise AssertionError("mutated paired witness accepted")
    objects = [Path(p) for p in qualified["objects"]]
    assert all(census.sha(p) == h for p, h in qualified["objects"].items())
    rebuilt = build_program(
        objects, RAW / "relink", target="gemmini", max_loaded_bytes=None
    )
    assert census.sha(rebuilt.elf) == qualified["elf_sha256"]
    assert audit_elf(rebuilt.elf.read_bytes())["status"] == "pass"
    recipe = backends.harness_build_recipe("gemmini")
    linkscript = derived_link_script(
        recipe.load_address, recipe.link_script, RAW / "relink"
    )
    link_objects = [
        *objects,
        *(RAW / "relink" / (p.stem + ".o") for p in recipe.support_sources),
    ]
    recipe_record = {
        "recipe": dataclasses.asdict(recipe),
        "support_commands": [
            recipe.compile_command(source=p, output=RAW / "relink" / (p.stem + ".o"))
            for p in recipe.support_sources
        ],
        "support_compile_cwd": str(RAW / "relink"),
        "link_command": recipe.link_command(
            objects=link_objects, output=rebuilt.elf, link_script=linkscript
        ),
        "ordered_link_objects": list(map(str, link_objects)),
        "byte_exact_relink": True,
    }
    (RAW / "actual_recipe.json").write_text(
        json.dumps(recipe_record, default=str, indent=2) + "\n"
    )
    paths = set()
    primary = []
    for qpath in [
        census.OUT / "qualification.json",
        target.OUT / "qualification.json",
        packet.OUT / "qualification.json",
    ]:
        q = json.loads(qpath.read_text())
        primary.append(qpath)
        for path, digest in q.get("pins", {}).items():
            if census.sha(path) != digest:
                snapshots = json.loads((RAW / "before_format.json").read_text())
                if path not in snapshots or snapshots[path]["sha256"] != digest:
                    raise AssertionError("unexplained primary pin change: " + path)
                assert census.sha(snapshots[path]["immutable_snapshot"]) == digest
                paths.add(Path(snapshots[path]["immutable_snapshot"]))
            else:
                paths.add(Path(path))
        paths.update(Path(p) for p in q.get("objects", {}))
    for directory in [census.OUT, target.OUT, packet.OUT, RAW]:
        paths.update(p for p in directory.rglob("*") if p.is_file())
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and (
            str(path).startswith(str(census.CORE / "src"))
            or str(path).startswith(str(census.HERE / "mlir_oot"))
        ):
            paths.add(Path(path))
    paths.update(Path(p) for p in recipe.support_sources)
    paths.add(Path(recipe.link_script))
    paths.update(
        [
            Path(recipe.compiler),
            target.GCC.with_name("spike"),
            target.LLVM / "clang",
            target.LLVM / "llvm-link",
            target.LLVM / "opt",
            target.LLVM / "llvm-nm",
            target.LLVM / "llvm-objdump",
        ]
    )
    paths.update(
        census.HERE / "tests" / name
        for name in [
            "source_interval_hierarchy_census.py",
            "source_interval_hierarchy_target.py",
            "source_interval_hierarchy_packet.py",
            "source_interval_hierarchy_seal.py",
        ]
    )
    paths.update(
        [
            census.CORE / "src/merlin/llvmlower/AGENT.md",
            census.CORE / "docs/design/agent_compiler_performance.md",
            census.CORE / "merlin/tests/ir/test_source_interval_hierarchy.py",
            census.CORE
            / "out/artifacts/probes/source-table-hierarchy-validation-20261007/tests.log",
            census.CORE
            / "out/artifacts/probes/source-table-hierarchy-validation-20261007/final_tests.log",
            census.CORE
            / "out/artifacts/probes/source-table-hierarchy-validation-20261007/initial_identifier_refusal.log",
        ]
    )
    paths.update(
        p
        for p in (
            census.HERE
            / "out/artifacts/probes/source-interval-hierarchy-census-20261007"
        ).rglob("*")
        if p.is_file()
    )
    pins = {str(p): census.sha(p) for p in sorted(paths) if p.is_file()}
    receipt = {
        "schema": "source_interval_hierarchy_complete_M8_release_v1",
        "status": "qualified_capsule_cost_unknown",
        "core_commit": "47ab40250",
        "core_base": "c0f40f8f8d100b841c26fe8bbd6c09e79b6de17c",
        "hypothesis": "Explicit source-wide small-table certificate reduces fine-table requests; added exact finishing work is measured, memory cost unknown.",
        "ownership": "Merlin source-wide table/refinement/use/effect API. OOT current FRM/RNE ABI/codegen/link/audit/capsule.",
        "control": "Actual current2085 zero observer/source16-bit table complete firstM8 helper; no whole-model inference.",
        "candidate": "Same exact source/i8 contract; fixed explicit13bit64KiB then unchanged16bit512KiB and original continuation.",
        "before_after": {
            "compiled_retired_control_mean": (1726425 + 1726425) / 2,
            "compiled_retired_candidate_mean": 1764710,
            "retired_delta_percent": (1764710 / 1726425 - 1) * 100,
            "hardware_cycles": "UNKNOWN",
            "whole_cycles": "UNKNOWN",
        },
        "all22_original_contexts": json.loads(
            (census.OUT / "qualification.json").read_text()
        )["sums"],
        "source_wide_table_derivation": "All source cells reconstructed by existing rounded evaluator, explicit partitions and aggregate budget; no sampled endpoint proof or output-based selectors.",
        "typed_witness": "22 original closed observers; one actual selected M8 source helper bound through canonical DAG, quant factor, finite scale scanner and zero-observer cells.",
        "qualification": "65 focused source/native/refusal cases; actual target all5FRM/7sticky45056 original source bytes/guards and unchanged helper objects in fresh named-storage ABBA successor.",
        "effect_scope": "Explicit nontrapping, gradual RNE, unobserved flags/error/interposition effects. NonRNE follows original source; no sticky-flag parity or universal undefined-source claim.",
        "actual_elf": qualified["elf_path"],
        "actual_elf_sha256": qualified["elf_sha256"],
        "parser_path": str(census.HERE / "tests/source_interval_hierarchy_packet.py"),
        "parser_function": "parse_report",
        "parser_negative_cases": 5,
        "expected_storage": parsed["named_storage"][0],
        "exact_source_45056_digest": "b702909ca80dfe2a",
        "ABBA": [0, 1, 1, 0],
        "cold_warm_scope": qualified["cold_warm_scope"],
        "named_storage_and_tables": qualified["symbols"],
        "private_arena": qualified["private_arena"],
        "source_replay_scope": "13375 original source cold calls unchanged across all22 native contexts; exact compiled source continuation retained. No dynamic counter prediction of whole hardware.",
        "initial_failures": [
            "Missing configured nativeClang produced4setup failures; explicit actualcompiler used forfinal65pass.",
            "Activation Cname original shadowed local source variable; saved transcript and generic Cscope collision refusal added.",
            "First census assumedrank3 arrays but actualcapturesflat; saved initialdriver/partialdeclaration and complete correctedv2 separate.",
        ],
        "original_primary_receipts": {str(p): census.sha(p) for p in primary},
        "regenerated_C_coarse_fine_and_relinked_ELF_byteexact": True,
        "prospective_model": "UNKNOWN; physical table traffic, stack/frame/branches and changed working-set regime are unpriced. No fitting to current/previous completeM8 labels.",
        "promotion": "Held forONE completecost stockpair/rootreview; no main/default/whole promotion",
        "pins": pins,
        "token_usage_available": False,
    }
    census.save(OUT / "receipt.json", receipt)
    for src in primary:
        name = src.parent.name + "_qualification.json"
        (OUT / name).write_bytes(src.read_bytes())
    print(
        json.dumps(
            {
                "receipt": str(OUT / "receipt.json"),
                "sha256": census.sha(OUT / "receipt.json"),
                "pins": len(pins),
                "elf": qualified["elf_path"],
                "elf_sha256": qualified["elf_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
