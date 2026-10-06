"""Close source scopes and export independently measured operand features."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from merlin.targetgen.elf_lanes import executable_sections, instruction_words

from mlir_oot.executed_features import _symbol_ranges, parse_pc_histogram
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.operand_features import read_telemetry, summarize
from mlir_oot.tables import rtl_facts as F


def pin(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def check_pin(value):
    assert pin(value["path"])["sha256"] == value["sha256"]


def combine(data, entries):
    keys = {(row["scope_id"], row["invocation"]) for row in entries}
    selected = dict(
        data,
        rows=[
            row for row in data["rows"] if (row["scope_id"], row["invocation"]) in keys
        ],
    )
    return summarize(selected)


def close(root, label):
    work = root / "out/executed_feature_probe" / ("final_" + label)
    identity_path = work / "replay_identity.json"
    identity = json.loads(identity_path.read_text())
    assert (
        identity["exit_code"] == 0
        and identity["stdout_byte_identity"]
        and identity["histogram_byte_identity"]
    )
    for value in identity["pins"].values():
        check_pin(value)
    build = json.loads(Path(identity["pins"]["build"]["path"]).read_text())
    for value in [
        *build["inputs"].values(),
        *build["outputs"].values(),
        build["hook"],
        build["builder"],
        build["spike_source"],
    ]:
        check_pin(value)
    assert build["outputs"]["spike"]["sha256"] == build["spike_source"]["sha256"]
    elf = Path(identity["pins"]["elf"]["path"])
    blob = elf.read_bytes()
    assert audit_elf(blob)["status"] == "pass"
    hist = parse_pc_histogram(Path(identity["pins"]["histogram"]["path"]).read_text())
    words = {
        pc: word
        for _, off, size, addr in executable_sections(blob)
        for pc, word in instruction_words(blob[off : off + size], addr)
    }
    data = read_telemetry(work / "operands.json")
    classes = Counter()
    for pc, count in hist.items():
        word = words.get(pc, 0)
        if word & 0x7F == F.CUSTOM_OPCODE:
            classes[word >> 25] += count
    observed = Counter()
    for row in data["rows"]:
        observed[row["funct"]] += row["commands"]
    assert classes == observed
    scopes_path = root / "out/executed_feature_probe" / f"{label}_scopes.json"
    scopes = json.loads(scopes_path.read_text())
    assert scopes["elf_sha256"] == pin(elf)["sha256"]
    assert (
        scopes["histogram_sha256"] == identity["pins"]["baseline_histogram"]["sha256"]
    )
    spec_path = root / "out/executed_feature_probe" / f"{label}_scopes.txt"
    assert scopes["scope_file_sha256"] == pin(spec_path)["sha256"]
    actual_ranges = {
        r["symbol"]: r
        for r in _symbol_ranges(elf, [r["symbol"] for r in scopes["scopes"]], "readelf")
    }
    names = {r["id"]: r["symbol"] for r in scopes["scopes"]}
    entries_by_scope = Counter(e["scope_id"] for e in data["entries"])
    closure = []
    for row in scopes["scopes"]:
        assert (row["start"], row["end"]) == (
            actual_ranges[row["symbol"]]["start"],
            actual_ranges[row["symbol"]]["end"],
        )
        assert words[row["entry_command_pc"]] & 0x7F == F.CUSTOM_OPCODE
        assert (
            row["observed_entry_count"]
            == hist[row["entry_command_pc"]]
            == entries_by_scope[row["id"]]
        )
        expected = Counter()
        for pc, count in hist.items():
            if (
                row["start"] <= pc < row["end"]
                and words.get(pc, 0) & 0x7F == F.CUSTOM_OPCODE
            ):
                expected[words[pc] >> 25] += count
        got = Counter()
        for event in data["rows"]:
            if event["scope_id"] == row["id"]:
                got[event["funct"]] += event["commands"]
        assert expected == got
        closure.append(
            {
                "symbol": row["symbol"],
                "scope_id": row["id"],
                "entries": entries_by_scope[row["id"]],
                "all_primitive_class_counts_match_exact_function_pc_histogram": True,
            }
        )
    for index, entry in enumerate(data["entries"]):
        selected = [
            r
            for r in data["rows"]
            if (r["scope_id"], r["invocation"])
            == (entry["scope_id"], entry["invocation"])
        ]
        assert min(r["first_event"] for r in selected) == entry["event"]
        next_event = (
            data["entries"][index + 1]["event"]
            if index + 1 < len(data["entries"])
            else data["total_commands"] + 1
        )
        assert max(r["last_event"] for r in selected) < next_event
    summary = summarize(data)
    provenance = [
        "elf_sha256:" + identity["pins"]["elf"]["sha256"],
        "telemetry_sha256:" + identity["pins"]["telemetry"]["sha256"],
        "engine_build_sha256:" + identity["pins"]["build"]["sha256"],
        "scope_file_sha256:" + pin(spec_path)["sha256"],
    ]
    for status in summary["feature_status"].values():
        status["provenance"] = provenance
    result = {
        "label": label,
        "identity_receipt": pin(identity_path),
        "scope_binding": pin(scopes_path),
        "scope_file": pin(spec_path),
        "telemetry_bytes": (work / "operands.json").stat().st_size,
        "elapsed_seconds": identity["elapsed_seconds"],
        "command_class_histogram_conservation": True,
        "source_function_primitive_class_conservation": closure,
        "ordered_entry_count": len(data["entries"]),
        "summary": summary,
        "engine_status": "Isolated copied Spike and separately compiled telemetry extension; production engine unchanged",
        "numeric_status": "Original baseline stdout and complete PC histogram are byte-identical",
        "feature_provenance": provenance,
    }
    return result, data, names


def export(root):
    root = Path(root)
    ref, ref_data, ref_names = close(root, "reference")
    current, current_data, current_names = close(root, "ours1874")
    reference_path = (
        root / "docs/perf_records/q1013_reference_heldout_layer_features.json"
    )
    reference = json.loads(reference_path.read_text())
    entries = [
        e
        for e in ref_data["entries"]
        if ref_names[e["scope_id"]].startswith(("fx_conv_", "fx_fc_"))
    ]
    assert len(entries) == 54
    for entry, row in zip(entries, reference["layers"], strict=True):
        expected = row["physical_function"] or (
            "fx_fc_54" if row["reference_index"] == 54 else None
        )
        assert ref_names[entry["scope_id"]] == expected
        observed = summarize(
            ref_data, scope_id=entry["scope_id"], invocation=entry["invocation"]
        )
        for status in observed["feature_status"].values():
            status["provenance"] = ref["feature_provenance"]
        row["features"].update(observed["features"])
        row["feature_status"].update(observed["feature_status"])
        row["operand_scope"] = dict(
            observed["scope"],
            physical_function=expected,
            source_interval_closure="Actual ordered entry markers match all54source-paired layer functions; complete body command classes match exact PC histogram",
        )
    residual_entries = [
        e
        for e in ref_data["entries"]
        if ref_names[e["scope_id"]].startswith("fx_res_conv_")
    ]
    assert len(residual_entries) == 16
    reference["residual_aggregate_operand_features"] = combine(
        ref_data, residual_entries
    )
    for status in reference["residual_aggregate_operand_features"][
        "feature_status"
    ].values():
        status["provenance"] = ref["feature_provenance"]
    reference["residual_aggregate_operand_features"]["features"][
        "executed_instructions"
    ] = reference["other_aggregate_counter_scopes"]["Res add"]["executed_instructions"]
    reference["residual_aggregate_operand_features"]["hardware_cycles"] = reference[
        "other_aggregate_counter_scopes"
    ]["Res add"]["hardware_cycles"]
    reference["residual_aggregate_operand_features"]["role"] = "heldout_evaluation_only"
    reference["operand_qualification"] = ref
    reference["whole_program_target_features"] = ref["summary"]
    reference["whole_program_target_features"]["scope"]["caveat"] = (
        "Includes one main flush outside the54layer/residual timers; do not claim exact forward event-window identity"
    )
    reference["pins"]["prior_counter_only_dataset"] = pin(reference_path)
    out = root / "docs/perf_records/q1013_reference_heldout_operand_features.json"
    out.write_text(json.dumps(reference, indent=2) + "\n")
    base_path = root / "docs/perf_records/resnet1874_executed_layer_features.json"
    model = json.loads(base_path.read_text())
    assert len(current_data["entries"]) == len(model["layers"]) == 70
    for entry, row in zip(current_data["entries"], model["layers"], strict=True):
        assert (
            current_names[entry["scope_id"]] == row["symbol"]
            and entry["invocation"] == 1
        )
        observed = summarize(
            current_data, scope_id=entry["scope_id"], invocation=entry["invocation"]
        )
        for status in observed["feature_status"].values():
            status["provenance"] = current["feature_provenance"]
        row["features"].update(observed["features"])
        row["feature_status"].update(observed["feature_status"])
        row["operand_scope"] = dict(
            observed["scope"],
            physical_function=row["symbol"],
            source_interval_closure="All70ordered source catalog boundary names match actual primitive entries; complete body command classes match PC histogram",
        )
        row["census"].pop("nofsm_audit", None)
    model["operand_qualification"] = current
    model["whole_program_target_features"] = current["summary"]
    model["pins"]["prior_counter_only_dataset"] = pin(base_path)
    model_out = root / "docs/perf_records/resnet1874_observed_operand_features.json"
    model_out.write_text(json.dumps(model, indent=2) + "\n")
    tail_path = root / "out/executed_feature_probe/final_tail/replay_identity.json"
    tail = json.loads(tail_path.read_text())
    for value in tail["pins"].values():
        check_pin(value)
    tail_features = summarize(
        read_telemetry(root / "out/executed_feature_probe/final_tail/operands.json")
    )
    assert (
        tail_features["features"]["requested_dma_load_bytes"],
        tail_features["features"]["requested_dma_store_bytes"],
    ) == (6890, 9636)
    receipt = {
        "schema": "gemmini_spike_operand_telemetry_qualification_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "owner": "reference_parity",
        "token_usage_available": False,
        "qualification": {
            "reference": ref,
            "current1874": current,
            "independent_tail": {"identity": pin(tail_path), "features": tail_features},
        },
        "outputs": {
            "reference_heldout": pin(out),
            "current_independent": pin(model_out),
        },
        "production_engine_modified": False,
        "hardware_jobs_submitted": False,
        "accuracy_gates_changed": False,
        "cycles_fitted": False,
        "driver": pin(Path(__file__)),
        "limitations": [
            "Nominal padded array work is not physical activity or a cycle floor.",
            "DMA bytes are instruction requested payload; DRAM bursts/reuse/overlap are unknown.",
            "Reference numeric source differs from current immutable capture; timing labels stay heldout.",
            "Ordered entries and complete source PC-class conservation close actual body commands, not every wrapper host instruction.",
        ],
    }
    final = root / "docs/perf_records/spike_operand_telemetry_qualification.json"
    final.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "qualification": str(final),
                "reference": str(out),
                "current": str(model_out),
                "reference_summary": ref["summary"]["features"],
                "current_summary": current["summary"]["features"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    export(args.artifact_root)
