"""Join observed CPU classes to already qualified source/hardware scopes.

Reference aliases are grouped by physical body, never divided between calls.
The grouped hardware labels remain held out of coefficient fitting.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from operand_telemetry_probe import check_pin, pin

from mlir_oot.cpu_opcode_census import CLASSES, instruction_index
from mlir_oot.executed_features import parse_pc_histogram


def body_counts(index, histogram, scope):
    counts = Counter(dict.fromkeys(CLASSES, 0))
    for pc, count in histogram.items():
        if scope["start"] <= pc < scope["end"]:
            if pc not in index:
                raise ValueError("Scoped histogram PC is not an instruction boundary")
            counts[index[pc][0]] += count
    return dict(counts)


def export(root):
    records = root / "docs/perf_records"
    paths = {
        "reference": records / "q1013_reference_heldout_operand_features.json",
        "ours1874": records / "resnet1874_observed_operand_features.json",
    }
    header = Path(
        "/scratch2/agustin/chipyard/.conda-env/riscv-tools/include/riscv/encoding.h"
    )
    decoder = root / "mlir_oot/cpu_opcode_census.py"
    outputs = {}
    for label, path in paths.items():
        data = json.loads(path.read_text())
        qualification = data["operand_qualification"]
        check_pin(qualification["identity_receipt"])
        identity = json.loads(
            Path(qualification["identity_receipt"]["path"]).read_text()
        )
        for value in identity["pins"].values():
            check_pin(value)
        check_pin(qualification["scope_binding"])
        scopes = json.loads(Path(qualification["scope_binding"]["path"]).read_text())
        by_name = {s["symbol"]: s for s in scopes["scopes"]}
        elf = Path(identity["pins"]["elf"]["path"])
        hist_path = Path(identity["pins"]["histogram"]["path"])
        index = instruction_index(elf.read_bytes())
        histogram = parse_pc_histogram(hist_path.read_text())
        grouped = defaultdict(list)
        for layer in data["layers"]:
            name = layer.get("symbol") or layer["physical_function"] or "fx_fc_54"
            grouped[name].append(layer)
        observations = []
        for name, layers in grouped.items():
            cpu = body_counts(index, histogram, by_name[name])
            feature = {"cpu_opcode_classes": cpu}
            selected_pcs = {
                pc
                for pc in histogram
                if by_name[name]["start"] <= pc < by_name[name]["end"]
            }
            feature["unique_executed_pcs"] = len(selected_pcs)
            feature["touched_instruction_bytes"] = sum(
                index[pc][1] for pc in selected_pcs
            )
            for key in layers[0]["features"]:
                if isinstance(layers[0]["features"][key], int):
                    feature[key] = sum(r["features"][key] for r in layers)
                elif key == "primitive_commands":
                    feature[key] = {
                        k: sum(r["features"][key][k] for r in layers)
                        for k in layers[0]["features"][key]
                    }
            body_instructions = sum(cpu.values())
            if label == "ours1874":
                assert len(layers) == 1
                assert (
                    body_instructions == layers[0]["kernel_body_retired_instructions"]
                )
            provenance = [
                "elf_sha256:" + pin(elf)["sha256"],
                "pc_histogram_sha256:" + pin(hist_path)["sha256"],
                "opcode_decoder_sha256:" + pin(decoder)["sha256"],
                "installed_spike_encoding_header_sha256:" + pin(header)["sha256"],
            ]
            cpu_status = {
                "/features/cpu_opcode_classes": {
                    "status": "observed_functional_execution",
                    "unit": "instruction",
                    "provenance": provenance,
                }
            }
            for key, unit in (
                ("unique_executed_pcs", "unique_pc"),
                ("touched_instruction_bytes", "instruction_byte"),
            ):
                cpu_status["/features/" + key] = dict(
                    cpu_status["/features/cpu_opcode_classes"], unit=unit
                )
            for key in cpu:
                cpu_status["/features/cpu_opcode_classes/" + key] = cpu_status[
                    "/features/cpu_opcode_classes"
                ]
            observations.append(
                {
                    "scope_id": label + "_physical_body_group_" + name,
                    "physical_function": name,
                    "function_range": by_name[name],
                    "invocations": len(layers),
                    "source_intervals": [
                        {
                            "scope_id": r["scope_id"],
                            "shape": r.get("geometry", r.get("shape")),
                            "category": r.get("geometry_class", r.get("category")),
                            "hardware_cycles": r["hardware_cycles"],
                            "retired_instructions": r["features"][
                                "executed_instructions"
                            ],
                        }
                        for r in layers
                    ],
                    "features": feature,
                    "cpu_feature_status": cpu_status,
                    "prior_operand_feature_status": layers[0]["feature_status"],
                    "hardware_cycles": sum(r["hardware_cycles"] for r in layers),
                    "body_retired_instructions": body_instructions,
                    "counter_interval_minus_body_instructions": feature[
                        "executed_instructions"
                    ]
                    - body_instructions,
                    "role": (
                        "heldout_evaluation_only"
                        if label == "reference"
                        else "independent_historical_control_observation"
                    ),
                    "scope_caveat": "CPU counts cover the exact physical function PC range; target commands cover the same body invocations; the measured wrapper/glue interval includes separately reported extra instructions. Their hardware cost and CPU/accelerator overlap remain UNKNOWN.",
                    "per_invocation_cpu_classes": (
                        "Observed for the unique invocation"
                        if len(layers) == 1
                        else "UNKNOWN: aggregate PC histogram cannot attribute data-dependent paths to individual invocations"
                    ),
                }
            )
        result = {
            "schema": "gemmini_scoped_cpu_operand_features_v1",
            "label": label,
            "pins": {
                "prior_operand_dataset": pin(path),
                "elf": pin(elf),
                "histogram": pin(hist_path),
                "identity_receipt": qualification["identity_receipt"],
                "scope_binding": qualification["scope_binding"],
                "decoder": pin(decoder),
                "spike_encoding_header": pin(header),
                "export_driver": pin(__file__),
            },
            "isa_classification": "RV64GC encoding classes; qualified engine replay establishes actual ISA execution",
            "role": (
                "heldout_evaluation_only"
                if label == "reference"
                else "independent_historical_control_observation"
            ),
            "fit_authorization": "All reference1876 timing remains excluded from coefficient fitting",
            "observations": observations,
            "conservation": {
                "source_interval_count": sum(r["invocations"] for r in observations),
                "physical_function_groups": len(observations),
                "body_retired_instructions": sum(
                    r["body_retired_instructions"] for r in observations
                ),
                "counter_interval_minus_body_instructions": sum(
                    r["counter_interval_minus_body_instructions"] for r in observations
                ),
                "hardware_interval_cycles": sum(
                    r["hardware_cycles"] for r in observations
                ),
                "unknown_encodings": sum(
                    r["features"]["cpu_opcode_classes"]["unknown_encoding"]
                    for r in observations
                ),
            },
            "unknown": {
                "opcode_latency": "UNKNOWN",
                "cpu_accelerator_overlap": "UNKNOWN",
                "physical_dram_traffic": "UNKNOWN",
                "instruction_scope_hardware_overhead": "UNKNOWN",
                "instruction_cache_line_footprint": "UNKNOWN: cache geometry not pinned",
                "instruction_cache_misses": "UNKNOWN: unordered histogram has no fetch chronology",
            },
        }
        out = records / (
            "q1013_reference_grouped_cpu_operand_features.json"
            if label == "reference"
            else "resnet1874_cpu_operand_features.json"
        )
        out.write_text(json.dumps(result, indent=2) + "\n")
        outputs[label] = result["conservation"]
    assert outputs["reference"]["source_interval_count"] == 54
    assert outputs["reference"]["physical_function_groups"] == 24
    assert outputs["reference"]["counter_interval_minus_body_instructions"] == 1624
    assert outputs["ours1874"]["source_interval_count"] == 70
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    print(json.dumps(export(parser.parse_args().artifact_root), indent=2))
