"""Check independent evidence and unit corruptions against a real qualified packet."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from cpu_service_cycle_screen_probe import verify_packet


def check(packet_path, output):
    original = json.loads(packet_path.read_text())
    evidence = verify_packet(original)
    engine = next(iter(original["observations"]))
    cases = []

    def refusal(name, mutate):
        packet = copy.deepcopy(original)
        mutate(packet)
        try:
            verify_packet(packet)
        except ValueError as exc:
            cases.append({"case": name, "refused": str(exc)})
        else:
            raise AssertionError("accepted evidence corruption: " + name)

    refusal("missing_measurement_evidence", lambda p: p.pop("measurement_refs"))
    refusal("stale_console_digest", lambda p: p["measurement_refs"][engine]["stdout"].update(sha256="0" * 64))
    refusal("different_executable", lambda p: p["built"].update(elf_sha256="0" * 64))
    refusal("partition_changed_after_seal", lambda p: p["manifest"]["cases"][1].update(partition="training"))
    refusal("partial_report", lambda p: p["observations"][engine]["report"]["rows"].pop())
    refusal("fabricated_feature", lambda p: p["observations"][engine]["features"][3]["features"].update(fp_div_ops=7))
    refusal("changed_counter_units", lambda p: p["observations"][engine]["features"][3]["metric"].update(mcycle_status="observed_hardware_counter" if engine == "spike" else "retired_instruction_proxy"))
    refusal("unobserved_function_footprint", lambda p: p["function_sizes"].update(empty=p["function_sizes"]["empty"] + 4))
    refusal("changed_final_audit", lambda p: p["nofsm_audit"].update(status="fail"))
    record = {"schema": "cpu_service_screen_evidence_rejections_v1", "packet": str(packet_path),
              "reclosed_artifact_digest_count": len(evidence), "status": "pass", "cases": cases,
              "source_and_measurement_artifacts_mutated": False, "cycle_labels_fitted": False}
    output.write_text(json.dumps(record, indent=2) + "\n")
    print("CPU_SERVICE_EVIDENCE_REFUSALS_PASS", len(cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check(args.packet, args.out)
