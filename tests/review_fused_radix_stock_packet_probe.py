"""Review a complete-group fused readout experiment before stock measurement."""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from merlin.llvmlower.radix_integer_reconstruct import c_fused_header
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups
from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review(args):
    require(not args.output.exists(), "root release output must be fresh")
    q = json.loads(args.packet.read_text())
    for path, expected in q["pins"].items():
        require(sha(path) == expected, "changed group qualification: " + path)
    require(q["control_job"] == 2064 and q["generic_core_final"] ==
            "63578e4b7372fcb7772962781f7cd46c08f3c7f9", "source/control version differs")
    stock = json.loads(args.control_review.read_text())
    for path, expected in stock["pins"].items():
        require(sha(path) == expected, "current control terminal changed")
    require(stock["job_id"] == 2064 and stock["stock_cycles"] == 4256146700 and
            sha(q["control_elf"]) == stock["nofsm"]["elf_sha256"] == q["control_sha256"],
            "actual measured control not reproduced")
    candidate = Path(q["candidate_elf"])
    require(sha(candidate) == q["candidate_sha256"], "candidate final ELF changed")
    audit = audit_elf(candidate.read_bytes())
    require(audit["status"] == "pass" and not audit["forbidden"] and not audit["unknown"],
            "every executable section must contain zero forbidden/unknown custom instructions")
    base = candidate.parent.parent
    plan = plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=192)
    require(json.loads((base / "canonical_plan.json").read_text()) ==
            json.loads(json.dumps(asdict(plan))), "completed readout range/prefix proof differs")
    header = c_fused_header(plan)
    for flavor in ("native_numeric", "target_numeric"):
        require((candidate.parent / flavor / "provider.c").read_text().count(header) == 1,
                "actual numeric source does not contain exact generic helper")
    native_path = base / "native_v2/validation.json"
    native = json.loads(native_path.read_text())
    old_base = Path("/scratch/agustin/tmp/gemmini-rms-selective-replay-20261007/out/rms4_fast_bounded_floor")
    old_native = json.loads((old_base / "native/validation.json").read_text())
    new_output, old_output = (np.load(path, allow_pickle=False) for path in
                              (base / "native_v2/output.npy", old_base / "native/output.npy"))
    require(new_output.dtype == old_output.dtype == np.float32 and new_output.size == 1600
            and new_output.shape == old_output.shape and new_output.tobytes() == old_output.tobytes(),
            "original1600 complete native observations differ")
    require(native["group_stats"] == old_native["group_stats"] and native["calls"][:3] == [48, 0, 0]
            and native["product_calls"] == 23040 and not native["callback_errors"]
            and native["bitwise_mismatches"] == native["elementwise_failures"] == 0,
            "complete48 source coverage, original gate or numeric replay changed")
    stdout_path = base / "histogram/spike.stdout"
    stdout = stdout_path.read_text()
    old_stdout = (old_base / "histogram/spike.stdout").read_text()
    normalize = lambda text: re.sub(r"^WORKSPACE_GROUP_INSTRUCTIONS \d+$", "ROI", text, flags=re.M)
    require(normalize(stdout) == normalize(old_stdout), "complete target consumer/stats/guards differ")
    terminal = json.loads((base / "histogram/terminal.json").read_text())
    require(terminal["exit_code"] == 0 and terminal["elf_sha256"] == sha(candidate)
            and terminal["stdout_sha256"] == sha(stdout_path), "strict terminal binding differs")
    layout = json.loads((base / "layout_probe.json").read_text())
    workspace = q["workspace"]
    require(workspace["required_bytes"] == 124061504 and workspace["delta_bytes"] == 1048576,
            "actual queried workspace changed")
    planes = layout["planes"]
    require(len(planes) == 5 and all(start % 64 == 0 and end - start == 524288 for start, end in planes)
            and all(planes[index][1] <= planes[index + 1][0] for index in range(4))
            and layout["center_offset"] + layout["center_bytes"] <= planes[0][0]
            and planes[-1][1] <= workspace["required_bytes"], "completed plane lifetimes/layout overlap")
    refusal = json.loads((base / "obsolete_capacity_refusal.json").read_text())
    require(refusal["status"] == "PASS" and refusal["product_calls"] == 0
            and refusal["private_public_outputs_untouched"] and refusal["statistics_untouched"],
            "obsolete capacity refusal must happen before writes/callbacks")
    fallback = q["actual_retained_source_fallback"]
    require(fallback["all_original_bf16_bits_exact"] and fallback["source_words"] == 196608
            and fallback["input_bytes_unchanged"] and fallback["guards_unchanged"]
            and fallback["output_descriptor_unchanged"], "retained source fallback proof differs")
    group = q["strict_group"]
    require(group["product_callbacks"] == 480 and group["requested_readback_bytes"] == 86507520
            and group["candidate_roi_instructions"] == 1806165115
            and group["control_roi_instructions"] == 1896853389, "actual complete group work differs")
    ratio = group["candidate_roi_instructions"] / group["control_roi_instructions"]
    result = {
        "schema": "root_fused_radix_complete_group_stock_release_v1",
        "status": "RELEASED_ONE_COMPLETE_GROUP_STOCK_OBSERVATION",
        "candidate": str(candidate), "candidate_elf_sha256": sha(candidate),
        "control_job": 2064, "control_cycles": stock["stock_cycles"],
        "control_elf_sha256": sha(q["control_elf"]), "qualification_pins_reclosed": len(q["pins"]),
        "native_original_words_exact": 1600, "native_source_groups": 48,
        "retained_source_fallback_words_exact": 196608,
        "numeric_gate": {"atol": .03125, "rtol": .02},
        "strict_group": group, "workspace": workspace, "final_noFSM": audit,
        "uart_required": {"pass_marker": "WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS",
                          "stats": group["stats"], "carrier_differences": 3},
        "hardware_alias": stock["hardware_identity"]["hw_config"],
        "hardware_identity_required": stock["hardware_identity"], "allowed_runs": 1,
        "conditional_pretiming_screen": {
            "hypothesis": "same complete callback workload; retired-instruction ratio scales2064 group cycles",
            "candidate_group_cycles": stock["stock_cycles"] * ratio, "ratio": ratio,
            "new_labels_fitted": False, "resolved_prediction": False,
            "unpriced": ["five live readout planes and changed physical write addresses",
                         "removed integer load/store passes, dependency/cache/frame/host-device overlap"],
            "scope": "Frozen diagnostic before this new hardware label; not a qualified cost/ranking model or whole projection.",
        },
        "whole_hardware_cycles": "UNKNOWN;1906 champion unchanged", "default_promotion": False,
        "integration_scope": "Generic exact reconstruction helper inside frozen experimental RMS4/bounded-floor numeric provider. Fresh exact normal pool/native ranked ABI adapter validates coverage and capacity separately; this is not a production approximate normal binder seal.",
        "requirements": ["Recovery submits once after root commit; exact staged ELF/config/stockbitstream/terminal.",
                         "Original complete786432i8/1024BF16 scales/8stats/carrier3/guards and input contract pass.",
                         "Retain frozen count-ratio diagnostic and uncertainty; no whole projection/default promotion."],
        "pins": {str(path.resolve()): sha(path) for path in
                 (args.packet, args.control_review, candidate, stdout_path, native_path,
                  base / "native_v2/output.npy", old_base / "native/output.npy", Path(__file__))},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "qualification_pins_reclosed", "candidate_elf_sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "control-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
