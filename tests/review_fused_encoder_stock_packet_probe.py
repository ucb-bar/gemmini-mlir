"""Reclose the already qualified generic encoder composition before one stock run."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from merlin.common.paths import data_path
from merlin.llvmlower.fused_encoded_witness import prepare_fused_encoded_witness
from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review(args):
    require(not args.output.exists(), "release must be fresh")
    q = json.loads(args.packet.read_text())
    capacity = json.loads(args.capacity.read_text())
    stock = json.loads(args.control_review.read_text())
    for record in (q, capacity, stock):
        for path, digest in record["pins"].items():
            require(sha(path) == digest, "changed evidence: " + path)
    require(q["control_job"] == stock["job_id"] == 2069 and
            q["generic_core"] == "63578e4b7372fcb7772962781f7cd46c08f3c7f9",
            "source/control differs")
    require(sha(q["control_elf"]) == stock["nofsm"]["elf_sha256"] == q["control_sha256"],
            "measured control was not reproduced")
    candidate = Path(q["candidate_elf"])
    base = candidate.parent.parent
    old = Path("/scratch/agustin/tmp/gemmini-fused-radix-readout-20261007/out/fused_reconstruction")
    require(sha(candidate) == q["candidate_sha256"], "candidate ELF differs")
    for flavor in ("native_numeric", "target_numeric"):
        control = base / "control" / flavor
        changed = base / "candidate" / flavor
        require((control / "provider.c").read_bytes() ==
                (old / "candidate" / flavor / "provider.c").read_bytes(), "control C differs")
        require(prepare_fused_encoded_witness((control / "provider.c").read_text()) ==
                (changed / "provider.c").read_text(), "actual generic source transform differs")
        require((control / "bf16_radix_pack.h").read_bytes() ==
                (data_path("runtime", "c") / "bf16_radix_pack.h").read_bytes(),
                "canonical source quantization differs")
        object_name = "provider.so" if flavor == "native_numeric" else "provider.o"
        require((control / object_name).read_bytes() ==
                (old / "candidate" / flavor / object_name).read_bytes(), "control object differs")
    audit = audit_elf(candidate.read_bytes())
    require(audit["status"] == "pass" and not audit["forbidden"] and not audit["unknown"],
            "all executable sections must contain zero forbidden/unknown custom instructions")
    native_path = Path(q["native"]["receipt"])
    native = json.loads(native_path.read_text())
    previous = json.loads((old / "native_v2/validation.json").read_text())
    output = np.load(base / "native_v2/output.npy", allow_pickle=False)
    original = Path("/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle/golden.npy")
    golden = np.load(original, allow_pickle=False)
    require(sha(original) == native["original_golden_sha256"] and
            output.dtype == golden.dtype == np.float32 and output.size == 1600 and
            output.shape == golden.shape and output.tobytes() == golden.tobytes(),
            "original complete native outputs differ")
    require(native["group_stats"] == previous["group_stats"] and
            native["calls"][:3] == [48, 0, 0] and native["product_calls"] == 23040 and
            not native["callback_errors"] and native["bitwise_mismatches"] ==
            native["elementwise_failures"] == 0, "native coverage/accuracy differs")
    stdout_path = base / "histogram/spike.stdout"
    stdout = stdout_path.read_text()
    normalize = lambda text: re.sub(r"^WORKSPACE_GROUP_INSTRUCTIONS \d+$", "ROI", text, flags=re.M)
    require(normalize(stdout) == normalize((old / "histogram/spike.stdout").read_text()),
            "complete target consumer/stats/guards differ")
    terminal = json.loads((base / "histogram/terminal.json").read_text())
    group = q["strict_group"]
    require(terminal["exit_code"] == 0 and terminal["elf_sha256"] == sha(candidate) and
            terminal["stdout_sha256"] == sha(stdout_path) and
            re.findall(r"^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$", stdout, re.M) ==
            [str(group["candidate_instructions"])], "strict terminal differs")
    require(group["control_instructions"] == 1806165115 and
            group["candidate_instructions"] == 1734429991 and group["callbacks"] == 480 and
            group["readback_bytes"] == 86507520 and group["original_i8"] == 786432 and
            group["original_scales"] == 1024, "complete work differs")
    require(capacity["status"] == "PASS" and capacity["queried_bytes"] == 124061504 and
            capacity["queried_alignment"] == 8 and
            capacity["unaligned_workspace_refused_before_publication"], "workspace ABI differs")
    refusal = capacity["capacity_refusal"]
    require(refusal["refused_capacities"] == [0, 123012928, 124061503] and
            refusal["product_calls"] == 0 and refusal["private_public_outputs_untouched"] and
            refusal["statistics_untouched"], "refusal must precede writes/callbacks")
    fallback = q["retained_source_fallback"]
    require(fallback["status"] == "PASS" and fallback["source_words"] == 196608 and
            fallback["all_original_bf16_bits_exact"] and fallback["input_bytes_unchanged"] and
            fallback["output_descriptor_unchanged"] and fallback["guards_unchanged"],
            "retained fallback differs")
    ratio = group["candidate_instructions"] / group["control_instructions"]
    result = {
        "schema": "root_fused_encoder_complete_group_stock_release_v1",
        "status": "RELEASED_ONE_COMPLETE_GROUP_STOCK_OBSERVATION",
        "candidate": str(candidate), "candidate_elf_sha256": sha(candidate),
        "control_job": 2069, "control_cycles": stock["stock_cycles"],
        "control_elf_sha256": q["control_sha256"],
        "qualification_pins_reclosed": len(q["pins"]),
        "capacity_pins_reclosed": len(capacity["pins"]),
        "generic_transform_independently_rederived": True,
        "native_original_words_exact": 1600, "native_source_groups": 48,
        "retained_source_fallback_words_exact": 196608,
        "numeric_gate": {"atol": .03125, "rtol": .02},
        "strict_group": group, "workspace": {"required_bytes": 124061504, "alignment": 8},
        "final_noFSM": audit,
        "uart_required": {"pass_marker": "WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS",
                          "stats": group["stats"], "carrier_differences": 3},
        "hardware_alias": stock["hardware_identity"]["hw_config"],
        "hardware_identity_required": stock["hardware_identity"], "allowed_runs": 1,
        "conditional_pretiming_screen": {
            "hypothesis": "same complete callback workload; retired-instruction ratio scales2069 group cycles",
            "candidate_group_cycles": stock["stock_cycles"] * ratio, "ratio": ratio,
            "new_labels_fitted": False, "resolved_prediction": False,
            "unpriced": ["physical traffic, cache and frame changes", "dependency/host-device overlap"],
            "scope": "Frozen diagnostic only; no qualified cost/ranking model or whole projection."},
        "whole_hardware_cycles": "UNKNOWN;1906 champion258621872969 unchanged",
        "default_promotion": False,
        "integration_scope": q["native"]["normal_scope"],
        "source_contract": q["source_contract"],
        "requirements": ["Recovery submits once after root commit; exact staged ELF/config/stock bitstream/terminal.",
                         "Original complete consumer/stats/carrier/inputs/guards pass; no whole projection.",
                         "Attention stage diagnosis remains primary; do not start another small composition campaign."],
        "pins": {str(path.resolve()): sha(path) for path in
                 (args.packet, args.capacity, args.control_review, candidate, stdout_path, native_path,
                  base / "native_v2/output.npy", original, Path(__file__))},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("status", "qualification_pins_reclosed", "candidate_elf_sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "capacity", "control-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
