"""Independently close the frozen explicit RMS4 group before one stock run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def close(args):
    if args.output.exists():
        raise ValueError("review requires a fresh output")
    packet = json.loads(args.packet.read_text())
    historical_path = Path(packet["historical_qualification"])
    historical = json.loads(historical_path.read_text())
    pins = packet["pins"]
    for name, expected in pins.items():
        if digest(Path(name)) != expected:
            raise ValueError("changed artifact: " + name)
    if len(historical["pins"]) != 223 or any(pins.get(p) != h for p, h in historical["pins"].items()):
        raise ValueError("original positive qualification was not preserved")
    base = args.packet.parent.parent.parent / "out/rms4_fast"
    candidate = base / "candidate/model.elf"
    control = base / "control/model.elf"
    target = packet["candidate"]
    for name, path in (("candidate", candidate), ("control", control)):
        if digest(path) != target[name + "_elf_sha256"]:
            raise ValueError("frozen final ELF changed")
        audit = audit_elf(path.read_bytes())
        if audit["status"] != "pass" or audit["forbidden"] or audit["unknown"]:
            raise ValueError("final executable custom instruction audit failed")
    if digest(control) != "ec3215988a150a0aea70eb10562b69c8bb617a4a5188138789790af247a01039":
        raise ValueError("stock2024 control identity differs")
    gold_path = Path(packet["accuracy"]["reference"])
    output_path = base / "native/output.npy"
    gold, output = np.load(gold_path), np.load(output_path)
    if gold.dtype != np.float32 or output.dtype != np.float32 or gold.size != 1600 or output.size != 1600:
        raise ValueError("original whole source output shape/type differs")
    gold, output = gold.reshape(-1), output.reshape(-1)
    failures = int(np.count_nonzero(~(np.isfinite(output) & (np.abs(output.astype(np.float64)-gold.astype(np.float64)) <= .03125+.02*np.abs(gold.astype(np.float64))))))
    bit_differences = int(np.count_nonzero(output.view(np.uint32) != gold.view(np.uint32)))
    if failures or bit_differences:
        raise ValueError("fresh original whole native numeric gate failed")
    native = json.loads((base / "native/validation.json").read_text())
    if native["calls"][:3] != [48,0,0] or native["product_calls"] != 23040 or native["callback_errors"] or native["original_golden_sha256"] != digest(gold_path):
        raise ValueError("all48 native source-group coverage differs")
    terminal = json.loads(Path(target["receipt"]).read_text())
    stdout_path = base / "histogram/spike.stdout"
    stdout = stdout_path.read_text()
    if terminal["exit_code"] != 0 or not terminal["same_elf"] or terminal["elf_sha256"] != digest(candidate) or terminal["stdout_sha256"] != digest(stdout_path):
        raise ValueError("production strict terminal does not bind candidate/stdout")
    counters = re.findall(r"^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$", stdout, re.M)
    stats = re.findall(r"^WORKSPACE_STAT (\d+) (\d+)$", stdout, re.M)
    if counters != [str(target["complete_roi_retired_instructions"])] or stats != [(str(i),str(v)) for i,v in enumerate(target["stats"])]:
        raise ValueError("strict ROI or complete consumer statistics differ")
    if "WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS" not in stdout or re.search(r"^UNOBSERVED_CARRIER_DIFFERENCES (\d+)$",stdout,re.M).group(1) != "3" or "CAPSULE UNEXPECTED SOURCE REFUSAL" in stdout:
        raise ValueError("complete consumer/guard/carrier gate failed")
    probe = args.packet.parent.parent.parent / "out/rms4_stock_review/probe.stdout"
    if probe.read_text().strip() != "RMS4_TARGET_REFUSAL_PASS cases=65 modes=5":
        raise ValueError("independent refusal/rounding target coverage differs")
    if packet["accuracy"]["original_gate"] != {"atol":.03125,"rtol":.02,"elements":1600} or not packet["policy"]["approximate_not_certificate"]:
        raise ValueError("explicit approximation/user gate scope differs")
    review = {
        "schema":"root_rms4_fast_stock_packet_review_v1",
        "status":"RELEASED_ONE_STOCK_COMPLETE_GROUP_EXPERIMENT",
        "created_utc":args.utc,
        "pins_reclosed":len(pins),"historical_positive_pins_preserved":223,
        "native_original_elements":1600,"native_original_elementwise_failures":failures,"native_bit_differences":bit_differences,
        "native_source_groups":48,"native_products":23040,"native_fallbacks":0,
        "target_refusal_cases":65,"target_rounding_modes":5,
        "candidate":str(candidate),"candidate_elf_sha256":digest(candidate),
        "control_elf_sha256":digest(control),"strict_roi_instructions":int(counters[0]),
        "control_roi_instructions":target["control_instructions"],
        "instruction_improvement_fraction":1-int(counters[0])/target["control_instructions"],
        "hardware_alias":"alveo_u250_firesim_gemmini_rocket_stock",
        "hardware_config":"FireSimGemminiRocketConfig","allowed_runs":1,
        "release_scope":"Original complete 12-head source group only; explicit experimental RMS4 approximation, immutable ELF and complete consumer validation. No whole run or default promotion.",
        "uart_required":{"roi_label":"WORKSPACE_GROUP_INSTRUCTIONS","pass_marker":packet["uart_contract"]["pass_marker"],"stats":target["stats"],"carrier_differences":3,"output_checksum":"NOT_EMITTED; full linked compiled-consumer comparison required, no checksum inferred"},
        "rejections":packet["uart_contract"]["required_rejections"],
        "whole_hardware_cycles":"UNKNOWN","whole_prediction":"UNKNOWN",
        "pins":{str(p):digest(p) for p in (args.packet,historical_path,output_path,gold_path,stdout_path,Path(__file__))},
    }
    args.output.write_text(json.dumps(review,indent=2)+"\n")
    print(json.dumps({k:review[k] for k in ("status","pins_reclosed","native_original_elementwise_failures","strict_roi_instructions","instruction_improvement_fraction")}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("packet","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--utc",required=True)
    close(parser.parse_args())
