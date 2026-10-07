"""Review the controlled normal integer-observation whole-model composition."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review(args):
    require(not args.output.exists(), "whole release output must be fresh")
    q = json.loads(args.packet.read_text())
    for path, expected in q["pins"].items():
        require(sha(path) == expected, "changed whole qualification: " + path)
    whole = q["whole"]
    candidate = Path(whole["ELF_path"])
    require(sha(candidate) == whole["ELF_sha256"], "final candidate ELF changed")
    audit = audit_elf(candidate.read_bytes())
    require(audit["status"] == "pass" and not audit["forbidden"] and not audit["unknown"],
            "every executable section must have zero forbidden/unknown custom instructions")
    adapter_path = Path(whole["reference_adapter_path"])
    adapter = json.loads(adapter_path.read_text())
    for key in ("reference", "torch_golden", "spike_console", "controlled_link", "source_qualification"):
        require(sha(adapter[key + "_path"]) == adapter[key + "_sha256"], "changed original adapter dependency")
    build_path = Path(adapter["controlled_link_path"])
    build = json.loads(build_path.read_text())
    old_build_path = next(Path(path) for path in build["pins"]
                          if path.endswith("masked-contraction-normal-whole-v3-20261007/build.json"))
    old_build = json.loads(old_build_path.read_text())
    for path, digest in build["candidate_objects"].items():
        require(sha(path) == digest, "actual candidate object differs")
        if Path(path).name != "model.o":
            require(old_build["candidate_objects"].get(path) == digest, "nonmodel object changed")
    require(len(build["candidate_objects"]) == len(old_build["candidate_objects"]) == 12
            and build["all155devicebindings"] and build["original_11_nonmodel_objects_unchanged"],
            "target binding/linked leaf coverage changed")
    baseline = Path(build["baseline_reproduction_argv"][-1])
    require(sha(baseline) == build["baseline_elf_sha256"] ==
            "77dbb5d85b4ddd2a0bc69e445398e85141aa404eedecc1c733998c5ebe9a79f7",
            "actual2062 final ELF not reproduced")
    original_model = next(Path(path) for path in old_build["candidate_objects"] if Path(path).name == "model.o")
    control_model = candidate.parent.parent.parent / "control/target/model.o"
    require(sha(control_model) == sha(original_model), "normal control object not reproduced")
    def normalize(argv):
        return ["MODEL" if Path(value).name == "model.o" else "OUTPUT" if index == len(argv) - 1 else value
                for index, value in enumerate(argv)]
    require(normalize(build["candidate_link_argv"]) == normalize(old_build["candidate_link_argv"]),
            "other link options or ordered object leaves changed")
    reference, gold = (np.load(adapter[key], allow_pickle=False)
                       for key in ("reference_path", "torch_golden_path"))
    require(reference.dtype == np.float32 and reference.shape == gold.shape == (1, 8, 32000),
            "original whole output type/extent changed")
    raw = hashlib.sha256(reference.astype("<f4", copy=False).tobytes()).hexdigest()
    require(raw == "ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3"
            and np.allclose(reference, gold, atol=.03125, rtol=.02), "original compiled/Torch gate failed")
    console_path = Path(adapter["spike_console_path"])
    console = console_path.read_text()
    require(re.findall(r"^OUT_SHA256 f32le (\d+) (\d+) ([a-f0-9]{64})$", console, re.M) ==
            [("256000", "1024000", raw)] and console.splitlines().count("DONE") == 1
            and console.splitlines().count("METRIC memref_rank_mismatch 0") == 1
            and re.findall(r"^METRIC cycles (\d+)$", console, re.M) == ["126586933"]
            and adapter["exit_code"] == 0 and adapter["elf_sha256"] == sha(candidate),
            "complete strict original output/terminal/counter differs")
    stock = json.loads(args.cost_review.read_text())
    for path, digest in stock["pins"].items():
        require(sha(path) == digest, "complete helper stock review changed")
    require(stock["job_id"] == 2067 and stock["fraction_lower"] > 0,
            "complete actual stock cost gate missing")
    result = {
        "schema": "root_closed_i8_normal_whole_stock_release_v1",
        "status": "RELEASED_ONE_CONTROLLED_WHOLE_STOCK_OBSERVATION",
        "qualification_pins_reclosed": len(q["pins"]), "candidate": str(candidate),
        "candidate_elf_sha256": sha(candidate), "standard_adapter": str(adapter_path),
        "control_job": 2062, "control_cycles": 410147055, "control_elf_sha256": sha(baseline),
        "normal_control_object_and_finalELF_reproduced": True,
        "original_compiled_words": 256000, "raw_original_sha256": raw,
        "original_Torch_gate": {"atol": .03125, "rtol": .02, "passed": True},
        "unchanged_nonmodel_linked_objects": 11, "target_boundaries": 155,
        "final_noFSM": audit, "strict_retirement_proxy": 126586933,
        "whole_cycle_prediction": "UNKNOWN", "default_promotion": False,
        "hardware_alias": "alveo_u250_firesim_gemmini_rocket_stock",
        "hardware_config": "FireSimGemminiRocketConfig", "allowed_runs": 1,
        "hardware_identity_required": stock["hardware_identity"],
        "requires": ["Recovery submits once after root commit; actual exact staged ELF/config/stockbitstream/terminal.",
                     "Complete256000 original rawdigest/DONE/rank0/standardadapter and originalTorch gate.",
                     "One complete model-forward cycle observation compared with2062; helper14.07% is not a whole forecast."],
        "scope": "Normal pre-object host callback on frozen2062 source plus controlled whole link. Direct already-certified integer publication; table/source/cold continuation and155 device bindings unchanged. Explicit opt-in; not a fresh whole upstream recapture or default promotion.",
        "pins": {str(path.resolve()): sha(path) for path in
                 (args.packet, args.cost_review, candidate, adapter_path, build_path, old_build_path,
                  console_path, Path(adapter["reference_path"]), Path(adapter["torch_golden_path"]), Path(__file__))},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "qualification_pins_reclosed", "candidate_elf_sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "cost-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
