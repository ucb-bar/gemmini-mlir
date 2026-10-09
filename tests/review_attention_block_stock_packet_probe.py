"""Release a complete current-source attention cost capsule after strict closure."""

import argparse
import json
import re
from pathlib import Path

from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def review(args):
    require(not args.output.exists(), "release must be fresh")
    q = json.loads(args.packet.read_text())
    for path, digest in q["pins"].items():
        require(sha(path) == digest, "changed block evidence: " + path)
    raw_path = Path(q["raw_qualification"])
    require(sha(raw_path) == q["raw_qualification_sha256"], "raw target qualification differs")
    raw = json.loads(raw_path.read_text())
    for path, digest in raw["pins"].items():
        require(sha(path) == digest, "changed raw target dependency: " + path)
    require(raw["status"] == "pass" and raw["returncode"] == 0 and
            q["status"] == "exact_source_target_block_qualified" and not q["candidate_new_policy"],
            "block must be current source, fully qualified")
    elf = Path(raw["elf"])
    require(sha(elf) == raw["elf_sha256"], "final ELF differs")
    audit = audit_elf(elf.read_bytes())
    require(audit["status"] == "pass" and not audit["forbidden"] and not audit["unknown"],
            "all executable sections noFSM refused")
    stdout = raw_path.parent / "spike.stdout"
    text = stdout.read_text()
    modes = re.findall(r"^ATTENTION_MODE_PASS mode=(\d+) sticky=(\d+) words=(\d+) digest=([0-9a-f]+)$", text, re.M)
    require(modes == [(str(mode), str(sticky), "16384", "9e1846e713bbd21d")
                      for mode in range(5) for sticky in range(2)], "mode/consumer/input guard closure differs")
    pattern = (r"^ATTENTION_BLOCK_ROW id=(\d+) sample=(\d+) cycles=(\d+) instructions=(\d+) "
               r"digest=([0-9a-f]+) score=([0-9a-f]+) p=([0-9a-f]+) endpoint=([0-9a-f]+) quant=([0-9a-f]+)$")
    entries = re.findall(pattern, text, re.M)
    require(len(entries) == 4 and [int(e[0]) for e in entries] == [0, 1, 1, 0], "complete ABBA differs")
    for actual, expected in zip(entries, q["complete_cost_scope"]["rows"]):
        for index, field in ((0, "id"), (1, "sample"), (2, "cycles"), (3, "instructions")):
            require(int(actual[index]) == expected[field], "strict row counter differs")
        for index, field in ((4, "digest"), (5, "score"), (6, "p"), (7, "endpoint"), (8, "quant")):
            require(int(actual[index], 16) == expected[field], "source addresses/observation differ")
    require(len({e[5:] for e in entries}) == 1 and
            "ATTENTION_BLOCK_PASS original16384i8 all5modes sticky2 inputs sameheap sourceendpointgate" in text,
            "complete source publication or common-address scope differs")
    source_gates = re.findall(r"^ATTENTION_SOURCE_GATE .*endpointfail=(\d+) i8changed=(\d+)$", text, re.M)
    require(source_gates == [("0", "0")] * 6, "original source output gate differs")
    stock = json.loads(args.hardware_review.read_text())
    for path, digest in stock["pins"].items():
        require(sha(path) == digest, "stock identity evidence differs")
    result = {
        "schema": "root_complete_current_attention_block_stock_release_v1",
        "status": "RELEASED_ONE_COMPLETE_ATTENTION_BLOCK_STOCK_OBSERVATION",
        "candidate": str(elf), "candidate_elf_sha256": sha(elf),
        "qualification_pins_reclosed": len(q["pins"]), "raw_target_pins_reclosed": len(raw["pins"]),
        "final_noFSM": audit, "allowed_runs": 1,
        "hardware_alias": stock["hardware_identity"]["hw_config"],
        "hardware_identity_required": stock["hardware_identity"],
        "uart_required": {"pass_marker": "ATTENTION_BLOCK_PASS original16384i8 all5modes sticky2 inputs sameheap sourceendpointgate",
                          "words": 16384, "digest": "9e1846e713bbd21d", "ABBA": [0, 1, 1, 0],
                          "strict_retired_instructions": [int(e[3]) for e in entries],
                          "all5FRM_x2sticky": True, "source_endpoint_gate": True},
        "complete_cost_scope": q["complete_cost_scope"],
        "native_vs_actualtarget_source_math": q["native_vs_actualtarget_source_math"],
        "numeric_gate": {"atol": .03125, "rtol": .02, "original_consumer_i8_exact": True},
        "candidate_policy": "Existing current2070 masked-QK, original remaining source pipeline; no new numerical policy.",
        "performance_model": "UNKNOWN; measure complete allocation/math/requant/free on stock. Do not multiply first-context cost by22 or infer whole gain.",
        "whole_default_promotion": False,
        "requirements": ["Recovery submits once after root commit; exact staged ELF/stock hardware identity.",
                         "All original source comparisons, guards, mode checks and common-address ABBA pass.",
                         "Retain hardware mcycle and retired-instruction metrics separately; Spike cycles are functional work only.",
                         "This prices already implemented attention; rejected approximate i8 attention remains unqualified."],
        "pins": {str(path.resolve()): sha(path) for path in
                 (args.packet, args.hardware_review, raw_path, elf, stdout, Path(__file__))},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "candidate_elf_sha256", "qualification_pins_reclosed")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "hardware-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
