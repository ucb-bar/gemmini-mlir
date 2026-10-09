"""Reclose the complete-cost negative without treating reduced callbacks as a win."""

import ast
import argparse
import json
import re
import subprocess
from pathlib import Path

from mlir_oot.no_fsm_audit import audit_elf
from review_closed_i8_stock_capsule_probe import require, sha


def executable_ast(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def review(args):
    require(not args.output.exists(), "negative review must be fresh")
    packet = json.loads(args.packet.read_text())
    for path, digest in packet["pins"].items():
        require(sha(path) == digest, "negative dependency changed: " + path)
    core = args.core
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip()
    require(head == packet["core_head"], "generic core head changed")
    relative = "src/merlin/llvmlower/radix_digit_norm_enclosure.py"
    module_path = core / relative
    old = subprocess.check_output(["git", "show", "6f97a6623:" + relative], cwd=core)
    require(executable_ast(old) == executable_ast(module_path.read_bytes()),
            "compiled norm implementation changed beyond docstrings")
    console_paths = [Path(path) for path in packet["pins"]
                     if path.endswith("/norm_readout/strict/stdout")]
    require(len(console_paths) == 1, "actual complete strict cost log missing")
    console = console_paths[0].read_text().replace("\r", "")
    metrics = packet["retired_instructions"]
    statistics = packet["native"]["statistics"]
    require(re.findall(r"^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$", console, re.M)
            == [str(metrics["digit_norm_partial"])], "complete strict counter differs")
    require(re.findall(r"^WORKSPACE_STAT (\d+) (\d+)$", console, re.M)
            == [(str(i), str(n)) for i, n in enumerate(statistics)], "native/strict statistics differ")
    require("WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS" in console
            and "CAPSULE UNEXPECTED SOURCE REFUSAL" not in console,
            "original consumer or source admission failed")
    require(packet["native"]["guards_inputs_unchanged"]
            and not packet["native"]["callback_errors"] and packet["native"]["status"] == "PASS",
            "native original consumer failed")
    elf_paths = [Path(path) for path, digest in packet["pins"].items()
                 if digest == packet["candidate_elf"]]
    require(len(elf_paths) == 1, "actual final ELF missing")
    audit = audit_elf(elf_paths[0].read_bytes())
    require(audit["status"] == "pass", "all executable section noFSM audit failed")
    increase = metrics["digit_norm_partial"] / metrics["exact2024"] - 1
    increase_prior = metrics["digit_norm_partial"] / metrics["canonical_partial_negative"] - 1
    require(increase > 0 and increase_prior > 0
            and increase == packet["increase_vs_exact"]
            and increase_prior == packet["increase_vs_canonical_negative"],
            "actual complete-cost negative changed")
    paths = [args.packet, module_path, core / "merlin/tests/ir/test_radix_digit_norm_enclosure.py",
             console_paths[0], elf_paths[0], Path(__file__)]
    result = {
        "schema": "root_digit_norm_partial_complete_negative_review_v1",
        "status": "CORRECT_BUT_REJECTED_COMPLETE_INSTRUCTION_COST",
        "pins_reclosed": len(packet["pins"]), "core_head": head,
        "compiled_core_executable_AST_unchanged": True,
        "module_byte_identity": "FALSE; later alias documentation changes only a docstring",
        "original_consumer_native_strict_and_guards_pass": True,
        "retired_instructions": metrics, "fraction_increase_vs_exact": increase,
        "fraction_increase_vs_prior_negative": increase_prior,
        "callback_count": packet["callbacks"], "requested_readback_bytes": packet["readback_bytes"],
        "digit_scan_words": packet["extra_digit_scan_words"], "metadata_bytes": packet["metadata_bytes"],
        "workspace_bytes": packet["workspace_bytes"], "statistics": statistics, "nofsm": audit,
        "source_review": "Checked complete-vector/stride/extent admission, signed byte range, owner/generation/source binding and weighted integer prefixes. Private C metadata authenticity and immutable lifetime remain compiler obligations, not authentication against arbitrary writes.",
        "installed_qualification": "NOT_PERFORMED; source-only prototype, no upstream/default promotion",
        "hardware_cycles": "UNKNOWN; no hardware submission", "whole_run": False,
        "pins": {str(path.resolve()): sha(path) for path in paths},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                     ("status", "pins_reclosed", "fraction_increase_vs_exact", "fraction_increase_vs_prior_negative")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "core", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
