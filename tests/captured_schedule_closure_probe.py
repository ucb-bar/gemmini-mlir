"""Reclose complete source-bound, common-address schedule capsules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mlir_oot.no_fsm_audit import audit_elf


def check(receipt: dict) -> None:
    for path, expected in receipt["pins"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"artifact changed: {path}")
    for pair in receipt["pairs"]:
        fixture = json.loads(Path(pair["fixture_receipt"]).read_text())
        results = []
        for filename in pair["results"]:
            path = Path(filename)
            result = json.loads(path.read_text())
            if (
                result["status"] != "pass"
                or not result["run_completed"]
                or result["run_returncode"] != 0
                or result["strict_target"]["status"] != "pass"
            ):
                raise ValueError("capsule did not close strict and hardware execution")
            if result["guards_checked"] != 4096:
                raise ValueError("dirty output guard check missing")
            if (
                result["fixture_receipt_sha256"]
                != hashlib.sha256(
                    Path(pair["fixture_receipt"]).read_bytes()
                ).hexdigest()
            ):
                raise ValueError("capsule is bound to another fixture")
            expected_type_bytes = 1 if fixture["shape"]["output_dtype"] == "i8" else 4
            extent = (
                Path(pair["fixture_receipt"])
                .parent.joinpath("expected.bin")
                .stat()
                .st_size
            )
            if result["outputs_checked"] * expected_type_bytes != extent:
                raise ValueError("partial output check")
            inputs = sum(
                Path(pair["fixture_receipt"])
                .parent.joinpath(name + ".bin")
                .stat()
                .st_size
                for name in ("a", "b")
            )
            if result["immutable_input_bytes_checked"] != inputs:
                raise ValueError("input preservation check missing")
            marker = f"CAPTURED_CONV_PASS N={result['outputs_checked']} SCHEDULE={result['schedule']}"
            if (
                marker not in result["stdout"]
                or marker not in (path.parent / "spike.stdout").read_text()
            ):
                raise ValueError("complete target/source checks missing")
            if result["engine_sha256"] != receipt["engine_sha256"]:
                raise ValueError("paired engine changed")
            if audit_elf((path.parent / "layer.elf").read_bytes())["status"] != "pass":
                raise ValueError("linked capsule gained forbidden instructions")
            if result["kernel_cycles"] is None:
                raise ValueError("ROI timer missing")
            results.append(result)
        if (
            len(results) != 2
            or results[0]["common_operand_addresses"]
            != results[1]["common_operand_addresses"]
        ):
            raise ValueError("pair does not use common operand/output addresses")
        control = Path(pair["results"][0]).parent / "kernel.o"
        if (
            hashlib.sha256(control.read_bytes()).hexdigest()
            != fixture["source_kernel_object_sha256"]
        ):
            raise ValueError("control implementation changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    check(receipt)
    print(f"CAPTURED_SCHEDULE_CLOSURE_PASS {len(receipt['pins'])} pins")


if __name__ == "__main__":
    main()
