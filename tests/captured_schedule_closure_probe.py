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
        paired = pair.get("paired_readout", False)
        results = []
        for filename in pair["results"]:
            path = Path(filename)
            result = json.loads(path.read_text())
            if (
                result["status"] != "pass"
                or not result["run_completed"]
                or result["run_returncode"] != 0
                or result.get("strict_target", result.get("strict_spike", {})).get(
                    "status"
                )
                != "pass"
            ):
                raise ValueError("capsule did not close strict and hardware execution")
            if result["guards_checked"] != (8192 if paired else 4096):
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
            expected_extent = (
                extent // 4 * 2 if paired else extent // expected_type_bytes
            )
            if result["outputs_checked"] != expected_extent:
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
            marker = (
                f"PAIRED_CONV_PASS N={result['outputs_checked'] // 2}"
                if paired
                else f"CAPTURED_CONV_PASS N={result['outputs_checked']} SCHEDULE={result['schedule']}"
            )
            if (
                marker not in result["stdout"]
                or marker not in (path.parent / "spike.stdout").read_text()
            ):
                raise ValueError("complete target/source checks missing")
            if result["engine_sha256"] != receipt["engine_sha256"]:
                raise ValueError("paired engine changed")
            if audit_elf((path.parent / "layer.elf").read_bytes())["status"] != "pass":
                raise ValueError("linked capsule gained forbidden instructions")
            if result.get("kernel_cycles", result.get("cycles")) is None:
                raise ValueError("ROI timer missing")
            results.append(result)
        if (
            len(results) != 2
            or results[0]["common_operand_addresses"]
            != results[1]["common_operand_addresses"]
        ):
            raise ValueError("pair does not use common operand/output addresses")
        control = Path(pair["results"][0]).parent / "kernel.o"
        control_sha = fixture.get("source_kernel_object_sha256")
        if paired:
            from mlir_oot.golden_conv import ConvShape
            from mlir_oot.readout_store_plan import PairedReadoutPlan

            manifest = json.loads(Path(pair["paired_manifest"]).read_text())
            route = next(
                r for r in manifest["routes"] if r["symbol"] == pair["source_symbol"]
            )
            proof = route["paired_readout"]["proof"]
            plan = PairedReadoutPlan(
                tuple(proof["source_scales"]),
                tuple(proof["store_scales"]),
                proof["lo"],
                proof["hi"],
                proof["relu"],
            )
            plan.require_conv_producer(ConvShape(**fixture["shape"]))
            if (
                tuple(fixture["source_numeric_proof"]["source_scales"])
                != plan.source_scales
                or plan.certificate() != proof
                or any(r["proof"] != proof for r in results)
            ):
                raise ValueError("paired source/certificate changed")
            control_sha = route["compilation"]["object_sha256"]
        if hashlib.sha256(control.read_bytes()).hexdigest() != control_sha:
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
