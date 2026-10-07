"""Compare source-bound paired resident command retention before timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from test_resident_conv_command_loops import executed_commands
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, StringAttr
from xdsl.dialects.llvm import LLVM
from xdsl.parser import Parser

from mlir_oot.conv_schedule import source_stride_resource_layout
from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_resident_conv import GoldenResidentConv
from mlir_oot.ir.gemmini_dialect import GEMMINI
from mlir_oot.readout_store_plan import PairedReadoutPlan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--source-symbol", required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.paired_manifest.read_text())
    routes = [r for r in manifest["routes"] if r["symbol"] == args.source_symbol]
    if len(routes) != 1 or not routes[0]["paired_readout"]["applied"]:
        raise ValueError("one paired semantic route required")
    route = routes[0]
    proof = route["paired_readout"]["proof"]
    plan = PairedReadoutPlan(
        tuple(proof["source_scales"]),
        tuple(proof["store_scales"]),
        proof["lo"],
        proof["hi"],
        proof["relu"],
    )
    if plan.certificate() != proof:
        raise ValueError("complete source certificate changed")
    fixture = json.loads((args.fixture / "receipt.json").read_text())
    shape = ConvShape(**fixture["shape"])
    if list(plan.source_scales) != fixture["source_numeric_proof"]["source_scales"]:
        raise ValueError("fixture source scales differ")
    resident, resources = source_stride_resource_layout(shape)
    source_dir = args.paired_manifest.parent / args.source_symbol
    source_ir, source_obj = source_dir / "kernel.gemmini.mlir", source_dir / "kernel.o"
    if sha(source_obj) != route["compilation"]["object_sha256"]:
        raise ValueError("source object is stale")
    if sha(source_ir) != route["compilation"]["target_ir_sha256"]:
        raise ValueError("source target module is stale")
    context = Context(allow_unregistered=True)
    for dialect in (Builtin, LLVM, GEMMINI):
        context.load_dialect(dialect)
    actual = Parser(context, source_ir.read_text()).parse_module()
    streams = [executed_commands(actual)]
    compiles = {}
    for name, compact in (("default", False), ("compact", True)):
        generator = GoldenResidentConv(
            resident.conv,
            rows_per_tile=resident.rows_per_tile,
            source_stride=True,
            weight_base=resident.explicit_weight_base,
            row_residue=resident.row_residue,
            compact_commands=compact,
            store_plan=plan,
        )
        module = generator.build()
        module.body.block.first_op.properties["sym_name"] = StringAttr(route["kernel"])
        streams.append(executed_commands(module))
        if streams[-1] != streams[0]:
            raise ValueError(
                f"{name}: actual command order, fields or DMA pointers changed"
            )
        compiles[name] = compile_module(module, args.llvm_bin, root / name)
    for name, source in (
        ("source", source_obj),
        ("default", root / "default/kernel.o"),
        ("compact", root / "compact/kernel.o"),
    ):
        subprocess.run(
            [
                str(args.llvm_bin / "llvm-objcopy"),
                "--dump-section",
                f".text={root / (name + '.text')}",
                str(source),
                str(root / (name + ".object_copy")),
            ],
            check=True,
            capture_output=True,
        )
    default_equal = (root / "source.text").read_bytes() == (
        root / "default.text"
    ).read_bytes()
    if not default_equal:
        raise ValueError("default actual source object text changed")
    result = {
        "schema": "paired_source_stride_retention_prelabel_v1",
        "status": "PASS",
        "source_symbol": args.source_symbol,
        "source_kernel": route["kernel"],
        "source_target_module": str(source_ir),
        "source_target_ir_sha256": sha(source_ir),
        "source_object": str(source_obj),
        "source_object_sha256": sha(source_obj),
        "fixture_receipt_sha256": sha(args.fixture / "receipt.json"),
        "paired_manifest_sha256": sha(args.paired_manifest),
        "source_proof": proof,
        "resources": resources,
        "default_source_text_byte_identical": default_equal,
        "all_three_command_pointer_streams_identical": True,
        "command_count": len(streams[0]),
        "command_counts": dict(Counter(row[0] for row in streams[0])),
        "encoded_sequence_repr_sha256": hashlib.sha256(
            repr(streams[0]).encode()
        ).hexdigest(),
        "compile_receipts": compiles,
        "text_bytes": {
            name: (root / (name + ".text")).stat().st_size
            for name in ("source", "default", "compact")
        },
        "hypotheses_before_labels": {
            "primitive_only": "Tie: same complete issued command and requested-transfer stream",
            "issue_footprint": "Retained bounded ordinary loops may reduce fetch footprint but add issue/address arithmetic",
            "admissible_cycle_prediction": "UNKNOWN: no qualified mixed command/CPU overlap and code-footprint price",
            "numeric_semantics": "Same source K order, both paired stores, final fence and exact decoder",
        },
        "timing_labels": None,
    }
    (root / "prelabel.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
