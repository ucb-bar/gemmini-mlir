"""Close actual normal result ownership and the pinned GSIM panel regime."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from merlin.llvmlower.fresh_tensor_writer import (
    FreshTensorWriterContract,
    prove_lowered_writer_result,
)
from xdsl.dialects import llvm
from xdsl.parser import Parser

from mlir_oot.captured_residual_bundle import binding_attributes
from mlir_oot.frontend.parse import context, parse_module
from mlir_oot.golden_streamed_resadd import panel_storage


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in (
        "fresh-module",
        "host-llvm",
        "normal-host-object",
        "linked-host-object",
        "allocator-reproduction",
        "gsim-receipt",
        "fir",
        "dts",
        "route-bundle",
        "llvm-bin",
        "out",
    ):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--route-index", type=int, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = json.loads(args.route_bundle.read_text())
    route = bundle["routes"][args.route_index]
    m, n, symbol = route["m"], route["n"], route["symbol"]
    module = parse_module(args.fresh_module.read_text())
    wrapper = next(
        op
        for op in module.body.block.ops
        if op.name == "func.func" and op.sym_name.data == symbol
    )
    expected_attrs = binding_attributes(
        route, bundle["source_sha256"], bundle["capture_receipt_sha256"]
    )
    assert all(
        wrapper.attributes.get(key) == value for key, value in expected_attrs.items()
    )
    assert [value.type.get_shape() for value in wrapper.body.block.args] == [(m, n)] * 3
    ops = tuple(wrapper.body.block.ops)
    assert [op.name for op in ops] == [
        "bufferization.to_buffer",
        "bufferization.to_buffer",
        "memref.alloc",
        "func.call",
        "bufferization.to_tensor",
        "func.return",
    ]
    assert all("read_only" in op.properties for op in ops[:2])
    assert ops[2].alignment.value.data == 64
    assert tuple(ops[3].arguments) == tuple(op.results[0] for op in ops[:3])
    borrowed = ops[3].callee.root_reference.data
    assert "restrict" in ops[4].properties and "writable" in ops[4].properties
    assert tuple(ops[4].operands) == (ops[2].results[0],)
    assert tuple(ops[5].arguments) == (ops[4].results[0],)
    declaration = next(
        op
        for op in module.body.block.ops
        if op.name == "func.func" and op.sym_name.data == borrowed
    )
    assert not declaration.body.blocks
    assert [
        attr.data["bufferization.access"].data for attr in declaration.arg_attrs
    ] == ["read", "read", "write"]
    imported = args.out / "imported_host.mlir"
    importer = args.llvm_bin / "mlir-translate"
    argv = [
        str(importer),
        "--import-llvm",
        "--mlir-print-op-generic",
        str(args.host_llvm),
        "-o",
        str(imported),
    ]
    subprocess.run(argv, check=True, capture_output=True)
    ctx = context()
    ctx.load_dialect(llvm.LLVM)
    lowered = Parser(ctx, imported.read_text()).parse_module()
    functions = {
        op.sym_name.data: op
        for op in lowered.body.block.ops
        if isinstance(op, llvm.FuncOp)
    }
    calls = [
        op
        for op in lowered.walk()
        if isinstance(op, llvm.CallOp)
        and op.callee is not None
        and op.callee.root_reference.data == symbol
    ]
    assert len(calls) == 1
    contract = FreshTensorWriterContract(symbol, 2, (2,), borrowed, 64)
    writer = prove_lowered_writer_result(
        functions[symbol],
        calls[0],
        contract,
        argument_shapes=((m, n),) * 3,
        element_bytes=(1, 1, 1),
        allocator_alignment=64,
    )
    assert args.normal_host_object.read_bytes() == args.linked_host_object.read_bytes()
    runtime = json.loads(args.allocator_reproduction.read_text())
    assert runtime["object_identity"]
    for key, sha_key in (
        ("source", "source_sha256"),
        ("old_object", "old_sha256"),
        ("new_object", "new_sha256"),
    ):
        assert pin(runtime[key])["sha256"] == runtime[sha_key]
    receipt = json.loads(args.gsim_receipt.read_text())
    assert pin(args.fir)["sha256"] == receipt["firrtl_sha256"]
    for row in receipt["inputs"][:2]:
        assert pin(row["path"])["sha256"] == row["sha256"]
    dts = args.dts.read_text()
    assert re.findall(r"d-cache-block-size\s*=\s*<(\d+)>", dts) == ["64"]
    assert re.findall(r"(?<![\w-])cache-block-size\s*=\s*<(\d+)>", dts) == ["64"]
    fir = args.fir.read_text()
    writer_module = fir.split("  module StreamWriter :", 1)[1].split("\n  module ", 1)[
        0
    ]
    lg_sizes = re.findall(
        r"connect write_packets_\d+\.lg_size, UInt<3>\(0h([0-9a-f]+)\)", writer_module
    )
    assert lg_sizes == ["4", "5", "6"]
    storage = panel_storage(m, coherence_granule_bytes=64, dma_max_request_bytes=64)
    paths = [
        value
        for value in vars(args).values()
        if isinstance(value, Path) and value.is_file()
    ]
    paths += [
        imported,
        importer,
        Path(runtime["source"]),
        Path(runtime["old_object"]),
        Path(runtime["new_object"]),
        *(Path(row["path"]) for row in receipt["inputs"][:2]),
    ]
    result = {
        "schema": "source_bound_streamed_normal_storage_v1",
        "route": route,
        "typed_fresh_writer": "read-only A/B; fresh full-writing C; restrict/writable returned owner; borrowed void ABI",
        "actual_lowered_writer": writer,
        "host_object_identical_to_control": True,
        "allocator": runtime,
        "allocator_semantics": "Retained linked source uses monotone bump(n,64), checks arena bounds and has no-op free. Distinct live allocations stay disjoint; reset is between inferences, never inside this borrowed call.",
        "panel_storage": storage,
        "gsim_fir_reproduced_byte_exact": True,
        "gsim_engine_sha256": receipt["binary_sha256"],
        "gsim_dma_write_lg_sizes": lg_sizes,
        "storage_effects": "Generated correction reads immutable A/B and writes only its completed C panel; target pending writes are in the next disjoint transaction partition. No helper allocation/free/retention or RoCC commands.",
        "runtime_alignment_scope": "Explicit linked allocator source/object closure, not earlier typed allocation alignment or sampled pointer values.",
        "stock_bitstream_to_elaboration_lineage": "UNKNOWN; matching quintuplet and current stock FIR do not bind dirty supplied bitstream build",
        "hardware_eligibility": "Pinned GSIM regime only, conditional on exact source/predictor/fence/borrowed-effect qualification; stock held",
        "normal_route_enabled": False,
        "actual_overlap": "UNKNOWN",
        "importer_argv": argv,
        "pins": [pin(path) for path in dict.fromkeys(paths)],
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("STREAMED_NORMAL_STORAGE_PASS", len(result["pins"]), symbol, flush=True)


if __name__ == "__main__":
    main()
