"""Ahead-of-time Gemmini device library for every exact integer model contraction.

Identical shapes share one specialized kernel.  Each distinct schedule receives a
stable symbol, and batched wrappers call their own private core.  This produces
one object for the model compiler to link after host scheduling and allocation.
The catalog alone is not an executable whole model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr

from .contraction_patterns import match_integer_gemm
from .frontend.parse import parse_module
from .golden_batched_gemm import build as build_batched
from .golden_contraction_upstream import choose_shape
from .golden_device_compile import compile_module
from .golden_gemm import GoldenGemm
from .no_fsm_audit import audit_elf


def _symbol(key: dict) -> str:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return "gemmini_golden_" + digest


def build_catalog(source: str, *, max_kernels: int | None = None, large_n: bool = False) -> tuple[ModuleOp, dict]:
    if max_kernels is not None and max_kernels <= 0:
        raise ValueError("max_kernels must be positive")
    source_module = parse_module(source)
    module = ModuleOp([])
    bindings = []
    kernels: dict[str, dict] = {}
    matched_total = 0
    for ordinal, op in enumerate(source_module.walk()):
        dims = match_integer_gemm(op)
        if dims is None:
            continue
        matched_total += 1
        shape = choose_shape(dims,large_n=large_n)
        batched = len(op.operands[0].type.get_shape()) == 3
        key = {"batch": dims.batch if batched else 0,
               "shape": asdict(shape), "batched": batched}
        symbol = _symbol(key)
        if symbol not in kernels and (max_kernels is None or len(kernels) < max_kernels):
            local = build_batched(dims.batch, shape) if batched else GoldenGemm(shape).build()
            for fn in list(local.body.blocks[0].ops):
                old = fn.sym_name.data
                new = symbol if old.endswith("batched_gemm") or not batched else symbol + "_core"
                fn.properties["sym_name"] = StringAttr(new)
                for nested in fn.walk():
                    if nested.name == "llvm.call":
                        nested.properties["callee"] = SymbolRefAttr(
                            StringAttr(symbol + "_core"), ArrayAttr([]))
                fn.detach()
                module.body.blocks[0].add_op(fn)
            kernels[symbol] = {"symbol": symbol, "dimensions": asdict(dims),
                               "schedule": asdict(shape), "batched": batched}
        if symbol in kernels:
            bindings.append({"source_operation_ordinal": ordinal,
                             "region": getattr(op.attributes.get("prov.region_id"), "data", ""),
                             "symbol": symbol,
                             "tensor_types": [str(x.type) for x in op.operands[:2]] +
                                             [str(op.results[0].type)]})
    module.attributes["gemmini.golden_catalog"] = StringAttr(
        json.dumps({"kernels": len(kernels), "bindings": len(bindings)}, sort_keys=True))
    module.verify()
    return module, {"schema": "gemmini_golden_device_catalog_v1",
                    "abi": {"argument_order": ["lhs", "rhs", "out"],
                            "pointee_layout": "dense_row_major",
                            "batch_call": "whole_batch",
                            "dtypes": ["i8", "i8", "i32"]},
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "schedule_policy": "large_n_grouped_b_v1" if large_n else "default",
                    "matched_contractions": matched_total,
                    "covered_contractions": len(bindings),
                    "coverage_complete": len(bindings) == matched_total,
                    "unique_kernels": len(kernels),
                    "kernels": list(kernels.values()),
                    "bindings": bindings}


def compile_catalog(source_path: Path, llvm_bin: Path, workdir: Path,
                    *, max_kernels: int | None = None, large_n: bool = False) -> dict:
    # The model compiler may overwrite its prepared path during offload. Keep
    # the exact bytes consumed here so bindings remain independently replayable.
    source_bytes = source_path.read_bytes()
    module, manifest = build_catalog(source_bytes.decode('utf-8'), max_kernels=max_kernels,large_n=large_n)
    if not manifest["covered_contractions"]:
        raise ValueError("source has no exact integer GEMM contractions")
    receipt = compile_module(module, llvm_bin, workdir)
    source_snapshot = workdir / 'catalog_source.mlir'
    source_snapshot.write_bytes(source_bytes)
    result = {**manifest, "compilation": receipt,
              "source_snapshot": str(source_snapshot.resolve()),
              "whole_model_memory_bound": False,
              "whole_model_correctness_verified": False}
    (workdir / "device_catalog.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def merlin_builder(llvm_bin: Path, *, large_n: bool = False):
    """Compile Merlin's final prepared IR at its pre-offload source boundary."""
    compiler = Path(llvm_bin)

    def build(prepared: Path, workdir: Path) -> tuple[Path, Path]:
        prepared, workdir = Path(prepared), Path(workdir)
        compile_catalog(prepared, compiler, workdir,large_n=large_n)
        return workdir / "device_catalog.json", workdir / "kernel.o"

    return build


def merlin_identity_view_transform(prepared: Path, workdir: Path) -> Path:
    """Apply Merlin's proven 1x1 im2col view before catalog source binding."""
    try:
        from merlin.llvmlower.im2col_identity_view import rewrite_prepared_file
    except ImportError as exc:
        raise RuntimeError("Merlin's identity im2col pass is required") from exc
    prepared, workdir = Path(prepared), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    transformed, report = rewrite_prepared_file(prepared, workdir)
    transformed = Path(transformed)
    # The upstream pass parses its own IR; this target additionally verifies
    # the mixed i8*i8->i32 named-matmul form it will compile and bind.
    parse_module(transformed.read_text())
    (workdir / "identity_view_report.json").write_text(
        json.dumps({"schema": "gemmini_identity_view_transform_v1",
                    "source_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
                    "prepared_sha256": hashlib.sha256(transformed.read_bytes()).hexdigest(),
                    **report.to_dict()}, indent=2) + "\n"
    )
    return transformed


def final_elf_audit(elf: Path) -> None:
    """Refuse a linked model that gained any forbidden hardware-loop instruction."""
    elf = Path(elf)
    report = audit_elf(elf.read_bytes())
    elf.with_suffix(".nofsm_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        raise ValueError(f"linked model has forbidden device instructions: {elf}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--max-kernels", type=int,
                    help="compile a partial catalog for debugging; receipt flags incomplete coverage")
    ap.add_argument("--large-n", action="store_true", help="opt in to grouped B loads and short-M A reuse")
    args = ap.parse_args()
    report = compile_catalog(args.input, args.llvm_bin, args.workdir,
                             max_kernels=args.max_kernels,large_n=args.large_n)
    print(json.dumps({k: report[k] for k in ("matched_contractions", "covered_contractions",
                                             "coverage_complete", "unique_kernels")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
