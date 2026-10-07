"""Separate SPAD real-D execute writes from unsupported ACC execute reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from affine_rectifier_capability_probe import generate, module, pin
from merlin.perf.layer_bench.build import build_program
from xdsl.dialects.builtin import IntegerAttr, i64

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.no_fsm_audit import audit_elf


def build(args):
    work = args.workdir.resolve()
    generate(work)
    target = module()
    function = target.ops.first
    load = G.MvinOp(operands=[[function.body.blocks.first.args[2]]], result_types=[[]])
    load.attributes.update(
        {
            key: IntegerAttr(value, i64)
            for key, value in {
                "local": 8192,
                "rows": 16,
                "cols": 16,
                "load_id": 0,
            }.items()
        }
    )
    original_load = next(
        op
        for op in function.body.blocks.first.ops
        if op.name == "gemmini.mvin" and op.a("local") == 0
    )
    function.body.blocks.first.insert_op_after(load, original_load)

    for op in target.walk():
        if op.name == "gemmini.compute" and op.a("a") & (1 << 31):
            op.attributes["a"] = IntegerAttr(0, i64)
            for key, value in {"bd": 8192, "bd_cols": 16, "bd_rows": 16}.items():
                op.attributes[key] = IntegerAttr(value, i64)
    target.verify()
    source = (work / "probe.c").read_text()
    source = source.replace("clipped(seed[j])", "clipped(2*increment[j])").replace(
        "clipped(-seed[j])", "clipped(2*increment[j])"
    )
    source = source.replace(
        'printf("CAP_PASS pairs=256 repeats=2 scaled_acc_reads=2 spad_writes=2 seeded_acc_add=1\\n")',
        'printf("SPAD_CAP_PASS pairs=256 repeats=2 acc_execute_reads=0 spad_writes=2 seeded_acc_add=1\\n")',
    )
    (work / "probe.c").write_text(source)
    compilation = compile_module(target, args.llvm_bin, work / "device")
    header = args.core.resolve() / "merlin/runtime/c/benchmark_buffer.h"
    built = build_program(
        [work / "probe.c", work / "device/kernel.o"],
        work / "build",
        target="gemmini",
        extra_cflags=[
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I",
            str(header.parent),
        ],
    )
    audit = audit_elf(built.elf.read_bytes())
    (work / "built.json").write_text(
        json.dumps(
            {
                "elf": pin(built.elf),
                "device": compilation,
                "audit": audit,
                "scope": "SPAD execute reads and private SPAD writes only; ACC load/update/readback separately checked. No ACC execute read and no STORE_SPAD.",
            },
            indent=2,
        )
        + "\n"
    )
    print(built.elf, built.elf_sha256)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--core", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    build(p.parse_args())
