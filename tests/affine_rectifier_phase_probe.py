"""Minimize the first ACC execute-path RTL failure at completed-command boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from affine_rectifier_capability_probe import generate, module, pin
from merlin.perf.layer_bench.build import build_program
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i64

from mlir_oot.codegen.builder import PTR, FnBuilder
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.tables import rtl_facts as F


def build(args):
    work = args.workdir.resolve()
    generate(work)
    original = module()
    function = original.ops.first
    segments = []
    current = []
    for op in function.body.blocks.first.ops:
        if isinstance(op, llvm.ReturnOp):
            continue
        current.append(op)
        if op.name == "gemmini.fence":
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    functions = []
    for index, segment in enumerate(segments):
        fb = FnBuilder([PTR] * 6)
        mapping = dict(zip(function.body.blocks.first.args, fb.entry.args, strict=True))
        for op in segment:
            fb.add(op.clone(value_mapper=mapping))
        functions.append(
            llvm.FuncOp(
                "affine_cap_step_" + str(index),
                llvm.LLVMFunctionType([PTR] * 6),
                linkage=llvm.LinkageAttr("external"),
                body=fb.finish(),
            )
        )
    result = ModuleOp(functions)
    result.attributes["gemmini.dim"] = IntegerAttr(F.DIM, i64)
    result.verify()
    source = (work / "probe.c").read_text()
    signature = (
        "(const int32_t*a,const int8_t*b,const int8_t*c,int8_t*d,int8_t*e,int32_t*f)"
    )
    declarations = "\n".join(
        "extern void affine_cap_step_" + str(index) + signature + ";"
        for index in range(len(segments))
    )
    calls = "\n".join(
        'printf("CAP_PHASE begin='
        + str(index)
        + '\\n");affine_cap_step_'
        + str(index)
        + '(a,b,c,d,e,f);printf("CAP_PHASE end='
        + str(index)
        + '\\n");'
        for index in range(len(segments))
    )
    source = source.replace(
        "extern void affine_rectifier_capability(const int32_t*,const int8_t*,const int8_t*,int8_t*,int8_t*,int32_t*);",
        declarations
        + "\nvoid affine_rectifier_capability"
        + signature
        + "{"
        + calls
        + "}",
    )
    (work / "probe.c").write_text(source)
    compilation = compile_module(result, args.llvm_bin, work / "device")
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
                "segments": [[op.name for op in segment] for segment in segments],
                "scope": "Extra function/printf/fence diagnostic; no performance attribution",
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
