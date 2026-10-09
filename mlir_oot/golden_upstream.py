"""Compile a supported upstream 1x1 conv capsule through the golden device path.

This is a strict single-layer bridge: it reads typed merlin_iface IR, uses the
shared semantic planner, and refuses any geometry or epilogue the dense GEMM
kernel cannot preserve exactly.  Whole-model graph binding is a separate step.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .frontend import reader
from .golden_device_compile import compile_module
from .golden_gemm import GoldenGemm, Shape, _ceil_div
from .golden_tuning import tune
from .lowering.plan import Builder, Contraction, LoweringDeclined
from .tables import rtl_facts as F


def extract(text: str) -> tuple[Shape, dict]:
    wl = reader.read(reader.parse_module(text))
    conv = [node for node in wl.nodes if node.kind == "conv2d"]
    if len(conv) != 1 or any(node.kind not in ("conv2d", "resident_pack", "evict")
                             for node in wl.nodes):
        raise LoweringDeclined("golden bridge needs one standalone conv2d",
                               op="conv2d")
    plan = Builder(wl).build()
    contractions = [task for task in plan.tasks if isinstance(task, Contraction)]
    if len(contractions) != 1 or len(plan.tasks) != 1:
        raise LoweringDeclined("golden bridge needs exactly one contraction", op="conv2d")
    c = contractions[0]
    node = conv[0]
    if c.lhs not in wl.tensors or plan.command_buffer.get("params", {}).get("im2col_recipes"):
        raise LoweringDeclined("convolution needs a gather the dense kernel cannot perform",
                               op="conv2d", shape=node.out_shape)
    if len(wl.tensors[c.lhs].shape) != 4 or wl.tensors[c.lhs].dtype != "i8":
        raise LoweringDeclined("golden bridge needs an NHWC i8 activation", op="conv2d")
    if wl.tensors[c.rhs].dtype != "i8":
        raise LoweringDeclined("golden bridge needs i8 weights", op="conv2d")
    stages = c.epilogue.stages
    if any(stage not in ("bias_add", "bias", "acc_scale", "relu") for stage in stages):
        raise LoweringDeclined("golden bridge cannot preserve this epilogue", op="conv2d")
    canonical = ["bias_add" if stage == "bias" else stage for stage in stages]
    if canonical != [stage for stage in ("bias_add", "acc_scale", "relu")
                     if stage in canonical]:
        raise LoweringDeclined("golden bridge needs bias, scale, ReLU in that order",
                               op="conv2d")
    if ("bias_add" in stages or "bias" in stages) != bool(c.epilogue.bias):
        raise LoweringDeclined("bias stage and operand disagree", op="conv2d")
    if c.epilogue.bias and (c.epilogue.bias not in wl.tensors or
                            wl.tensors[c.epilogue.bias].shape != [c.n] or
                            wl.tensors[c.epilogue.bias].dtype != "i32"):
        raise LoweringDeclined("golden bridge needs an i32 bias vector", op="conv2d")
    scale = c.epilogue.acc_scale if "acc_scale" in stages else 1.0
    shape = Shape(c.m, c.n, c.k, c.epilogue.output_dtype,
                  bias=bool(c.epilogue.bias), scale=scale,
                  relu="relu" in stages,
                  wide_store=c.n >= 64 and c.epilogue.output_dtype == "i8")
    try:
        shape, _ = tune(shape)
        shape = replace(shape, reuse_b=shape.bm > 1)
        if _ceil_div(shape.m, F.DIM) > shape.bm:
            try:
                cached = replace(shape, cache_b=True)
                cached.validate()
                shape = cached
            except ValueError:
                pass
        # The q1013 1x1 schedule uses 16x64 panel loads.  Apply that exact
        # four-block load form only where all physical panel rows fit and the
        # xDSL verifier can express it; the ordinary narrow path remains the
        # fallback for other channel counts.
        for flag, eligible in (("wide_a", shape.k in (32, 48, 64)),
                               ("wide_b", shape.n in (32, 48, 64))):
            if eligible:
                try:
                    candidate = replace(shape, **{flag: True})
                    candidate.validate()
                    shape = candidate
                except ValueError:
                    pass
        shape.validate()
    except ValueError as exc:
        raise LoweringDeclined(str(exc), op="conv2d", shape=node.out_shape) from exc
    binding = {"A": c.lhs, "B": c.rhs, "C": c.dst}
    if c.epilogue.bias:
        binding["bias"] = c.epilogue.bias
    return shape, {"pointer_binding": binding,
                   "input_tensor_shape": wl.tensors[c.lhs].shape,
                   "output_tensor_shape": plan.buffers[c.dst].shape,
                   "upstream_command_buffer": plan.command_buffer}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    shape, provenance = extract(args.input.read_text())
    receipt = compile_module(GoldenGemm(shape).build(), args.llvm_bin, args.workdir)
    result = {"schema": "gemmini_golden_upstream_1x1_v1",
              "shape": asdict(shape), **provenance, "compilation": receipt}
    (args.workdir / "upstream_binding.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"shape": result["shape"],
                      "pointer_binding": result["pointer_binding"],
                      "object_sha256": receipt["object_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
