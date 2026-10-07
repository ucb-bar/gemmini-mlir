"""Write the target-independent ``command_buffer.json`` for a workload.

This is the interface program restated in the frozen ABI's vocabulary -- every extent, dtype and
epilogue parameter is copied from the capsule that was handed in. It is also where the package
states its ADMISSION ROUTE: accelerated commands (A), a declared host lane (H,
``params.lane_placement``) or an explicit ``declined`` (D). The three are disjoint and one of them
is always stated: an empty command list with nothing said is a contract violation, not a decline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ir.workload import Epilogue, Workload
from .lowering import lanes

ABI_VERSION = "0.1"


def _epilogue_attrs(ep: Epilogue) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "epilogue": list(ep.stages),
        "output_dtype": ep.output_dtype,
    }
    if ep.acc_scale is not None:
        attrs["acc_scale"] = float(ep.acc_scale)
    if ep.requant_shift is not None:
        attrs["requant_shift"] = int(ep.requant_shift)
    if ep.bias is not None:
        attrs["bias"] = ep.bias
    if "maxpool" in ep.has:
        attrs["pool_in_dims"] = list(ep.pool_in_dims or ())
        attrs["pool_size"] = list(ep.pool_size or ())
        attrs["pool_stride"] = list(ep.pool_stride or ())
        attrs["pool_padding"] = list(ep.pool_padding)
        if any(ep.pool_padding):
            attrs["pool_pad_value"] = int(ep.pool_pad_value)
    return attrs


def build(wl: Workload, *, kernel_abi: dict[str, Any] | None = None,
          declined: dict[str, Any] | None = None) -> dict[str, Any]:
    cb: dict[str, Any] = {
        "abi_version": ABI_VERSION,
        "target": wl.target or "gemmini",
        "tensors": {},
        "commands": [],
    }
    for name, t in wl.tensors.items():
        cb["tensors"][name] = {"shape": list(t.shape), "dtype": t.dtype, "role": t.role}

    if declined is not None:
        cb["declined"] = declined
        cb["commands"] = []
        return cb

    conv_only_handles = _conv_only_handles(wl)
    for op in wl.ops:
        if op.kind == "resident_pack":
            if op.out in conv_only_handles:
                continue
            cb["commands"].append(
                {"opcode": "RES_PACK",
                 "operands": {"src": op.operands["src"], "dst": op.out},
                 "attributes": {"layout": op.attrs.get("layout") or "packed_rhs"}}
            )
        elif op.kind == "evict":
            if op.operands["handle"] in conv_only_handles:
                continue
            cb["commands"].append(
                {"opcode": "EVICT", "operands": {"handle": op.operands["handle"]}}
            )
        elif op.kind == "matmul":
            cb["commands"].append(
                {"opcode": "MATMUL_RESIDENT",
                 "operands": {"lhs": op.operands["lhs"], "rhs": op.operands["rhs"], "dst": op.out}}
            )
        elif op.kind == "commit":
            ep = op.epilogue or Epilogue()
            operands = {"src": op.operands["src"], "dst": op.out}
            if ep.bias and ("bias_add" in ep.has or "bias" in ep.has):
                operands["bias"] = ep.bias
            cb["commands"].append(
                {"opcode": "COMMIT", "operands": operands, "attributes": _epilogue_attrs(ep)}
            )
        elif op.kind == "conv2d":
            handle = op.operands["weight"]
            attrs = dict(_epilogue_attrs(op.epilogue or Epilogue()))
            attrs.update(
                {"kernel": list(op.attrs["kernel"]), "stride": list(op.attrs["stride"]),
                 "padding": list(op.attrs["padding"]), "dilation": list(op.attrs["dilation"]),
                 "layout": op.attrs.get("layout") or "nhwc"}
            )
            cb["commands"].append(
                {"opcode": "CONV2D",
                 "operands": {"ifm": op.operands["ifm"],
                              "weight": wl.residents.get(handle, handle),
                              "dst": op.out},
                 "attributes": attrs}
            )
        elif op.kind == "movement":
            attrs = {"output_dtype": op.attrs.get("output_dtype") or wl.tensors[op.out].dtype}
            if op.attrs.get("semantic"):
                attrs["semantic"] = op.attrs["semantic"]
            cb["commands"].append(
                {"opcode": "MOVEMENT",
                 "operands": {"src": op.operands["src"], "dst": op.out},
                 "attributes": attrs}
            )
        elif op.kind == "matmul_batched":
            cb["commands"].append(
                {"opcode": "BATCHED_MATMUL",
                 "operands": {"a": op.operands["a"], "w": op.operands["w"], "dst": op.out}}
            )
        elif op.kind in ("attention_qk", "attention_pv"):
            qk = op.kind == "attention_qk"
            ops_ = ({"q": op.operands["q"], "k": op.operands["k"]} if qk
                    else {"p": op.operands["p"], "v": op.operands["v"]})
            ops_["dst"] = op.out
            ep = op.epilogue or Epilogue()
            cb["commands"].append(
                {"opcode": "ATTENTION_QK" if qk else "ATTENTION_PV",
                 "operands": ops_,
                 "attributes": {"epilogue": list(ep.stages), "output_dtype": ep.output_dtype}}
            )
        elif op.kind == "bias_add":
            cb["commands"].append(
                {"opcode": "BIAS_ADD",
                 "operands": {"src": op.operands["src"], "bias": op.operands["bias"],
                              "dst": op.out},
                 "attributes": {"output_dtype": op.attrs.get("output_dtype")
                                or wl.tensors[op.out].dtype}}
            )
        elif op.kind == "residual_add":
            ep = op.epilogue or Epilogue()
            cb["commands"].append(
                {"opcode": "RESIDUAL_ADD",
                 "operands": {"lhs": op.operands["lhs"], "rhs": op.operands["rhs"],
                              "dst": op.out},
                 "attributes": {"lhs_scale": float(op.attrs.get("lhs_scale", 1.0)),
                                "rhs_scale": float(op.attrs.get("rhs_scale", 1.0)),
                                "bound_lsb": int(op.attrs.get("bound_lsb", 0)),
                                "epilogue": list(ep.stages),
                                "output_dtype": op.attrs.get("output_dtype")
                                or wl.tensors[op.out].dtype}}
            )
        elif op.kind in ("rmsnorm", "softmax", "rope"):
            keys = {"rmsnorm": ("src", "gamma"), "softmax": ("src",), "rope": ("src",)}[op.kind]
            ops_ = {k: op.operands[k] for k in keys if k in op.operands}
            ops_["dst"] = op.out
            attrs = {"output_dtype": op.attrs.get("output_dtype")
                     or wl.tensors[op.out].dtype}
            for k in ("eps", "theta", "scale", "axis", "causal"):
                if k in op.attrs:
                    attrs[k] = op.attrs[k]
            cb["commands"].append(
                {"opcode": op.kind.upper(), "operands": ops_, "attributes": attrs}
            )
        else:
            raise ValueError(f"no command-buffer opcode for interface op {op.kind!r}")

    # The ROUTING PLAN, published for every region this package placed: the host lane it refused a
    # region to (H) and the mesh lane it accepted one onto (A). A capsule asserting lane
    # composition is graded against these keys, and an accelerated program that publishes nothing
    # has an empty ledger -- which reads as "no lane carried anything".
    # A HYBRID kernel carries regions on both lanes, so its ledger has to name both -- reporting
    # every region as accelerated would claim the array ran a normalisation it has no path for.
    from .lowering import hybrid

    if hybrid.is_hybrid(wl):
        placement = list(wl.lane_placement) + lanes.hybrid_placement(wl)
    else:
        placement = list(wl.lane_placement) + lanes.mesh_placement(wl)
    if placement:
        cb.setdefault("params", {})["lane_placement"] = placement
    if wl.im2col_recipes:
        cb.setdefault("params", {})["im2col_recipes"] = wl.im2col_recipes
    if kernel_abi is not None:
        cb["kernel_abi"] = kernel_abi
    return cb


def _conv_only_handles(wl: Workload) -> set[str]:
    """Handles whose only consumer is a whole-op CONV2D: that command carries the weight itself."""
    used_by_matmul = {op.operands["rhs"] for op in wl.ops if op.kind == "matmul"}
    used_by_conv = {op.operands["weight"] for op in wl.ops if op.kind == "conv2d"}
    return {h for h in wl.residents if h in used_by_conv and h not in used_by_matmul}


def write(cb: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(cb, indent=2))
