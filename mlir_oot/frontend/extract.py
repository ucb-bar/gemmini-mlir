"""Build the target-agnostic :class:`Workload` from a VERIFIED ``merlin_iface`` module."""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import (
    ArrayAttr,
    BFloat16Type,
    Float16Type,
    Float32Type,
    Float64Type,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.ir import SSAValue

from ..ir.workload import Epilogue, Op, Workload
from . import iface_dialect as ifd


class InterfaceError(Exception):
    """The interface module is well-formed MLIR but is not a program this ABI defines."""


def dtype_name(t) -> str:
    if isinstance(t, IntegerType):
        return f"i{t.width.data}"
    if isinstance(t, BFloat16Type):
        return "bf16"
    if isinstance(t, Float16Type):
        return "f16"
    if isinstance(t, Float32Type):
        return "f32"
    if isinstance(t, Float64Type):
        return "f64"
    raise InterfaceError(f"unsupported element type {t}")


def tensor_shape_dtype(t) -> tuple[tuple[int, ...], str]:
    if not isinstance(t, TensorType):
        raise InterfaceError(f"expected a ranked tensor, got {t}")
    return tuple(int(d) for d in t.get_shape()), dtype_name(t.get_element_type())


def _s(op, key: str) -> str | None:
    a = op.attributes.get(key)
    return a.data if isinstance(a, StringAttr) else None


def _strs(op, key: str) -> tuple[str, ...]:
    a = op.attributes.get(key)
    if not isinstance(a, ArrayAttr):
        return ()
    return tuple(e.data for e in a.data if isinstance(e, StringAttr))


def _i(op, key: str) -> int | None:
    a = op.attributes.get(key)
    if isinstance(a, IntegerAttr):
        return int(a.value.data)
    return None


#: The positional operand names the ABI gives each whole-op mnemonic. Read from the contract's own
#: operand tables, so a command buffer names what the runner reads rather than an index.
_OPERAND_KEYS = {
    "rmsnorm": ("src", "gamma"),
    "rope": ("src",),
    "softmax": ("src",),
    "residual_add": ("lhs", "rhs"),
    "bias_add": ("src", "bias"),
    "k_chain": ("src", "weight", "weight2"),
    "depthwise_conv2d": ("src", "weight"),
}

def _ints(op, key: str) -> tuple[int, ...] | None:
    v = ifd.int_list(op, key)
    return None if v is None else tuple(v)


def epilogue_of(op) -> Epilogue:
    return Epilogue(
        stages=_strs(op, "epilogue"),
        output_dtype=_s(op, "output_dtype") or "i32",
        acc_scale=ifd.float_attr(op, "acc_scale"),
        requant_shift=_i(op, "requant_shift"),
        bias=_s(op, "bias"),
        pool_in_dims=_ints(op, "pool_in_dims"),
        pool_size=_ints(op, "pool_size"),
        pool_stride=_ints(op, "pool_stride"),
        pool_padding=_ints(op, "pool_padding") or (0, 0, 0, 0),
        pool_pad_value=_i(op, "pool_pad_value") or 0,
    )


def extract(module: ModuleOp) -> Workload:
    attrs = module.attributes
    version = attrs.get("merlin_iface.version")
    version = version.data if isinstance(version, StringAttr) else None
    if version != ifd.SUPPORTED_VERSION:
        raise InterfaceError(
            f"merlin_iface version {version!r} is not implemented (this package implements "
            f"{ifd.SUPPORTED_VERSION!r})"
        )
    target = attrs.get("merlin_iface.target")
    abi = attrs.get("merlin_iface.abi_version")
    wl = Workload(
        target=target.data if isinstance(target, StringAttr) else "gemmini",
        abi_version=abi.data if isinstance(abi, StringAttr) else "0.1",
    )

    names: dict[SSAValue, str] = {}
    n_acc = 0
    n_res = 0

    for op in module.body.block.ops:
        if isinstance(op, ifd.TensorOp):
            name = _s(op, "name")
            shape, dt = tensor_shape_dtype(op.results[0].type)
            wl.declare(name, shape, dt, _s(op, "role"))
            names[op.results[0]] = name
        elif isinstance(op, ifd.ResidentPackOp):
            src = names[op.operands[0]]
            hid = f"res{n_res}"
            n_res += 1
            names[op.results[0]] = hid
            wl.residents[hid] = src
            wl.ops.append(Op("resident_pack", hid, {"src": src}, {"layout": _s(op, "layout")}))
        elif isinstance(op, ifd.MatmulOp):
            lhs = names[op.operands[0]]
            rhs = names[op.operands[1]]
            aid = f"acc{n_acc}"
            n_acc += 1
            names[op.results[0]] = aid
            w = wl.tensors[wl.residents[rhs]]
            a = wl.tensors[lhs]
            if len(a.shape) != 2 or len(w.shape) != 2 or a.shape[1] != w.shape[0]:
                raise InterfaceError(
                    f"matmul shapes do not contract: lhs {a.shape} against weight {w.shape}"
                )
            wl.ops.append(
                Op(
                    "matmul",
                    aid,
                    {"lhs": lhs, "rhs": rhs},
                    {},
                    shape={"m": a.shape[0], "k": a.shape[1], "n": w.shape[1]},
                )
            )
        elif isinstance(op, ifd.CommitOp):
            aid = names[op.operands[0]]
            name = _s(op, "name")
            shape, dt = tensor_shape_dtype(op.results[0].type)
            wl.declare(name, shape, dt, "output")
            names[op.results[0]] = name
            wl.ops.append(Op("commit", name, {"src": aid}, {}, epilogue=epilogue_of(op)))
        elif isinstance(op, ifd.EvictOp):
            wl.ops.append(Op("evict", "", {"handle": names[op.operands[0]]}))
        elif isinstance(op, ifd.Conv2dOp):
            ifm = names[op.operands[0]]
            wname = names[op.operands[1]]
            name = _s(op, "name")
            shape, dt = tensor_shape_dtype(op.results[0].type)
            wl.declare(name, shape, dt, "output")
            names[op.results[0]] = name
            k = _ints(op, "kernel")
            wl.ops.append(
                Op(
                    "conv2d",
                    name,
                    {"ifm": ifm, "weight": wname},
                    {
                        "kernel": k,
                        "stride": _ints(op, "stride") or (1, 1),
                        "padding": _ints(op, "padding") or (0, 0, 0, 0),
                        "dilation": _ints(op, "dilation") or (1, 1),
                        "layout": _s(op, "layout") or "nhwc",
                    },
                    epilogue=epilogue_of(op),
                )
            )
        elif isinstance(op, ifd.MovementOp):
            src = names[op.operands[0]]
            name = _s(op, "name")
            shape, dt = tensor_shape_dtype(op.results[0].type)
            wl.declare(name, shape, dt, "output")
            names[op.results[0]] = name
            wl.ops.append(
                Op(
                    "movement",
                    name,
                    {"src": src},
                    {"output_dtype": _s(op, "output_dtype") or dt, "semantic": _s(op, "semantic")},
                )
            )
        elif isinstance(op, ifd.MatmulBatchedOp):
            a, w = names[op.operands[0]], names[op.operands[1]]
            name = _s(op, "name")
            shape, dt = tensor_shape_dtype(op.results[0].type)
            wl.declare(name, shape, dt, "output")
            names[op.results[0]] = name
            wl.ops.append(
                Op(
                    "matmul_batched",
                    name,
                    {"a": a, "w": w},
                    {"output_dtype": _s(op, "output_dtype") or dt},
                    epilogue=epilogue_of(op),
                )
            )
        elif isinstance(op, (ifd.AttentionQkOp, ifd.AttentionPvOp)):
            lhs, rhs = names[op.operands[0]], names[op.operands[1]]
            name = _s(op, "name")
            shape, dt = tensor_shape_dtype(op.results[0].type)
            wl.declare(name, shape, dt, "output")
            names[op.results[0]] = name
            qk = isinstance(op, ifd.AttentionQkOp)
            wl.ops.append(
                Op(
                    "attention_qk" if qk else "attention_pv",
                    name,
                    ({"q": lhs, "k": rhs} if qk else {"p": lhs, "v": rhs}),
                    {},
                    epilogue=epilogue_of(op),
                )
            )
        elif isinstance(op, ifd.BiasAddOp):
            src, bias = names[op.operands[0]], names[op.operands[1]]
            name = _s(op, "name")
            shape, dt = tensor_shape_dtype(op.results[0].type)
            wl.declare(name, shape, dt, "output")
            names[op.results[0]] = name
            wl.ops.append(
                Op(
                    "bias_add",
                    name,
                    {"src": src, "bias": bias},
                    {"output_dtype": _s(op, "output_dtype") or dt},
                )
            )
        elif isinstance(op, (ifd.SoftmaxOp, ifd.RmsnormOp, ifd.RopeOp)):
            name = _s(op, "name")
            shape, dt = tensor_shape_dtype(op.results[0].type)
            wl.declare(name, shape, dt, "output")
            names[op.results[0]] = name
            kind = op.name.split(".")[-1]
            keys = _OPERAND_KEYS[kind]
            operand_names = {keys[i]: names[v] for i, v in enumerate(op.operands)}
            extra: dict[str, Any] = {"output_dtype": _s(op, "output_dtype") or dt}
            for key in ("eps", "theta", "scale"):
                v = ifd.float_attr(op, key)
                if v is not None:
                    extra[key] = v
            for key in ("axis", "rotary_dim", "position_offset"):
                v = _i(op, key)
                if v is not None:
                    extra[key] = v
            causal = ifd.bool_attr(op, "causal")
            if causal is not None:
                extra["causal"] = causal
            sem = _s(op, "semantic")
            if sem:
                extra["semantic"] = sem
            wl.ops.append(Op(kind, name, operand_names, extra, epilogue=epilogue_of(op)))
        elif isinstance(op, (ifd.KChainOp, ifd.ResidualAddOp, ifd.DepthwiseConv2dOp)):
            name = _s(op, "name")
            shape, dt = tensor_shape_dtype(op.results[0].type)
            wl.declare(name, shape, dt, "output")
            names[op.results[0]] = name
            kind = op.name.split(".")[-1]
            # The ABI names each whole-op's operands; a positional fallback keeps a shape this
            # reader has no row for readable rather than unnameable.
            keys = _OPERAND_KEYS.get(kind)
            operand_names = {
                (keys[i] if keys and i < len(keys) else f"in{i}"): names[v]
                for i, v in enumerate(op.operands)
            }
            extra: dict[str, Any] = {}
            for key in ("kernel", "stride", "padding", "dilation"):
                v = _ints(op, key)
                if v is not None:
                    extra[key] = v
            for key in ("lhs_scale", "rhs_scale"):
                v = ifd.float_attr(op, key)
                if v is not None:
                    extra[key] = v
            b = _i(op, "bound_lsb")
            if b is not None:
                extra["bound_lsb"] = b
            extra["output_dtype"] = _s(op, "output_dtype") or dt
            wl.ops.append(Op(kind, name, operand_names, extra, epilogue=epilogue_of(op)))
        else:
            raise InterfaceError(f"operation {op.name!r} is not part of merlin_iface v0.1")
    return wl
