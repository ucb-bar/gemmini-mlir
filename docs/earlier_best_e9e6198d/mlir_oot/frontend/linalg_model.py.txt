"""Turn a parsed ``linalg-on-tensors`` module into the target-agnostic workload model.

The reader is structural: xDSL parses the real IR (``func.func @forward`` with linalg/tensor/arith/
math ops, plus the producer's unregistered ``quant_ext`` extensions), and this module walks the
operand graph. Which regions this target can actually execute is decided in the lowering, not here.
"""

from __future__ import annotations

from typing import Any

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, math, tensor
from xdsl.dialects.builtin import Builtin, ModuleOp, StringAttr, TensorType
from xdsl.parser import Parser

from ..ir.workload import Op, Workload
from .extract import dtype_name


def parse_module(text: str) -> ModuleOp:
    ctx = Context(allow_unregistered=True)
    for d in (Builtin, func.Func, arith.Arith, linalg.Linalg, tensor.Tensor, math.Math):
        try:
            ctx.load_dialect(d)
        except Exception:  # a dialect this xDSL build does not ship stays unregistered
            pass
    module = Parser(ctx, text).parse_module()
    return module


def _region_family(op) -> str:
    """The producer's own ``prov.family`` label for a region, when it carries one."""
    a = op.attributes.get("prov.family") or op.properties.get("prov.family")
    if isinstance(a, StringAttr):
        return a.data
    return op.name.split(".")[0]


def _region_id(op) -> str | None:
    a = op.attributes.get("prov.region_id") or op.properties.get("prov.region_id")
    return a.data if isinstance(a, StringAttr) else None


def build_workload(text: str) -> Workload:
    wl = Workload(target="gemmini", abi_version="0.1", grammar="linalg-on-tensors")
    try:
        module = parse_module(text)
        module.verify()
    except Exception as exc:  # noqa: BLE001
        # A producer extension this reader cannot render structurally is a STATED gap, not a tool
        # failure: the entrypoint still answers, and the buffer says what was not read.
        wl.declined = {
            "reason": (
                "the linalg-on-tensors module carries a producer construct this reader does not "
                f"render structurally ({type(exc).__name__}: {str(exc)[:160]}); no region of it was "
                "lowered"
            ),
            "op": "forward",
        }
        return wl

    entry = None
    for op in module.body.block.ops:
        if isinstance(op, func.FuncOp) and op.sym_name.data == "forward":
            entry = op
            break
    if entry is None:
        for op in module.body.block.ops:
            if isinstance(op, func.FuncOp):
                entry = op
                break
    if entry is None:
        wl.declined = {"reason": "the linalg module declares no entry function"}
        return wl

    block = entry.body.block
    names = argument_names(entry)
    for i, arg in enumerate(block.args):
        if isinstance(arg.type, TensorType):
            shape = tuple(int(d) for d in arg.type.get_shape())
            wl.declare(names[i], shape, dtype_name(arg.type.get_element_type()), "input")
    ret = None
    for op in block.ops:
        if isinstance(op, func.ReturnOp):
            ret = op
    if ret is not None:
        for j, v in enumerate(ret.operands):
            if isinstance(v.type, TensorType):
                shape = tuple(int(d) for d in v.type.get_shape())
                wl.declare(f"Y{j}", shape, dtype_name(v.type.get_element_type()), "output")

    wl.ops = []
    wl.host_module = module
    wl.host_entry = entry
    regions = host_lane_regions(block)
    if "prov.weights_file" in module.attributes:
        # A WHOLE MODEL carries its own weight file and is dispatched layer by layer by the
        # harness's own whole-model path, which decides placement and calls this backend per
        # accelerable tile. Declaring a per-region placement for the same program from here would
        # be a second, competing routing plan for work this package never sees whole -- so the
        # honest statement about the WHOLE-PROGRAM lowering is that there is none.
        families = sorted({r["family"] for r in regions})
        wl.declined = {
            "reason": (
                "this package lowers the merlin_iface v0.1 integer program grammar; a whole "
                "linalg-on-tensors model is dispatched per layer by the harness's own model path, "
                "and this package emits no whole-program lowering for it (region families: "
                + ", ".join(families)
                + ")"
            ),
            "op": "forward",
        }
        return wl
    wl.lane_placement = regions
    if not wl.lane_placement:
        wl.declined = {"reason": "the linalg module declares no region to place"}
    return wl


#: The ABI's own tensor-role vocabulary. The linalg-on-tensors grammar carries NO tensor names --
#: it identifies operands positionally -- while the command buffer identifies every tensor BY NAME,
#: and the runner materialises leaf data and reads outputs back by that name. So a name has to be
#: derived, and the only defensible source is the ABI's own vocabulary: the activation is `X`, the
#: stationary operand is `W`, the per-channel vector is `B`, an attention region's three
#: equally-shaped operands are `Q`, `K`, `V`, and committed results are `Y0`, `Y1`, ... -- exactly
#: what every op in the merlin_iface half of the same ABI spells.
ROLE_NAMES = ("X", "W", "B")
ATTENTION_NAMES = ("Q", "K", "V")


def argument_names(entry) -> list[str]:
    """Names for an entry function's arguments, derived from the region's own structure."""
    args = list(entry.body.block.args)
    shapes = [
        tuple(int(d) for d in a.type.get_shape()) if isinstance(a.type, TensorType) else ()
        for a in args
    ]
    contractions = sum(
        1 for op in entry.body.block.ops if op.name in ("linalg.matmul", "linalg.batch_matmul")
    )
    if len(args) == 3 and len(set(shapes)) == 1 and contractions >= 2:
        return list(ATTENTION_NAMES)
    out: list[str] = []
    for i in range(len(args)):
        if i < len(ROLE_NAMES):
            out.append(ROLE_NAMES[i])
        else:
            out.append(f"{ROLE_NAMES[-1]}{i - len(ROLE_NAMES) + 1}")
    return out


#: The datapath this target's RTL declares, and therefore the operand encodings the mesh HAS.
#: ``rtl.facts.load_facts("gemmini")["facts"]["datapaths"]`` -> input i8 (scratchpad ``UInt<8>``),
#: accumulator i32 (``AccumulatorMem SInt<32>``). There is no floating-point operand port.
ACCELERATED_ELEMENT_TYPES = ("i8",)


def _element_types(op) -> set[str]:
    out: set[str] = set()
    for v in list(op.operands) + list(op.results):
        t = v.type
        if isinstance(t, TensorType):
            try:
                out.add(dtype_name(t.get_element_type()))
            except Exception:  # noqa: BLE001 - an element type this reader does not name
                out.add("unknown")
    return out


def host_lane_regions(block) -> list[dict[str, Any]]:
    """Group the entry block's operations into declared host-lane regions, with the reason.

    A region is refused HERE because of the OPERAND ENCODING, not because of its family: this
    target's readout does apply activation, accumulator scale and pooling, so "the family has no
    datapath" would be a claim its own derived capability contradicts. What it has no encoding for
    is a floating-point operand -- the mesh reads i8 and accumulates i32, and that is an RTL fact.
    """
    regions: dict[str, dict[str, Any]] = {}
    for op in block.walk():
        if isinstance(op, (func.ReturnOp, func.FuncOp)):
            continue
        rid = _region_id(op) or _region_family(op)
        family = _region_family(op)
        types = _element_types(op)
        entry = regions.setdefault(
            rid,
            {"region": rid, "lane": "host", "family": family, "dtypes": set(), "ops": 0},
        )
        entry["dtypes"] |= types
        entry["ops"] += 1
    out: list[dict[str, Any]] = []
    for rid, e in regions.items():
        dtypes = sorted(d for d in e["dtypes"] if d)
        float_operands = [d for d in dtypes if not d.startswith("i")]
        if float_operands:
            reason = (
                "this target's mesh reads i8 operands into an i32 accumulator (RTL-derived "
                "datapaths: scratchpad UInt<8>, AccumulatorMem SInt<32>); the region's operands are "
                + ", ".join(float_operands)
                + ", for which the unit has no operand encoding, so the region is placed on the "
                "host lane rather than quantised behind the program's back"
            )
        else:
            reason = (
                "the region carries no contraction this weight-stationary mesh can issue "
                "(operand element types: " + ", ".join(dtypes or ["unknown"]) + ")"
            )
        out.append(
            {
                "region": rid,
                "lane": "host",
                "family": e["family"],
                "operations": e["ops"],
                "dtypes": dtypes,
                "reason": reason,
            }
        )
    return out
