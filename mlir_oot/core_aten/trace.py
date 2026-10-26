"""Decode a CPU-looped Gemmini-dialect kernel into its executed primitive stream.

The golden GEMM/convolution builders emit Gemmini ops inside ordinary LLVM loops.
Tracing the function with symbolic pointer arguments yields the exact command
sequence, which the hazard lint (``lint.lint``) and the bit-level model
(``stream.simulate``) then check like any recipe stream.
"""

from __future__ import annotations

from xdsl.dialects import llvm

from ..ir import gemmini_dialect as G
from ..tables import isa
from .stream import Arg, Ins, Stream

_KIND = {G.FlushOp: "flush", G.ConfigExOp: "config_ex", G.ConfigLdOp: "config_ld", G.ConfigStOp: "config_st",
         G.MvinOp: "mvin", G.MvoutOp: "mvout", G.PreloadOp: "preload", G.ComputeOp: "compute", G.FenceOp: "fence"}

_DEFAULTS = {
    "config_ex": dict(act=0, acc_scale=1.0, a_stride=1, c_stride=1, a_transpose=0, b_transpose=0),
    "config_ld": dict(scale=1.0, shrunk=0, block_stride=isa.DIM, pixel_repeats=1),
    "config_st": dict(acc_act=0, acc_scale=1.0),
    "compute": dict(bd=isa.GARBAGE_ADDR, bd_cols=isa.DIM, bd_rows=isa.DIM, accumulate=0),
    "flush": dict(skip=0),
}
_KEEP = {
    "config_ex": ("dataflow", "act", "acc_scale", "a_stride", "c_stride", "a_transpose", "b_transpose"),
    "config_ld": ("load_id", "stride", "scale", "shrunk", "block_stride", "pixel_repeats"),
    "config_st": ("stride", "acc_act", "acc_scale"),
    "mvin": ("local", "cols", "rows", "load_id"),
    "mvout": ("local", "cols", "rows"),
    "preload": ("bd", "c", "bd_cols", "bd_rows", "c_cols", "c_rows"),
    "compute": ("a", "bd", "a_cols", "a_rows", "bd_cols", "bd_rows", "accumulate"),
    "flush": ("skip",),
    "fence": (),
}


def stream_from_module(module, args: list[Arg], symbol: str | None = None) -> Stream:
    from merlin.llvmlower.static_llvm_cfg import StaticInt, StaticPointer, trace_static_function

    if ('gemmini.runtime_scale' in module.attributes or
            any('gemmini.runtime_scale' in op.attributes
                or 'scale_operand' in op.attributes or 'scale_operand' in op.properties
                for op in module.walk())):
        raise ValueError('unsupported dynamic trace: runtime CONFIG_ST scale requires decoded emitted stream')
    fns = [op for op in module.walk() if isinstance(op, llvm.FuncOp)
           and (symbol is None or op.sym_name.data == symbol) and op.body.blocks]
    fn = fns[-1]
    pointers = [StaticPointer(i, StaticInt(0, 64)) for i in range(len(fn.body.blocks.first.args))]
    s = Stream(list(args))
    for step in trace_static_function(fn, pointers, observe=lambda op: isinstance(op, G._GemminiOp),
                                      pointer_index_bits=64, max_steps=200_000_000):
        op = step.operation
        kind = _KIND[type(op)]
        attrs = dict(_DEFAULTS.get(kind, {}))
        attrs.update({k: op.a(k) for k in _KEEP[kind] if op.a(k) is not None})
        arg, off = None, 0
        if kind in ("mvin", "mvout"):
            ptr = step.inputs[0]
            if kind == "mvin":
                attrs["zero"] = int(ptr.base is None)
            if ptr.base is not None:
                arg, off = ptr.base, ptr.offset.signed
        elif kind == "compute" and step.inputs:
            attrs["a"] = step.inputs[0].value
        elif kind == "preload" and step.inputs:
            rows = dict(zip(op.dynamic_rows(), step.inputs))
            if "bd" in rows:
                attrs["bd"] = rows["bd"].value
            if "c" in rows:
                attrs["c"] = isa.acc_addr(rows["c"].value, accumulate=bool(op.a("c_accumulate", 0)))
        if kind == "config_ld":
            attrs["shrunk"] = int(bool(attrs["shrunk"]))
        if "acc_scale" in attrs:
            attrs["acc_scale"] = float(attrs["acc_scale"])
        if "scale" in attrs:
            attrs["scale"] = float(attrs["scale"])
        s.ins.append(Ins(kind, tuple(sorted(attrs.items())), arg, off))
    return s
