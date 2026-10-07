"""Route Q: carry a FLOAT contraction on an integer mesh, by quantising it in the compiler.

This target's mesh has no floating-point operand port (RTL-derived: the scratchpad is ``UInt<8>``
and the accumulator ``SInt<32>``), so :func:`..iface_to_gemmini.require_mesh_dtypes` refuses a
contraction whose operands are f32/bf16. Refusing is the right answer for a region that arrives in
the LINALG grammar -- there the interface is describing a program, and a host lane is where a float
contraction belongs.

A region that arrives in the ACCELERATOR ABI (``merlin_iface``) is a different statement: it is a
request to issue this contraction on the unit. This pass answers that request the way a quantising
compiler does -- it does not reinterpret four float bytes as integer data, it DERIVES a per-tensor
symmetric scale from the operand at run time, stages the operand as i8, contracts on the mesh into
the i32 accumulator, and rescales the readout back into the float output buffer.

The scale derivation is emitted code, not a constant: nothing here reads a value out of a capsule.
It is also grid-aware. An operand that already lies on a uniform grid of at most 127 levels -- what
a dequantised int8 tensor is -- is recovered EXACTLY, because the emitted prologue recognises the
grid instead of re-quantising onto a coarser one of its own; see :mod:`..codegen.quant_lane`.

The rewrite is structural and happens on the Workload IR, before the interface->target pass runs,
so the mesh lowering below it is the SAME weight-stationary emitter every integer capsule uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir.workload import Epilogue, Op, Workload
from ..target import facts, layout

#: Operand element types this route can read and stage. Both are float containers the mesh has no
#: port for; an operand already in an integer container does not need this route at all.
FLOAT_OPERANDS = ("f32", "bf16")
#: The output container the readout can be written back into IN PLACE. The mesh commits an i32
#: accumulator word, so the buffer the harness allocated must be exactly four bytes per element --
#: a bf16 output would need a second buffer and is left to the host lane.
FLOAT_RESULTS = ("f32",)
#: Interface mnemonics this route rewrites. A conv2d's lhs is a DERIVED im2col stream rather than a
#: declared tensor, so staging it needs the im2col materialised first; that is not attempted here.
CONTRACTIONS = ("matmul", "matmul_batched")
#: Ops allowed to appear alongside them.
PASSTHROUGH = ("resident_pack", "evict", "commit")

#: Staging lives in the kernel's own stack frame. The harness's frame preflight
#: (``kernel.stack_frame.json``) reports a 65536-byte static ceiling; half of it is the budget this
#: route will spend, so the rest of the frame is never the thing that overflows.
STAGING_BUDGET_BYTES = 32768

#: The mesh's operand range is symmetric and excludes the asymmetric end of the container, so a
#: negated operand is representable and the readout stays sign-symmetric.
QMAX = 127


@dataclass
class Staged:
    """One DRAM operand and the i8 image of it this route stages on the stack."""

    src: str
    dst: str
    rows: int
    pitch: int
    src_dtype: str

    @property
    def elems(self) -> int:
        return self.rows * self.pitch


@dataclass
class Rescale:
    """One committed output: i32 accumulator words in place, rescaled back to the float grid."""

    out: str
    rows: int
    pitch: int
    #: the staged operands whose derived scales multiply into this output's scale
    operands: list[str]
    out_dtype: str


@dataclass
class QuantPlan:
    staged: list[Staged] = field(default_factory=list)
    rescales: list[Rescale] = field(default_factory=list)

    @property
    def staged_bytes(self) -> int:
        return sum(s.elems for s in self.staged)

    def reason(self) -> str:
        names = ", ".join(sorted({s.src_dtype for s in self.staged}))
        return (
            f"issued on this target's {facts.DIM}x{facts.DIM} systolic mesh: the region's operands "
            f"are {names}, for which the RTL-derived datapath (scratchpad UInt<8>, AccumulatorMem "
            "SInt<32>) has no port, so the compiler derives a per-tensor symmetric scale from each "
            "operand at run time, stages it as i8, contracts on the mesh and rescales the i32 "
            "accumulator back onto the float grid"
        )


def _float_operand_tensors(wl: Workload, op: Op) -> list[str]:
    """The DECLARED tensors this contraction reads, resolving a resident handle to its weight."""
    if op.kind == "matmul":
        keys = ("lhs", "rhs")
    else:
        keys = ("a", "w")
    out: list[str] = []
    for key in keys:
        ref = op.operands.get(key)
        if ref is None:
            return []
        out.append(wl.residents.get(ref, ref))
    return out


def _result_of(wl: Workload, op: Op, commits: dict[str, Op]) -> Op | None:
    """The op that WRITES this contraction's result tensor: itself, or the commit that reads it."""
    return commits.get(op.out) if op.kind == "matmul" else op


def plan(wl: Workload) -> QuantPlan | None:
    """The plan for ``wl``, or None when this route does not apply to it.

    None is not a failure: the caller falls through to the host-lane route, which is what a float
    contraction outside the accelerator ABI is supposed to get.
    """
    if wl.grammar != "merlin_iface" or not wl.ops:
        return None
    if any(op.kind not in CONTRACTIONS + PASSTHROUGH for op in wl.ops):
        return None
    contractions = [op for op in wl.ops if op.kind in CONTRACTIONS]
    if not contractions:
        return None

    commits = {op.operands["src"]: op for op in wl.ops if op.kind == "commit"}
    p = QuantPlan()
    seen: dict[str, Staged] = {}
    for op in contractions:
        operands = _float_operand_tensors(wl, op)
        if len(operands) != 2:
            return None
        for name in operands:
            decl = wl.tensors.get(name)
            if decl is None or decl.dtype not in FLOAT_OPERANDS:
                return None
        writer = _result_of(wl, op, commits)
        if writer is None:
            return None
        ep = writer.epilogue or Epilogue()
        if ep.stages:
            # A fused float readout stage (bias/scale/relu/pool) runs in a domain the accumulator
            # is not in yet. Staging the contraction without it would drop the stage silently.
            return None
        out = wl.tensors.get(writer.out)
        if out is None or out.dtype not in FLOAT_RESULTS:
            return None
        for name in operands:
            if name not in seen:
                decl = wl.tensors[name]
                seen[name] = Staged(
                    src=name,
                    dst=f"{name}$q",
                    rows=layout.row_count(decl.shape),
                    pitch=layout.row_pitch(decl.shape),
                    src_dtype=decl.dtype,
                )
                p.staged.append(seen[name])
        p.rescales.append(
            Rescale(
                out=writer.out,
                rows=layout.row_count(out.shape),
                pitch=layout.row_pitch(out.shape),
                operands=[seen[n].dst for n in operands],
                out_dtype=out.dtype,
            )
        )
    if p.staged_bytes > STAGING_BUDGET_BYTES:
        return None
    return p


def apply(wl: Workload, p: QuantPlan) -> None:
    """Rewrite ``wl`` in place onto the staged i8 operands and the raw i32 readout.

    Every extent, pitch and dtype below this point is then read from the rewritten declarations by
    exactly the same integer path the i8 capsules take -- this pass adds no second lowering.
    """
    for s in p.staged:
        src = wl.tensors[s.src]
        wl.declare(s.dst, src.shape, "i8", "staging")
    swap = {s.src: s.dst for s in p.staged}
    for handle, weight in list(wl.residents.items()):
        if weight in swap:
            wl.residents[handle] = swap[weight]
    for op in wl.ops:
        if op.kind == "resident_pack":
            src = op.operands.get("src")
            if src in swap:
                op.operands["src"] = swap[src]
        elif op.kind in CONTRACTIONS:
            for key, ref in list(op.operands.items()):
                if ref in swap:
                    op.operands[key] = swap[ref]
    outs = {r.out for r in p.rescales}
    for r in p.rescales:
        # The accumulator word and the float element are both four bytes, so the committed rows go
        # straight into the buffer the harness allocated and the rescale runs over it in place --
        # the readout declares the container it actually writes, which is the raw accumulator.
        wl.tensors[r.out].dtype = "i32"
    for op in wl.ops:
        if op.kind in CONTRACTIONS + ("commit",) and op.out in outs:
            op.epilogue = Epilogue(stages=(), output_dtype="i32")
    wl.quant = p
