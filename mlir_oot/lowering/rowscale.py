"""Hoist a per-ROW scale out of a contraction, so the contraction itself stays exact integer work.

A normalisation feeding a contraction is not two independent regions. ``rmsnorm`` divides every
element of row ``i`` by ONE scalar -- the row's own RMS -- and the contraction that follows sums over
COLUMNS, so the scalar is constant across everything the sum touches and commutes with it::

    (x[i,:] / r[i] * g) @ W   ==   ( (x[i,:] * g) @ W ) / r[i]

The left side is what the interface writes. The right side is an exact INTEGER contraction of two
declared i8 tensors, followed by one division per output element -- which is the form this target can
actually issue, because its operand port reads integers and its accumulator is exact.

This is worth more than a convenience. Rounding the normalisation to its declared integer container
FIRST and contracting that is a different function: measured against the capsule's own declared
stimulus, the two disagree on 250 of 256 elements by up to 13. The fused operation is defined on the
UNROUNDED normalisation, and hoisting the row scale is how a compiler reaches that definition without
a float operand port -- not by approximating it, but by reassociating into arithmetic the hardware
already does exactly.

``x[i,k] * g[k]`` is the product of two i8 values, so it always fits i16 -- a bound that comes from
the declared containers and holds for every value they can carry. The staged operand is declared that
width, so the wide-operand split below it (:mod:`.widen`) needs two digits rather than four.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.workload import Epilogue, Op, TensorDecl, Workload

#: The product of two i8 operands, which is what the hoisted operand holds.
PRODUCT_DTYPE = "i16"
#: The contraction's readout: the raw accumulator word, because the row scale has not been applied.
PARTIAL_DTYPE = "i32"


@dataclass
class RowScalePlan:
    """One ``rmsnorm`` whose row scale is hoisted past the contraction that consumes it."""

    norm: Op
    matmul: Op
    commit: Op
    src: str
    gamma: str
    weighted: str
    partial: str

    def reason(self) -> str:
        return (
            "the normalisation's divisor is constant across the row the contraction sums over, so "
            "it is hoisted past the contraction: the array contracts the exact integer product of "
            "the declared operands and the row scale is applied once, at the readout. The fused "
            "operation is defined on the UNROUNDED normalisation, which no rounding of the "
            "intermediate into its declared container can reproduce"
        )


def plan(wl: Workload) -> RowScalePlan | None:
    """The ``rmsnorm -> matmul -> commit`` chain this pass can reassociate, if ``wl`` has one."""
    commits = {op.operands["src"]: op for op in wl.ops if op.kind == "commit"}
    for norm in wl.ops:
        if norm.kind != "rmsnorm":
            continue
        mm = next(
            (o for o in wl.ops if o.kind == "matmul" and o.operands.get("lhs") == norm.out), None
        )
        if mm is None:
            continue
        commit = commits.get(mm.out)
        if commit is None:
            continue
        if commit.epilogue is not None and commit.epilogue.has:
            continue        # the readout is a function of the SCALED sum; not attempted here
        src, gamma = norm.operands.get("src"), norm.operands.get("gamma")
        if src not in wl.tensors or gamma not in wl.tensors:
            continue
        if wl.tensors[src].dtype != "i8" or wl.tensors[gamma].dtype != "i8":
            continue        # the i16 product bound is what makes the hoist cheap AND exact
        if wl.tensors[commit.out].dtype != PARTIAL_DTYPE:
            continue
        if tuple(wl.tensors[norm.out].shape) != tuple(wl.tensors[src].shape):
            continue
        return RowScalePlan(
            norm=norm, matmul=mm, commit=commit, src=src, gamma=gamma,
            weighted=f"{src}${gamma}", partial=f"{commit.out}$z",
        )
    return None


def apply(wl: Workload, p: RowScalePlan) -> None:
    """Rewrite ``wl`` in place: weight the operand, contract it, then apply the row scale."""
    src = wl.tensors[p.src]
    out = wl.tensors[p.commit.out]
    wl.tensors[p.weighted] = TensorDecl(p.weighted, tuple(src.shape), PRODUCT_DTYPE, "scratch")
    wl.tensors[p.partial] = TensorDecl(p.partial, tuple(out.shape), PARTIAL_DTYPE, "scratch")
    wl.scratch = list(wl.scratch) + [p.weighted, p.partial]

    rebuilt: list[Op] = []
    for op in wl.ops:
        if op is p.norm:
            # The interface DECLARES this intermediate, so it is still computed and still written;
            # what changes is that the contraction no longer reads it.
            rebuilt.append(op)
            rebuilt.append(
                Op("row_weight", p.weighted, {"src": p.src, "gamma": p.gamma}, {})
            )
            continue
        if op is p.matmul:
            rebuilt.append(
                Op("matmul", op.out, {**op.operands, "lhs": p.weighted}, dict(op.attrs),
                   epilogue=op.epilogue, shape=dict(op.shape))
            )
            continue
        if op is p.commit:
            rebuilt.append(Op("commit", p.partial, dict(op.operands), dict(op.attrs),
                              epilogue=Epilogue(output_dtype=PARTIAL_DTYPE)))
            rebuilt.append(
                Op("row_normalize", op.out, {"src": p.partial, "ref": p.src},
                   {"eps": float(p.norm.attrs.get("eps", 0.0))})
            )
            continue
        rebuilt.append(op)
    wl.ops = rebuilt
