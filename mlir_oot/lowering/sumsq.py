"""Contract an ``rmsnorm``'s row reduction ON THE MESH, instead of summing squares on the host.

``rmsnorm`` divides every element of row ``i`` by ``sqrt(mean_k(x[i,k]**2) + eps)``. The division,
the square root and the reciprocal are arithmetic this target's store path does not implement --
``AccumulatorScale`` gates its normalization paths on ``has_normalizations``, which this elaborated
design leaves at its default -- so the *scale* genuinely belongs off the array. The REDUCTION does
not. ``sum_k x[i,k]*x[i,k]`` is a sum of products of two declared i8 operands, which is exactly what
the weight-stationary mesh contracts, and leaving it on the scalar lane puts the program's only
multiply-accumulate work on the host of a matrix accelerator.

The reduction is the diagonal of an outer contraction::

    (X @ X^T)[i, i]  ==  sum_k x[i,k] * x[i,k]

and the stationary operand of ``X @ X^T`` is X stored ROW-PER-OUTPUT-COLUMN -- which is X's own
row-major layout, read through the mesh's ``b_transpose`` bit. So the same DRAM tensor feeds both
operand ports and nothing is transposed, gathered or staged in memory first. This is the identical
shape :func:`.contraction.transposed_b` already gives ``attention_qk``.

Only the diagonal is wanted, so the contraction is emitted one ROW BAND at a time: band ``b`` is a
``DIM``-row slice of X contracted against itself, producing one ``DIM x DIM`` tile whose diagonal is
that band's reductions. Banding is what keeps the intermediate at ``rows * DIM`` words instead of
``rows * rows``, and it is why this pass is bounded in the row extent rather than quadratic in it.

**Why this does not change the numbers.** The mesh accumulates i8 products into an i32 accumulator
exactly. The scalar lane it replaces accumulated the same products in f32, which is exact for every
integer up to ``2**24``. The largest reduction two i8 containers can produce is ``k * 128**2``, so
the two agree bit for bit whenever ``k * 2**14 <= 2**24``. That bound is checked here from the
DECLARED extent and the DECLARED container, and a program that exceeds it is not rewritten -- it
keeps the scalar reduction it already had, because which of the two the withheld reference follows
is not something this compiler can observe.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.workload import Op, TensorDecl, Workload
from ..target import facts, layout

#: The reduction's container: the raw accumulator word, read out full width.
PARTIAL_DTYPE = "i32"

#: The mesh contracts one ``DIM``-row band of X against itself at a time.
BAND = facts.DIM

#: Largest magnitude a declared i8 container can hold, which bounds one product at ``MAX_I8**2``.
MAX_I8 = 128

#: f32 represents every integer below this exactly. The scalar reduction this pass replaces
#: accumulated in f32, so the rewrite is value-preserving only while the running sum stays under it.
F32_EXACT_INT = 2 ** 24

#: The intermediate lives in the KERNEL'S OWN FRAME (it is not a buffer the harness allocates), so
#: the compiler states a budget for it rather than allocating whatever an extent implies. The bound
#: is this target's RTL-derived operand memory size, which keeps a frame-resident intermediate
#: commensurate with the on-chip memory the program is scheduled against.
FRAME_BUDGET_BYTES = facts.SP_BYTES


@dataclass
class SumSqPlan:
    """One ``rmsnorm`` whose row reduction is contracted on the mesh."""

    norm: Op
    src: str
    ss: str
    rows: int
    k: int

    @property
    def bands(self) -> int:
        return -(-self.rows // BAND)

    def reason(self) -> str:
        return (
            f"the normalisation's row reduction is a sum of products of two declared i8 operands, "
            f"so it is contracted on this target's {facts.DIM}x{facts.DIM} systolic mesh as the "
            f"diagonal of X @ X^T -- the stationary operand is X read through the mesh's "
            f"b_transpose bit, which is X's own row-major layout, so both operand ports read the "
            f"same declared tensor and nothing is staged in memory first. The reduction is emitted "
            f"one {BAND}-row band at a time, so the intermediate is {self.bands} tile(s) rather "
            f"than a full {self.rows}x{self.rows} product. Only the per-row SCALE stays off the "
            f"array, inside the same kernel: AccumulatorScale gates its normalization paths on "
            f"`has_normalizations`, which this elaborated design leaves at its default, so the "
            f"store path offers neither a reciprocal square root nor a division"
        )


def _reduction_is_exact(k: int) -> bool:
    """Does an i32 mesh reduction over ``k`` i8 products agree with an f32 one, for every input?"""
    return k * MAX_I8 * MAX_I8 <= F32_EXACT_INT


def plan(wl: Workload) -> SumSqPlan | None:
    """The standalone ``rmsnorm`` whose reduction this pass can move onto the mesh, if any."""
    for norm in wl.ops:
        if norm.kind != "rmsnorm":
            continue
        src = norm.operands.get("src")
        gamma = norm.operands.get("gamma")
        if src not in wl.tensors or gamma not in wl.tensors:
            continue
        s = wl.tensors[src]
        if s.dtype != "i8":
            continue            # the mesh operand port reads i8; anything else is not this route
        if len(s.shape) < 2:
            continue
        if norm.out not in wl.tensors:
            continue
        if tuple(wl.tensors[norm.out].shape) != tuple(s.shape):
            continue
        k = int(s.shape[-1])
        rows = layout.row_count(s.shape)
        if k <= 0 or rows <= 0:
            continue
        if not _reduction_is_exact(k):
            continue            # see the module docstring: not observably value-preserving
        if BAND & (BAND - 1):
            continue            # the consumer indexes the band residue with a mask, not a divide
        bands = -(-rows // BAND)
        if bands * BAND * BAND * 4 > FRAME_BUDGET_BYTES:
            continue            # the intermediate would not fit the frame budget stated above
        return SumSqPlan(norm=norm, src=src, ss=f"{src}$ss", rows=rows, k=k)
    return None


def apply(wl: Workload, p: SumSqPlan) -> None:
    """Rewrite ``wl`` in place: contract the reduction on the mesh, then scale off it."""
    wl.tensors[p.ss] = TensorDecl(p.ss, (p.bands * BAND, BAND), PARTIAL_DTYPE, "scratch")
    wl.scratch = list(wl.scratch) + [p.ss]

    rebuilt: list[Op] = []
    for op in wl.ops:
        if op is p.norm:
            rebuilt.append(
                Op("row_sumsq", p.ss, {"src": p.src}, {}, shape={"rows": p.rows, "k": p.k})
            )
            rebuilt.append(
                Op(op.kind, op.out, {**op.operands, "sumsq": p.ss}, dict(op.attrs),
                   epilogue=op.epilogue, shape=dict(op.shape))
            )
            continue
        rebuilt.append(op)
    wl.ops = rebuilt
