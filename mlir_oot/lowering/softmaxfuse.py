"""Fuse a softmax into the contraction that consumes its weights.

``softmax`` produces attention WEIGHTS: a row of them sums to one, so they are not integers, and the
interface's declared integer container for them is a description of the buffer rather than of the
arithmetic. Rounding them into it before the contraction is a different function -- the same mistake
:mod:`.rowscale` documents for ``rmsnorm``, where putting the intermediate back in its declared
container first disagreed with the fused definition on 250 of 256 elements.

So the contraction that consumes the weights reads them BEFORE they are put in a container, from the
same stage that produced them. The declared intermediate is still written, because the interface
declares it; what changes is that nothing reads it.

This leaves the weighted sum off the array -- the weights are not integers and this datapath's
operand port only reads integers, which is a statement about the hardware and is DECLARED as a host
placement rather than hidden. The contraction that CAN go on the array (the query-key product, whose
operands are both declared i8) still does.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.workload import Op, TensorDecl, Workload

#: The weights are re-derived in single precision, which is what the scalar lane computes in, so a
#: weight read back out of this buffer is the SAME f32 the inline expansion would have produced.
WEIGHT_DTYPE = "f32"


@dataclass
class SoftmaxFusion:
    """One softmax whose weights are consumed by a contraction in the same program."""

    softmax: Op
    consumer: Op
    weights_src: str
    value: str

    def reason(self) -> str:
        return (
            "the softmax weights this contraction sums are not integers, and this target's "
            "RTL-derived operand port reads integers only (scratchpad UInt<8>), so the weighted sum "
            "runs on the scalar lane against the UNROUNDED weights -- the fused operation is "
            "defined on those, and no rounding of them into the declared intermediate container "
            "reproduces it. The query-key product, whose operands the port does encode, stays on "
            "the array"
        )


#: contraction mnemonic -> (the role that reads the weights, the role that reads the values)
CONSUMERS = {"attention_pv": ("p", "v"), "matmul": ("lhs", "rhs")}


def plan(wl: Workload) -> SoftmaxFusion | None:
    """The ``softmax -> contraction`` pair this pass fuses, if ``wl`` has one."""
    for sm in wl.ops:
        if sm.kind != "softmax":
            continue
        for op in wl.ops:
            roles = CONSUMERS.get(op.kind)
            if roles is None:
                continue
            weight_role, value_role = roles
            if op.operands.get(weight_role) != sm.out:
                continue
            value = wl.residents.get(op.operands.get(value_role, ""),
                                     op.operands.get(value_role, ""))
            if value not in wl.tensors:
                continue
            if op.epilogue is not None and op.epilogue.has:
                continue        # the readout is a function of the weighted sum; not attempted here
            return SoftmaxFusion(softmax=sm, consumer=op, weights_src=sm.operands["src"],
                                 value=value)
    return None


def apply(wl: Workload, f: SoftmaxFusion) -> None:
    """Rewrite ``wl`` in place: the consumer reads the softmax's SOURCE and re-derives the weights.

    One row of weights is derived ONCE into a kernel-frame buffer and then read by every output
    column, rather than re-derived inside the column loop. The weighted sum contracts ``k`` weights
    for each of ``n`` output columns, so deriving them in place costs ``n * k`` exponentials per row
    where ``k`` suffice -- and the exponential is the expensive term, expanded from base arithmetic
    because the bare-metal harness links no math library. The buffer holds the identical f32 the
    inline expansion produced, so this changes the program's COST and not its arithmetic.
    """
    src = wl.tensors[f.weights_src]
    row = f"{f.consumer.out}$w"
    wl.tensors[row] = TensorDecl(row, (1, int(src.shape[-1])), WEIGHT_DTYPE, "scratch")
    wl.scratch = list(wl.scratch) + [row]

    rebuilt: list[Op] = []
    for op in wl.ops:
        if op is f.consumer:
            rebuilt.append(
                Op("softmax_weighted_sum", op.out,
                   {"src": f.weights_src, "value": f.value, "weights": row},
                   {**f.softmax.attrs, "transposed": op.kind == "matmul"})
            )
            continue
        rebuilt.append(op)
    wl.ops = rebuilt
