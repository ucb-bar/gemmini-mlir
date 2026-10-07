"""Route W: contract an operand WIDER than the mesh's operand port, exactly, on the mesh.

The RTL-derived datapath reads i8 operands (scratchpad ``UInt<8>``) into an i32 accumulator, so
:func:`..iface_to_gemmini.require_mesh_dtypes` refuses a contraction whose left operand is an i32
tensor. A program can legitimately ask for exactly that: an off-mesh stage that commits an i32
intermediate, immediately contracted against an i8 weight.

Refusing it to the host lane computes the right numbers while the array issues nothing, which is not
what a capsule asking for the array's own instruction classes is asking for. This pass answers the
request the way any compiler carries a wide multiply on a narrow multiplier -- by SPLITTING the wide
operand into radix-256 digits the port can encode, issuing one contraction per digit, and recombining
the partial products with the shifts the split implies::

    H = sum_s d_s * 256**s        with every d_s in [-128, 127]
    H @ W = sum_s 256**s * (d_s @ W)

The digits are BALANCED (each one signed, in the operand port's own symmetric range) rather than the
plain unsigned bytes, because an unsigned byte does not fit a signed port and correcting for the bias
would need a column sum of the weight that this datapath has no cheaper way to produce. The split is
the standard one: ``r_0 = H``, ``r_{s+1} = floor((r_s + 128) / 256)``, ``d_s = r_s - 256*r_{s+1}``.

Exactness: with ``n = width/8`` digits the residual is ``r_n * 256**n``, and ``256**n == 2**width`` is
zero in the output's own container -- so the identity holds exactly in the arithmetic the result is
compared in, which is the declared integer container and not a wider one.

The rewrite is structural and happens on the Workload IR, on a COPY, so the mesh lowering below it is
the same weight-stationary emitter every other integer contraction uses, and the command buffer goes
on describing the program the interface declared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir.workload import Op, TensorDecl, Workload
from ..target import facts, layout

#: Element types this route can SPLIT: integer containers wider than the mesh's operand port.
WIDE_OPERANDS = {"i16": 2, "i32": 4}
#: What the port itself encodes -- one digit is exactly one of these.
DIGIT_DTYPE = "i8"
RADIX_BITS = 8
#: The partial products are read out raw, so the readout container is the accumulator's own width.
PARTIAL_DTYPE = "i32"

#: Staging lives in the kernel's own stack frame, under the same ceiling route Q respects.
STAGING_BUDGET_BYTES = 32768

#: contraction mnemonic -> (wide operand role, narrow operand role). Only the role named first is
#: split; a contraction with TWO wide operands would need the cross terms as well and is refused.
SPLITTABLE = {
    "matmul": ("lhs", "rhs"),
    "attention_pv": ("p", "v"),
    "attention_qk": ("q", "k"),
}


@dataclass
class WidenPlan:
    """One contraction whose wide operand this route splits, and the buffers that takes."""

    target: str                      # the op's own output tensor name
    kind: str
    wide: str                        # the declared tensor being split
    digits: int
    digit_names: list[str] = field(default_factory=list)
    partial_names: list[str] = field(default_factory=list)

    @property
    def scratch(self) -> list[str]:
        return list(self.digit_names) + list(self.partial_names)


def _bytes_of(wl: Workload, name: str, dtype: str) -> int:
    from .iface_to_gemmini import dtype_bytes

    shape = wl.tensors[name].shape
    return layout.row_count(shape) * layout.row_pitch(shape) * dtype_bytes(dtype)


def plan(wl: Workload) -> list[WidenPlan] | None:
    """The contractions in ``wl`` this route can carry, or ``None`` when it does not apply.

    Refuses rather than half-applies: a contraction whose readout asks for anything but the raw
    accumulator word cannot have its epilogue applied per digit (the epilogue is a function of the
    SUM, not of each partial), and one with two wide operands would need the cross terms too.
    """
    commits = {op.operands["src"]: op for op in wl.ops if op.kind == "commit"}
    out: list[WidenPlan] = []
    for op in wl.ops:
        roles = SPLITTABLE.get(op.kind)
        if roles is None:
            continue
        wide_role, narrow_role = roles
        wide = wl.residents.get(op.operands.get(wide_role, ""), op.operands.get(wide_role, ""))
        narrow = wl.residents.get(op.operands.get(narrow_role, ""), op.operands.get(narrow_role, ""))
        if wide not in wl.tensors or narrow not in wl.tensors:
            continue
        width = WIDE_OPERANDS.get(wl.tensors[wide].dtype)
        if width is None:
            continue
        if wl.tensors[narrow].dtype != DIGIT_DTYPE:
            return None     # both operands wide: the split would need the cross terms as well
        result = commits.get(op.out) if op.kind == "matmul" else op
        if result is None:
            return None
        if result.epilogue is not None and result.epilogue.has:
            return None     # the readout is a function of the SUM, not of one digit's partial
        if wl.tensors[result.out].dtype != PARTIAL_DTYPE:
            return None
        out.append(WidenPlan(target=result.out, kind=op.kind, wide=wide, digits=width))
    if not out:
        return None
    return out


def reason(plans: list[WidenPlan]) -> str:
    names = ", ".join(sorted({p.wide for p in plans}))
    digits = max(p.digits for p in plans)
    return (
        f"issued on this target's {facts.DIM}x{facts.DIM} systolic mesh: operand(s) {names} are "
        f"wider than the RTL-derived operand port (scratchpad UInt<8>), so the compiler splits each "
        f"one into {digits} balanced radix-{1 << RADIX_BITS} digits the port DOES encode, contracts "
        f"each digit against the same resident weight, and recombines the partial products with the "
        f"shifts the split implies -- exactly, in the output's own container"
    )


def apply(wl: Workload, plans: list[WidenPlan]) -> None:
    """Rewrite ``wl`` in place: split, one contraction per digit, recombine.

    Raises ``ValueError`` when the staging the split needs does not fit the kernel's frame budget,
    so the caller states that as a refusal rather than emitting a kernel that overflows its stack.
    """
    by_target = {p.target: p for p in plans}
    budget = 0
    for p in plans:
        src = wl.tensors[p.wide]
        outd = wl.tensors[p.target]
        for s in range(p.digits):
            d = f"{p.wide}$d{s}"
            wl.tensors[d] = TensorDecl(d, tuple(src.shape), DIGIT_DTYPE, "scratch")
            p.digit_names.append(d)
            q = f"{p.target}$p{s}"
            wl.tensors[q] = TensorDecl(q, tuple(outd.shape), PARTIAL_DTYPE, "scratch")
            p.partial_names.append(q)
            budget += _bytes_of(wl, d, DIGIT_DTYPE) + _bytes_of(wl, q, PARTIAL_DTYPE)
    if budget > STAGING_BUDGET_BYTES:
        raise ValueError(
            f"splitting this contraction's wide operand needs {budget} bytes of kernel staging, "
            f"over the {STAGING_BUDGET_BYTES}-byte frame budget this package holds itself to"
        )

    commits = {op.operands["src"]: op for op in wl.ops if op.kind == "commit"}
    consumed_commit = {c.out for c in commits.values() if c.out in by_target}
    rebuilt: list[Op] = []
    for op in wl.ops:
        target = None
        if op.kind in SPLITTABLE:
            result = commits.get(op.out) if op.kind == "matmul" else op
            target = result.out if result is not None else None
        if target is None or target not in by_target:
            if op.kind == "commit" and op.out in consumed_commit:
                continue            # emitted per digit, just below
            rebuilt.append(op)
            continue

        p = by_target[target]
        wide_role = SPLITTABLE[op.kind][0]
        rebuilt.append(
            Op("int_digits", p.digit_names[0], {"src": p.wide},
               {"outs": list(p.digit_names), "radix_bits": RADIX_BITS})
        )
        for s, (d, q) in enumerate(zip(p.digit_names, p.partial_names)):
            sub = Op(op.kind, f"{op.out}$w{s}", dict(op.operands), dict(op.attrs),
                     epilogue=op.epilogue, shape=dict(op.shape))
            sub.operands[wide_role] = d
            if op.kind == "matmul":
                rebuilt.append(sub)
                rebuilt.append(Op("commit", q, {"src": sub.out}, {}))
            else:
                sub.out = q
                rebuilt.append(sub)
        rebuilt.append(
            Op("digit_combine", target,
               {f"src{s}": q for s, q in enumerate(p.partial_names)},
               {"radix_bits": RADIX_BITS, "digits": p.digits})
        )
    wl.ops = rebuilt
    wl.scratch = list(wl.scratch) + [
        n for p in plans for n in p.scratch if n not in wl.scratch
    ]
