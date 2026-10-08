"""Explicit dense signed-i8 product pairs sharing one exact i32 destination.

A/B point to plane-major arrays [planes,M,K] and [planes,K,N]. Each pair is
reduced into the same resident accumulator block, then stored once. Callers
provide a proven absolute i32 bound; this is integer sum semantics, no float
reassociation or radix policy is inferred by this target provider.
"""

from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerAttr, i64

from .golden_gemm import GoldenGemm, Shape, _ceil_div
from .tables import isa
from .tables import rtl_facts as F


@dataclass(frozen=True)
class OperandResidencyPlan:
    """Complete, call-local storage of only the referenced signed-i8 planes.

    Admission proves resource legality, not profitability. A caller may select
    the existing panel schedule when ``admitted`` is false. No pointers or
    contents survive a call, and no input-format/numeric permission is inferred.
    """

    lhs_indices: tuple[int, ...]
    rhs_indices: tuple[int, ...]
    lhs_rows: int
    rhs_rows: int
    accumulator_rows: int
    admitted: bool
    reason: str


def operand_residency_plan(
    shape: Shape, pairs: tuple[tuple[int, int], ...]
) -> OperandResidencyPlan:
    """Price physical tile storage from the actual shape and referenced planes."""
    shape.validate()
    if not pairs or any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(index) is not int or index < 0 for index in pair)
        for pair in pairs
    ):
        raise ValueError("residency planning requires explicit nonnegative plane pairs")
    lhs = tuple(sorted({pair[0] for pair in pairs}))
    rhs = tuple(sorted({pair[1] for pair in pairs}))
    mt, nt, kt = (_ceil_div(extent, F.DIM) for extent in (shape.m, shape.n, shape.k))
    a_rows = len(lhs) * mt * kt * F.DIM
    b_rows = len(rhs) * kt * nt * F.DIM
    acc_rows = shape.bm * shape.bn * F.DIM
    fits = a_rows + b_rows <= F.SPAD_ROWS and acc_rows <= F.ACC_ROWS
    return OperandResidencyPlan(
        lhs,
        rhs,
        a_rows,
        b_rows,
        acc_rows,
        fits,
        "complete operands fit" if fits else "complete operands exceed target storage",
    )


class GoldenProductSum(GoldenGemm):
    def __init__(
        self,
        shape: Shape,
        *,
        pairs: tuple[tuple[int, int], ...],
        lhs_planes: int,
        rhs_planes: int,
        absolute_bound: int,
        lhs_magnitude_bound: int = 128,
        rhs_magnitude_bound: int = 128,
        resident_operands: bool = False,
    ):
        if type(resident_operands) is not bool:
            raise ValueError("operand residency selection must be boolean")
        if resident_operands and shape.separate_b_bank:
            raise ValueError(
                "complete operand residency requires its own contiguous bank placement"
            )
        if shape.output_dtype != "i32" or shape.bias:
            raise ValueError("product sums require unbiased i32 output")
        if any(
            (
                shape.cache_a,
                shape.cache_b,
                shape.wide_a,
                shape.pipeline_m,
                shape.prefetch_b,
                shape.prefetch_m,
                shape.banked_m,
            )
        ):
            raise ValueError(
                "product sum currently requires independently loaded K panels"
            )
        if not pairs or min(lhs_planes, rhs_planes) <= 0:
            raise ValueError("nonempty pairs and positive plane counts required")
        if type(absolute_bound) is not int or not 0 <= absolute_bound < 1 << 31:
            raise ValueError("explicit signed-i32 absolute bound required")
        if any(
            type(v) is not int or not 0 <= v <= 128
            for v in (lhs_magnitude_bound, rhs_magnitude_bound)
        ):
            raise ValueError("signed-i8 operand magnitude domains required")
        required = len(pairs) * shape.k * lhs_magnitude_bound * rhs_magnitude_bound
        if absolute_bound < required:
            raise ValueError("accumulator bound does not cover every product prefix")
        # Caller-owned source proof must establish these operand domains; no
        # value scan or benchmark-specific assumption is emitted here.
        for a, b in pairs:
            if (
                type(a) is not int
                or type(b) is not int
                or not (0 <= a < lhs_planes and 0 <= b < rhs_planes)
            ):
                raise ValueError("plane index outside declared input span")
        self.pairs = tuple(pairs)
        self._first_pair = True
        super().__init__(shape)
        self.resident_operands = resident_operands
        self.residency = (
            operand_residency_plan(shape, self.pairs) if resident_operands else None
        )
        if self.residency is not None and not self.residency.admitted:
            raise ValueError(self.residency.reason)
        if resident_operands and (
            any(type(v) is not int for v in (lhs_planes, rhs_planes))
            or max(
                lhs_planes * shape.m * shape.k,
                rhs_planes * shape.k * shape.n,
                shape.m * shape.n * 4,
            )
            >= 1 << 63
        ):
            raise ValueError(
                "resident operand source span exceeds signed pointer indexing"
            )

    def _emit_work(self):
        if self.residency is not None:
            s = self.shape
            mt, nt, kt = (_ceil_div(extent, F.DIM) for extent in (s.m, s.n, s.k))
            # CONFIG_LD's existing DIM block stride places each <=4-tile
            # packet in the same disjoint tile layout consumed below. Tail
            # packets use exact source extents, never speculative reads.
            for slot, plane in enumerate(self.residency.lhs_indices):
                for mi in range(mt):
                    rows = min(F.DIM, s.m - mi * F.DIM)
                    for ki in range(0, kt, 4):
                        ptr = self._ptr(
                            self.a,
                            self.fb.const(plane * s.m + mi * F.DIM),
                            s.k,
                            self.fb.const(ki * F.DIM),
                        )
                        self._rocc(
                            "mvin",
                            {
                                "local": (slot * mt * kt + mi * kt + ki) * F.DIM,
                                "rows": rows,
                                "cols": min(4 * F.DIM, s.k - ki * F.DIM),
                                "load_id": 0,
                            },
                            ptr,
                        )
            for slot, plane in enumerate(self.residency.rhs_indices):
                for ki in range(kt):
                    rows = min(F.DIM, s.k - ki * F.DIM)
                    for ni in range(0, nt, 4 if s.wide_b else 1):
                        width = 4 if s.wide_b else 1
                        ptr = self._ptr(
                            self.b,
                            self.fb.const(plane * s.k + ki * F.DIM),
                            s.n,
                            self.fb.const(ni * F.DIM),
                        )
                        self._rocc(
                            "mvin",
                            {
                                "local": self.residency.lhs_rows
                                + (slot * kt * nt + ki * nt + ni) * F.DIM,
                                "rows": rows,
                                "cols": min(width * F.DIM, s.n - ni * F.DIM),
                                "load_id": 1,
                            },
                            ptr,
                        )
        super()._emit_work()

    def _resident_k_tile(self, m0, n0, k0, mr, nr, kr, first, ap, bp):
        s, plan = self.shape, self.residency
        assert plan is not None
        mt, nt, kt = (_ceil_div(extent, F.DIM) for extent in (s.m, s.n, s.k))
        a_base = plan.lhs_indices.index(ap) * mt * kt * F.DIM
        b_base = plan.lhs_rows + plan.rhs_indices.index(bp) * kt * nt * F.DIM
        a_row = self.fb.add_i(
            self.fb.const(a_base), self.fb.mul_i(m0, self.fb.const(kt * F.DIM))
        )
        a_row = self.fb.add_i(a_row, self.fb.mul_i(k0, self.fb.const(F.DIM)))
        b_row = self.fb.add_i(
            self.fb.const(b_base), self.fb.mul_i(k0, self.fb.const(nt * F.DIM))
        )
        b_row = self.fb.add_i(b_row, self.fb.mul_i(n0, self.fb.const(F.DIM)))

        def tile(a, d):
            rows, cols = mr[a], nr[d]
            attrs = {
                "c": isa.acc_addr((a * s.bn + d) * F.DIM, accumulate=not first),
                "bd_cols": cols,
                "bd_rows": kr,
                "c_cols": cols,
                "c_rows": rows,
            }
            if not s.reuse_b or a == 0:
                attrs.update(
                    bd_min=b_base,
                    bd_max=b_base + (kt * nt - 1) * F.DIM,
                    bd_reserved_rows=plan.lhs_rows + plan.rhs_rows,
                    bd_alignment=F.DIM,
                )
                address = (
                    b_row if d == 0 else self.fb.add_i(b_row, self.fb.const(d * F.DIM))
                )
                self._rocc("preload", attrs, address)
            else:
                attrs["bd"] = isa.GARBAGE_ADDR
                self._rocc("preload", attrs)
            address = (
                a_row if a == 0 else self.fb.add_i(a_row, self.fb.const(a * kt * F.DIM))
            )
            self._rocc(
                "compute",
                {
                    "a_max": a_base + (mt * kt - 1) * F.DIM,
                    "a_reserved_rows": plan.lhs_rows,
                    "a_cols": kr,
                    "a_rows": rows,
                    "accumulate": s.reuse_b and a != 0,
                },
                address,
            )

        if s.reuse_b:
            for d in range(len(nr)):
                for a in range(len(mr)):
                    tile(a, d)
        else:
            for a in range(len(mr)):
                for d in range(len(nr)):
                    tile(a, d)

    def _k_tile(self, m0, n0, k0, mr, nr, kr, first, *args, **kwargs):
        return super()._k_tile(
            m0, n0, k0, mr, nr, kr, first and self._first_pair, *args, **kwargs
        )

    def _reduce_output_block(self, m0, n0, mr, nr, slot=0, prefetch=None):
        if self.residency is not None:
            for index, (ap, bp) in enumerate(self.pairs):
                self._resident_k_tile(
                    m0,
                    n0,
                    self.fb.const(0),
                    mr,
                    nr,
                    min(F.DIM, self.shape.k),
                    index == 0,
                    ap,
                    bp,
                )
                full = self.shape.k // F.DIM
                if full > 1:
                    self.fb.for_loop(
                        1,
                        full,
                        1,
                        lambda ki, ap=ap, bp=bp: self._resident_k_tile(
                            m0, n0, ki, mr, nr, F.DIM, False, ap, bp
                        ),
                    )
                if self.shape.k > F.DIM and self.shape.k % F.DIM:
                    self._resident_k_tile(
                        m0,
                        n0,
                        self.fb.const(full),
                        mr,
                        nr,
                        self.shape.k % F.DIM,
                        False,
                        ap,
                        bp,
                    )
            return
        a, b = self.a, self.b
        s = self.shape
        for index, (ap, bp) in enumerate(self.pairs):
            self._first_pair = index == 0
            self.a = self._ptr(a, self.fb.const(ap * s.m), s.k, self.fb.const(0))
            self.b = self._ptr(b, self.fb.const(bp * s.k), s.n, self.fb.const(0))
            super()._reduce_output_block(m0, n0, mr, nr, slot, prefetch)
        self.a, self.b = a, b
        self._first_pair = True

    def build_batched(self, batch):
        if self.resident_operands:
            raise ValueError(
                "resident product planes require a separately admitted batched ABI"
            )
        return super().build_batched(batch)

    def _finish(self, symbol, batch=1):
        module = super()._finish(symbol, batch)
        if self.resident_operands:
            module.attributes["gemmini.resident_product_operands"] = IntegerAttr(1, i64)
        return module
