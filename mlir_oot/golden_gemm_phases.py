"""Explicit cached-input GEMM issue phases for an owned host/device pipeline.

This provider emits no automatic routing decision. The caller proves identical
immutable A, source integer contractions, complete output ownership and exclusive
device use from begin through the terminal wait. Each issue borrows two disjoint
host destinations until wait; the subsequent host consumer may overlap the next
issue only after its own destinations have become ready. No LOOP_* command is
used. Target resources, instructions and protocol are owned here.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64

from .codegen.builder import PTR, FnBuilder
from .golden_gemm import GoldenGemm, Shape, _ceil_div
from .tables import isa
from .tables import rtl_facts as F


@dataclass(frozen=True)
class PairedIssueResources:
    m: int
    n: int
    k: int
    columns: int
    a_rows: int
    accumulator_rows_per_output: int
    total_accumulator_rows: int
    host_bytes_per_slot: int
    slots: int = 2


class GoldenGemmPhases(GoldenGemm):
    """Two sibling products per issue, with an explicit begin/wait protocol.

    Current admission is one complete M block, at least two K panels, exact
    column tiles and an independently supplied fixed issue width dividing N.
    Both products use the established resident-A / prefetched-B primitive
    schedule. Private scratch and output storage are never allocated here.
    """

    def __init__(self, shape: Shape, *, columns: int):
        if (
            type(columns) is not int
            or columns <= 0
            or columns % F.DIM
            or shape.n % columns
            or shape.output_dtype != "i32"
            or shape.bias
            or shape.scale != 1.0
            or shape.relu
            or not shape.cache_a
            or not shape.prefetch_b
            or shape.pipeline_m
            or shape.cache_b
            or shape.wide_a
            or shape.bm != _ceil_div(shape.m, F.DIM)
            or shape.bn != columns // F.DIM
        ):
            raise ValueError(
                "paired issue requires explicit exact-width cached-A i32 products"
            )
        super().__init__(shape)
        rows = shape.bm * shape.bn * F.DIM
        if 2 * rows > F.ACC_ROWS:
            raise ValueError("paired products exceed the accumulator capacity")
        self.resources = PairedIssueResources(
            shape.m,
            shape.n,
            shape.k,
            columns,
            shape.bm * _ceil_div(shape.k, F.DIM) * F.DIM,
            rows,
            2 * rows,
            2 * shape.m * columns * 4,
        )
        self._accumulator_offset = 0

    def _rocc(self, kind, attrs, pointer=None):
        attrs = dict(attrs)
        if kind == "preload" and self._accumulator_offset:
            # Preserve the source first-K overwrite and subsequent accumulation
            # flags while placing the sibling in a disjoint accumulator span.
            attrs["c"] += self._accumulator_offset
        super()._rocc(kind, attrs, pointer)

    def _function(self, symbol, *, terminal_wait):
        if terminal_wait:
            self._rocc("fence", {})
        function = llvm.FuncOp(
            symbol,
            llvm.LLVMFunctionType([arg.type for arg in self.fb.entry.args]),
            linkage=llvm.LinkageAttr("external"),
            body=self.fb.finish(),
        )
        return function

    def _reset(self, arguments):
        self.fb = FnBuilder(arguments)
        self._accumulator_offset = 0

    def build(self, *, prefix: str):
        if (
            not isinstance(prefix, str)
            or not prefix.isascii()
            or not prefix.isidentifier()
        ):
            raise ValueError("explicit ordinary symbol prefix required")
        shape = self.shape
        self._reset([PTR])
        self.a = self.fb.entry.args[0]
        self._emit_config()
        # _emit_config uses the original row pitch for weights; compact owned
        # tile destinations have the explicit issue width as their row pitch.
        self._rocc(
            "config_st",
            {
                "stride": self.resources.columns * 4,
                "acc_act": isa.NO_ACTIVATION,
                "acc_scale": 1.0,
            },
        )
        kt = _ceil_div(shape.k, F.DIM)
        for a in range(shape.bm):
            rows = min(F.DIM, shape.m - a * F.DIM)
            for ki in range(kt):
                pointer = self._ptr(
                    self.a, self.fb.const(a * F.DIM), shape.k, self.fb.const(ki * F.DIM)
                )
                self._rocc(
                    "mvin",
                    {
                        "local": (a * kt + ki) * F.DIM,
                        "rows": rows,
                        "cols": min(F.DIM, shape.k - ki * F.DIM),
                        "load_id": 0,
                    },
                    pointer,
                )
        begin = self._function(prefix + "_begin", terminal_wait=False)

        self._reset([PTR, PTR, PTR, PTR, i64])
        bg, bu, cg, cu, column = self.fb.entry.args
        # The bridge validates column in [0,N-width] and aligned to width.
        # Convert the source element coordinate to the provider tile coordinate.
        n0 = self.fb.add(llvm.UDivOp(column, self.fb.const(F.DIM))).results[0]
        mr = tuple(min(F.DIM, shape.m - a * F.DIM) for a in range(shape.bm))
        nr = (F.DIM,) * shape.bn
        m0 = self.fb.const(0)
        for weight, output, base in (
            (bg, cg, 0),
            (bu, cu, self.resources.accumulator_rows_per_output),
        ):
            self.b, self.c = weight, output
            self._accumulator_offset = base
            self._reduce_output_block(m0, n0, mr, nr)
            for a, rows in enumerate(mr):
                for d, cols in enumerate(nr):
                    pointer = self._ptr(
                        output,
                        self.fb.const(a * F.DIM),
                        self.resources.columns,
                        self.fb.const(d * F.DIM),
                        4,
                    )
                    acc = base + (a * shape.bn + d) * F.DIM
                    self._rocc(
                        "mvout",
                        {
                            "local": isa.acc_addr(acc, full_row=True),
                            "rows": rows,
                            "cols": cols,
                        },
                        pointer,
                    )
        issue = self._function(prefix + "_issue", terminal_wait=False)
        self._reset([])
        wait = self._function(prefix + "_wait", terminal_wait=True)
        module = ModuleOp([begin, issue, wait])
        module.attributes["gemmini.dim"] = IntegerAttr(F.DIM, i64)
        module.attributes["gemmini.paired_issue_protocol"] = StringAttr(
            "exclusive begin; issue with disjoint borrowed destinations; wait before host read/reuse; terminal wait"
        )
        module.verify()
        return module
