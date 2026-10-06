"""Complete-A/full-K-B residency with accumulator-bounded output stripes."""

from dataclasses import replace

from xdsl.dialects.builtin import IntegerAttr, i64

from .golden_gemm import GoldenGemm, _ceil_div, _groups
from .tables import isa
from .tables import rtl_facts as F


class GoldenResidentStripeGemm(GoldenGemm):
    def __init__(self, shape, *, stripe_tiles=None):
        shape.validate()
        if shape.bias:
            raise ValueError("resident stripes require a zero accumulator seed")
        self.plane = _ceil_div(shape.m, F.DIM) * F.DIM
        self.kt = _ceil_div(shape.k, F.DIM)
        self.input_rows = self.plane * self.kt
        bn = min(shape.bn, _ceil_div(shape.n, F.DIM))
        self.weight_rows = self.kt * bn * F.DIM
        self.bbase = F.SPAD_ROWS - self.weight_rows
        if self.input_rows > self.bbase:
            raise ValueError("complete A and full reduction B exceed scratchpad")
        capacity = F.ACC_ROWS // (bn * F.DIM)
        if stripe_tiles is None:
            stripe_tiles = min(_ceil_div(shape.m, F.DIM), capacity)
        if type(stripe_tiles) is not int or not 1 <= stripe_tiles <= capacity:
            raise ValueError("output stripe exceeds accumulator capacity")
        self.stripe_tiles = stripe_tiles
        super().__init__(
            replace(
                shape,
                bm=stripe_tiles,
                bn=bn,
                cache_a=False,
                cache_b=False,
                pipeline_m=False,
                prefetch_m=False,
                banked_m=False,
                wide_a=False,
                wide_b=True,
                separate_b_bank=False,
                prefetch_b=False,
                reuse_b=True,
                wide_store=shape.output_dtype == "i8",
            )
        )

    def _emit_work(self):
        s = self.shape
        self._rocc(
            "config_ld", {"stride": s.k, "block_stride": self.plane, "load_id": 0}
        )
        for ki in range(0, self.kt, 4):
            cols = min(4 * F.DIM, s.k - ki * F.DIM)
            for row in range(0, s.m, F.DIM):
                ptr = self._ptr(
                    self.a, self.fb.const(row), s.k, self.fb.const(ki * F.DIM)
                )
                self._rocc(
                    "mvin",
                    {
                        "local": ki * self.plane + row,
                        "rows": min(F.DIM, s.m - row),
                        "cols": cols,
                        "load_id": 0,
                    },
                    ptr,
                )

        def channel(n0, nr):
            for ki in range(self.kt):
                self._load_b_panel(
                    n0,
                    self.fb.const(ki),
                    nr,
                    min(F.DIM, s.k - ki * F.DIM),
                    self.bbase // F.DIM + ki * s.bn,
                )

            def stripe(m0, mr, max_tile):
                for ki in range(self.kt):
                    kr = min(F.DIM, s.k - ki * F.DIM)
                    for d, cols in enumerate(nr):
                        for a, rows in enumerate(mr):
                            address = self.fb.add_i(
                                self._tile(m0, a), self.fb.const(ki * self.plane)
                            )
                            bound = (max_tile + a) * F.DIM + ki * self.plane
                            if bound + rows > self.input_rows:
                                raise ValueError(
                                    "resident A compute exceeds reserved extent"
                                )
                            self._rocc(
                                "preload",
                                {
                                    "bd": self.bbase + (ki * s.bn + d) * F.DIM
                                    if a == 0
                                    else isa.GARBAGE_ADDR,
                                    "c": isa.acc_addr(
                                        (a * s.bn + d) * F.DIM, accumulate=ki != 0
                                    ),
                                    "bd_rows": kr,
                                    "bd_cols": cols,
                                    "c_rows": rows,
                                    "c_cols": cols,
                                },
                            )
                            self._rocc(
                                "compute",
                                {
                                    "a_cols": kr,
                                    "a_rows": rows,
                                    "accumulate": a != 0,
                                    "a_max": bound,
                                    "a_reserved_rows": self.input_rows,
                                },
                                address,
                            )
                step = 4 if s.output_dtype == "i8" else 1
                for a, rows in enumerate(mr):
                    for d in range(0, len(nr), step):
                        ptr = self._ptr(
                            self.c,
                            self._tile(m0, a),
                            s.n,
                            self._tile(n0, d),
                            4 if s.output_dtype == "i32" else 1,
                        )
                        self._rocc(
                            "mvout",
                            {
                                "local": isa.acc_addr(
                                    (a * s.bn + d) * F.DIM,
                                    full_row=s.output_dtype == "i32",
                                ),
                                "rows": rows,
                                "cols": sum(nr[d : d + step]),
                            },
                            ptr,
                        )

            for start, stop, step, widths in _groups(s.m, s.bm):
                if stop - start == step:
                    stripe(self.fb.const(start), widths, start)
                else:
                    self.fb.for_loop(
                        start,
                        stop,
                        step,
                        lambda m, w=widths, bound=stop - step: stripe(m, w, bound),
                    )

        self._for_groups(_groups(s.n, s.bn), channel)

    def build(self):
        self._emit_config()
        self._emit_work()
        module = self._finish("gemmini_golden_gemm")
        module.attributes["gemmini.resident_stripe_input_rows"] = IntegerAttr(
            self.input_rows, i64
        )
        module.attributes["gemmini.resident_stripe_weight_rows"] = IntegerAttr(
            self.weight_rows, i64
        )
        return module


def command_counts(shape, *, stripe_tiles=None):
    g = GoldenResidentStripeGemm(shape, stripe_tiles=stripe_tiles)
    s = g.shape
    mt, nt = _ceil_div(s.m, F.DIM), _ceil_div(s.n, F.DIM)
    ng = sum(_ceil_div(min(s.bn, nt - d), 4) for d in range(0, nt, s.bn))
    return {
        "mvin_a": mt * _ceil_div(g.kt, 4),
        "mvin_b": g.kt * ng,
        "compute": mt * nt * g.kt,
        "preload": mt * nt * g.kt,
        "mvout": mt * (nt if s.output_dtype == "i32" else ng),
        "requested_activation_load_bytes": s.m * s.k,
        "requested_weight_load_bytes": s.k * s.n,
        "resident_input_rows": g.input_rows,
        "resident_weight_rows": g.weight_rows,
        "stripe_tiles": g.stripe_tiles,
    }
