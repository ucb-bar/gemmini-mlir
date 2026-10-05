"""Direct NHWC/HWIO 3x3 convolution; CPU loops and primitive Gemmini only.

A whole output row is tiled across the array. Each tap gathers strided NHWC
pixels directly into scratchpad; only boundary lanes are zero-filled. No host
im2col allocation or copying is performed. B stays stationary across row tiles.
"""
from dataclasses import asdict, dataclass
import json
from xdsl.dialects import llvm
from xdsl.dialects.builtin import StringAttr
from .golden_gemm import GoldenGemm, Shape, _groups, _ceil_div
from .tables import rtl_facts as F, isa
from .codegen.builder import PTR


@dataclass(frozen=True)
class ConvShape:
    h: int
    w: int
    cin: int
    cout: int
    stride: int = 1
    bn: int = 4
    output_dtype: str = "i32"
    scale: float = 1.0
    relu: bool = False
    wide_b: bool = False

    @property
    def oh(self):
        return (self.h - 1) // self.stride + 1

    @property
    def ow(self):
        return (self.w - 1) // self.stride + 1

    def validate(self):
        if min(self.h, self.w, self.cin, self.cout, self.bn) <= 0:
            raise ValueError("positive convolution extents required")
        if self.stride not in (1, 2):
            raise ValueError("only stride 1 or 2 supported")
        if _ceil_div(self.ow, F.DIM) * self.bn * F.DIM > F.ACC_ROWS:
            raise ValueError("whole output row exceeds accumulator capacity")


class GoldenConv(GoldenGemm):
    def __init__(self, conv: ConvShape):
        conv.validate()
        self.conv = conv
        super().__init__(Shape(conv.ow, conv.cout, conv.cin,
            bm=_ceil_div(conv.ow, F.DIM), bn=conv.bn,
            output_dtype=conv.output_dtype, scale=conv.scale,
            relu=conv.relu, wide_store=True, reuse_b=True))

    def build(self):
        s = self.conv
        self._emit_config()
        self._rocc("config_ld", {"stride": s.stride * s.cin, "load_id": 0})
        zero = self.fb.add(llvm.IntToPtrOp(self.fb.const(0))).results[0]
        widths = tuple(min(F.DIM, s.ow - x) for x in range(0, s.ow, F.DIM))
        b_base = len(widths) * F.DIM

        def row(y, taps):
            def channel(n0, nr):
                for kh in taps:
                    iy = self.fb.add_i(self.fb.mul_i(y, self.fb.const(s.stride)), self.fb.const(kh - 1))
                    for kw in range(3):
                        def reduction(ci, kr, first):
                            for a, rows in enumerate(widths):
                                x = a * F.DIM
                                valid = [r for r in range(rows) if 0 <= (x+r)*s.stride+kw-1 < s.w]
                                if len(valid) != rows:
                                    self._rocc("mvin", {"local": a*F.DIM, "rows": rows, "cols": kr, "load_id": 0}, zero)
                                if valid:
                                    left, count = valid[0], len(valid)
                                    pixel = self.fb.add_i(self.fb.mul_i(iy, self.fb.const(s.w)), self.fb.const((x+left)*s.stride+kw-1))
                                    ptr = self._ptr(self.a, pixel, s.cin, ci)
                                    self._rocc("mvin", {"local": a*F.DIM+left, "rows": count, "cols": kr, "load_id": 0}, ptr)
                            krow = self.fb.add_i(ci, self.fb.const((kh*3+kw)*s.cin))
                            if s.wide_b:
                                for d in range(0,len(nr),4):
                                    ptr = self._ptr(self.b,krow,s.cout,self._tile(n0,d))
                                    self._rocc("mvin", {"local":b_base+d*F.DIM,"rows":kr,"cols":sum(nr[d:d+4]),"load_id":1},ptr)
                            for d, cols in enumerate(nr):
                                if not s.wide_b:
                                    ptr = self._ptr(self.b, krow, s.cout, self._tile(n0,d))
                                    self._rocc("mvin", {"local": b_base+d*F.DIM, "rows": kr, "cols": cols, "load_id": 1}, ptr)
                                for a, rows in enumerate(widths):
                                    self._rocc("preload", {"bd": b_base+d*F.DIM if a == 0 else isa.GARBAGE_ADDR,
                                        "c": isa.acc_addr((a*s.bn+d)*F.DIM, accumulate=not first),
                                        "bd_cols": cols, "bd_rows": kr, "c_cols": cols, "c_rows": rows})
                                    self._rocc("compute", {"a": a*F.DIM, "a_cols": kr, "a_rows": rows, "accumulate": a != 0})
                        reduction(self.fb.const(0), min(F.DIM,s.cin), kh == taps[0] and kw == 0)
                        if s.cin // F.DIM > 1:
                            self.fb.for_loop(F.DIM, s.cin//F.DIM*F.DIM, F.DIM,
                                lambda ci: reduction(ci,F.DIM,False))
                        if s.cin > F.DIM and s.cin % F.DIM:
                            reduction(self.fb.const(s.cin//F.DIM*F.DIM),s.cin%F.DIM,False)
                for a, rows in enumerate(widths):
                    pixel = self.fb.add_i(self.fb.mul_i(y,self.fb.const(s.ow)),self.fb.const(a*F.DIM))
                    store_tiles = 4 if s.output_dtype == "i8" else 1
                    for d in range(0,len(nr),store_tiles):
                        cols = sum(nr[d:d+store_tiles])
                        ptr = self._ptr(self.c,pixel,s.cout,self._tile(n0,d),4 if s.output_dtype == "i32" else 1)
                        self._rocc("mvout", {"local":isa.acc_addr((a*s.bn+d)*F.DIM,full_row=s.output_dtype == "i32"),"rows":rows,"cols":cols},ptr)
            self._for_groups(_groups(s.cout,s.bn), channel)
        # Group rows with identical padding; interior rows share one CPU loop.
        groups = []
        for y in range(s.oh):
            taps = tuple(kh for kh in range(3) if 0 <= y*s.stride+kh-1 < s.h)
            if groups and groups[-1][2] == taps:
                groups[-1] = (groups[-1][0],y+1,taps)
            else:
                groups.append((y,y+1,taps))
        for start, stop, taps in groups:
            self.fb.for_loop(start,stop,1,lambda y,t=taps:row(y,t))
        module = self._finish("gemmini_golden_conv")
        module.attributes["gemmini.conv_shape"] = StringAttr(json.dumps(asdict(s),sort_keys=True))
        module.attributes["gemmini.conv_layout"] = StringAttr("NHWC,HWIO,NHWC;3x3;pad1")
        return module


def command_counts(s: ConvShape) -> dict[str, int]:
    """Exact dynamic primitive counts for this schedule (no throughput estimate)."""
    s.validate()
    result = dict(config=5, fence=2, flush=1, mvin_a=0, mvin_b=0,
                  preload=0, compute=0, mvout=0, host_im2col_bytes=0)
    widths = tuple(min(F.DIM,s.ow-x) for x in range(0,s.ow,F.DIM))
    kt = _ceil_div(s.cin,F.DIM)
    nt = _ceil_div(s.cout,F.DIM)
    blocks = _ceil_div(nt,s.bn)
    for y in range(s.oh):
        taps = sum(0 <= y*s.stride+kh-1 < s.h for kh in range(3))
        result['mvin_b'] += taps*3*kt*(sum(_ceil_div(min(s.bn,nt-d),4) for d in range(0,nt,s.bn)) if s.wide_b else nt)
        result['compute'] += taps*3*kt*nt*len(widths)
        result['mvout'] += (sum(_ceil_div(min(s.bn,nt-d),4) for d in range(0,nt,s.bn)) if s.output_dtype == 'i8' else nt)*len(widths)
        for kw in range(3):
            for a, rows in enumerate(widths):
                valid = sum(0 <= (a*F.DIM+r)*s.stride+kw-1 < s.w for r in range(rows))
                result['mvin_a'] += taps*kt*blocks*((valid != rows)+(valid > 0))
    result['preload'] = result['compute']
    result['padded_array_issue_cycles'] = result['compute']*F.DIM
    return result
