"""Fill array M tiles across convolution rows, retaining all spatial outputs.

Explicit NHWC halo values and HWIO reduction order are preserved. DMA gathers
split at source rows and array tile boundaries. Optional wide A loads cache up
to four adjacent input-channel tiles, reducing DMA commands without host im2col.
Only primitive Gemmini commands and ordinary CPU loops are emitted.
"""
from dataclasses import asdict
import json
from xdsl.dialects.builtin import StringAttr
from .golden_gemm import GoldenGemm, Shape, _ceil_div, _groups
from .tables import rtl_facts as F, isa


def spatial_runs(s):
    """(tile, lane, y, x, count), covering each output pixel exactly once."""
    runs = []
    for tile, start in enumerate(range(0, s.oh * s.ow, F.DIM)):
        remaining = min(F.DIM, s.oh * s.ow - start)
        lane = 0
        while remaining:
            y, x = divmod(start + lane, s.ow)
            count = min(remaining, s.ow - x)
            runs.append((tile, lane, y, x, count))
            lane += count
            remaining -= count
    return runs


def eligible(s):
    return s.explicit_halo and _ceil_div(s.oh*s.ow, F.DIM)*s.bn*F.DIM <= F.ACC_ROWS


class GoldenFlatConv(GoldenGemm):
    def __init__(self, s, *, wide_a=False):
        s.validate()
        if not eligible(s):
            raise ValueError('flat spatial image requires explicit halo and accumulator capacity')
        self.conv = s
        self.wide_a = wide_a
        super().__init__(Shape(s.oh*s.ow, s.cout, s.cin,
            bm=_ceil_div(s.oh*s.ow, F.DIM), bn=s.bn,
            output_dtype=s.output_dtype, scale=s.scale, relu=s.relu,
            wide_store=True, reuse_b=True))
        if (self.shape.bm * (4 if wide_a else 1) + s.bn) * F.DIM > F.SPAD_ROWS:
            raise ValueError('flat A panel and B channel block exceed scratchpad')

    def build(self):
        s = self.conv
        self._emit_config()
        self._rocc('config_ld', {'stride':s.stride*s.cin, 'load_id':0})
        widths = tuple(min(F.DIM, s.oh*s.ow-start) for start in range(0, s.oh*s.ow, F.DIM))
        runs = spatial_runs(s)
        panel_tiles = 4 if self.wide_a else 1
        panel_width = panel_tiles * F.DIM
        bbase = len(widths) * panel_width

        def channel(n0, nr):
            for kh in range(3):
                for kw in range(3):
                    def panel(ci, channels, first):
                        for tile, lane, y, x, rows in runs:
                            pixel = (y*s.stride+kh)*(s.w+2) + x*s.stride+kw
                            ptr = self._ptr(self.a, self.fb.const(pixel), s.cin, ci)
                            self._rocc('mvin', {'local':tile*panel_width+lane,
                                'rows':rows, 'cols':channels, 'load_id':0}, ptr)
                        for ki in range(_ceil_div(channels, F.DIM)):
                            kr = min(F.DIM, channels-ki*F.DIM)
                            krow = self.fb.add_i(ci, self.fb.const((kh*3+kw)*s.cin+ki*F.DIM))
                            for d in range(0, len(nr), 4):
                                ptr = self._ptr(self.b, krow, s.cout, self._tile(n0,d))
                                self._rocc('mvin', {'local':bbase+d*F.DIM,
                                    'rows':kr, 'cols':sum(nr[d:d+4]), 'load_id':1}, ptr)
                            for d, cols in enumerate(nr):
                                for a, rows in enumerate(widths):
                                    self._rocc('preload', {
                                        'bd':bbase+d*F.DIM if a==0 else isa.GARBAGE_ADDR,
                                        'c':isa.acc_addr((a*s.bn+d)*F.DIM, accumulate=not (first and ki==0)),
                                        'bd_cols':cols, 'bd_rows':kr, 'c_cols':cols, 'c_rows':rows})
                                    self._rocc('compute', {'a':a*panel_width+ki*F.DIM,
                                        'a_cols':kr, 'a_rows':rows, 'accumulate':a!=0})
                    panel(self.fb.const(0), min(panel_width,s.cin), kh==0 and kw==0)
                    full = s.cin // panel_width
                    if full > 1:
                        self.fb.for_loop(panel_width, full*panel_width, panel_width,
                            lambda ci: panel(ci,panel_width,False))
                    if s.cin > panel_width and s.cin % panel_width:
                        panel(self.fb.const(full*panel_width), s.cin % panel_width, False)
            for a, rows in enumerate(widths):
                step = 4 if s.output_dtype=='i8' else 1
                for d in range(0, len(nr), step):
                    ptr = self._ptr(self.c, self.fb.const(a*F.DIM), s.cout,
                        self._tile(n0,d), 4 if s.output_dtype=='i32' else 1)
                    self._rocc('mvout', {'local':isa.acc_addr((a*s.bn+d)*F.DIM,
                        full_row=s.output_dtype=='i32'), 'rows':rows, 'cols':sum(nr[d:d+step])}, ptr)
        self._for_groups(_groups(s.cout,s.bn), channel)
        module = self._finish('gemmini_golden_flat_conv')
        module.attributes['gemmini.flat_conv_shape'] = StringAttr(json.dumps(asdict(s),sort_keys=True))
        module.attributes['gemmini.flat_conv_wide_a'] = StringAttr(str(self.wide_a))
        return module


def command_counts(s, *, wide_a=False):
    s.validate()
    if not eligible(s):
        raise ValueError('flat spatial schedule exceeds capacity or lacks explicit halo')
    mt, nt, kt = (_ceil_div(s.oh*s.ow,F.DIM), _ceil_div(s.cout,F.DIM), _ceil_div(s.cin,F.DIM))
    groups = _ceil_div(nt,s.bn)
    bloads = sum(_ceil_div(min(s.bn,nt-d),4) for d in range(0,nt,s.bn))
    compute = 9*kt*nt*mt
    return dict(compute=compute, preload=compute,
        mvin_a=9*_ceil_div(s.cin,64 if wide_a else 16)*len(spatial_runs(s))*groups,
        mvin_b=9*kt*bloads, mvout=mt*(nt if s.output_dtype=='i32' else bloads),
        padded_array_issue_cycles=compute*F.DIM, host_im2col_bytes=0,
        weight_bytes=9*s.cin*s.cout, spatial_tiles=mt, gather_fragments=len(spatial_runs(s)))
