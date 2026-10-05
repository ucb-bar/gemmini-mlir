"""Fill array M tiles across convolution rows within complete-row bands.

Explicit NHWC halo values and HWIO reduction order are preserved. DMA gathers
split at source rows and array tile boundaries. Each band retains its output accumulators. Optional wide A loads cache up
to four adjacent input-channel tiles, reducing DMA commands without host im2col.
Only primitive Gemmini commands and ordinary CPU loops are emitted.
"""
from dataclasses import asdict
import json
from xdsl.dialects.builtin import StringAttr
from .golden_gemm import GoldenGemm, Shape, _ceil_div, _groups
from .tables import rtl_facts as F, isa


def spatial_runs(s, rows=None):
    """(tile, lane, y, x, count), covering each output pixel exactly once."""
    rows = s.oh if rows is None else rows
    runs = []
    for tile, start in enumerate(range(0, rows * s.ow, F.DIM)):
        remaining = min(F.DIM, rows * s.ow - start)
        lane = 0
        while remaining:
            y, x = divmod(start + lane, s.ow)
            count = min(remaining, s.ow - x)
            runs.append((tile, lane, y, x, count))
            lane += count
            remaining -= count
    return runs


def eligible(s, band_rows=None):
    rows = s.oh if band_rows is None else band_rows
    return (s.explicit_halo and 0 < rows <= s.oh and
        _ceil_div(rows*s.ow, F.DIM)*s.bn*F.DIM <= F.ACC_ROWS)


def choose_band_rows(s):
    """Minimize padded spatial tiles, then weight passes, within accumulator capacity."""
    s.validate()
    maximum = min(s.oh, F.ACC_ROWS // (s.bn*F.DIM) * F.DIM // s.ow)
    if not s.explicit_halo or maximum < 1:
        raise ValueError('complete output row requires explicit halo and accumulator capacity')
    def cost(rows):
        full, tail = divmod(s.oh,rows)
        tiles = full*_ceil_div(rows*s.ow,F.DIM) + _ceil_div(tail*s.ow,F.DIM)
        return tiles, _ceil_div(s.oh,rows), -rows
    return min(range(1,maximum+1),key=cost)


class GoldenFlatConv(GoldenGemm):
    def __init__(self, s, *, wide_a=False, separate_b_bank=False, band_rows=None):
        s.validate()
        if not eligible(s, band_rows):
            raise ValueError('spatial band requires explicit halo and accumulator capacity')
        self.conv = s
        self.band_rows = s.oh if band_rows is None else band_rows
        self.wide_a = wide_a
        self.separate_b_bank = separate_b_bank
        super().__init__(Shape(self.band_rows*s.ow, s.cout, s.cin,
            bm=_ceil_div(self.band_rows*s.ow, F.DIM), bn=s.bn,
            output_dtype=s.output_dtype, scale=s.scale, relu=s.relu,
            wide_store=True, reuse_b=True))
        if (self.shape.bm * (4 if wide_a else 1) + s.bn) * F.DIM > F.SPAD_ROWS:
            raise ValueError('flat A panel and B channel block exceed scratchpad')

    def build(self):
        s = self.conv
        self._emit_config()
        self._rocc('config_ld', {'stride':s.stride*s.cin, 'load_id':0})
        def band(y0, band_height):
            widths = tuple(min(F.DIM, band_height*s.ow-start) for start in range(0, band_height*s.ow, F.DIM))
            runs = spatial_runs(s,band_height)
            panel_tiles = 4 if self.wide_a else 1
            panel_width = panel_tiles * F.DIM
            bbase = F.SPAD_ROWS // 2 if self.separate_b_bank else len(widths) * panel_width
            if len(widths)*panel_width > bbase or bbase+s.bn*F.DIM > F.SPAD_ROWS:
                raise ValueError("A and B scratchpad ranges overlap or exceed capacity")

            def channel(n0, nr):
                for kh in range(3):
                    for kw in range(3):
                        def panel(ci, channels, first):
                            for tile, lane, y, x, rows in runs:
                                iy = self.fb.add_i(self.fb.mul_i(y0,self.fb.const(s.stride)),self.fb.const(y*s.stride+kh))
                                pixel = self.fb.add_i(self.fb.mul_i(iy,self.fb.const(s.w+2)),self.fb.const(x*s.stride+kw))
                                ptr = self._ptr(self.a, pixel, s.cin, ci)
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
                        pixel = self.fb.add_i(self.fb.mul_i(y0,self.fb.const(s.ow)),self.fb.const(a*F.DIM))
                        ptr = self._ptr(self.c, pixel, s.cout,
                            self._tile(n0,d), 4 if s.output_dtype=='i32' else 1)
                        self._rocc('mvout', {'local':isa.acc_addr((a*s.bn+d)*F.DIM,
                            full_row=s.output_dtype=='i32'), 'rows':rows, 'cols':sum(nr[d:d+step])}, ptr)
            self._for_groups(_groups(s.cout,s.bn), channel)
        full, tail = divmod(s.oh,self.band_rows)
        if full == 1:
            band(self.fb.const(0),self.band_rows)
        else:
            self.fb.for_loop(0,full*self.band_rows,self.band_rows,
                lambda y0: band(y0,self.band_rows))
        if tail:
            band(self.fb.const(full*self.band_rows),tail)
        module = self._finish('gemmini_golden_flat_conv')
        module.attributes['gemmini.flat_conv_band_rows'] = StringAttr(str(self.band_rows))
        module.attributes['gemmini.flat_conv_shape'] = StringAttr(json.dumps(asdict(s),sort_keys=True))
        module.attributes['gemmini.flat_conv_wide_a'] = StringAttr(str(self.wide_a))
        module.attributes['gemmini.flat_conv_separate_b_bank'] = StringAttr(str(self.separate_b_bank))
        return module


def command_counts(s, *, wide_a=False, band_rows=None):
    s.validate()
    if not eligible(s,band_rows):
        raise ValueError('spatial band exceeds capacity or lacks explicit halo')
    rows = s.oh if band_rows is None else band_rows
    full, tail = divmod(s.oh,rows)
    bands = [rows]*full + ([tail] if tail else [])
    mt = sum(_ceil_div(r*s.ow,F.DIM) for r in bands)
    runs = sum(len(spatial_runs(s,r)) for r in bands)
    nt, kt = _ceil_div(s.cout,F.DIM), _ceil_div(s.cin,F.DIM)
    groups = _ceil_div(nt,s.bn)
    bloads = sum(_ceil_div(min(s.bn,nt-d),4) for d in range(0,nt,s.bn))
    compute = 9*kt*nt*mt
    return dict(compute=compute, preload=compute,
        mvin_a=9*_ceil_div(s.cin,64 if wide_a else 16)*runs*groups,
        mvin_b=9*kt*bloads*len(bands), mvout=mt*(nt if s.output_dtype=='i32' else bloads),
        padded_array_issue_cycles=compute*F.DIM, host_im2col_bytes=0,
        weight_bytes=9*s.cin*s.cout*len(bands), spatial_tiles=mt,
        gather_fragments=runs, band_rows=rows, bands=len(bands))
