"""Fill array M tiles across convolution rows within complete-row bands.

Explicit NHWC halo values and HWIO reduction order are preserved. DMA gathers
split at source rows and array tile boundaries. Each band retains its output accumulators. Optional wide A loads cache up
to four adjacent input-channel tiles, reducing DMA commands without host im2col.
Only primitive Gemmini commands and ordinary CPU loops are emitted.
"""
from dataclasses import asdict
import json
from xdsl.dialects import llvm
from xdsl.dialects.builtin import StringAttr
from .emission_options import FlatConvEmissionOptions
from .golden_gemm import GoldenGemm, Shape, _ceil_div, _groups
from .codegen.builder import PTR
from .readout_store_plan import PairedReadoutPlan
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


def eligible(s, band_rows=None, *, virtual_padding=False):
    rows = s.oh if band_rows is None else band_rows
    return ((s.explicit_halo or virtual_padding) and 0 < rows <= s.oh and
        _ceil_div(rows*s.ow, F.DIM)*s.bn*F.DIM <= F.ACC_ROWS)


def choose_band_rows(s, *, virtual_padding=False):
    """Minimize padded spatial tiles, then weight passes, within accumulator capacity."""
    s.validate()
    maximum = min(s.oh, F.ACC_ROWS // (s.bn*F.DIM) * F.DIM // s.ow)
    if not (s.explicit_halo or virtual_padding) or maximum < 1:
        raise ValueError('complete output row requires explicit halo and accumulator capacity')
    def cost(rows):
        full, tail = divmod(s.oh,rows)
        tiles = full*_ceil_div(rows*s.ow,F.DIM) + _ceil_div(tail*s.ow,F.DIM)
        return tiles, _ceil_div(s.oh,rows), -rows
    return min(range(1,maximum+1),key=cost)


def padding_segments(s, y, x, rows, kh, kw):
    """Partition a spatial run into proven in-bounds DMA and zero-fill lanes."""
    iy=y*s.stride+kh-1
    valid=[r for r in range(rows) if 0<=iy<s.h and 0<=(x+r)*s.stride+kw-1<s.w]
    if not valid:return [(0,rows,True)]
    first,last=valid[0],valid[-1]+1
    assert valid==list(range(first,last))
    return ([(0,first,True)] if first else [])+[(first,last-first,False)]+([(last,rows-last,True)] if last<rows else [])


def virtual_band_groups(s, rows):
    """CPU loop groups have identical vertical validity for every relative row/tap."""
    groups=[];signatures=[]
    for start in range(0,s.oh,rows):
        height=min(rows,s.oh-start)
        signature=tuple(0<=(start+y)*s.stride+kh-1<s.h for y in range(height) for kh in range(3))
        if groups and signatures[-1]==signature and groups[-1][2]==height:
            groups[-1]=(groups[-1][0],start+rows,height)
        else:groups.append((start,start+rows,height));signatures.append(signature)
    return groups


class GoldenFlatConv(GoldenGemm):
    emission_options_type = FlatConvEmissionOptions
    emission_shape_attribute = "conv"

    def __init__(self, s, *, wide_a=False, separate_b_bank=False, band_rows=None, virtual_padding=False, pingpong_b=False, loop_spatial=False, store_plan=None):
        s.validate()
        if type(loop_spatial) is not bool:
            raise ValueError("spatial command loop selection must be boolean")
        if store_plan is not None:
            if type(store_plan) is not PairedReadoutPlan:
                raise ValueError("typed paired readout plan required")
            store_plan.require_conv_producer(s)
        self.store_plan = store_plan
        self.loop_spatial = loop_spatial
        if virtual_padding and s.explicit_halo:raise ValueError("virtual padding requires unpadded input shape")
        if not eligible(s, band_rows,virtual_padding=virtual_padding):
            raise ValueError('spatial band requires explicit halo and accumulator capacity')
        if pingpong_b and (not separate_b_bank or not wide_a or s.bn*F.DIM>F.SPAD_BANK_ROWS):
            raise ValueError("B pingpong requires wide A and separate B bank capacity")
        self.pingpong_b = pingpong_b
        self.conv = s
        self.virtual_padding=virtual_padding
        self.band_rows = s.oh if band_rows is None else band_rows
        self.wide_a = wide_a
        self.separate_b_bank = separate_b_bank
        super().__init__(Shape(self.band_rows*s.ow, s.cout, s.cin,
            bm=_ceil_div(self.band_rows*s.ow, F.DIM), bn=s.bn,
            output_dtype=s.output_dtype, scale=s.scale, relu=s.relu,
            wide_store=True, reuse_b=True))
        self.second_output = self.fb.entry.insert_arg(PTR, 3) if store_plan is not None else None
        if (self.shape.bm * (4 if wide_a else 1) + s.bn) * F.DIM > F.SPAD_ROWS:
            raise ValueError('flat A panel and B channel block exceed scratchpad')

    def build(self):
        s = self.conv
        self._emit_config()
        self._rocc('config_ld', {'stride':s.stride*s.cin, 'load_id':0})
        zero=self.fb.add(llvm.IntToPtrOp(self.fb.const(0))).results[0] if self.virtual_padding else None
        def band(y0, band_height, sample_y=0):
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
                                segments=padding_segments(s,sample_y+y,x,rows,kh,kw) if self.virtual_padding else [(0,rows,False)]
                                for offset,count,is_zero in segments:
                                    ptr=zero
                                    if not is_zero:
                                        halo=0 if self.virtual_padding else 1
                                        iy = self.fb.add_i(self.fb.mul_i(y0,self.fb.const(s.stride)),self.fb.const(y*s.stride+kh-(1-halo)))
                                        pixel = self.fb.add_i(self.fb.mul_i(iy,self.fb.const(s.w+2*halo)),self.fb.const((x+offset)*s.stride+kw-(1-halo)))
                                        ptr = self._ptr(self.a, pixel, s.cin, ci)
                                    self._rocc('mvin', {'local':tile*panel_width+lane+offset,
                                        'rows':count, 'cols':channels, 'load_id':0}, ptr)
                            for ki in range(_ceil_div(channels, F.DIM)):
                                current_bbase=bbase+(ki%2)*F.SPAD_BANK_ROWS if self.pingpong_b else bbase
                                kr = min(F.DIM, channels-ki*F.DIM)
                                krow = self.fb.add_i(ci, self.fb.const((kh*3+kw)*s.cin+ki*F.DIM))
                                for d in range(0, len(nr), 4):
                                    ptr = self._ptr(self.b, krow, s.cout, self._tile(n0,d))
                                    self._rocc('mvin', {'local':current_bbase+d*F.DIM,
                                        'rows':kr, 'cols':sum(nr[d:d+4]), 'load_id':1}, ptr)
                                for d, cols in enumerate(nr):
                                    def spatial(a, rows, dynamic=False):
                                        initialize = not (first and ki==0)
                                        if dynamic:
                                            address = self.fb.add_i(self.fb.mul_i(a,self.fb.const(s.bn*F.DIM)),self.fb.const(d*F.DIM))
                                            self._rocc('preload', {
                                                'bd':isa.GARBAGE_ADDR, 'c_accumulate':int(initialize),
                                                'c_max':(len(widths)-1)*s.bn*F.DIM+d*F.DIM,
                                                'c_reserved_rows':len(widths)*s.bn*F.DIM,
                                                'bd_cols':cols, 'bd_rows':kr, 'c_cols':cols, 'c_rows':rows},address)
                                            address = self.fb.add_i(self.fb.mul_i(a,self.fb.const(panel_width)),self.fb.const(ki*F.DIM))
                                            self._rocc('compute', {'a_cols':kr, 'a_rows':rows, 'accumulate':True,
                                                'a_max':(len(widths)-1)*panel_width+ki*F.DIM,
                                                'a_reserved_rows':len(widths)*panel_width},address)
                                            return
                                        self._rocc('preload', {
                                            'bd':current_bbase+d*F.DIM if a==0 else isa.GARBAGE_ADDR,
                                            'c':isa.acc_addr((a*s.bn+d)*F.DIM, accumulate=initialize),
                                            'bd_cols':cols, 'bd_rows':kr, 'c_cols':cols, 'c_rows':rows})
                                        self._rocc('compute', {'a':a*panel_width+ki*F.DIM,
                                            'a_cols':kr, 'a_rows':rows, 'accumulate':a!=0})
                                    if self.loop_spatial:
                                        spatial(0,widths[0])
                                        full = sum(rows==F.DIM for rows in widths)
                                        if full>1:self.fb.for_loop(1,full,1,lambda a:spatial(a,F.DIM,True),retain_loop=True)
                                        if widths[-1] != F.DIM and len(widths)>1:
                                            spatial(len(widths)-1,widths[-1])
                                    else:
                                        for a, rows in enumerate(widths):spatial(a,rows)
                        panel(self.fb.const(0), min(panel_width,s.cin), kh==0 and kw==0)
                        full = s.cin // panel_width
                        if full > 1:
                            self.fb.for_loop(panel_width, full*panel_width, panel_width,
                                lambda ci: panel(ci,panel_width,False),retain_loop=self.loop_spatial)
                        if s.cin > panel_width and s.cin % panel_width:
                            panel(self.fb.const(full*panel_width), s.cin % panel_width, False)
                def store(output, dtype):
                    for a, rows in enumerate(widths):
                        step = 4 if dtype=='i8' else 1
                        for d in range(0, len(nr), step):
                            pixel = self.fb.add_i(self.fb.mul_i(y0,self.fb.const(s.ow)),self.fb.const(a*F.DIM))
                            ptr = self._ptr(output, pixel, s.cout,
                                self._tile(n0,d), 4 if dtype=='i32' else 1)
                            self._rocc('mvout', {'local':isa.acc_addr((a*s.bn+d)*F.DIM,
                                full_row=dtype=='i32'), 'rows':rows, 'cols':sum(nr[d:d+step])}, ptr)
                if self.store_plan is None:
                    store(self.c, s.output_dtype)
                else:
                    # Both passes consume the same completed accumulator block;
                    # no compute/load modifies it between these ordered stores.
                    for output, scale in zip((self.c, self.second_output), self.store_plan.store_scales):
                        self._rocc('config_st', {'stride':s.cout, 'acc_scale':scale,
                            'acc_act':isa.RELU if self.store_plan.relu else isa.NO_ACTIVATION})
                        store(output, 'i8')
            self._for_groups(_groups(s.cout,s.bn), channel,retain_loop=self.loop_spatial)
        if self.virtual_padding:
            for start,stop,height in virtual_band_groups(s,self.band_rows):
                if stop-start==self.band_rows:band(self.fb.const(start),height,start)
                else:self.fb.for_loop(start,stop,self.band_rows,lambda y0,h=height,sample=start:band(y0,h,sample),retain_loop=self.loop_spatial)
        else:
            full, tail = divmod(s.oh,self.band_rows)
            if full == 1:
                band(self.fb.const(0),self.band_rows)
            else:
                self.fb.for_loop(0,full*self.band_rows,self.band_rows,
                    lambda y0: band(y0,self.band_rows),retain_loop=self.loop_spatial)
            if tail:
                band(self.fb.const(full*self.band_rows),tail)
        module = self._finish('gemmini_golden_flat_conv')
        if self.virtual_padding:module.attributes['gemmini.virtual_padding']=StringAttr('static-zero-pad1-bounded-DMA')
        module.attributes['gemmini.flat_conv_band_rows'] = StringAttr(str(self.band_rows))
        module.attributes['gemmini.flat_conv_shape'] = StringAttr(json.dumps(asdict(s),sort_keys=True))
        module.attributes['gemmini.flat_conv_wide_a'] = StringAttr(str(self.wide_a))
        if self.pingpong_b:module.attributes['gemmini.flat_conv_pingpong_b']=StringAttr('two-independent-banks')
        if self.loop_spatial:module.attributes['gemmini.flat_conv_loop_spatial']=StringAttr('exact bounded ordinary CPU spatial command loop')
        module.attributes['gemmini.flat_conv_separate_b_bank'] = StringAttr(str(self.separate_b_bank))
        if self.store_plan is not None:
            module.attributes['gemmini.paired_readout'] = StringAttr(json.dumps(
                dict(certificate=self.store_plan.certificate(),
                     abi='A:i8*,B:i8*,first:i8*,second:i8*',
                     storage='Both outputs have OH*OW*Cout bytes, are disjoint from each other and inputs, and remain stable until the caller decoder finishes',
                     lifetime='Both ordered store passes precede every subsequent accumulator overwrite; final fence completes both outputs'), sort_keys=True))
        return module


def command_counts(s, *, wide_a=False, band_rows=None,virtual_padding=False):
    s.validate()
    if not eligible(s,band_rows,virtual_padding=virtual_padding):
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
    aloads=9*runs
    if virtual_padding:
        aloads=sum(len(padding_segments(s,start+y,x,n,kh,kw))
            for start in range(0,s.oh,rows)
            for _,_,y,x,n in spatial_runs(s,min(rows,s.oh-start))
            for kh in range(3) for kw in range(3))
    return dict(compute=compute, preload=compute,
        mvin_a=aloads*_ceil_div(s.cin,64 if wide_a else 16)*groups,
        mvin_b=9*kt*bloads*len(bands), mvout=mt*(nt if s.output_dtype=='i32' else bloads),
        padded_array_issue_cycles=compute*F.DIM, host_im2col_bytes=0,
        weight_bytes=9*s.cin*s.cout*len(bands), spatial_tiles=mt,
        gather_fragments=runs, band_rows=rows, bands=len(bands))
