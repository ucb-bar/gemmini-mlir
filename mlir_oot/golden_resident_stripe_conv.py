"""Primitive convolution with complete input and per-channel-block weight residency.

The padded activation is channel-tile-major.  One full 3x3 reduction for the
current output-channel block is cached alongside it, and accumulator capacity
partitions output rows into stripes.  Admission depends only on shape, numeric
policy and declared resources; no capture identity participates in selection.
"""
from dataclasses import asdict
import json

from xdsl.dialects import llvm
from xdsl.dialects.builtin import StringAttr

from .golden_gemm import GoldenGemm, Shape, _ceil_div, _groups
from .tables import isa, rtl_facts as F


class GoldenResidentStripeConv(GoldenGemm):
    def __init__(self, conv, *, stripe_rows=None):
        conv.validate()
        if conv.explicit_halo or conv.stride != 1 or conv.cin % F.DIM:
            raise ValueError('resident stripes need unpadded stride1 and aligned Cin')
        self.conv = conv
        self.plane = (conv.h + 2) * (conv.w + 2)
        self.kt = conv.cin // F.DIM
        self.xt = _ceil_div(conv.w, F.DIM)
        self.input_rows = self.kt * self.plane
        self.weight_rows = 9 * self.kt * conv.bn * F.DIM
        self.bbase = F.SPAD_ROWS - self.weight_rows
        if self.bbase < self.input_rows:
            raise ValueError('resident activation and full reduction weights exceed scratchpad')
        if self.plane >= 1 << 16:
            raise ValueError('resident channel-plane stride exceeds configuration field')
        capacity = F.ACC_ROWS // (self.xt * conv.bn * F.DIM)
        if stripe_rows is None:
            stripe_rows = min(conv.h, capacity)
        if type(stripe_rows) is not int or not 1 <= stripe_rows <= min(conv.h, capacity):
            raise ValueError('resident output stripe exceeds accumulator capacity')
        self.stripe_rows = stripe_rows
        super().__init__(Shape(conv.h * conv.w, conv.cout, conv.cin,
            bm=stripe_rows * self.xt, bn=conv.bn, output_dtype=conv.output_dtype,
            scale=conv.scale, relu=conv.relu, wide_store=True, reuse_b=True))

    def build(self):
        s = self.conv
        pw = s.w + 2
        self._emit_config()
        self._rocc('config_ld', {'stride':s.cin, 'block_stride':self.plane, 'load_id':0})
        zero = self.fb.add(llvm.IntToPtrOp(self.fb.const(0))).results[0]
        # The partitions are disjoint, cover every halo/activation lane, and
        # never issue a DMA larger than the declared DIM row limit.
        for ci in range(0, s.cin, 4 * F.DIM):
            cols = min(4 * F.DIM, s.cin - ci)
            base = ci // F.DIM * self.plane
            for y in range(s.h + 2):
                if y in (0, s.h + 1):
                    for x in range(0, pw, F.DIM):
                        self._rocc('mvin', {'local':base+y*pw+x,
                            'rows':min(F.DIM,pw-x), 'cols':cols, 'load_id':0}, zero)
                else:
                    self._rocc('mvin', {'local':base+y*pw, 'rows':1,
                        'cols':cols, 'load_id':0}, zero)
                    for x in range(0, s.w, F.DIM):
                        ptr = self._ptr(self.a, self.fb.const((y-1)*s.w+x),
                            s.cin, self.fb.const(ci))
                        self._rocc('mvin', {'local':base+y*pw+x+1,
                            'rows':min(F.DIM,s.w-x), 'cols':cols, 'load_id':0}, ptr)
                    self._rocc('mvin', {'local':base+y*pw+s.w+1, 'rows':1,
                        'cols':cols, 'load_id':0}, zero)

        def channel(n0, nr):
            for kh in range(3):
                for kw in range(3):
                    for ki in range(self.kt):
                        panel = ((kh*3+kw)*self.kt+ki) * s.bn * F.DIM
                        for d in range(0, len(nr), 4):
                            ptr = self._ptr(self.b, self.fb.const((kh*3+kw)*s.cin+ki*F.DIM),
                                s.cout, self._tile(n0,d))
                            self._rocc('mvin', {'local':self.bbase+panel+d*F.DIM,
                                'rows':F.DIM, 'cols':sum(nr[d:d+4]), 'load_id':1}, ptr)

            def stripe(y0, count, max_y0):
                for kh in range(3):
                    for kw in range(3):
                        for ki in range(self.kt):
                            panel = ((kh*3+kw)*self.kt+ki) * s.bn * F.DIM
                            first = kh == kw == ki == 0
                            for d, cols in enumerate(nr):
                                for ry in range(count):
                                    for a in range(self.xt):
                                        tile = ry*self.xt+a
                                        rows = min(F.DIM,s.w-a*F.DIM)
                                        offset = ki*self.plane+(ry+kh)*pw+a*F.DIM+kw
                                        address = self.fb.add_i(self.fb.mul_i(y0,self.fb.const(pw)),self.fb.const(offset))
                                        bound = max_y0*pw+offset
                                        if bound+rows > self.input_rows:
                                            raise ValueError('resident compute exceeds activation extent')
                                        self._rocc('preload', {'bd':self.bbase+panel+d*F.DIM if tile == 0 else isa.GARBAGE_ADDR,
                                            'c':isa.acc_addr((tile*s.bn+d)*F.DIM,accumulate=not first),
                                            'bd_cols':cols,'bd_rows':F.DIM,'c_cols':cols,'c_rows':rows})
                                        self._rocc('compute', {'a_cols':F.DIM,'a_rows':rows,
                                            'accumulate':tile != 0,'a_max':bound,
                                            'a_reserved_rows':self.input_rows},address)
                for ry in range(count):
                    for a in range(self.xt):
                        tile = ry*self.xt+a
                        rows = min(F.DIM,s.w-a*F.DIM)
                        pixel = self.fb.add_i(self.fb.mul_i(y0,self.fb.const(s.w)),self.fb.const(ry*s.w+a*F.DIM))
                        step = 4 if s.output_dtype == 'i8' else 1
                        for d in range(0,len(nr),step):
                            ptr = self._ptr(self.c,pixel,s.cout,self._tile(n0,d),4 if s.output_dtype == 'i32' else 1)
                            self._rocc('mvout', {'local':isa.acc_addr((tile*s.bn+d)*F.DIM,
                                full_row=s.output_dtype == 'i32'),'rows':rows,'cols':sum(nr[d:d+step])},ptr)

            full, tail = divmod(s.h, self.stripe_rows)
            self.fb.for_loop(0,full*self.stripe_rows,self.stripe_rows,
                lambda y:stripe(y,self.stripe_rows,(full-1)*self.stripe_rows))
            if tail:
                stripe(self.fb.const(full*self.stripe_rows),tail,full*self.stripe_rows)

        self._for_groups(_groups(s.cout,s.bn),channel)
        module = self._finish('gemmini_golden_resident_stripe_conv')
        module.attributes['gemmini.resident_stripe_conv_shape'] = StringAttr(json.dumps(asdict(s),sort_keys=True))
        module.attributes['gemmini.resident_stripe_rows'] = StringAttr(str(self.stripe_rows))
        module.attributes['gemmini.resident_stripe_layout'] = StringAttr('complete channel planes; full reduction weights per N block; bounded ACC row stripes')
        return module


def command_counts(conv, *, stripe_rows=None):
    g = GoldenResidentStripeConv(conv,stripe_rows=stripe_rows)
    nt = _ceil_div(conv.cout,F.DIM)
    bloads = sum(_ceil_div(min(conv.bn,nt-d),4) for d in range(0,nt,conv.bn))
    computes = 9*g.kt*nt*conv.h*g.xt
    return dict(mvin_a=(2*_ceil_div(conv.w+2,F.DIM)+conv.h*(2+g.xt))*_ceil_div(conv.cin,4*F.DIM),
        mvin_b=9*g.kt*bloads,compute=computes,preload=computes,
        mvout=conv.h*g.xt*(nt if conv.output_dtype == 'i32' else bloads),
        padded_array_issue_cycles=computes*F.DIM,activation_dram_bytes=conv.h*conv.w*conv.cin,
        weight_dram_bytes=9*conv.cin*conv.cout,input_plane_stride=g.plane,
        resident_input_rows=g.input_rows,resident_weight_rows=g.weight_rows,
        weight_base_row=g.bbase,stripe_rows=g.stripe_rows,host_im2col_bytes=0)
