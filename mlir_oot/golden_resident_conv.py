"""Primitive exact 3x3 convolution with one resident, virtually padded input.

A channel tile owns a complete padded spatial plane. CONFIG_LD block_stride
places wide input loads directly into this layout; shifted resident rows
supply all nine taps without repeated input DMA. Current family is stride1
with each output row fitting one DIM tile. Resource admission is explicit.
"""
from dataclasses import asdict
import json
from xdsl.dialects import llvm
from xdsl.dialects.builtin import StringAttr
from .golden_gemm import GoldenGemm, Shape, _ceil_div, _groups
from .tables import rtl_facts as F, isa


class GoldenResidentConv(GoldenGemm):
    def __init__(self, s):
        s.validate()
        if s.explicit_halo or s.stride != 1 or s.w + 2 > F.DIM or s.cin % F.DIM:
            raise ValueError('resident convolution needs unpadded stride1, width+halo<=DIM and aligned Cin')
        self.plane = (s.h+2)*(s.w+2)
        self.bbase = 2*F.SPAD_BANK_ROWS
        if _ceil_div(s.cin,F.DIM)*self.plane > self.bbase:
            raise ValueError('resident input overlaps weight banks')
        if self.bbase+s.bn*F.DIM > F.SPAD_ROWS or s.h*s.bn*F.DIM > F.ACC_ROWS:
            raise ValueError('resident convolution output/weight panel exceeds resources')
        self.conv=s
        super().__init__(Shape(s.h*s.w,s.cout,s.cin,bm=s.h,bn=s.bn,
            output_dtype=s.output_dtype,scale=s.scale,relu=s.relu,wide_store=True,reuse_b=True))

    def build(self):
        s=self.conv;pw=s.w+2
        self._emit_config()
        self._rocc('config_ld',{'stride':s.cin,'block_stride':self.plane,'load_id':0})
        zero=self.fb.add(llvm.IntToPtrOp(self.fb.const(0))).results[0]
        # These loads partition the scratch input exactly: no overwrite races.
        for ci in range(0,s.cin,4*F.DIM):
            cols=min(4*F.DIM,s.cin-ci);base=(ci//F.DIM)*self.plane
            for y in range(s.h+2):
                row=base+y*pw
                if y in (0,s.h+1):
                    self._rocc('mvin',{'local':row,'rows':pw,'cols':cols,'load_id':0},zero)
                else:
                    self._rocc('mvin',{'local':row,'rows':1,'cols':cols,'load_id':0},zero)
                    ptr=self._ptr(self.a,self.fb.const((y-1)*s.w),s.cin,self.fb.const(ci))
                    self._rocc('mvin',{'local':row+1,'rows':s.w,'cols':cols,'load_id':0},ptr)
                    self._rocc('mvin',{'local':row+s.w+1,'rows':1,'cols':cols,'load_id':0},zero)
        def channel(n0,nr):
            for kh in range(3):
                for kw in range(3):
                    for ci in range(0,s.cin,F.DIM):
                        for d in range(0,len(nr),4):
                            ptr=self._ptr(self.b,self.fb.const((kh*3+kw)*s.cin+ci),s.cout,self._tile(n0,d))
                            self._rocc('mvin',{'local':self.bbase+d*F.DIM,'rows':F.DIM,'cols':sum(nr[d:d+4]),'load_id':1},ptr)
                        for d,cols in enumerate(nr):
                            for y in range(s.h):
                                self._rocc('preload',{'bd':self.bbase+d*F.DIM if y==0 else isa.GARBAGE_ADDR,
                                    'c':isa.acc_addr((y*s.bn+d)*F.DIM,accumulate=not(kh==0 and kw==0 and ci==0)),
                                    'bd_cols':cols,'bd_rows':F.DIM,'c_cols':cols,'c_rows':s.w})
                                self._rocc('compute',{'a':(ci//F.DIM)*self.plane+(y+kh)*pw+kw,
                                    'a_cols':F.DIM,'a_rows':s.w,'accumulate':y!=0})
            for y in range(s.h):
                for d in range(0,len(nr),4 if s.output_dtype=='i8' else 1):
                    step=4 if s.output_dtype=='i8' else 1
                    ptr=self._ptr(self.c,self.fb.const(y*s.w),s.cout,self._tile(n0,d),4 if s.output_dtype=='i32' else 1)
                    self._rocc('mvout',{'local':isa.acc_addr((y*s.bn+d)*F.DIM,full_row=s.output_dtype=='i32'),
                        'rows':s.w,'cols':sum(nr[d:d+step])},ptr)
        self._for_groups(_groups(s.cout,s.bn),channel)
        module=self._finish('gemmini_golden_resident_conv')
        module.attributes['gemmini.resident_conv_shape']=StringAttr(json.dumps(asdict(s),sort_keys=True))
        module.attributes['gemmini.resident_conv_layout']=StringAttr('channel-tile-major padded spatial planes; disjoint input DMA partitions')
        return module


def command_counts(s):
    """Exact primitive counts and transferred activation bytes for this schedule."""
    g=GoldenResidentConv(s)
    kt,nt=_ceil_div(s.cin,F.DIM),_ceil_div(s.cout,F.DIM)
    bloads=sum(_ceil_div(min(s.bn,nt-d),4) for d in range(0,nt,s.bn))
    compute=9*kt*nt*s.h
    return dict(mvin_a=(2+3*s.h)*_ceil_div(s.cin,4*F.DIM),
        mvin_b=9*kt*bloads,compute=compute,preload=compute,
        mvout=s.h*(nt if s.output_dtype=='i32' else bloads),
        padded_array_issue_cycles=compute*F.DIM,
        activation_dram_bytes=s.h*s.w*s.cin,
        resident_input_rows=kt*g.plane,input_plane_stride=g.plane,
        accumulator_rows=s.h*s.bn*F.DIM,weight_base_row=g.bbase,
        host_im2col_bytes=0)
