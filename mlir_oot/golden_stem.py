"""Primitive packed-K 7x7 stride-2 stem on explicit NHWC halo.

Each input row supplies contiguous kw*Cin=21 bytes through overlapping stride-6
DMA. A 16+5 reduction split uses original HWIO weights without host packing or
reads beyond the seven-pixel window. No host im2col or hardware loop commands.
"""
from dataclasses import dataclass,asdict
import json
from xdsl.dialects.builtin import StringAttr
from .golden_gemm import GoldenGemm,Shape,_ceil_div,_groups
from .tables import rtl_facts as F,isa

@dataclass(frozen=True)
class StemShape:
    h:int
    w:int
    cout:int=64
    bn:int=4
    output_dtype:str='i32'
    scale:float=1.0
    relu:bool=False
    @property
    def oh(self):return (self.h+1)//2
    @property
    def ow(self):return (self.w+1)//2
    def validate(self):
        if min(self.h,self.w,self.cout,self.bn)<=0:raise ValueError('positive stem extents required')
        if _ceil_div(self.ow,F.DIM)*self.bn*F.DIM>F.ACC_ROWS:raise ValueError('stem row exceeds accumulator capacity')
        if self.output_dtype not in ('i32','i8'):raise ValueError('stem output must be i32 or i8')

class GoldenStem(GoldenGemm):
    def __init__(self,stem):
        stem.validate();self.stem=stem
        super().__init__(Shape(stem.ow,stem.cout,21,bm=_ceil_div(stem.ow,F.DIM),bn=stem.bn,output_dtype=stem.output_dtype,scale=stem.scale,relu=stem.relu,wide_store=True,reuse_b=True))
    def build(self):
        s=self.stem;self._emit_config();self._rocc('config_ld',{'stride':6,'load_id':0})
        widths=tuple(min(F.DIM,s.ow-x) for x in range(0,s.ow,F.DIM));bbase=len(widths)*2*F.DIM
        def row(y):
            def channel(n0,nr):
                for kh in range(7):
                    iy=self.fb.add_i(self.fb.mul_i(y,self.fb.const(2)),self.fb.const(kh))
                    for a,rows in enumerate(widths):
                        pixel=self.fb.add_i(self.fb.mul_i(iy,self.fb.const(s.w+6)),self.fb.const(a*F.DIM*2))
                        ptr=self._ptr(self.a,pixel,3,self.fb.const(0))
                        self._rocc('mvin',{'local':a*2*F.DIM,'rows':rows,'cols':21,'load_id':0},ptr)
                    for ki,kr in [(0,16),(1,5)]:
                        krow=self.fb.const(kh*21+ki*16)
                        for d in range(0,len(nr),4):
                            ptr=self._ptr(self.b,krow,s.cout,self._tile(n0,d))
                            self._rocc('mvin',{'local':bbase+d*F.DIM,'rows':kr,'cols':sum(nr[d:d+4]),'load_id':1},ptr)
                        for d,cols in enumerate(nr):
                            for a,rows in enumerate(widths):
                                self._rocc('preload',{'bd':bbase+d*F.DIM if a==0 else isa.GARBAGE_ADDR,'c':isa.acc_addr((a*s.bn+d)*F.DIM,accumulate=not(kh==0 and ki==0)),'bd_cols':cols,'bd_rows':kr,'c_cols':cols,'c_rows':rows})
                                self._rocc('compute',{'a':(a*2+ki)*F.DIM,'a_cols':kr,'a_rows':rows,'accumulate':a!=0})
                for a,rows in enumerate(widths):
                    pixel=self.fb.add_i(self.fb.mul_i(y,self.fb.const(s.ow)),self.fb.const(a*F.DIM))
                    step=4 if s.output_dtype=='i8' else 1
                    for d in range(0,len(nr),step):
                        ptr=self._ptr(self.c,pixel,s.cout,self._tile(n0,d),4 if s.output_dtype=='i32' else 1)
                        self._rocc('mvout',{'local':isa.acc_addr((a*s.bn+d)*F.DIM,full_row=s.output_dtype=='i32'),'rows':rows,'cols':sum(nr[d:d+step])},ptr)
            self._for_groups(_groups(s.cout,s.bn),channel)
        self.fb.for_loop(0,s.oh,1,row)
        module=self._finish('gemmini_golden_stem');module.attributes['gemmini.stem_shape']=StringAttr(json.dumps(asdict(s),sort_keys=True));return module


def command_counts(s):
    s.validate();mt=_ceil_div(s.ow,F.DIM);nt=_ceil_div(s.cout,F.DIM);groups=_ceil_div(nt,s.bn)
    computes=s.oh*7*2*mt*nt
    return {'compute':computes,'preload':computes,'mvin_a':s.oh*7*mt*groups,'mvin_b':s.oh*14*sum(_ceil_div(min(s.bn,nt-d),4) for d in range(0,nt,s.bn)),'mvout':s.oh*mt*(nt if s.output_dtype=='i32' else sum(_ceil_div(min(s.bn,nt-d),4) for d in range(0,nt,s.bn))),'padded_array_issue_cycles':computes*F.DIM,'host_im2col_bytes':0,'host_weight_pack_bytes':0}
