"""Packed RGB stem with spatial-contiguous accumulator bands and hardware pool.

A band retains up to nine complete convolution rows for four pooled rows. One
16-channel block occupies the accumulator at a time; cache all A panels and B
weights in scratchpad. Adjacent bands recompute their shared convolution row.
This intentional 11.6% full-stem compute overhead avoids host intermediates.
"""
from dataclasses import dataclass,asdict
import json
from xdsl.dialects.builtin import StringAttr
from .golden_gemm import GoldenGemm,Shape,_ceil_div,_groups
from .tables import rtl_facts as F,isa

@dataclass(frozen=True)
class StemPoolShape:
    h:int
    w:int
    cout:int=64
    scale:float=1.0
    pooled_rows:int=4
    @property
    def oh(self):return (self.h+1)//2
    @property
    def ow(self):return (self.w+1)//2
    @property
    def ph(self):return (self.oh+1)//2
    @property
    def pw(self):return (self.ow+1)//2
    def bands(self):
        return [(p,min(self.pooled_rows,self.ph-p),max(0,p*2-1),min(self.oh,(p+min(self.pooled_rows,self.ph-p)-1)*2+2)) for p in range(0,self.ph,self.pooled_rows)]
    def validate(self):
        if min(self.h,self.w,self.cout,self.pooled_rows)<=0:raise ValueError('positive pooled stem extents required')
        if self.cout>64:raise ValueError('pooled stem supports at most64 output channels')
        if max(self.oh,self.ow,self.ph,self.pw)>255:raise ValueError('pool configuration dimension overflow')
        rows=max(end-start for _,_,start,end in self.bands());mt=_ceil_div(self.ow,16);nt=_ceil_div(self.cout,16)
        if rows*self.ow>F.ACC_ROWS:raise ValueError('spatial accumulator band exceeds capacity')
        if rows*7*mt*2*16+14*nt*16>F.SPAD_ROWS:raise ValueError('cached band and weights exceed scratchpad')
        if self.scale<=0:raise ValueError('positive proven scalar scale required')

class GoldenStemPool(GoldenGemm):
    def __init__(self,s,*,loop_spatial=False):
        if type(loop_spatial) is not bool:raise ValueError('spatial CPU loop option requires explicit boolean')
        s.validate();self.pool=s;self.loop_spatial=loop_spatial
        super().__init__(Shape(s.ow,s.cout,21,bm=_ceil_div(s.ow,16),bn=1,output_dtype='i8',scale=s.scale,relu=True))

    def _compact_spatial(self,rows,widths,kh,ki,kr,cols,d,bbase):
        """Retain ordinary CPU loops over proven resident A/ACC tile addresses.

        Full-width bands are affine in one tile index. A partial final tile
        instead keeps row and spatial indices separate, so no DMA or compute
        crosses a logical row. The first real-B preload remains separate.
        """
        s=self.pool;mt=len(widths);accumulate=not(kh==0 and ki==0)
        self._rocc('preload',{'bd':bbase+((kh*2+ki)*_ceil_div(s.cout,16)+d)*16,
            'c':isa.acc_addr(0,accumulate=accumulate),'bd_cols':cols,'bd_rows':kr,
            'c_cols':cols,'c_rows':widths[0]})
        self._rocc('compute',{'a':(kh*rows*mt*2+ki)*16,'a_cols':kr,
            'a_rows':widths[0],'accumulate':False})
        cmax=(rows-1)*s.ow+(mt-1)*16
        amax=((kh*rows+rows-1)*mt+mt-1)*32+ki*16
        def emit(crow,arow,nrows):
            self._rocc('preload',{'bd':isa.GARBAGE_ADDR,'c_accumulate':int(accumulate),
                'c_max':min(cmax,rows*s.ow-nrows),'c_reserved_rows':rows*s.ow,'bd_cols':cols,'bd_rows':kr,
                'c_cols':cols,'c_rows':nrows},crow)
            self._rocc('compute',{'a_cols':kr,'a_rows':nrows,'accumulate':True,
                'a_max':amax,'a_reserved_rows':bbase},arow)
        if all(width==16 for width in widths):
            def tile(index):
                emit(self.fb.mul_i(index,self.fb.const(16)),
                    self.fb.add_i(self.fb.mul_i(index,self.fb.const(32)),
                        self.fb.const(kh*rows*mt*32+ki*16)),16)
            self.fb.for_loop(1,rows*mt,1,tile,retain_loop=True)
            return
        full=sum(width==16 for width in widths)
        def tile(ry,a,nrows):
            crow=self.fb.add_i(self.fb.mul_i(ry,self.fb.const(s.ow)),self.fb.mul_i(a,self.fb.const(16)))
            panel=self.fb.add_i(self.fb.mul_i(ry,self.fb.const(mt)),a)
            arow=self.fb.add_i(self.fb.mul_i(panel,self.fb.const(32)),self.fb.const(kh*rows*mt*32+ki*16))
            emit(crow,arow,nrows)
        # Finish row zero before advancing the source reduction's spatial scan.
        if full>1:self.fb.for_loop(1,full,1,lambda a:tile(self.fb.const(0),a,16),retain_loop=True)
        if mt>1:tile(self.fb.const(0),self.fb.const(mt-1),widths[-1])
        def row(ry):
            self.fb.for_loop(0,full,1,lambda a:tile(ry,a,16),retain_loop=True)
            tile(ry,self.fb.const(mt-1),widths[-1])
        self.fb.for_loop(1,rows,1,row,retain_loop=True)
    def build(self):
        s=self.pool;self._emit_config();self._rocc('config_ld',{'stride':6,'load_id':0})
        widths=tuple(min(16,s.ow-x) for x in range(0,s.ow,16));mt=len(widths);nt=_ceil_div(s.cout,16);maxrows=max(end-start for _,_,start,end in s.bands());bbase=maxrows*7*mt*2*16
        # All original147xCout weights remain cached in14 K blocks.
        for kh in range(7):
            for ki,kr in [(0,16),(1,5)]:
                ptr=self._ptr(self.b,self.fb.const(kh*21+ki*16),s.cout,self.fb.const(0))
                self._rocc('mvin',{'local':bbase+(kh*2+ki)*nt*16,'rows':kr,'cols':s.cout,'load_id':1},ptr)
        groups=[]
        for p,count,start,end in s.bands():
            key=(count,end-start,int(p==0))
            if groups and groups[-1][2]==key:groups[-1]=(groups[-1][0],p+s.pooled_rows,key)
            else:groups.append((p,p+s.pooled_rows,key))
        def band(p,count,rows,upad):
            start=self.fb.add_i(self.fb.mul_i(p,self.fb.const(2)),self.fb.const(0 if upad else -1))
            for kh in range(7):
                for ry in range(rows):
                    cy=self.fb.add_i(start,self.fb.const(ry));iy=self.fb.add_i(self.fb.mul_i(cy,self.fb.const(2)),self.fb.const(kh))
                    for a,nrows in enumerate(widths):
                        pixel=self.fb.add_i(self.fb.mul_i(iy,self.fb.const(s.w+6)),self.fb.const(a*32))
                        ptr=self._ptr(self.a,pixel,3,self.fb.const(0));local=((kh*rows+ry)*mt+a)*2*16
                        self._rocc('mvin',{'local':local,'rows':nrows,'cols':21,'load_id':0},ptr)
            # Channel blocks are static so every scratch/accumulator address is proven.
            for d in range(nt):
                cols=min(16,s.cout-d*16)
                for kh in range(7):
                    for ki,kr in [(0,16),(1,5)]:
                        if self.loop_spatial:
                            self._compact_spatial(rows,widths,kh,ki,kr,cols,d,bbase)
                            continue
                        for ry in range(rows):
                            for a,nrows in enumerate(widths):
                                first=ry==0 and a==0
                                self._rocc('preload',{'bd':bbase+((kh*2+ki)*nt+d)*16 if first else isa.GARBAGE_ADDR,'c':isa.acc_addr(ry*s.ow+a*16,accumulate=not(kh==0 and ki==0)),'bd_cols':cols,'bd_rows':kr,'c_cols':cols,'c_rows':nrows})
                                self._rocc('compute',{'a':(((kh*rows+ry)*mt+a)*2+ki)*16,'a_cols':kr,'a_rows':nrows,'accumulate':not first})
                # <=2 pooled rows keeps one command's window visits <=1024.
                for q in range(0,count,2):
                    pc=min(2,count-q);offset=0 if q==0 else q*2-upad;pad=upad if q==0 else 0
                    if pc*s.pw*9>1024:raise ValueError('pool command exceeds tracker bound')
                    self._rocc('config_st',{'stride':s.cout,'acc_act':isa.RELU,'acc_scale':s.scale,'pool_stride':2,'pool_size':3,'pool_out_dim':s.pw,'porows':pc,'pocols':s.pw,'orows':rows-offset,'ocols':s.ow,'upad':pad,'lpad':1})
                    py=self.fb.add_i(p,self.fb.const(q));ptr=self._ptr(self.c,py,s.pw*s.cout,self.fb.const(d*16))
                    self._rocc('mvout',{'local':isa.acc_addr(offset*s.ow),'rows':1,'cols':cols},ptr)
                # Stock RS models pooling as reading to the end of its start bank.
                # A spatial band may cross that bank, so drain before C reuse.
                self._rocc('fence',{})
        for start,stop,(count,rows,upad) in groups:self.fb.for_loop(start,stop,s.pooled_rows,lambda p,c=count,r=rows,u=upad:band(p,c,r,u),retain_loop=self.loop_spatial)
        module=self._finish('gemmini_golden_stem_pool');module.attributes['gemmini.stem_pool_shape']=StringAttr(json.dumps(asdict(s),sort_keys=True));return module


def command_counts(s):
    s.validate();rows=sum(end-start for _,_,start,end in s.bands());mt=_ceil_div(s.ow,16);nt=_ceil_div(s.cout,16);computes=rows*mt*nt*14
    return {'compute':computes,'preload':computes,'mvin_a':rows*mt*7,'mvin_b':14,'pool_mvout':sum(_ceil_div(count,2)*nt for _,count,_,_ in s.bands()),'convolution_rows_computed':rows,'convolution_rows_required':s.oh,'padded_array_issue_cycles':computes*16,'phase_fences':len(s.bands())*nt,'host_intermediate_bytes':0,'host_im2col_bytes':0}
