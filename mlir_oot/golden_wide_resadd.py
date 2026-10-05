"""Exact wide-integer residual using resident signed-i8 diagonal chunks.

ABI A,B,C,coefficient_tables. Tables contain diagonal 127, p%127, q%127
in that order, each 16x16 i8. Inputs/output are contiguous Mx64 i8, M%16=0.
The binder proves the final scale against every source operand pair separately.
No load rounding, intermediate readout, CPU duplication or FSM commands occur.
"""
import math
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp,StringAttr,IntegerAttr,i64
from .codegen.builder import FnBuilder,PTR
from .golden_gemm import GoldenGemm,Shape
from .tables import isa,rtl_facts as F


def tables(p,q):
    return bytes((v if i==j else 0) for v in (127,p%127,q%127) for i in range(16) for j in range(16))


def build(m,p,q,scale,*,relu=True):
    if type(m) is not int or m<=0 or m%16:raise ValueError('positive M divisible by16 required')
    if any(type(x) is not int or not 1<=x<=32767 for x in (p,q)):raise ValueError('coefficients must be integers1..32767')
    if not math.isfinite(scale) or scale<=0:raise ValueError('positive finite readout scale required')
    e=GoldenGemm(Shape(m,64,16,bn=4));e.fb=FnBuilder([PTR]*4)
    a,b,c,coeff=e.fb.entry.args
    e._rocc('fence',{});e._rocc('flush',{})
    e._rocc('config_ex',{'dataflow':isa.WEIGHT_STATIONARY})
    for lid in (0,1):e._rocc('config_ld',{'stride':64,'load_id':lid})
    e._rocc('config_ld',{'stride':16,'load_id':2})
    e._rocc('config_st',{'stride':64,'acc_act':isa.RELU if relu else isa.NO_ACTIVATION,'acc_scale':scale})
    for index in range(3):
        ptr=e._ptr(coeff,e.fb.const(0),1,e.fb.const(index*256))
        e._rocc('mvin',{'local':8192+index*16,'rows':16,'cols':16,'load_id':2},ptr)
    def panel(row):
        for ptr,base,lid in ((a,0,0),(b,4096,1)):
            e._rocc('mvin',{'local':base,'rows':16,'cols':64,'load_id':lid},e._ptr(ptr,row,64,e.fb.const(0)))
        first=True
        for value,base,remainder_tile in ((p,0,1),(q,4096,2)):
            full,remainder=divmod(value,127)
            def chunk(weight,load_weight,initialize):
                for d in range(4):
                    e._rocc('preload',{'bd':weight if load_weight and d==0 else isa.GARBAGE_ADDR,
                        'c':isa.acc_addr(d*16,accumulate=not initialize),'bd_cols':16,'bd_rows':16,'c_cols':16,'c_rows':16})
                    e._rocc('compute',{'a':base+d*16,'a_cols':16,'a_rows':16,'accumulate':not (load_weight and d==0)})
            if full:
                chunk(8192,True,first);first=False
                if full>1:e.fb.for_loop(1,full,1,lambda unused:chunk(8192,False,False))
            if remainder:
                chunk(8192+remainder_tile*16,True,first);first=False
        e._rocc('mvout',{'local':isa.acc_addr(0),'rows':16,'cols':64},e._ptr(c,row,64,e.fb.const(0)))
    e.fb.for_loop(0,m,16,panel)
    e._rocc('fence',{})
    fn=llvm.FuncOp('gemmini_golden_wide_resadd',llvm.LLVMFunctionType([PTR]*4),linkage=llvm.LinkageAttr('external'),body=e.fb.finish())
    module=ModuleOp([fn]);module.attributes['gemmini.dim']=IntegerAttr(F.DIM,i64)
    module.attributes['gemmini.golden_wide_resadd']=StringAttr(f'M{m}:N64:p{p}:q{q}:scale{scale}:relu{int(relu)}')
    module.verify();return module
