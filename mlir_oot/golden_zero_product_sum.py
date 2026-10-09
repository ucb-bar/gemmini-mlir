"""Runtime omission using immutable encoded-i8 panel nonzero metadata.

Five-pointer ABI adds A/B flags [plane,ceil(M or N/DIM)]. Flags are the exact
OR of nonzero encoded bytes in every logical row of a complete DIM-row panel.
Merlin owns producer joins/domain/lifetime proof. This provider consumes those
explicit facts, never floating dtype assumptions. Dense blocks retain the
original schedule; sparse blocks zero every owned accumulator tile before
conditionally loading/computing terms. Every logical output is always stored.
Encoded inputs and flags must be immutable together and must not overlap the
complete destination span. This API does not infer that alias/lifetime proof.
"""
from collections import Counter

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64
from xdsl.ir import Block

from .codegen.builder import PTR, FnBuilder
from .golden_gemm import _ceil_div
from .golden_product_sum import GoldenProductSum
from .tables import isa
from .tables import rtl_facts as F


def command_census(shape, pairs, lhs_nonzero, rhs_nonzero):
    """Count emitted primitive commands for explicit immutable panel flags.

    This is a schedule census, never a latency/occupancy estimate. Configuration
    and CPU metadata/branch costs are not primitive commands and remain separate.
    """
    from .golden_gemm import _groups
    if (lhs_nonzero.rows!=shape.m or rhs_nonzero.rows!=shape.n
            or lhs_nonzero.reduction_length!=shape.k or rhs_nonzero.reduction_length!=shape.k
            or lhs_nonzero.block_rows!=F.DIM or rhs_nonzero.block_rows!=F.DIM):
        raise ValueError('complete shape-bound DIM-row encoded summaries required')
    counts=Counter()
    for mstart,mstop,mstep,mr in _groups(shape.m,shape.bm):
        for nstart,nstop,nstep,nr in _groups(shape.n,shape.bn):
            for m0 in range(mstart,mstop,mstep):
                for n0 in range(nstart,nstop,nstep):
                    active=[([lhs_nonzero.active(ap,m0+a) for a in range(len(mr))],
                             [rhs_nonzero.active(bp,n0+b) for b in range(len(nr))])
                            for ap,bp in pairs]
                    dense=all(all(aa) and all(bb) for aa,bb in active)
                    counts['dense_output_blocks' if dense else 'sparse_output_blocks']+=1
                    counts['summary_byte_reads_logical']+=len(pairs)*(len(mr)+len(nr))
                    counts['mvout']+=len(mr)*len(nr)
                    if not dense:counts['mvin_zero_acc']+=len(mr)*len(nr)
                    kt=_ceil_div(shape.k,F.DIM)
                    for aa,bb in active:
                        if not any(aa) or not any(bb):
                            counts['omitted_pair_blocks']+=1
                            continue
                        compute=len(mr)*len(nr) if dense else sum(aa)*sum(bb)
                        counts['mvin_a']+=kt*(len(mr) if dense else sum(aa))
                        counts['mvin_b']+=kt*(_ceil_div(len(nr),4) if shape.wide_b else len(nr))
                        counts['preload']+=kt*compute
                        counts['compute']+=kt*compute
    return dict(counts)


class GoldenZeroProductSum(GoldenProductSum):
    def __init__(self, shape, **kwargs):
        if kwargs.get("resident_operands", False):
            raise ValueError(
                "encoded zero metadata has a distinct panel placement; "
                "resident operand composition requires a separate contract"
            )
        super().__init__(shape, **kwargs)
        self.fb = FnBuilder([PTR] * 5)
        self.a,self.b,self.c,self.aflags,self.bflags=self.fb.entry.args
        self._active = None

    def _if(self, cond, yes, no=None):
        then,other,merge=Block(),Block(),Block()
        for block in (then,other,merge):self.fb.region.add_block(block)
        self.fb.add(llvm.CondBrOp(cond,then,[],other,[]))
        self.fb.blk=then;yes();self.fb.add(llvm.BrOp(merge))
        self.fb.blk=other
        if no is not None:no()
        self.fb.add(llvm.BrOp(merge));self.fb.blk=merge

    def _nz(self, pointer, plane, tile, tiles):
        index=self.fb.add_i(self.fb.const(plane*tiles),tile)
        value=self.fb.load_i64(pointer,index,'i8')
        return self.fb.add(llvm.ICmpOp(value,self.fb.const(0),
            IntegerAttr(llvm.ICmpPredicateFlag.NE.to_int(),i64))).results[0]

    def _or(self,values):
        value=values[0]
        for other in values[1:]:value=self.fb.add(llvm.OrOp(value,other)).results[0]
        return value

    def _k_tile(self,m0,n0,k0,mr,nr,kr,first,*args,**kwargs):
        if self._active is None:
            return super()._k_tile(m0,n0,k0,mr,nr,kr,first,*args,**kwargs)
        aa,bb=self._active;s=self.shape;krow=self._tile(k0,0)
        for a,rows in enumerate(mr):
            def load(a=a,rows=rows):
                ptr=self._ptr(self.a,self._tile(m0,a),s.k,krow)
                self._rocc('mvin',{'local':a*F.DIM,'rows':rows,'cols':kr,'load_id':0},ptr)
            self._if(aa[a],load)
        # Preserve supported wide B DMA and its original scratchpad span.
        self._load_b_panel(n0,k0,nr,kr,s.bm)
        for d,cols in enumerate(nr):
            def column(d=d,cols=cols):
                for a,rows in enumerate(mr):
                    def compute(a=a,rows=rows):
                        def emit(reuse):
                            self._rocc('preload',{'bd':isa.GARBAGE_ADDR if reuse else (s.bm+d)*F.DIM,
                                'c':isa.acc_addr((a*s.bn+d)*F.DIM,accumulate=True),
                                'bd_cols':cols,'bd_rows':kr,'c_cols':cols,'c_rows':rows})
                            self._rocc('compute',{'a':a*F.DIM,'a_cols':kr,'a_rows':rows,'accumulate':reuse})
                        if s.reuse_b and a:
                            self._if(self._or(aa[:a]),lambda:emit(True),lambda:emit(False))
                        else:emit(False)
                    self._if(aa[a],compute)
            self._if(bb[d],column)

    def _reduce_output_block(self,m0,n0,mr,nr,slot=0,prefetch=None):
        s=self.shape;a,b=self.a,self.b
        active=[];all_flags=[]
        for ap,bp in self.pairs:
            aa=[self._nz(self.aflags,ap,self.fb.add_i(m0,self.fb.const(t)),_ceil_div(s.m,F.DIM)) for t in range(len(mr))]
            bb=[self._nz(self.bflags,bp,self.fb.add_i(n0,self.fb.const(t)),_ceil_div(s.n,F.DIM)) for t in range(len(nr))]
            active.append((aa,bb));all_flags.extend(aa+bb)
        dense=all_flags[0]
        for value in all_flags[1:]:dense=self.fb.add(llvm.AndOp(dense,value)).results[0]
        def normal():super(GoldenZeroProductSum,self)._reduce_output_block(m0,n0,mr,nr,slot,prefetch)
        def sparse():
            zero=self.fb.add(llvm.IntToPtrOp(self.fb.const(0))).results[0]
            # ACC DMA accepts one DIM-wide i32 tile; no unproved wide-zero path.
            for ai,rows in enumerate(mr):
                for bi,cols in enumerate(nr):
                    self._rocc('mvin',{'local':isa.acc_addr((ai*s.bn+bi)*F.DIM),
                        'rows':rows,'cols':cols,'load_id':2},zero)
            for (ap,bp),(aa,bb) in zip(self.pairs,active):
                self.a=self._ptr(a,self.fb.const(ap*s.m),s.k,self.fb.const(0))
                self.b=self._ptr(b,self.fb.const(bp*s.k),s.n,self.fb.const(0))
                self._active=(aa,bb)
                condition=self.fb.add(llvm.AndOp(self._or(aa),self._or(bb))).results[0]
                self._if(condition,lambda:super(GoldenProductSum,self)._reduce_output_block(m0,n0,mr,nr,slot,prefetch))
            self.a,self.b=a,b;self._active=None
        self._if(dense,normal,sparse)

    def build(self):
        self._emit_config()
        self._rocc('config_ld',{'stride':0,'load_id':2})
        self._emit_work();self._rocc('fence',{})
        fn=llvm.FuncOp('gemmini_golden_zero_product_sum',llvm.LLVMFunctionType([PTR]*5),
            linkage=llvm.LinkageAttr('external'),body=self.fb.finish())
        module=ModuleOp([fn]);module.attributes['gemmini.dim']=IntegerAttr(F.DIM,i64)
        module.attributes['gemmini.encoded_i8_nonzero']=StringAttr('plane-major DIM-row complete immutable summaries')
        module.attributes['gemmini.product_pairs']=StringAttr(str(self.pairs))
        module.attributes['gemmini.product_shape']=StringAttr(f'{self.shape.m}x{self.shape.k}x{self.shape.n}:i32:bm{self.shape.bm}:bn{self.shape.bn}')
        module.verify();return module
