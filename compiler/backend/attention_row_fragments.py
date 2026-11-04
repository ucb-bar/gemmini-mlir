"""Exact fragmented transport for a logical concatenation of disjoint matrices.

Only row-address transport changes. Each logical transfer is partitioned into
ordered disjoint original-matrix rectangles. The shared dense encoder retains
all arithmetic, readout configuration and completion ownership.
"""
from . import isa
from .attention_row_stack import _RowView
from merlin.compile.scheduling.affine_gather_runs import (
    ConvolutionGatherRuns, emit_convolution_gather_runs,
)
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter


def _plan(members,m,width):
    # The exact 1x1 geometry is simply a flattened batch of member row axes.
    # It grants row partitions only; no convolution or payload gather is run.
    return ConvolutionGatherRuns((members,1,m,width),(members,1,m),
                                (1,1),(1,1),(1,1),isa.DIM)


def check_fragment_projection(members,m,k,n):
    """Exhaust every row-boundary residue and primitive row count.

    Fragmentation is translation invariant by member rows. Terminal partial
    groups have identical partitions, and the logical dense bounds separately
    prove the final member index. Local displacements partition [0,count), so
    every SPAD/ACC row and each original output row keeps its exact owner.
    """
    queries=0
    for width in set((k,n)):
        # Include one extra member for worst-case primitive row counts. This
        # reference only checks the periodic partition; actual bounds use the
        # consumed dense schedule and the emitted plan uses actual members.
        plan=_plan(members+isa.DIM,m,width)
        for first in range(m):
            for count in range(1,isa.DIM+1):
                used=0
                for offset,local,rows in plan.fragments(first,count):
                    if (offset!=(first+used)*width or local!=used or rows<=0
                            or rows>isa.DIM or (first+used)%m+rows>m):
                        raise ValueError('invalid exact member row partition')
                    used+=rows
                if used!=count:raise ValueError('incomplete member row partition')
                queries+=1
    return dict(periodic_rectangle_queries=queries,
        projection='fragment source=(logical row+used) divmod member rows; local row=original local row+used',
        coverage='ordered disjoint fragments partition each actual DMA rectangle',
        accumulator_mapping='same column/flags; local accumulator row advances by exact fragment displacement')


class FragmentRowView(_RowView):
    def __init__(self,ctx,parts,m,k,n,dtype):
        super().__init__(ctx,parts,m,k,n,dtype)
        self._pending={}
        self._plans={name:_plan(len(parts),m,shape[1])for name,shape in self._shapes.items()}

    def ptr(self,name,elem_offset=0):
        value=super().ptr(name,elem_offset)
        if name in self._shapes:self._pending[value]=(name,elem_offset)
        return value

    def emit(self,funct,rs1,rs2,*args,**kwargs):
        pending=self._pending.pop(rs1,None) if not isinstance(rs1,int) else None
        if pending is None:return self._ctx.emit(funct,rs1,rs2,*args,**kwargs)
        name,offset=pending
        expected=isa.FUNCT['mvin'] if name=='_row_stack_lhs' else isa.FUNCT['mvout']
        if funct!=expected:raise ValueError('fragment row pointer consumed by unexpected instruction')
        from .dense_readout import _pack
        fb=self.fb;b=FnBuilderControlAdapter(fb)
        width=self._shapes[name][1]
        offset=b.value(offset)
        first=fb.udiv_i(offset,fb.const(width))
        column=fb.urem_i(offset,fb.const(width))
        word=b.value(rs2)
        rows=fb.and_i(fb.ashr_i(word,48),fb.const(65535))
        columns=fb.and_i(fb.ashr_i(word,32),fb.const(65535))
        address=fb.and_i(word,fb.const(0xffffffff))
        for source,used,length in emit_convolution_gather_runs(
                self._plans[name],b,first,rows,max_count=min(isa.DIM,len(self._parts)*self._m)):
            def issue(source=source,used=used,length=length):
                pointer=super(FragmentRowView,self).ptr(name,fb.add_i(source,column))
                self._ctx.emit(funct,pointer,_pack(fb,fb.add_i(address,b.value(used)),length,columns),*args,**kwargs)
            b.branch(b.cmp('ult',0,length),issue,lambda:None)
