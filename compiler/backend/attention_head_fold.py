"""Fold contiguous GQA head rows while preserving every source product.

A shared KV matrix is read by adjacent query heads. Their M rows and result
rows are contiguous, so [G,M,K] -> [G*M,K] is a pointer-preserving view. The
reduction remains K; no head is reduced, broadcast, copied or dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from merlin.compile.scheduling.batched_dense import (
    BatchMatrixView, BatchedDenseProblem, counted_batched_dense_products,
)
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program


@dataclass(frozen=True)
class HeadFoldPlan:
    groups: int
    heads: int
    m: int
    k: int
    n: int
    source: BatchedDenseProblem
    folded: BatchedDenseProblem
    kernel: object

    def revalidate(self):
        dimensions = (self.groups,self.heads,self.m,self.k,self.n)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError('head fold requires positive integer shape extents')
        if type(self.source) is not BatchedDenseProblem or type(self.folded) is not BatchedDenseProblem:
            raise ValueError('head fold requires typed original and folded products')
        self.source.__post_init__(); self.folded.__post_init__()
        source=BatchedDenseProblem((self.groups,self.heads),
            BatchMatrixView((self.groups,self.heads,self.m,self.k),(0,1)),
            BatchMatrixView((self.groups,self.k,self.n),(0,)))
        folded=BatchedDenseProblem((self.groups,),
            BatchMatrixView((self.groups,self.heads*self.m,self.k),(0,)),
            BatchMatrixView((self.groups,self.k,self.n),(0,)))
        if self.source!=source or self.folded!=folded:
            raise ValueError('head fold changed source batch axes or complete tensor extents')
        expected = counted_batched_dense_products(self.folded,
            name='attention_folded_head_rows', instruction='attention_folded_product')
        # The canonical digest also distinguishes Python-equal values such
        # as a Boolean loop extent versus the required integer extent.
        if (type(self.kernel) is not type(expected) or self.kernel != expected
            or self.kernel.digest() != expected.digest()):
            raise ValueError('folded attention counted source offsets changed')
        # Exact affine forms over arbitrary (group, head, row, column, K).
        # Integer coefficient equality proves the address identity on the
        # entire domain, without enumerating points or depending on an SMT
        # solver. All constants and products below are unbounded Python ints.
        g,h,r,c,t = tuple(tuple(int(i == j) for i in range(5)) for j in range(5))
        def add(*forms):
            return tuple(sum(values) for values in zip(*forms,strict=True))
        def scale(form, factor):
            return tuple(value*factor for value in form)
        def equal(left, right):
            if left != right:
                raise ValueError('head row fold affine source projection changed')
        row = add(scale(h,self.m),r)
        equal(add(row,scale(r,-1)),scale(h,self.m))
        row_max = (self.heads-1)*self.m+(self.m-1)
        if row_max != self.heads*self.m-1:
            raise ValueError('head row fold does not cover the complete row interval')
        # For every head, [head*M, head*M+M-1] has M elements and
        # its successor starts at the previous endpoint+1. These disjoint
        # adjacent intervals cover [0,heads*M). Because 0<=row_in_head<M,
        # Euclidean division uniquely recovers (head,row_in_head). Conversely
        # any folded row in that interval has a quotient in [0,heads) and a
        # remainder in [0,M), and quotient*M+remainder is that original row.
        if ((self.heads*self.m)//self.m != self.heads
            or divmod(row_max,self.m) != (self.heads-1,self.m-1)):
            raise ValueError('head row fold lacks the complete Euclidean inverse')
        source_row = add(scale(add(scale(g,self.heads),h),self.m),r)
        folded_row = add(scale(g,self.heads*self.m),row)
        source_lhs = add(scale(source_row,self.k),t)
        folded_lhs = add(scale(folded_row,self.k),t)
        source_rhs = add(scale(add(scale(g,self.k),t),self.n),c)
        folded_rhs = add(scale(g,self.k*self.n),scale(t,self.n),c)
        source_out = add(scale(source_row,self.n),c)
        folded_out = add(scale(folded_row,self.n),c)
        extents = (self.groups,self.heads,self.m,self.n,self.k)
        address_bounds = {}
        for name,original,folded,elements,widths in (
            ('lhs',source_lhs,folded_lhs,self.groups*self.heads*self.m*self.k,(1,)),
            ('rhs',source_rhs,folded_rhs,self.groups*self.k*self.n,(1,)),
            ('output',source_out,folded_out,self.groups*self.heads*self.m*self.n,(1,4)),
        ):
            equal(original,folded)
            # Nonnegative coefficients make the affine minimum zero and the
            # maximum occur at every domain endpoint simultaneously. Check
            # the last byte too, rather than only a tile's starting address.
            if any(value < 0 for value in original):
                raise ValueError('head fold address is not monotone on its domain')
            maximum = sum(coefficient*(extent-1)
                          for coefficient,extent in zip(original,extents,strict=True))
            if maximum != elements-1:
                raise ValueError('head fold address bounds changed the complete tensor extent')
            address_bounds[name] = {}
            for width in widths:
                equal(scale(original,width),scale(folded,width))
                last_byte = maximum*width+width-1
                if last_byte != elements*width-1 or not 0 <= last_byte < (1 << 63):
                    raise ValueError('head fold byte address exceeds its complete signed-index domain')
                address_bounds[name][str(width)] = {'minimum_byte':0,'maximum_byte':last_byte}
        return dict(schema='submission.attention_head_row_fold.v1',
            source=self.source.record(),folded=self.folded.record(),
            source_products=self.groups*self.heads,folded_products=self.groups,
            source_row='local_head * M + original_row',reduction_extent=self.k,
            kernel_sha256=self.kernel.digest(),address_projection='all original MAC and output addresses identical',
            row_bijection='local_head = folded_row // M; original_row = folded_row % M',
            numeric='each independent source row keeps its original K reduction and readout',
            proof='exact integer affine coefficients and Euclidean row bijection',
            byte_address_bounds=address_bounds)


def plan_head_fold(groups,heads,m,k,n):
    source=BatchedDenseProblem((groups,heads),
        BatchMatrixView((groups,heads,m,k),(0,1)),
        BatchMatrixView((groups,k,n),(0,)))
    folded=BatchedDenseProblem((groups,),
        BatchMatrixView((groups,heads*m,k),(0,)),
        BatchMatrixView((groups,k,n),(0,)))
    kernel=counted_batched_dense_products(folded,
        name='attention_folded_head_rows',instruction='attention_folded_product')
    plan=HeadFoldPlan(groups,heads,m,k,n,source,folded,kernel)
    plan.revalidate()
    return plan


def lower(ctx,left,right,dst,geometry,attrs):
    """Try complete KV partitions; refusal occurs before commands or aliases."""
    from .dense import lower_dense
    batches,hq,hkv,m,k,n=geometry
    heads=hq//hkv
    report={'mechanism':'attention_grouped_head_row_fold','applied':False}
    ctx.metadata.setdefault('attention_head_fold',[]).append(report)
    if heads<2:
        report['refusal']='no shared KV heads';return False
    # A shared [M,N] initializer would need replication across head rows. Keep
    # its existing per-product contract rather than silently changing the view.
    if attrs.get('accumulator_init') or attrs.get('block_scale') is not None or 'pool' in attrs.get('epilogue',()):
        report['refusal']='initializer, block scale or pooled readout needs original product boundary';return False
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    names={left,right,dst}|{attrs[key] for key in ('bias',) if attrs.get(key)}
    sources={a.get('source') for a in aliases.values() if isinstance(a,dict)}
    if dst in (left,right) or names & (set(aliases)|sources):
        report['refusal']='explicit aliases or output/input overlap';return False
    if attrs.get('bias') and ctx.spec(attrs['bias'])['shape']!=[n]:
        report['refusal']='bias does not share the original output columns';return False
    plan=plan_head_fold(batches*hkv,heads,m,k,n)
    proof=plan.revalidate()
    bridge=FnBuilderIndexAdapter(ctx.fb);source_index=ctx.current_source_index
    def alias(name,offset,shape):
        label=f'_attention_fold_{name}_{len(ctx.tensors)}'
        ctx.register(label,ctx.ptr(name,bridge.value(offset)),shape,ctx.spec(name)['dtype'])
        return label
    def emit(instruction,a):
        if instruction!='attention_folded_product':raise ValueError('unknown folded attention effect')
        ctx.current_source_index=source_index
        aa=alias(left,a['lhs_offset'],(heads*m,k))
        bb=alias(right,a['rhs_offset'],(k,n))
        cc=alias(dst,a['output_offset'],(heads*m,n))
        lower_dense(ctx,aa,bb,cc,m=heads*m,k=k,n=n,attributes=attrs)
        ctx.fence()
    emission=emit_index_program(plan.kernel,bridge,emit)
    ctx.current_source_index=source_index
    report.update(proof,applied=True,emission=emission,
        layout='pointer-preserving contiguous query-head rows and original result rows',
        completion='complete shared dense schedule before the next disjoint KV partition')
    return True
