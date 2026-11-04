"""Exact shared-RHS row packing for independent source-qualified products.

Discovery owns source independence and reordering. This consumer checks the
physical partition, packs every original LHS with checked identity DMA, invokes
the common dense scheduler, then scatters every complete output with identity
DMA. All numeric work remains in the unchanged dense device product/readout.
"""
from __future__ import annotations

from . import isa
from .attention_head_fold import plan_head_fold
from .attention_row_copy import plan_copy, emit_copy


def lower_packed_products(ctx,members):
    """Return None before mutation if a complete shared row group is refused.

    Members use the resident-products interface: lhs/rhs/dst/attributes and
    optional m/k/n. Every source row keeps its original reduction and columns.
    This initial alternative targets small M, where folding fills array rows.
    """
    members=tuple(members)
    if not 2<=len(members)<=32:return None
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    try:
        first=members[0];rhs=first['rhs']
        a,b,c=(ctx.spec(first[key]) for key in ('lhs','rhs','dst'))
        m,k=a['shape'];bk,n=b['shape'];dtype=c['dtype']
        if not 0<m<isa.DIM or bk!=k or dtype not in ('i8','i32'):return None
        # Packing is useful only if it eliminates primitive padded M tiles.
        # The M13 x3 witness has equal tile counts and regressed with copies.
        if (len(members)*m+isa.DIM-1)//isa.DIM>=len(members):return None
        def numeric(attrs):
            return {key:value for key,value in attrs.items() if not key.startswith(('merlin.','prov.'))}
        attrs=dict(first.get('attributes',{}));contract=numeric(attrs)
        if attrs.get('accumulator_init') or attrs.get('block_scale') is not None:return None
        if any(e not in ('bias_add','acc_scale','relu') for e in attrs.get('epilogue',())):return None
        if attrs.get('requant_shift') is not None:return None
        if attrs.get('output_dtype',dtype)!=dtype:return None
        inputs={rhs};outputs=[]
        for row in members:
            aa,bb,cc=(ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
            if row['rhs']!=rhs or aa['shape']!=[m,k] or bb['shape']!=[k,n] or cc['shape']!=[m,n]:return None
            if aa['dtype']!='i8' or bb['dtype']!='i8' or cc['dtype']!=dtype:return None
            if any(key in row and row[key]!=value for key,value in (('m',m),('k',k),('n',n))):return None
            if numeric(row.get('attributes',{}))!=contract:return None
            inputs.add(row['lhs']);outputs.append(row['dst'])
        if attrs.get('bias'):
            bias=attrs['bias'];bs=ctx.spec(bias)
            if bs['shape']!=[n] or bs['dtype']!='i32':return None
            inputs.add(bias)
        sources={v.get('source') for v in aliases.values() if isinstance(v,dict)}
        if len(set(outputs))!=len(outputs) or set(outputs)&inputs:return None
        if (inputs|set(outputs))&(set(aliases)|sources):return None
        # Prevalidate the actual movement geometry, tail/field limits, DMA
        # address projection and disjoint accumulator staging before emission.
        pack=plan_copy(m*k,'i8')
        scatter=plan_copy(m*n,dtype)
        fold=plan_head_fold(1,len(members),m,k,n).revalidate()
    except (ValueError,KeyError,TypeError,OverflowError):
        return None
    from .dense import lower_dense
    prefix=f'_shared_head_rows_{len(ctx.tensors)}'
    lhs_name=prefix+'_lhs';out_name=prefix+'_output'
    ctx.alloc(lhs_name,(len(members)*m,k),'i8')
    ctx.alloc(out_name,(len(members)*m,n),dtype)
    source_index=ctx.current_source_index
    def view(name,offset,elements):
        label=f'{prefix}_view_{len(ctx.tensors)}'
        ctx.register(label,ctx.ptr(name,offset),(1,elements),ctx.spec(name)['dtype'])
        return label
    def move(src,dst,plan):
        return emit_copy(ctx,src,dst,plan)
    packing=[];scattering=[]
    ctx.fence()
    for i,row in enumerate(members):
        packing.append(move(view(row['lhs'],0,m*k),view(lhs_name,i*m*k,m*k),pack))
    ctx.fence()
    lower_dense(ctx,lhs_name,rhs,out_name,m=len(members)*m,k=k,n=n,attributes=attrs)
    ctx.fence()
    for i,row in enumerate(members):
        ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index',source_index)
        scattering.append(move(view(out_name,i*m*n,m*n),view(row['dst'],0,m*n),scatter))
    ctx.fence();ctx.current_source_index=source_index
    report=dict(mechanism='source_shared_rhs_packed_rows',applied=True,
        members=len(members),source_products=[{'lhs':r['lhs'],'rhs':r['rhs'],'dst':r['dst']} for r in members],
        source_shape=[m,k,n],folded_shape=[len(members)*m,k,n],
        source_projection=fold,packing=packing,scattering=scattering,
        pack_plan_sha256=pack.kernel.digest(),scatter_plan_sha256=scatter.kernel.digest(),
        preparation_bytes=len(members)*m*k,
        scatter_bytes=len(members)*m*n*(1 if dtype=='i8' else 4),
        source_coverage='every member LHS packed exactly once; every original complete output scattered exactly once',
        completion='full DMA completion before dense; full dense completion before scatter; full completion before storage reuse',
        numeric='identical K reduction and declared dense readout; exact identity movement')
    ctx.metadata.setdefault('source_shared_rhs_packed_rows',[]).append(report)
    return report
