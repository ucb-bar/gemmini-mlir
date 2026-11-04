"""Optional full-width dense product followed by device column readout.

The product uses the common shared dense schedule and terminal encoder. A
private i32 matrix preserves every accumulator bit between that product and
the strided column loads; scale, ReLU and saturation execute in Gemmini.
"""
from __future__ import annotations

from . import isa
from .dense_readout import store_tile
from .column_batches import emit_batched_column_readout


def lower_column_readout(ctx,lhs,rhs,dst,*,m,k,n,attributes):
    from .dense import lower_dense
    attrs=dict(attributes)
    ep=attrs.get('epilogue',[])
    if attrs.get('output_dtype',ctx.spec(dst)['dtype'])!='i8':
        raise ValueError('column reload readout requires an i8 destination')
    if 'maxpool' in ep:
        raise ValueError('column reload readout requires a separate spatial pooling schedule')
    if attrs.get('requant_shift') is not None and 'requant' in ep:
        raise ValueError('half-up requantization has no exact Gemmini readout contract')
    index=getattr(ctx,'_dense_column_scratch_count',0)
    ctx._dense_column_scratch_count=index+1
    raw='_dense_column_raw_'+str(index)
    ctx.alloc(raw,(m,n),'i32')
    product_attrs=dict(attrs)
    for name in ('acc_scale_by_column','acc_scale','relu','requant_shift'):
        product_attrs.pop(name,None)
    product_attrs['epilogue']=[stage for stage in ep if stage=='bias_add']
    product_attrs['output_dtype']='i32'
    product_attrs['dense_column_readout']='recompute'
    receipt=lower_dense(ctx,lhs,rhs,raw,m=m,k=k,n=n,attributes=product_attrs)
    batching=emit_batched_column_readout(ctx,raw,dst,m,n,attrs)
    if batching is not None:
        receipt=dict(receipt)
        receipt['column_readout']={'route':'raw_reload','raw_storage_bytes':m*n*4,
                                   'raw_reload_bytes':m*n*4,'output_bytes':m*n,
                                   'product_tile_columns':isa.DIM,'batching':batching}
        return receipt
    fb=ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(n*4,channel=2,shrink=False))

    def column(col):
        def tile(row,rows):
            offset=fb.add_i(fb.mul_i(row,fb.const(n)),col)
            ctx.emit(*isa.mvin(ctx.ptr(raw,offset),isa.acc_addr(0),rows,1,channel=2,fb=fb))
            ctx.fence()
            store_tile(ctx,dst,0,rows,1,n,row,col,attrs)
        complete=m//isa.DIM*isa.DIM
        if complete:
            fb.for_loop(0,complete,isa.DIM,lambda row:tile(row,isa.DIM),retain_loop=True)
        if complete<m:
            tile(fb.const(complete),m-complete)
    fb.for_loop(0,n,1,column,retain_loop=True)
    receipt=dict(receipt)
    receipt['column_readout']={'route':'raw_reload','raw_storage_bytes':m*n*4,
                               'raw_reload_bytes':m*n*4,'output_bytes':m*n,
                               'product_tile_columns':isa.DIM}
    return receipt
