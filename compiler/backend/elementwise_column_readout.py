"""Exact literal column scales through the common accumulator terminal."""
from __future__ import annotations
import math
import struct
from . import isa


def lower_column_readout(ctx, operands, attributes):
    from .elementwise import ElementwiseError, _shape, _dtype
    from .dense_readout import store_tile
    from .source_groups import _path
    src,dst=operands['src'],operands['dst']
    shape=_shape(ctx,src)
    if not 2<=len(shape)<=4 or _shape(ctx,dst)!=shape:
        raise ElementwiseError('column readout requires identical contiguous rank2..4 shapes')
    if _dtype(ctx,src)!='i32' or _dtype(ctx,dst)!='i8':
        raise ElementwiseError('column readout requires signed i32 to i8')
    allowed={'acc_scale_by_column','epilogue','output_dtype','merlin.global_task','merlin.source_op_index'}
    if any(k not in allowed and not k.startswith('prov.') for k in attributes):
        raise ElementwiseError('column readout has unsupported numeric attributes')
    ep=attributes.get('epilogue')
    if ep not in (['acc_scale'],['acc_scale','relu']) or attributes.get('output_dtype','i8')!='i8':
        raise ElementwiseError('column readout requires scale then optional relu and i8 saturation')
    scales=attributes.get('acc_scale_by_column')
    n=shape[-1]
    if not isinstance(scales,(list,tuple)) or len(scales)!=n:
        raise ElementwiseError('column readout requires one literal f32 scale per lastaxis column')
    bits=[]
    for scale in scales:
        if type(scale) not in (int,float) or not math.isfinite(scale) or scale<0:
            raise ElementwiseError('column scale must be finite nonnegative f32')
        try:
            word=struct.pack('<f',scale)
        except (OverflowError,struct.error) as exc:
            raise ElementwiseError('column scale is not a finite f32') from exc
        if not math.isfinite(struct.unpack('<f',word)[0]) or struct.unpack('<f',word)[0]!=scale:
            raise ElementwiseError('column scale must preserve its original f32 value')
        bits.append(word)
    path=_path()
    acc={s['kind'] for s in path.get('accumulator_stages',()) if s.get('dtype')=='i32'}
    readout=path.get('readout',())
    scale_stages=[s for s in readout if s.get('kind')=='scale' and s.get('dtype')=='f32'
                  and s.get('rounding')=='half_even' and s.get('granularity')=='per_command'
                  and type(s.get('columns_per_scale')) is int and s['columns_per_scale']>0]
    if (path.get('accumulator',{}).get('dtype')!='i32' or 'preload' not in acc
            or not any(s.get('kind')=='convert' and s.get('to')=='f32' and s.get('rounding')=='half_even' for s in readout)
            or not scale_stages
            or not any(s.get('kind')=='requant' and s.get('to')=='i8' and s.get('rounding')=='half_even' and s.get('clamp')==[-128,127] for s in readout)
            or ('relu' in ep and not any(s.get('kind')=='activation' and s.get('function')=='relu' and s.get('domain')=='accumulator' for s in readout))):
        raise ElementwiseError('selected output path does not admit exact column readout')
    aliases=getattr(ctx,'cb',{}).get('params',{}).get('tensor_aliases',{})
    def root(name):
        seen=set()
        while name in aliases:
            if name in seen:raise ElementwiseError('column readout rejects cyclic tensor aliases')
            seen.add(name);name=aliases[name]['source']
        return name
    if root(src)==root(dst):
        raise ElementwiseError('column readout requires fresh nonalias output')
    # Literal bit runs preserve even the sign of a zero scale. Only leading
    # axes flatten: last-axis column identities and row strides stay intact.
    max_width=min(16,scale_stages[0]['columns_per_scale'])
    runs=[];col=0
    while col<n:
        end=col+1
        while end<n and end-col<max_width and bits[end]==bits[col]:end+=1
        runs.append((col,end-col,scales[col]));col=end
    m=math.prod(shape[:-1]);fb=ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(n*4,channel=0,shrink=False))
    def row_tile(row,rows):
        row=fb.const(row) if isinstance(row,int) else row
        for col,width,scale in runs:
            offset=fb.add_i(fb.mul_i(row,fb.const(n)),fb.const(col))
            ctx.emit(*isa.mvin(ctx.ptr(src,offset),isa.acc_addr(0),rows=rows,cols=width,channel=0))
            store_tile(ctx,dst,0,rows,width,n,row,col,
                       {'acc_scale':scale,'epilogue':ep,'output_dtype':'i8'})
    full=m//16*16
    if full:fb.for_loop(0,full,16,lambda row:row_tile(row,16),retain_loop=True)
    if m%16:row_tile(full,m%16)
    ctx.fence()
