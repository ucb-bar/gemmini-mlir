"""Independent same-RHS products folded into useful dense rows.

The caller proves that all products commute. This adapter preserves each input
and output buffer and source readout; only immutable i8 row movement and the
matrix row axis change. All movement uses the existing terminal ISA encoder.
"""
from __future__ import annotations

from . import isa



def fold_traffic(m,k,n,products,output_dtype):
    """Separate profitability premise for the exact row concatenation.

    Every original thin product consumes its RHS once per primitive output-row
    tile. Folding replaces `products` such row tiles with ceil(products*M/DIM).
    Packing adds one read/write of each LHS byte; scattering adds one read/write
    of each already-produced output byte. Its actual two DMA commands per copy
    rectangle also must be covered by eliminated RHS DMA commands. This retains
    small or very narrow RHS matrices even when row padding alone improves.
    These are traffic/issue premises, not a predicted cycle count.
    """
    d=isa.DIM
    folded_tiles=(products*m+d-1)//d
    removed_row_tiles=products-folded_tiles
    width=1 if output_dtype=='i8' else 4
    saved_rhs_bytes=removed_row_tiles*k*n
    added_copy_bytes=2*products*m*(k+n*width)
    saved_rhs_dma_commands=removed_row_tiles*((k+d-1)//d)*((n+d-1)//d)
    added_copy_dma_commands=2*products*((m+d-1)//d)*(((k+d-1)//d)+((n+d-1)//d))
    return dict(saved_rhs_bytes=saved_rhs_bytes,added_copy_bytes=added_copy_bytes,
        saved_rhs_dma_commands=saved_rhs_dma_commands,added_copy_dma_commands=added_copy_dma_commands,
        admitted=(removed_row_tiles>0 and saved_rhs_bytes>=added_copy_bytes
                  and saved_rhs_dma_commands>=added_copy_dma_commands),
        meaning='source bytes and actual primitive DMA descriptors; not cycles')


def _admit(ctx, members):
    if not 2 <= len(members) <= 32:
        return None
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    names = {r[key] for r in members for key in ('lhs', 'rhs', 'dst')}
    alias_sources = {a.get('source') for a in aliases.values() if isinstance(a, dict)}
    if names & (set(aliases) | alias_sources):
        return None
    outputs = [r['dst'] for r in members]
    inputs = {r[key] for r in members for key in ('lhs', 'rhs')}
    if len(set(outputs)) != len(outputs) or set(outputs) & inputs:
        return None
    first = members[0]
    a,b,c = (ctx.spec(first[key]) for key in ('lhs','rhs','dst'))
    if len(a['shape']) != 2 or len(b['shape']) != 2:
        return None
    m,k = a['shape']; bk,n = b['shape']
    if (any(type(v) is not int or v <= 0 for v in (m,k,n))
            or k != bk or not m < isa.DIM):
        return None
    # Packing cannot reduce the dense tile count when concatenation still
    # occupies one primitive row tile per source product. Keep that geometry
    # on its incumbent: it has no padding/reload benefit to pay for movement.
    if (len(members)*m + isa.DIM-1)//isa.DIM >= len(members):
        return None
    if not fold_traffic(m,k,n,len(members),c['dtype'])['admitted']:
        return None
    attrs = dict(first.get('attributes', {}))
    if attrs.get('output_dtype',c['dtype']) != c['dtype']:
        return None
    # A row-dependent initializer or bias would require its own proved packing
    # boundary. Keep this alternative to exact zero-initialized products.
    if attrs.get('bias') or attrs.get('accumulator_init'):
        return None
    if any(e not in ('acc_scale','relu') for e in attrs.get('epilogue', ())):
        return None
    if attrs.get('requant_shift') is not None or any(x.startswith('pool') for x in attrs):
        return None
    def numerical(row):
        return {key:value for key,value in row.items()
                if not key.startswith('merlin.') and not key.startswith('prov.')}
    for row in members:
        aa,bb,cc = (ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
        if (row['rhs'] != first['rhs'] or aa != a or bb != b or cc != c
                or aa['dtype'] != 'i8' or bb['dtype'] != 'i8'
                or cc['dtype'] not in ('i8','i32') or cc['shape'] != [m,n]
                or numerical(row.get('attributes',{})) != numerical(attrs)
                or any(key in row and row[key] != value for key,value in (('m',m),('k',k),('n',n)))):
            return None
    # Whole source tensor footprints remain representable in signed indexing.
    if max(len(members)*m*k,len(members)*m*n)*4 >= 1 << 63:
        return None
    return m,k,n,c['dtype'],attrs


def _copy_rows(ctx, src, dst, *, rows, columns, src_row=0, dst_row=0, dtype):
    """Copy a dense row interval through one exclusive accumulator tile.

    Every pair covers rows [r,r+height) and columns [c,c+width). These disjoint
    rectangles partition the complete interval, including both tails. Real
    drains publish loads before readout and readout before local row reuse.
    """
    fb = ctx.fb
    width = 1 if dtype == 'i8' else 4
    ctx.fence()
    ctx.emit(*isa.config_ld(columns*width, shrink=dtype=='i8'))
    ctx.emit(*isa.config_st(columns*width, scale=1.0, activation=0))
    def tile(r, height, c, count):
        rr = fb.const(r) if isinstance(r,int) else r
        cc = fb.const(c) if isinstance(c,int) else c
        common = fb.add_i(fb.mul_i(rr,fb.const(columns)),cc)
        source = fb.add_i(common,fb.const(src_row*columns))
        target = fb.add_i(common,fb.const(dst_row*columns))
        ctx.emit(*isa.mvin(ctx.ptr(src,source),isa.acc_addr(0),height,count,fb=fb))
        ctx.fence()
        ctx.emit(*isa.mvout(ctx.ptr(dst,target),isa.acc_addr(0,full=dtype=='i32'),height,count,fb=fb))
        ctx.fence()
    def column_tiles(r,height):
        full,tail = divmod(columns,isa.DIM)
        if full:
            fb.for_loop(0,full,1,lambda c:tile(r,height,fb.mul_i(c,fb.const(isa.DIM)),isa.DIM),retain_loop=True)
        if tail: tile(r,height,full*isa.DIM,tail)
    full,tail = divmod(rows,isa.DIM)
    if full:
        fb.for_loop(0,full,1,lambda r:column_tiles(fb.mul_i(r,fb.const(isa.DIM)),isa.DIM),retain_loop=True)
    if tail:column_tiles(full*isa.DIM,tail)


def lower_folded_rhs_products(ctx, members):
    """Return None without mutation unless complete source products admit.

    Packing is required because independent source buffers need not be adjacent.
    The packed row i*M+r reads member i's row r and its output is scattered to
    that same member. K order, signed products, accumulator and readout stay in
    the shared dense lowerer, including its fact-derived storage selection.
    """
    members = tuple(members)
    try:
        admitted = _admit(ctx,members)
    except (KeyError,TypeError,ValueError,OverflowError):
        return None
    if admitted is None:return None
    from .dense import lower_dense
    m,k,n,dtype,attrs = admitted
    saved = ctx.current_source_index
    prefix = '_folded_rhs_' + str(len(ctx.tensors))
    lhs,dst = prefix+'_lhs',prefix+'_dst'
    total = len(members)*m
    ctx.alloc(lhs,(total,k),'i8')
    ctx.alloc(dst,(total,n),dtype)
    for i,row in enumerate(members):
        ctx.current_source_index = row.get('attributes',{}).get('merlin.source_op_index',saved)
        _copy_rows(ctx,row['lhs'],lhs,rows=m,columns=k,dst_row=i*m,dtype='i8')
    ctx.current_source_index = attrs.get('merlin.source_op_index',saved)
    dense = lower_dense(ctx,lhs,members[0]['rhs'],dst,m=total,k=k,n=n,attributes=attrs)
    ctx.fence()
    for i,row in enumerate(members):
        ctx.current_source_index = row.get('attributes',{}).get('merlin.source_op_index',saved)
        _copy_rows(ctx,dst,row['dst'],rows=m,columns=n,src_row=i*m,dtype=dtype)
    ctx.current_source_index = saved
    receipt = dict(mechanism='independent_rhs_row_fold',products=len(members),
        traffic_admission=fold_traffic(m,k,n,len(members),dtype),
        rhs=members[0]['rhs'],source_shape=[m,k,n],folded_shape=[total,k,n],
        packed_input_bytes=total*k,scattered_output_bytes=total*n*(1 if dtype=='i8' else 4),
        row_map='folded_row = member_index*M + original_row; inverse uses divmod(M)',
        coverage='all member rows and columns exactly once; unchanged K order and source readout',
        completion='each staging DMA load/readout fully drained before local reuse; dense full completion before scatter',
        dense=dense,rhs_source_loads=((k+15)//16)*((n+15)//16),
        members=[dict(source_index=r.get('attributes',{}).get('merlin.source_op_index'),
                      lhs=r['lhs'],dst=r['dst'],row_interval=[i*m,(i+1)*m]) for i,r in enumerate(members)])
    ctx.metadata.setdefault('folded_rhs_products',[]).append(receipt)
    return receipt
