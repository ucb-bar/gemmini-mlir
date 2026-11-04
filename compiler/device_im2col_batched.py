"""Batch exact im2col byte DMA over adjacent output-x coordinates."""
from __future__ import annotations
import re
from device_im2col import emit_im2col as _scalar_rows


def emit_im2col(matrix, inp, shape, kernel, stride, dilation, padding,
                oh, ow, row_base, row_count, tag, *, row_mod_ow):
    """Materialize one block with invariant ``row_base % ow == row_mod_ow``.

    All schedule bounds are literals. Each load/store covers <=16 output
    positions and <=112 channels, occupying accumulator rows0..111. The
    caller owns the destination and must establish the phase invariant.
    """
    # Reuse the baseline geometry validation and its general fallback.
    fallback=_scalar_rows(matrix,inp,shape,kernel,stride,dilation,padding,
                          oh,ow,row_base,row_count,tag)
    if type(row_mod_ow) is not int or not 0<=row_mod_ow<ow:
        raise ValueError('DMA im2col phase must be an integer in [0, ow)')
    if type(row_base) is int and row_base%ow!=row_mod_ow:
        raise ValueError('DMA im2col row_base disagrees with promised phase')
    if ow==1:
        return fallback
    batch,h,w,ci=shape;kh,kw,_,_=kernel;sh,sw=stride;dh,dw=dilation
    pt,pl,pb,pr=padding;k=kh*kw*ci
    tag='batched_im2col_'+re.sub(r'[^A-Za-z0-9_]','_',str(tag))
    body=['gm_config_ex();',f'gm_config_ld({sw*ci},2,1);',f'gm_config_st({k},1.0f,0);']
    serial=0
    def row_piece(start,length,xstart):
        """One fixed x range of a spatial output row, start a C expression."""
        nonlocal serial
        serial+=1;t=f'{tag}_p{serial}'
        flat,bn,oy,iy=(t+'_'+n for n in ('flat','bn','oy','iy'))
        lines=['{',f'int {flat}=({row_base})+({start});',
               f'int {bn}={flat}/{oh*ow}, {oy}=({flat}/{ow})%{oh};']
        for ky in range(kh):
            for kx in range(kw):
                q=f'{t}_k{ky}_{kx}'
                # x*sw-pl+kx*dw is in [0,w) precisely on this interval.
                lower=max(xstart,min(xstart+length,-(-(pl-kx*dw)//sw)))
                upper=max(lower,min(xstart+length,(w-1+pl-kx*dw)//sw+1))
                boundaries=[(xstart,lower,False),(lower,upper,True),(upper,xstart+length,False)]
                lines += ['{',f'int {iy}={oy}*{sh}-{pt}+{ky*dh};']
                for seg,(lo,hi,xvalid) in enumerate(boundaries):
                    if lo>=hi:continue
                    # Split a possibly partial x run into full16 plus a
                    # literal tail, so row counts also remain static.
                    count=hi-lo;full=count//16*16
                    segments=[]
                    if full:segments.append((f'{q}_s{seg}_x',lo,lo+full,16,True))
                    if count%16:segments.append((str(lo+full),0,0,count%16,False))
                    for xvar,begin,end,rows,loop in segments:
                        if loop:lines.append(f'for(int {xvar}={begin};{xvar}<{end};{xvar}+=16) {{')
                        else:lines.append('{')
                        valid=f'{q}_v';offset=f'{q}_off';address=f'{q}_addr'
                        lines.append(f'gm_u64 {valid}=((unsigned){iy}<{h}) & {int(xvalid)}u;')
                        channel=f'{q}_c';fullci=ci//112*112
                        channel_parts=[]
                        if fullci:channel_parts.append((channel,112,True))
                        if ci%112:channel_parts.append((str(fullci),ci%112,False))
                        for chan,cols,cloop in channel_parts:
                            if cloop:lines.append(f'for(int {chan}=0;{chan}<{fullci};{chan}+=112) {{')
                            else:lines.append('{')
                            lines += [f'gm_u64 {offset}=(((gm_u64){bn}*{h}+(gm_u64){iy})*{w}+(gm_u64)({xvar})*{sw}-{pl}+{kx*dw})*{ci}+({chan});',
                                      f'gm_u64 {address}=((gm_u64)({inp})+{offset})*{valid};',
                                      f'gm_mvin((const void*){address},0x80000000u,{rows},{cols},2);',
                                      f'gm_mvout((void*)((gm_u64)({matrix})+((gm_u64)({start})+({xvar})-{xstart})*{k}+{(ky*kw+kx)*ci}+({chan})),0x80000000u,{rows},{cols});','}']
                        lines.append('}')
                lines.append('}')
        lines.append('}')
        return lines
    # Partial first row, a loop over complete rows, and partial final row.
    prefix=min(row_count,ow-row_mod_ow) if row_mod_ow else 0
    if prefix:body += row_piece('0',prefix,row_mod_ow)
    remaining=row_count-prefix
    full_rows=remaining//ow*ow
    if full_rows:
        r=tag+'_r'
        body += [f'for(int {r}={prefix};{r}<{prefix+full_rows};{r}+={ow}) {{']
        body += row_piece(r,ow,0)+['}']
    tail=remaining-full_rows
    if tail:body += row_piece(str(prefix+full_rows),tail,0)
    body.append('gm_fence();')
    return [],body
