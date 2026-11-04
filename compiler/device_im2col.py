"""Materialize NHWC im2col with exact signed-byte accelerator DMA copies."""
from __future__ import annotations
import re


def emit_im2col(matrix, inp, shape, kernel, stride, dilation, padding,
                oh, ow, row_base, row_count, tag):
    """Return (globals, statements) for one literal-size im2col row block.

    ``matrix`` and ``inp`` are C pointer expressions; ``row_base`` is a C
    integer expression or integer. The caller owns matrix[row_count*K].
    Accumulator rows 0,16,...,96 are temporary and fenced before returning
    to product. Shrink=1 i8 transfers support up to 112 columns; their
    16-column blocks advance by the configured block stride of 16 rows.
    No host tensor values, including padding bytes, are read or written.
    """
    if type(row_count) is not int or row_count<=0:
        raise ValueError('DMA im2col row_count must be a positive literal integer')
    batch,h,w,ci=map(int,shape)
    kh,kw,kci,co=map(int,kernel)
    sh,sw=map(int,stride);dh,dw=map(int,dilation)
    pt,pl,pb,pr=map(int,padding)
    if min(batch,h,w,ci,kh,kw,co,sh,sw,dh,dw,oh,ow,row_count)<=0 or ci!=kci or min(pt,pl,pb,pr)<0:
        raise ValueError('invalid DMA im2col geometry')
    if oh!=(h+pt+pb-(dh*(kh-1)+1))//sh+1 or ow!=(w+pl+pr-(dw*(kw-1)+1))//sw+1:
        raise ValueError('DMA im2col output geometry mismatch')
    tag='dma_im2col_'+re.sub(r'[^A-Za-z0-9_]','_',str(tag))
    k=kh*kw*ci
    row,ky,kx,c=(tag+'_'+s for s in ('row','ky','kx','c'))
    flat,bn,oy,ox,iy,ix,pos,valid,offset,address=(tag+'_'+s for s in ('flat','bn','oy','ox','iy','ix','pos','valid','offset','address'))
    body=['gm_config_ex();','gm_config_ld(0,2,1);',f'gm_config_st({k},1.0f,0);',
          f'for(int {row}=0;{row}<{row_count};++{row}) {{',
          f'int {flat}=({row_base})+{row};',
          f'int {bn}={flat}/{oh*ow}, {oy}=({flat}/{ow})%{oh}, {ox}={flat}%{ow};',
          f'for(int {ky}=0;{ky}<{kh};++{ky}) for(int {kx}=0;{kx}<{kw};++{kx}) {{',
          f'int {iy}={oy}*{sh}-{pt}+{ky}*{dh}, {ix}={ox}*{sw}-{pl}+{kx}*{dw};',
          f'int {pos}={row}*{k}+({ky}*{kw}+{kx})*{ci};',
          f'gm_u64 {valid}=((unsigned){iy}<{h}) & ((unsigned){ix}<{w});']
    # LoadController.scala72/109 defines rs1==0 as all_zeros. Address
    # arithmetic is unsigned, so invalid spatial coordinates never create a
    # C out-of-bounds pointer or execute a scalar tensor read.
    def copy(channel,cols):
        return [f'gm_u64 {offset}=(((gm_u64){bn}*{h}+(gm_u64){iy})*{w}+(gm_u64){ix})*{ci}+({channel});',
                f'gm_u64 {address}=((gm_u64)({inp})+{offset})*{valid};',
                f'gm_mvin((const void*){address},0x80000000u,1,{cols},2);',
                f'gm_mvout((void*)((gm_u64)({matrix})+(gm_u64){pos}+({channel})),0x80000000u,1,{cols});']
    channel_tile=112
    full=ci//channel_tile*channel_tile
    if full:
        body += [f'for(int {c}=0;{c}<{full};{c}+={channel_tile}) {{']+copy(c,channel_tile)+['}']
    if ci%channel_tile:
        body += ['{']+copy(str(full),ci%channel_tile)+['}']
    body += ['}','}','gm_fence();']
    return [],body
