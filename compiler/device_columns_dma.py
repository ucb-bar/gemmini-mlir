"""Optional per-column readout with all layout transport on device DMA.

The contraction uses accumulator rows0..63. Copy commands reserve rows512..527,
so preparing another K panel cannot overwrite a live partial sum.
"""


def runtime_header():
    return ''


def _copy_rows(src, dst, count, source_stride, dest_stride, width, tag):
    """Move up to sixteen strided scalar lanes per exact DMA copy pair."""
    full = '0xa0000200u' if width == 4 else '0x80000200u'
    return f'''gm_config_ld(({source_stride})*{width},0,{int(width == 1)});
      gm_config_st(({dest_stride})*{width},1.0f,0);
      for(int {tag}=0;{tag}<({count});{tag}+=16) {{
        int nr=gm_min16(({count})-{tag});
        gm_mvin((const void*)((gm_u64)({src})+(gm_u64){tag}*({source_stride})*{width}),0x80000200u,nr,1,0);
        gm_mvout((void*)((gm_u64)({dst})+(gm_u64){tag}*({dest_stride})*{width}),{full},nr,1);
      }}
      gm_fence();'''


def emit_column_product(a,b,y,m,k,n,attrs,cb,names,tag,m_capacity=None,*,force=False):
    # Share semantic admission and profitability with the tested implementation;
    # generate the DMA body independently, without rewriting generated C text.
    from device_columns import emit_column_product as admit
    selected = admit(a,b,y,m,k,n,attrs,cb,names,tag,m_capacity,force=force)
    if selected is None:
        return None
    from conv import _epilogue, _f32
    _, _, relu, bias, scales = _epilogue(attrs,cb,names,n)
    p=tag+'_column_dma'
    lhs,rhs,out,bp,gains=(p+x for x in ('_lhs','_rhs','_out','_bias','_gains'))
    glob=[f'static signed char {lhs}[16*64] __attribute__((aligned(64)));',
          f'static signed char {rhs}[64*64] __attribute__((aligned(64)));',
          f'static signed char {out}[16*64] __attribute__((aligned(64)));',
          f'static const float {gains}[{n}] = {{{",".join(_f32(x) for x in scales)}}};']
    bias_init=''
    if bias!='0':
        glob.append(f'static int {bp}[16*64] __attribute__((aligned(64)));')
        bias_copy=_copy_rows(f'({bias})+nb+c',f'{bp}+c*64','mr',0,1,4,'br')
        bias_init=f'''for(int c=0;c<nc;c++) {{ {bias_copy} }}
      gm_config_ld(64*4,2,0);
      for(int br=0;br<mr;br+=16) {{
        int bn=gm_min16(mr-br);
        gm_mvin({bp}+br,0x80000000u+br,nc,bn,2);
      }}'''
    lhs_copy=_copy_rows(f'((const signed char*)({b}))+(gm_u64)kb*{n}+nb+c',f'{lhs}+c*64','kr',n,1,1,'lq')
    rhs_copy=_copy_rows(f'((const signed char*)({a}))+(gm_u64)rb*{k}+kb+q',f'{rhs}+q*64','mr',k,1,1,'rr')
    out_copy=_copy_rows(f'{out}+c*64',f'((signed char*)({y}))+(gm_u64)rb*{n}+nb+c','mr',1,n,1,'orow')
    body=[f'''{{
  gm_config_ex();
  for(int nb=0;nb<{n};nb+=16) {{
    int nc=gm_min16({n}-nb);
    for(int rb=0;rb<({m});rb+=64) {{
      int mr=({m})-rb;if(mr>64)mr=64;
      {bias_init}
      for(int kb=0;kb<{k};kb+=64) {{
        int kr={k}-kb;if(kr>64)kr=64;
        for(int c=0;c<nc;c++) {{ {lhs_copy} }}
        for(int q=0;q<kr;q++) {{ {rhs_copy} }}
        gm_config_ld(64,0,0);
        gm_config_ld(64,1,0);
        gm_mvin({lhs},0,nc,kr,0);
        for(int q=0;q<kr;q+=16) {{
          int qn=gm_min16(kr-q);
          gm_mvin({rhs}+q*64,4096+(q/16)*64,qn,mr,1);
        }}
        for(int q=0;q<kr;q+=16) {{
          int qn=gm_min16(kr-q);
          for(int r=0;r<mr;r+=16) {{
            int rn=gm_min16(mr-r);
            gm_u32 ca=((kb || q || {int(bias!='0')})?0xc0000000u:0x80000000u)+r;
            gm_preload(4096+(q/16)*64+r,ca,qn,rn,nc);
            gm_compute(q,nc,qn);
          }}
        }}
        gm_fence();
      }}
      for(int r=0;r<mr;r+=16) {{
        int rn=gm_min16(mr-r);
        for(int c=0;c<nc;c++) {{
          gm_config_st(64,{gains}[nb+c],{relu});
          gm_mvout({out}+c*64+r,0x80000000u+r+c,1,rn);
        }}
      }}
      gm_fence();
      for(int c=0;c<nc;c++) {{ {out_copy} }}
    }}
  }}
}}''']
    return glob,body
