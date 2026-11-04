"""Static packed-A residency for up to four M subtiles.

Each K tile occupies exactly M scratchpad rows, including a short final M tile.
A stays in rows0..ceil(K/16)*M-1; RHS K256/N64 panels use8192..9215.
"""


def emit_static_product(a,b,y,M,K,N,width,scale,relu,bias,tag):
    from device_static_reuse import emit_static_product as previous
    if any(type(v) is not int for v in (M,K,N)):
        raise ValueError('static product requires integer dimensions')
    if not (17<=M<=64 and N>64 and K>16 and ((K+15)//16)*M<=8192):
        return previous(a,b,y,M,K,N,width,scale,relu,bias,tag)
    if width not in (1,4):
        raise ValueError('static product requires i8 or i32 output')
    serial=iter(range(1000000))
    def partition(size,tile,stem,emit):
        full=size//tile*tile
        parts=[]
        if full:
            var=f'{tag}_{stem}_{next(serial)}'
            parts += [f'for(int {var}=0;{var}<{full};{var}+={tile}) {{',emit(var,tile),'}']
        if full<size:
            parts += [emit(str(full),size-full)]
        return '\n'.join(parts)
    def load_a_m(ii,mr):
        def load_a_k(kk,kc):
            return f'gm_mvin(((const signed char*)({a}))+(gm_u64)({ii})*{K}+({kk}),(({kk})/16)*{M}+({ii}),{mr},{kc},0);'
        return partition(K,64,'load_a_k',load_a_k)
    def columns(jb,nb):
        pieces=[]
        if bias!='0':
            def bias_m(ii,mr):
                def bias_n(jj,nr):
                    return f'gm_mvin(({bias})+({jb})+({jj}),0x80000000u+(({ii})/16)*64+({jj}),{mr},{nr},2);'
                return partition(nb,16,'bias_n',bias_n)
            pieces += [partition(M,16,'bias_m',bias_m)]
        def reduction(kb,klen):
            def load_b(kk,kr):
                return f'gm_mvin(((const signed char*)({b}))+((gm_u64)({kb})+({kk}))*{N}+({jb}),8192+(({kk})/16)*64,{kr},{nb},1);'
            def compute_k(kk,kr):
                def compute_n(jj,nr):
                    def compute_m(ii,mr,first):
                        accum=f'((({kb})!=0) | (({kk})!=0) | {int(bias!="0")})'
                        address=f'8192+(({kk})/16)*64+({jj})' if first else '0xffffffffu'
                        a_addr=f'((({kb})+({kk}))/16)*{M}+({ii})'
                        compute=(f'gm_compute({a_addr},{mr},{kr});' if first else
                            f'GM_INSN(5,gm_shape({a_addr},{mr},{kr}),gm_shape(0xffffffffu,16,16));')
                        return f'''gm_preload({address},
              ({accum}?0xc0000000u:0x80000000u)+(({ii})/16)*64+({jj}),{kr},{nr},{mr});
            {compute}'''
                    parts=[compute_m('0',16,True)]
                    full=M//16*16
                    if full>16:
                        ii=f'{tag}_compute_m_{next(serial)}'
                        parts += [f'for(int {ii}=16;{ii}<{full};{ii}+=16) {{',compute_m(ii,16,False),'}']
                    if full<M:
                        parts += [compute_m(str(full),M-full,False)]
                    return '\n'.join(parts)
                return partition(nb,16,'compute_n',compute_n)
            return '\n'.join([partition(klen,16,'load_b_k',load_b),partition(klen,16,'compute_k',compute_k)])
        pieces += [partition(K,256,'kb',reduction)]
        def store_m(ii,mr):
            if width==4:
                def store_n(jj,nr):
                    return f'gm_mvout(((unsigned char*)({y}))+((gm_u64)({ii})*{N}+({jb})+({jj}))*4,0xa0000000u+(({ii})/16)*64+({jj}),{mr},{nr});'
                return partition(nb,16,'store_n',store_n)
            return f'gm_mvout(((unsigned char*)({y}))+(gm_u64)({ii})*{N}+({jb}),0x80000000u+(({ii})/16)*64,{mr},{nb});'
        pieces += [partition(M,16,'store_m',store_m)]
        return '\n'.join(pieces)
    load_bias_config='gm_config_ld(0,2,0);' if bias!='0' else ''
    body=[f'''{{
  gm_config_ex();
  /* CONFIG_MVIN block_stride=M packs every K tile into M live rows. */
  GM_INSN(0,((gm_u64)0x3f800000<<32)|((gm_u64){M}<<16)|1,{K});
  gm_config_ld({N},1,0);
  {load_bias_config}
  gm_config_st({N*width},{scale},{int(bool(relu))});
  {partition(M,16,'load_a_m',load_a_m)}
  {partition(N,64,'jb',columns)}
  gm_fence();
}}''']
    return [],body
