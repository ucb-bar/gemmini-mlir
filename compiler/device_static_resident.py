"""Literal-bound static contraction with complete thin-M A residency."""


def emit_static_product(a,b,y,M,K,N,width,scale,relu,bias,tag):
    from device_static import emit_static_product as panel
    if any(type(v) is not int for v in (M,K,N)):
        raise ValueError('static product requires integer dimensions')
    if not (1<=M<=16 and N>64 and K>16 and ((K+15)//16)*M<=8192):
        return panel(a,b,y,M,K,N,width,scale,relu,bias,tag)
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
    def load_a(kk,kc):
        return f'gm_mvin(((const signed char*)({a}))+({kk}),(({kk})/16)*{M},{M},{kc},0);'
    def columns(jb,nb):
        pieces=[]
        if bias!='0':
            def load_bias(jj,nr):
                return f'gm_mvin(({bias})+({jb})+({jj}),0x80000000u+({jj}),{M},{nr},2);'
            pieces += [partition(nb,16,'bias_n',load_bias)]
        def reduction(kb,klen):
            def load_b(kk,kr):
                return f'gm_mvin(((const signed char*)({b}))+((gm_u64)({kb})+({kk}))*{N}+({jb}),8192+(({kk})/16)*64,{kr},{nb},1);'
            def compute_k(kk,kr):
                def compute_n(jj,nr):
                    accum=f'((({kb})!=0) | (({kk})!=0) | {int(bias!="0")})'
                    return f'''gm_preload(8192+(({kk})/16)*64+({jj}),
          ({accum}?0xc0000000u:0x80000000u)+({jj}),{kr},{nr},{M});
        gm_compute((({kb})+({kk}))/16*{M},{M},{kr});'''
                return partition(nb,16,'compute_n',compute_n)
            return '\n'.join([partition(klen,16,'load_b_k',load_b),partition(klen,16,'compute_k',compute_k)])
        pieces += [partition(K,256,'kb',reduction)]
        if width==4:
            def store(jj,nr):
                return f'gm_mvout(((unsigned char*)({y}))+((gm_u64)({jb})+({jj}))*4,0xa0000000u+({jj}),{M},{nr});'
            pieces += [partition(nb,16,'store_n',store)]
        else:
            pieces += [f'gm_mvout(((unsigned char*)({y}))+({jb}),0x80000000u,{M},{nb});']
        return '\n'.join(pieces)
    load_bias_config = 'gm_config_ld(0,2,0);' if bias != '0' or K == 0 else ''
    body=[f'''{{
  gm_config_ex();
  /* CONFIG_MVIN.state0 block_stride=M packs short A tiles into live rows. */
  GM_INSN(0,((gm_u64)0x3f800000<<32)|((gm_u64){M}<<16)|1,{K});
  gm_config_ld({N},1,0);
  {load_bias_config}
  gm_config_st({N*width},{scale},{int(bool(relu))});
  {partition(K,64,'load_a_k',load_a)}
  {partition(N,64,'jb',columns)}
  gm_fence();
}}''']
    return [],body
