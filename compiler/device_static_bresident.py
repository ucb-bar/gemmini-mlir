"""Literal-bound panels retaining an entire short-reduction B panel."""


def emit_static_product(a,b,y,M,K,N,width,scale,relu,bias,tag):
    from device_static_resident import emit_static_product as previous
    if any(type(v) is not int for v in (M,K,N)):
        raise ValueError('static product requires integer dimensions')
    if not (M>64 and 0<K<=2048 and N>=16):
        return previous(a,b,y,M,K,N,width,scale,relu,bias,tag)
    if width not in (1,4):
        raise ValueError('static product requires i8 or i32 output')
    kpad=(K+15)//16*16
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
    def columns(jb,nb):
        def load_b(kk,kr):
            return f'gm_mvin(((const signed char*)({b}))+(gm_u64)({kk})*{N}+({jb}),8192+(({kk})/16)*64,{kr},{nb},1);'
        def rows(ib,mb):
            pieces=[]
            if bias!='0':
                def bias_m(ii,mr):
                    def bias_n(jj,nr):
                        return f'gm_mvin(({bias})+({jb})+({jj}),0x80000000u+(({ii})/16)*64+({jj}),{mr},{nr},2);'
                    return partition(nb,16,'bias_n',bias_n)
                pieces += [partition(mb,16,'bias_m',bias_m)]
            def load_a_m(ii,mr):
                def load_a_k(kk,kc):
                    return f'gm_mvin(((const signed char*)({a}))+((gm_u64)({ib})+({ii}))*{K}+({kk}),(({ii})/16)*{kpad}+({kk}),{mr},{kc},0);'
                return partition(K,64,'load_a_k',load_a_k)
            pieces += [partition(mb,16,'load_a_m',load_a_m)]
            def compute_k(kk,kr):
                def compute_n(jj,nr):
                    def compute_m(ii,mr):
                        accum=f'((({kk})!=0) | {int(bias!="0")})'
                        return f'''gm_preload(8192+(({kk})/16)*64+({jj}),
              ({accum}?0xc0000000u:0x80000000u)+(({ii})/16)*64+({jj}),{kr},{nr},{mr});
            gm_compute((({ii})/16)*{kpad}+({kk}),{mr},{kr});'''
                    return partition(mb,16,'compute_m',compute_m)
                return partition(nb,16,'compute_n',compute_n)
            pieces += [partition(K,16,'compute_k',compute_k)]
            def store_m(ii,mr):
                if width==4:
                    def store_n(jj,nr):
                        return f'gm_mvout(((unsigned char*)({y}))+(((gm_u64)({ib})+({ii}))*{N}+({jb})+({jj}))*4,0xa0000000u+(({ii})/16)*64+({jj}),{mr},{nr});'
                    return partition(nb,16,'store_n',store_n)
                return f'gm_mvout(((unsigned char*)({y}))+((gm_u64)({ib})+({ii}))*{N}+({jb}),0x80000000u+(({ii})/16)*64,{mr},{nb});'
            pieces += [partition(mb,16,'store_m',store_m)]
            return '\n'.join(pieces)
        return '\n'.join([partition(K,16,'load_b_k',load_b),partition(M,64,'ib',rows)])
    load_bias_config = 'gm_config_ld(0,2,0);' if bias != '0' or K == 0 else ''
    body=[f'''{{
  gm_config_ex();
  gm_config_ld({K},0,0);
  gm_config_ld({N},1,0);
  {load_bias_config}
  gm_config_st({N*width},{scale},{int(bool(relu))});
  /* A occupies at most 4*Kpad rows; B occupies 4*ceil(K/16)*16 rows. */
  {partition(N,64,'jb',columns)}
  gm_fence();
}}''']
    return [],body
