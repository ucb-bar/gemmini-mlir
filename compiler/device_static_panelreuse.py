"""Literal-bound ordinary panels with WS weights reused across M subtiles.

The first M subtile loads actual B and flips banks; later subtiles use garbage
preloads for C tags and funct5 to retain the active WS weights. Public RTL:
ExecuteController.scala90/94/426/796/833, MeshWithDelays.scala116, PE.scala119-130.
"""


def emit_static_product(a,b,y,M,K,N,width,scale,relu,bias,tag):
    if any(type(v) is not int or v < 0 for v in (M,K,N)):
        raise ValueError('static product requires nonnegative integer dimensions')
    if width not in (1,4):
        raise ValueError('static product requires i8 or i32 output')
    if M<=16 or K==0:
        from device_static_resident import emit_static_product as thin
        return thin(a,b,y,M,K,N,width,scale,relu,bias,tag)
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
    def mm(ib,mb):
        def nn(jb,nb):
            pieces=[]
            if bias!='0' or K==0:
                def bias_m(ii,mr):
                    def bias_n(jj,nr):
                        src=f'({bias})+({jb})+({jj})' if bias!='0' else '(const void*)0'
                        return f'gm_mvin({src},0x80000000u+(({ii})/16)*64+({jj}),{mr},{nr},2);'
                    return partition(nb,16,'bias_n',bias_n)
                pieces += [partition(mb,16,'bias_m',bias_m)]
            def reduction(kb,klen):
                def load_a(ii,mr):
                    def load_k(kk,kc):
                        return f'gm_mvin(((const signed char*)({a}))+((gm_u64)({ib})+({ii}))*{K}+({kb})+({kk}),(({ii})/16)*256+({kk}),{mr},{kc},0);'
                    return partition(klen,64,'load_a_k',load_k)
                def load_b(kk,kr):
                    return f'gm_mvin(((const signed char*)({b}))+((gm_u64)({kb})+({kk}))*{N}+({jb}),8192+(({kk})/16)*64,{kr},{nb},1);'
                def compute_k(kk,kr):
                    def compute_n(jj,nr):
                        def compute_m(ii,mr,first):
                            accum=f'((({kb})!=0) | (({kk})!=0) | {int(bias!="0")})'
                            address=f'8192+(({kk})/16)*64+({jj})' if first else '0xffffffffu'
                            compute=(f'gm_compute((({ii})/16)*256+({kk}),{mr},{kr});' if first else
                                f'GM_INSN(5,gm_shape((({ii})/16)*256+({kk}),{mr},{kr}),gm_shape(0xffffffffu,16,16));')
                            return f'''gm_preload({address},
                ({accum}?0xc0000000u:0x80000000u)+(({ii})/16)*64+({jj}),{kr},{nr},{mr});
              {compute}'''
                        parts=[compute_m('0',min(16,mb),True)]
                        full=mb//16*16
                        if full>16:
                            ii=f'{tag}_compute_m_{next(serial)}'
                            parts += [f'for(int {ii}=16;{ii}<{full};{ii}+=16) {{',
                                      compute_m(ii,16,False),'}']
                        if full and full<mb:
                            parts += [compute_m(str(full),mb-full,False)]
                        return '\n'.join(parts)
                    return partition(nb,16,'compute_n',compute_n)
                return '\n'.join([partition(mb,16,'load_a_m',load_a),partition(klen,16,'load_b_k',load_b),partition(klen,16,'compute_k',compute_k)])
            pieces += [partition(K,256,'kb',reduction)]
            def store_m(ii,mr):
                if width==4:
                    def store_n(jj,nr):
                        return f'gm_mvout(((unsigned char*)({y}))+(((gm_u64)({ib})+({ii}))*{N}+({jb})+({jj}))*4,0xa0000000u+(({ii})/16)*64+({jj}),{mr},{nr});'
                    return partition(nb,16,'store_n',store_n)
                return f'gm_mvout(((unsigned char*)({y}))+((gm_u64)({ib})+({ii}))*{N}+({jb}),0x80000000u+(({ii})/16)*64,{mr},{nb});'
            pieces += [partition(mb,16,'store_m',store_m)]
            return '\n'.join(pieces)
        return partition(N,64,'jb',nn)
    load_bias_config = 'gm_config_ld(0,2,0);' if bias != '0' or K == 0 else ''
    body=[f'''{{
  gm_config_ex();
  gm_config_ld({K},0,0);
  gm_config_ld({N},1,0);
  {load_bias_config}
  gm_config_st({N*width},{scale},{int(bool(relu))});
  {partition(M,64,'ib',mm)}
  gm_fence();
}}''']
    return [],body
