"""Gemmini StoreController maxpool over spatially retained accumulators.

Pooling is a device readout. Source bias is seeded before contraction, followed
by configured f32 narrowing/activation and the store path's windowed maximum.
"""
from isa_encoder import (ACC_ADDR, ACC_ACCUMULATE, GARBAGE_ADDR,
                         config_ex, config_load, config_store)


def pool_geometry(M, attrs):
    for key in ('pool_in_dims', 'pool_size', 'pool_stride'):
        if key not in attrs:
            raise ValueError('maxpool requires source ' + key)
    h, w = attrs['pool_in_dims']
    ph, pw = attrs['pool_size']
    sh, sw = attrs['pool_stride']
    pt, pl, pb, pr = attrs.get('pool_padding', [0,0,0,0])
    if min(h,w,ph,pw,sh,sw) <= 0 or min(pt,pl,pb,pr) < 0:
        raise ValueError('invalid source maxpool geometry')
    if M % (h*w):
        raise ValueError('pool input rows do not form complete source images')
    if ph != pw or sh != sw or ph > 3 or sh > 3:
        raise ValueError('store pooling requires square size/stride fitting its two-bit fields')
    if max(pt,pl,pb,pr) > 3:
        raise ValueError('store pooling padding exceeds its two-bit fields')
    if any((pt,pl,pb,pr)) and not (attrs.get('pool_pad_value') == 0 or ('relu' in attrs.get('epilogue',[]) and attrs.get('pool_pad_value',1) <= 0)):
        raise ValueError('store pooling padding contributes zero; source must explicitly agree')
    oh = (h+pt+pb-ph)//sh+1
    ow = (w+pl+pr-pw)//sw+1
    if ph-pt <= 0 or pw-pl <= 0 or (oh-1)*sh-pt >= h or (ow-1)*sw-pl >= w:
        raise ValueError('pooling windows containing only padding need a separate device fill')
    if min(oh,ow) <= 0 or max(h,w,oh,ow) > 255:
        raise ValueError('pool spatial extents do not fit nonzero eight-bit config fields')
    return dict(h=h,w=w,size=ph,stride=sh,pt=pt,pl=pl,oh=oh,ow=ow,batches=M//(h*w))


def config_pool(stride_bytes, attrs, g):
    rs1, rs2 = config_store(stride_bytes, scale=attrs.get('acc_scale',1.0),
                            activation=int('relu' in attrs.get('epilogue',[])))
    # Public ConfigMvoutRs1 register layout from rtl_facts/facts.json.
    rs1 |= g['stride'] << 4 | g['size'] << 6 | g['pt'] << 8 | g['pl'] << 10
    rs1 |= g['ow'] << 24 | g['oh'] << 32 | g['ow'] << 40
    rs1 |= g['h'] << 48 | g['w'] << 56
    return rs1, rs2


def emit_pooled_matmul(e,A,B,C,M,K,N,attrs,bias=None,lhs_stride=None,
                       rhs_stride=None,out_stride=None,rhs_resident=None,compute_provenance=None):
    from matmul_lowering import packed, extent, index2
    compute_provenance=compute_provenance or {}
    stages=attrs.get('epilogue',[])
    if not stages or stages[-1] != 'maxpool':
        raise ValueError('device pooling requires maxpool as final readout stage')
    if set(stages)-{'bias','bias_add','acc_scale','relu','maxpool'}:
        raise ValueError('unsupported stage before maxpool')
    if attrs.get('output_dtype','i32') != 'i8':
        raise ValueError('StoreController pooling requires narrowed i8 readout')
    if ('bias' in stages or 'bias_add' in stages) and bias is None:
        raise ValueError('source pooled bias operand is missing')
    if 'acc_scale_by_column' in attrs:
        raise ValueError('pooled column scales require independent column scheduling')
    g=pool_geometry(M,attrs)
    lhs_stride=K if lhs_stride is None else lhs_stride
    rhs_stride=N if rhs_stride is None else rhs_stride
    out_stride=N if out_stride is None else out_stride
    b=e.b
    if g['h']*g['w'] > 1024:
        # Keep only complete source-row bands resident. Adjacent output bands
        # recompute the halo required by their pooling windows, preserving both
        # source indexing and the configured output's global DRAM row pitch.
        ph, pw = attrs['pool_size']
        sh, sw = attrs['pool_stride']
        pt, pl, pb, pr = attrs.get('pool_padding',[0,0,0,0])
        max_input_rows = 1024//g['w']
        output_band = (max_input_rows-ph)//sh+1
        if output_band < 1:
            raise ValueError('one pooling source window exceeds accumulator row capacity')
        def one_batch(batch_id):
            for first_out in range(0,g['oh'],output_band):
                last_out=min(g['oh'],first_out+output_band)
                first_src=first_out*sh-pt
                last_src=(last_out-1)*sh-pt+ph
                lo,hi=max(0,first_src),min(g['h'],last_src)
                band=dict(attrs)
                band['pool_in_dims']=[hi-lo,g['w']]
                band['pool_padding']=[lo-first_src,pl,last_src-hi,pr]
                abase=b.add_i(b.mul_i(batch_id,e.value(g['h']*g['w'])),e.value(lo*g['w']))
                cbase=b.add_i(b.mul_i(batch_id,e.value(g['oh']*g['ow'])),e.value(first_out*g['ow']))
                emit_pooled_matmul(e,e.ptr(A,b.mul_i(abase,e.value(lhs_stride))),B,
                    e.ptr(C,b.mul_i(cbase,e.value(out_stride)),'i8'),(hi-lo)*g['w'],K,N,
                    band,bias=bias,lhs_stride=lhs_stride,rhs_stride=rhs_stride,
                    out_stride=out_stride,rhs_resident=rhs_resident,compute_provenance=compute_provenance)
        e.for_loop(0,g['batches'],1,one_batch)
        return
    e.insn('config',*config_ex(dataflow=1))
    e.insn('config',*config_load(lhs_stride,state_id=0))
    e.insn('config',*config_load(rhs_stride,state_id=1))
    e.insn('config',*config_load(0,state_id=2))
    def batch(batch_id):
        input_base=b.mul_i(batch_id,e.value(g['h']*g['w']))
        output_base=b.mul_i(batch_id,e.value(g['oh']*g['ow']))
        def columns(n):
            nn=extent(e,N,n)
            def rows(m):
                mm=extent(e,g['h']*g['w'],m)
                acc=b.add_i(e.value(ACC_ADDR),m)
                bias_addr=0 if bias is None else e.address(e.ptr(bias,n,'i32'))
                e.insn('mvin3',bias_addr,packed(e,acc,mm,nn))
                def reduction(k):
                    kk=extent(e,K,k)
                    lhsrow=b.add_i(input_base,m)
                    ap=e.ptr(A,index2(e,lhsrow,lhs_stride,k))
                    e.insn('mvin',e.address(ap),packed(e,0,mm,kk))
                    if rhs_resident is None:
                        local_b=16
                        bp=e.ptr(B,index2(e,k,rhs_stride,n))
                        e.insn('mvin2',e.address(bp),packed(e,local_b,kk,nn))
                    else:
                        base,nt=rhs_resident
                        tile=b.add_i(b.mul_i(b.udiv_i(k,e.value(16)),e.value(nt)),b.udiv_i(n,e.value(16)))
                        local_b=b.add_i(e.value(base),b.mul_i(tile,e.value(16)))
                    e.insn('preload',packed(e,local_b,kk,nn),packed(e,b.or_i(acc,e.value(ACC_ACCUMULATE)),mm,nn),**compute_provenance)
                    e.insn('compute',packed(e,0,mm,kk),packed(e,GARBAGE_ADDR,16,16),**compute_provenance)
                e.for_loop(0,K,16,reduction)
            e.for_loop(0,g['h']*g['w'],16,rows)
            e.insn('config',*config_pool(out_stride,attrs,g))
            dst=e.ptr(C,index2(e,output_base,out_stride,n),'i8')
            # rows=0 selects StoreController's configured pooling traversal;
            # porows/pocols supply the number of destination spatial positions.
            e.insn('mvout',e.address(dst),packed(e,ACC_ADDR,0,nn))
            e.fence()
        e.for_loop(0,N,16,columns)
    e.for_loop(0,g['batches'],1,batch)
    # No configured pooling leaks into following readouts.
    e.insn('config',*config_store(out_stride))
    e.fence()
