"""Source-correct contraction preparation using admitted host indexing IR.

Host code computes addresses and loop bounds only. Tensor values pass through
ordinary device DMA, the Gemmini mesh and accumulator readout.
"""
from __future__ import annotations



def conv_geometry(ifm_shape, attrs):
    shape = tuple(int(x) for x in ifm_shape)
    if len(shape) != 4 or any(x <= 0 for x in shape):
        raise ValueError('CONV2D requires a positive static NHWC shape')
    if attrs.get('layout', 'nhwc') != 'nhwc' or attrs.get('groups', 1) != 1:
        raise ValueError('CONV2D supports source NHWC ungrouped semantics')
    kernel = tuple(int(x) for x in attrs['kernel'])
    if len(kernel) != 4 or min(kernel) <= 0:
        raise ValueError('CONV2D requires [kh,kw,ci,co] kernel')
    kh, kw, ci, co = kernel
    batch, h, w, channels = shape
    if ci != channels:
        raise ValueError('CONV2D input channels differ from kernel')
    sh, sw = tuple(attrs.get('stride', [1, 1]))
    pt, pl, pb, pr = tuple(attrs.get('padding', [0, 0, 0, 0]))
    dh, dw = tuple(attrs.get('dilation', [1, 1]))
    if min(sh, sw, dh, dw) <= 0 or min(pt, pl, pb, pr) < 0:
        raise ValueError('CONV2D stride/dilation must be positive and padding nonnegative')
    oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
    ow = (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
    if min(oh, ow) <= 0:
        raise ValueError('CONV2D has no output positions')
    return dict(batch=batch, h=h, w=w, ci=ci, co=co, kh=kh, kw=kw,
                sh=sh, sw=sw, pt=pt, pl=pl, dh=dh, dw=dw,
                oh=oh, ow=ow, m=batch*oh*ow, k=kh*kw*ci, n=co)


def emit_device_im2col(e, ifm, g):
    """Materialize [M,K] with matrix DMA across consecutive output X.

    Preparation uses M*K DRAM bytes and sixteen scratchpad rows. A load's
    source row pitch is SW*CI; a store's destination row pitch is K, so up to
    sixteen output windows share each pair of ordinary DMA commands. Source
    padding is split into explicit zero-source intervals, preserving K order.
    """
    from isa_encoder import config_load, config_store
    from matmul_lowering import packed, extent, index2
    output=e.scratch(g['m']*g['k'],'i8')
    b=e.b; c=e.value
    mul=lambda a,z:b.mul_i(a,c(z))
    e.insn('config',*config_load(g['sw']*g['ci'],state_id=0))
    # Ordinary mvout reads scratchpad local address zero into DRAM. Funct23
    # STORE_SPAD carries a local destination, not this DRAM pointer form.
    e.insn('config',*config_store(g['k']))
    def batch(batch_id):
        def out_y(oy):
            output_row=mul(b.add_i(mul(batch_id,g['oh']),oy),g['ow'])
            base_y=b.sub_i(mul(oy,g['sh']),c(g['pt']))
            for ky in range(g['kh']):
                iy=b.add_i(base_y,c(ky*g['dh']))
                valid_y=b.and_i(b.mask_nonneg(iy),b.mask_neg(b.sub_i(iy,c(g['h']))))
                sy=b.clamp(iy,0,g['h']-1)
                source_row=mul(b.add_i(mul(batch_id,g['h']),sy),g['w']*g['ci'])
                for kx in range(g['kw']):
                    # Each interval either has only valid X positions or is
                    # explicit padding. No vector load crosses an image row.
                    first=max(0,min(g['ow'],-((kx*g['dw']-g['pl'])//g['sw'])))
                    last=max(0,min(g['ow'],(g['w']-1+g['pl']-kx*g['dw'])//g['sw']+1))
                    last=max(first,last)
                    column_base=(ky*g['kw']+kx)*g['ci']
                    def interval(lo,hi,padding):
                        if lo==hi:
                            return
                        def positions(ox,rows):
                            out_index=b.add_i(output_row,ox)
                            if not padding:
                                ix=b.add_i(mul(ox,g['sw']),c(kx*g['dw']-g['pl']))
                                source_base=b.add_i(source_row,mul(ix,g['ci']))
                            def channels(channel):
                                kk=c(16) if g['ci']%16==0 else extent(e,g['ci'],channel)
                                address=0 if padding else b.and_i(e.address(e.ptr(ifm,b.add_i(source_base,channel))),valid_y)
                                e.insn('mvin',address,packed(e,0,rows,kk))
                                column=b.add_i(c(column_base),channel)
                                dst=e.ptr(output,index2(e,out_index,g['k'],column))
                                e.insn('mvout',e.address(dst),packed(e,0,rows,kk))
                            e.for_loop(0,g['ci'],16,channels)
                        full=lo+(hi-lo)//16*16
                        e.for_loop(lo,full,16,lambda ox:positions(ox,16))
                        if full!=hi:
                            positions(c(full),hi-full)
                    interval(0,first,True)
                    interval(first,last,False)
                    interval(last,g['ow'],True)
        e.for_loop(0,g['oh'],1,out_y)
    e.for_loop(0,g['batch'],1,batch)
    # Local-address producer/consumer hazards order each scratchpad reuse;
    # publish every prepared DRAM byte before the common matrix DMA reads it.
    e.fence()
    return output


def emit_conv(e, ifm, weight, output, ifm_shape, attrs, *, bias=None, emit_matmul=None):
    """Gather source NHWC channels directly with device DMA, then contract.

    Host arithmetic computes addresses and loop bounds only. Every tensor value
    moves through mvin, the Gemmini mesh, and accumulator readout.
    """
    from isa_encoder import (ACC_ADDR, ACC_ACCUMULATE, ACC_FULL, GARBAGE_ADDR,
                             config_ex, config_load, config_store, float_bits)
    from matmul_lowering import packed, extent, index2
    g=conv_geometry(ifm_shape,attrs)
    # A short convolution reduction still consumes its zero initial value.
    # Keep this as an actual accumulator initialization plus accumulating
    # compute when the shared matrix path contains just one K tile.
    if 0 < g['k'] <= 16:
        attrs=dict(attrs, _initialize_accumulator=True)
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    preparation=knob('conv_preparation',default='gather',values=('gather','device_im2col','spad_panel','spad_vector','device_im2col_pool','spad_m_panel','spad_m_runs','regional_runs'),root=Path(__file__).parent)
    if preparation == 'regional_runs':
        # Choose the real scratch-panel producer only where the shared consumer
        # admits aligned K storage. Other source geometries retain prepared DMA.
        free_base = max(32, getattr(e, '_gemmini_resident_end', 32))
        use_runs = (g['ci'] % 16 == 0 and g['m'] > 16 and g['n'] > 16
                    and g['m'] * g['k'] * g['n'] >= 2**20
                    and free_base + g['k'] <= 16384
                    and 'maxpool' not in attrs.get('epilogue', [])
                    and 'acc_scale_by_column' not in attrs)
        preparation = 'spad_m_runs' if use_runs else 'device_im2col_pool'
    nontrivial=(g['kh'],g['kw'],g['sh'],g['sw'],g['pt'],g['pl'],g['oh'],g['ow']) != (1,1,1,1,0,0,g['h'],g['w'])
    if preparation in ('device_im2col','device_im2col_pool') and nontrivial and 'maxpool' not in attrs.get('epilogue',[]) and g['m']*g['k'] <= 64*1024*1024:
        prepared=emit_device_im2col(e,ifm,g)
        if emit_matmul is None:
            from matmul_lowering import emit_matmul
        return emit_matmul(e,prepared,weight,output,g['m'],g['k'],g['n'],attrs=attrs,bias=bias)
    if preparation=='device_im2col_pool' and 'maxpool' in attrs.get('epilogue',[]) and g['m']*g['k'] <= 64*1024*1024:
        from pooling_lowering import pool_geometry
        pooled_attrs=dict(attrs,pool_in_dims=[g['oh'],g['ow']])
        stages=pooled_attrs.get('epilogue',[])
        # Validate the shared readout before emitting any preparation. Unsupported
        # pool contracts retain the direct convolution path and its diagnostics.
        eligible=(stages[-1:] == ['maxpool'] and
                  not (set(stages)-{'bias','bias_add','acc_scale','relu','maxpool'}) and
                  pooled_attrs.get('output_dtype','i32') == 'i8' and
                  'acc_scale_by_column' not in pooled_attrs and
                  (not set(stages)&{'bias','bias_add'} or bias is not None) and
                  isinstance(pooled_attrs.get('acc_scale',1.0),(int,float)) and
                  not ('relu' in stages and pooled_attrs.get('acc_scale',1.0)<0))
        if eligible:
            try:
                pg=pool_geometry(g['m'],pooled_attrs)
                ph=pooled_attrs['pool_size'][0]
                eligible=ph*pg['w']<=1024
            except (ValueError,TypeError):
                eligible=False
        if eligible:
            if emit_matmul is None:
                from matmul_lowering import emit_matmul
            prepared=emit_device_im2col(e,ifm,g) if nontrivial else ifm
            return emit_matmul(e,prepared,weight,output,g['m'],g['k'],g['n'],attrs=pooled_attrs,bias=bias)
    if (g['kh'],g['kw'],g['sh'],g['sw'],g['pt'],g['pl'],g['oh'],g['ow']) == (1,1,1,1,0,0,g['h'],g['w']):
        if emit_matmul is None:
            from matmul_lowering import emit_matmul
        return emit_matmul(e,ifm,weight,output,g['m'],g['k'],g['n'],attrs=attrs,bias=bias)
    stages=attrs.get('epilogue',[])
    if set(stages)-{'bias','bias_add','acc_scale','relu','maxpool'}:
        raise ValueError('unsupported convolution device readout')
    dtype=attrs.get('output_dtype','i32')
    if dtype not in ('i8','i32') or (dtype=='i32' and set(stages)&{'acc_scale','relu','maxpool'}):
        raise ValueError('convolution readout dtype does not admit its epilogue')
    if set(stages)&{'bias','bias_add'} and bias is None:
        raise ValueError('convolution source bias is missing')
    scales=attrs.get('acc_scale_by_column')
    scale=attrs.get('acc_scale',1.0)
    if scales is not None:
        if len(scales)!=g['n'] or dtype!='i8' or 'acc_scale' not in stages:
            raise ValueError('convolution column scale extent/dtype is invalid')
        if 'relu' in stages and any(float(v)<0 for v in scales):
            raise ValueError('negative post-scale relu requires a separate device epilogue')
        scale_table=e.constant_array([float_bits(v) for v in scales],'i32')
    elif 'relu' in stages and float(scale)<0:
        raise ValueError('negative post-scale relu requires a separate device epilogue')
    b=e.b
    c=e.value
    mul=lambda a,z:b.mul_i(a,c(z))
    div=lambda a,z:b.udiv_i(a,c(z))
    rem=lambda a,z:b.urem_i(a,c(z))
    # Cache only activation panels that fit beside every live RHS resident.
    # Each kernel-point/channel chunk owns a complete physical 16x16 slot.
    chunks=(g['ci']+15)//16
    panel_base=max(32,getattr(e,'_gemmini_resident_end',32))
    panel_rows=g['kh']*g['kw']*chunks*16
    cache_activations=(preparation in ('spad_panel','spad_vector') and 'maxpool' not in stages
                       and panel_base+panel_rows<=16384)
    def activation_slot(kp,channel):
        return b.add_i(c(panel_base),mul(b.add_i(mul(kp,chunks),div(channel,16)),16))
    def gather_chunk(m,live_rows,kp,channel,kk,local,*,initialize=True):
        ky,kx=div(kp,g['kw']),rem(kp,g['kw'])
        if initialize:
            # The streaming path keeps its original per-chunk ordering.
            e.insn('mvin',0,packed(e,local,16,16))
            e.fence()
        def gather(r):
            row=b.add_i(m,r)
            batch=div(row,g['oh']*g['ow'])
            oy,ox=rem(div(row,g['ow']),g['oh']),rem(row,g['ow'])
            iy=b.add_i(b.sub_i(mul(oy,g['sh']),c(g['pt'])),mul(ky,g['dh']))
            ix=b.add_i(b.sub_i(mul(ox,g['sw']),c(g['pl'])),mul(kx,g['dw']))
            valid=b.and_i(b.mask_nonneg(iy),b.mask_neg(b.sub_i(iy,c(g['h']))))
            valid=b.and_i(valid,b.and_i(b.mask_nonneg(ix),b.mask_neg(b.sub_i(ix,c(g['w'])))))
            valid=b.and_i(valid,b.mask_neg(b.sub_i(row,c(g['m']))))
            safe_batch=b.clamp(batch,0,g['batch']-1)
            sy,sx=b.clamp(iy,0,g['h']-1),b.clamp(ix,0,g['w']-1)
            pos=b.add_i(mul(b.add_i(mul(b.add_i(mul(safe_batch,g['h']),sy),g['w']),sx),g['ci']),channel)
            address=b.and_i(e.address(e.ptr(ifm,pos)),valid)
            e.insn('mvin',address,packed(e,b.add_i(c(local),r),1,kk))
        e.for_loop(0,live_rows,1,gather)
    def gather_vector_or_scalar(m,live_rows,kp,channel,kk,local):
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import IntegerAttr, i64
        from xdsl.ir import Block
        batch=div(m,g['oh']*g['ow'])
        oy,ox=rem(div(m,g['ow']),g['oh']),rem(m,g['ow'])
        ky,kx=div(kp,g['kw']),rem(kp,g['kw'])
        iy=b.add_i(b.sub_i(mul(oy,g['sh']),c(g['pt'])),mul(ky,g['dh']))
        ix=b.add_i(b.sub_i(mul(ox,g['sw']),c(g['pl'])),mul(kx,g['dw']))
        last_x=b.add_i(ix,c((live_rows-1)*g['sw']))
        valid=b.mask_nonneg(b.sub_i(c(g['ow']-live_rows),ox))
        valid=b.and_i(valid,b.and_i(b.mask_nonneg(iy),b.mask_neg(b.sub_i(iy,c(g['h'])))))
        valid=b.and_i(valid,b.and_i(b.mask_nonneg(ix),b.mask_neg(b.sub_i(last_x,c(g['w'])))))
        condition=b.add(llvm.ICmpOp(valid,c(0),IntegerAttr(llvm.ICmpPredicateFlag.NE.to_int(),i64))).results[0]
        vector,scalar,done=Block(),Block(),Block()
        for block in (vector,scalar,done):
            b.region.add_block(block)
        b.add(llvm.CondBrOp(condition,vector,[],scalar,[]))
        b.blk=vector
        pos=b.add_i(mul(b.add_i(mul(b.add_i(mul(batch,g['h']),iy),g['w']),ix),g['ci']),channel)
        e.insn('config',*config_load(g['sw']*g['ci'],state_id=0))
        e.insn('mvin',e.address(e.ptr(ifm,pos)),packed(e,local,live_rows,kk))
        e.insn('config',*config_load(g['ci'],state_id=0))
        b.add(llvm.BrOp(done))
        b.blk=scalar
        gather_chunk(m,live_rows,kp,channel,kk,local,initialize=False)
        b.add(llvm.BrOp(done))
        b.blk=done
    def gather_row_runs(m,live_rows,kp,local_base):
        """Fill one CI-aligned kernel-point panel with maximal source runs.

        Runs end at the logical tile, output image row, or source-X validity
        boundary. Padding uses the same device zero writer as scalar gathers.
        Channel chunks share the address geometry and remain in physical K slots.
        """
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import IntegerAttr, i64
        from xdsl.ir import Block
        from merlin.llvmlower.llvm_loop_metadata import disable_loop_unroll
        ky,kx=div(kp,g['kw']),rem(kp,g['kw'])
        header=Block(arg_types=[i64])
        run_body,done=Block(),Block()
        for block in (header,run_body,done):
            b.region.add_block(block)
        b.add(llvm.BrOp(header,c(0)))
        b.blk=header
        r=header.args[0]
        cond=b.add(llvm.ICmpOp(r,c(live_rows),IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(),i64))).results[0]
        b.add(llvm.CondBrOp(cond,run_body,[],done,[]))
        b.blk=run_body
        row=b.add_i(m,r)
        batch=div(row,g['oh']*g['ow'])
        oy,ox=rem(div(row,g['ow']),g['oh']),rem(row,g['ow'])
        iy=b.add_i(b.sub_i(mul(oy,g['sh']),c(g['pt'])),mul(ky,g['dh']))
        ix=b.add_i(b.sub_i(mul(ox,g['sw']),c(g['pl'])),mul(kx,g['dw']))
        bound=b.smin(b.sub_i(c(live_rows),r),b.sub_i(c(g['ow']),ox))
        valid_y=b.and_i(b.mask_nonneg(iy),b.mask_neg(b.sub_i(iy,c(g['h']))))
        left=b.mask_neg(ix)
        right=b.mask_nonneg(b.sub_i(ix,c(g['w'])))
        left_count=div(b.add_i(b.max0(b.sub_i(c(0),ix)),c(g['sw']-1)),g['sw'])
        sx=b.clamp(ix,0,g['w']-1)
        valid_count=b.add_i(div(b.sub_i(c(g['w']-1),sx),g['sw']),c(1))
        # Every selected length is positive: left padding ends at the first
        # nonnegative X, an interior run ends before X reaches W, and right/Y
        # padding spans the remainder of this output image row.
        length=b.blend(valid_y,
            b.blend(left,b.smin(bound,left_count),b.blend(right,bound,b.smin(bound,valid_count))),bound)
        valid=b.and_i(valid_y,b.notmask(b.or_i(left,right)))
        safe_batch=b.clamp(batch,0,g['batch']-1)
        sy=b.clamp(iy,0,g['h']-1)
        pos=mul(b.add_i(mul(b.add_i(mul(safe_batch,g['h']),sy),g['w']),sx),g['ci'])
        first_slot=b.add_i(local_base,b.add_i(mul(kp,g['ci']),r))
        def channels(channel):
            address=b.and_i(e.address(e.ptr(ifm,b.add_i(pos,channel))),valid)
            e.insn('mvin',address,packed(e,b.add_i(first_slot,channel),length,16))
        e.for_loop(0,g['ci'],16,channels)
        latch=b.add(llvm.BrOp(header,b.add_i(r,length)))
        disable_loop_unroll(latch)
        b.blk=done
    def prepare_panel(m,live_rows):
        # Complete the previous panel's device readers before reusing its SP.
        e.fence()
        if g['ci']%16:
            # Distinct slots can be initialized together. Drain these zero
            # producers once before any narrower row gather overlaps them.
            def zero_slot(offset):
                e.insn('mvin',0,packed(e,b.add_i(c(panel_base),offset),16,16))
            e.for_loop(0,panel_rows,16,zero_slot)
            e.fence()
        # With aligned CI, each gather overwrites all sixteen columns in every
        # consumed row, including padding via source zero. Compute uses exactly
        # live_rows, so the unused M-tail rows require no initialization.
        def kernelpoint(kp):
            def channels(channel):
                kk=c(16) if g['ci']%16==0 else extent(e,g['ci'],channel)
                if preparation=='spad_vector' and 1<live_rows<=g['ow']:
                    gather_vector_or_scalar(m,live_rows,kp,channel,kk,activation_slot(kp,channel))
                else:
                    gather_chunk(m,live_rows,kp,channel,kk,activation_slot(kp,channel),initialize=False)
            e.for_loop(0,g['ci'],16,channels)
        e.for_loop(0,g['kh']*g['kw'],1,kernelpoint)
    def compute_tile(m,mm,n,nn,acc,live_rows,*,cached=False):
        bias_addr=0 if bias is None else e.address(e.ptr(bias,n,'i32'))
        e.insn('mvin3',bias_addr,packed(e,acc,mm,nn))
        def kernelpoint(kp):
            def channels(channel):
                kk=c(16) if g['ci']%16==0 else extent(e,g['ci'],channel)
                local=activation_slot(kp,channel) if cached else c(0)
                if not cached:
                    gather_chunk(m,live_rows,kp,channel,kk,local)
                weight_row=b.add_i(mul(kp,g['ci']),channel)
                wp=e.ptr(weight,index2(e,weight_row,g['n'],n))
                e.insn('mvin2',e.address(wp),packed(e,16,kk,nn))
                e.insn('preload',packed(e,16,kk,nn),packed(e,b.or_i(acc,c(ACC_ACCUMULATE)),mm,nn))
                e.insn('compute',packed(e,local,mm,kk),packed(e,GARBAGE_ADDR,16,16))
            e.for_loop(0,g['ci'],16,channels)
        e.for_loop(0,g['kh']*g['kw'],1,kernelpoint)
    def readout_config(n, pool=None):
        if pool is None:
            rs1,rs2=config_store(g['n']*(4 if dtype=='i32' else 1),scale=scale,activation=int('relu' in stages))
        else:
            from pooling_lowering import config_pool
            rs1,rs2=config_pool(g['n'],attrs,pool)
        if scales is not None:
            bits=b.and_i(b.load_i64(scale_table,n,'i32'),c(0xffffffff))
            rs2=b.or_i(c(g['n']),b.shl_i(bits,32))
        e.insn('config',rs1,rs2)
    if (preparation in ('spad_m_panel','spad_m_runs') and 'maxpool' not in stages and scales is None
            and g['ci']%16==0 and g['m']>16 and g['n']>16
            and panel_base+(g['k']//16)*16<=16384):
        if emit_matmul is None:
            from matmul_lowering import emit_matmul
        def load_source_panel(emitter,base,first_m,panel_count,k_tiles):
            # The shared consumer owns the completion fence and panel lifetime.
            # All CI-aligned source/zero gathers overwrite every consumed lane.
            source_stride=g['sw']*g['ci'] if preparation=='spad_m_runs' else g['ci']
            emitter.insn('config',*config_load(source_stride,state_id=0))
            def tile(local_m,live_rows):
                m=b.add_i(first_m,local_m)
                local_base=b.add_i(c(base),mul(local_m,k_tiles))
                def kernelpoint(kp):
                    if preparation=='spad_m_runs':
                        gather_row_runs(m,live_rows,kp,local_base)
                    else:
                        def channels(channel):
                            slot=b.add_i(local_base,mul(b.add_i(mul(kp,g['ci']//16),div(channel,16)),16))
                            if 1<live_rows<=g['ow']:
                                gather_vector_or_scalar(m,live_rows,kp,channel,c(16),slot)
                            else:
                                gather_chunk(m,live_rows,kp,channel,c(16),slot,initialize=False)
                        emitter.for_loop(0,g['ci'],16,channels)
                emitter.for_loop(0,g['kh']*g['kw'],1,kernelpoint)
            full=panel_count//16*16
            emitter.for_loop(0,full,16,lambda m:tile(m,16))
            if full!=panel_count:
                tile(c(full),panel_count-full)
        return emit_matmul(e,ifm,weight,output,g['m'],g['k'],g['n'],attrs=attrs,
                           bias=bias,lhs_panel_loader=load_source_panel)
    e.insn('config',*config_ex(dataflow=1))
    e.insn('config',*config_load(g['ci'],state_id=0))
    e.insn('config',*config_load(g['n'],state_id=1))
    e.insn('config',*config_load(0,state_id=2))
    e.insn('config',*config_store(g['n']*(4 if dtype=='i32' else 1),scale=scale,activation=int('relu' in stages)))
    if 'maxpool' in stages:
        from pooling_lowering import pool_geometry
        if stages[-1]!='maxpool':
            raise ValueError('convolution maxpool must be the final readout stage')
        pg=pool_geometry(g['m'],attrs)
        ph,pw=attrs['pool_size']; sh,sw=attrs['pool_stride']
        pt,pl,pb,pr=attrs.get('pool_padding',[0,0,0,0])
        band_output_rows=max(1,((1024//pg['w'])-ph)//sh+1)
        if ph*pg['w']>1024:
            raise ValueError('pool window row band exceeds accumulator capacity')
        def batch(batch_id):
            for first_out in range(0,pg['oh'],band_output_rows):
                last_out=min(pg['oh'],first_out+band_output_rows)
                first_src=first_out*sh-pt
                last_src=(last_out-1)*sh-pt+ph
                lo,hi=max(0,first_src),min(pg['h'],last_src)
                band=dict(attrs)
                band['pool_in_dims']=[hi-lo,pg['w']]
                band['pool_padding']=[lo-first_src,pl,last_src-hi,pr]
                bg=pool_geometry((hi-lo)*pg['w'],band)
                source_start=b.add_i(mul(batch_id,pg['h']*pg['w']),c(lo*pg['w']))
                output_start=b.add_i(mul(batch_id,pg['oh']*pg['ow']),c(first_out*pg['ow']))
                def columns(n):
                    nn=(c(16) if g['n']%16==0 else extent(e,g['n'],n)) if scales is None else c(1)
                    def rows(local_m,live_rows):
                        m=b.add_i(source_start,local_m)
                        acc=b.add_i(c(ACC_ADDR),local_m)
                        compute_tile(m,c(live_rows),n,nn,acc,live_rows)
                    count=(hi-lo)*pg['w']
                    full=count//16*16
                    e.for_loop(0,full,16,lambda m: rows(m,16))
                    if full!=count:
                        rows(c(full),count-full)
                    readout_config(n,bg)
                    dst=e.ptr(output,index2(e,output_start,g['n'],n),'i8')
                    e.insn('mvout',e.address(dst),packed(e,ACC_ADDR,0,nn))
                    e.fence()
                e.for_loop(0,g['n'],16 if scales is None else 1,columns)
        e.for_loop(0,pg['batches'],1,batch)
        e.insn('config',*config_store(g['n']))
    else:
        def rows(m,live_rows):
            mm=c(live_rows)
            if cache_activations:
                prepare_panel(m,live_rows)
            def columns(n):
                nn=(c(16) if g['n']%16==0 else extent(e,g['n'],n)) if scales is None else c(1)
                readout_config(n)
                compute_tile(m,mm,n,nn,c(ACC_ADDR),live_rows,cached=cache_activations)
                dst=e.ptr(output,index2(e,m,g['n'],n),dtype)
                e.insn('mvout',e.address(dst),packed(e,ACC_ADDR|(ACC_FULL if dtype=='i32' else 0),mm,nn))
            e.for_loop(0,g['n'],16 if scales is None else 1,columns)
        full=g['m']//16*16
        e.for_loop(0,full,16,lambda m: rows(m,16))
        if full!=g['m']:
            rows(c(full),g['m']-full)
    e.fence()


def emit_batched_matmul(e, lhs, rhs, output, lhs_shape, rhs_shape, output_shape,
                        attrs=None, *, bias=None, emit_matmul=None):
    from merlin.runtime.commandbuffer import batched_matmul_geometry
    if emit_matmul is None:
        from matmul_lowering import emit_matmul
    g = batched_matmul_geometry(lhs_shape, rhs_shape, output_shape)
    attrs = attrs or {}
    def one(index):
        b = e.b
        off = lambda size: b.mul_i(index, b.const(size))
        emit_matmul(e, e.ptr(lhs, off(g.lhs_slice_elements)),
                    e.ptr(rhs, off(g.rhs_slice_elements)),
                    e.ptr(output, off(g.output_slice_elements), attrs.get('output_dtype', 'i32')),
                    g.m, g.k, g.n, attrs=attrs, bias=bias)
    e.for_loop(0, g.batch_count, 1, one)


def emit_attention(e, lhs, rhs, output, lhs_shape, rhs_shape, attrs=None, *,
                   qk=False, emit_matmul=None):
    """QK contracts trailing head dims; PV consumes already row-major [S,D]."""
    if emit_matmul is None:
        from matmul_lowering import emit_matmul
    attrs = attrs or {}
    if qk:
        from attention_lowering import emit_attention_qk
        return emit_attention_qk(e,lhs,rhs,output,lhs_shape,rhs_shape,attrs)
    ls,rs=tuple(lhs_shape),tuple(rhs_shape)
    if len(ls)!=2 or len(rs)!=2 or ls[1]!=rs[0]:
        raise ValueError('attention PV requires [M,S] by [S,D] source matrices')
    return emit_matmul(e,lhs,rhs,output,ls[0],ls[1],rs[1],attrs=attrs)


def lower_conv(e, cb, command, pointers):
    operands, attrs = command['operands'], command.get('attributes', {})
    source = operands['ifm']
    bias = pointers[attrs['bias']] if attrs.get('bias') else None
    return emit_conv(e, pointers[source], pointers[operands['weight']], pointers[operands['dst']],
                     cb['tensors'][source]['shape'], attrs, bias=bias)


def lower_batched(e, cb, command, pointers):
    operands, attrs = command['operands'], command.get('attributes', {})
    lhs, rhs, dst = (operands[key] for key in ('a', 'w', 'dst'))
    bias = pointers[attrs['bias']] if attrs.get('bias') else None
    return emit_batched_matmul(e, pointers[lhs], pointers[rhs], pointers[dst],
                               cb['tensors'][lhs]['shape'], cb['tensors'][rhs]['shape'],
                               cb['tensors'][dst]['shape'], attrs, bias=bias)


def lower_attention(e, cb, command, pointers):
    operands, attrs = command['operands'], command.get('attributes', {})
    qk = command['opcode'] == 'ATTENTION_QK'
    lhs, rhs = (operands[key] for key in (('q', 'k') if qk else ('p', 'v')))
    return emit_attention(e, pointers[lhs], pointers[rhs], pointers[operands['dst']],
                          cb['tensors'][lhs]['shape'], cb['tensors'][rhs]['shape'], attrs, qk=qk)
