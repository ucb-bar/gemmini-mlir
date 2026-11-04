"""Standalone accelerator load, accumulator addition and spatial maximum routes.

Scalar instructions compute addresses and counted-loop control only. Numeric
screens are compile-time exhaustive checks of the declared i8 operand domain.
"""
from functools import lru_cache
from math import prod, isfinite
import struct

ACC = 1 << 31
ADD = 1 << 30


def _f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def _sat(x):
    return max(-128, min(127, x))


@lru_cache(maxsize=128)
def residual_screen(lhs_scale, rhs_scale, bound_lsb, relu=False):
    """Prove a load/add/readout binding against the single-round source.

    ReadoutFacet's operand-sum contract permits moving a common factor from
    saturating operand loads to the accumulator readout. First keep the direct
    binding if it meets the declared bound; otherwise screen the common-factor
    binding over every signed-i8 input pair. Neither path changes the reference
    rounding or infers an output tolerance.
    """
    from merlin.runtime.tensor import Tensor
    a, b = _f32(lhs_scale), _f32(rhs_scale)
    if not isfinite(a) or not isfinite(b):
        raise ValueError('residual operand scales must be finite f32')
    lhs = [x for x in range(-128, 128) for _ in range(256)]
    rhs = list(range(-128, 128)) * 256
    reference = Tensor((256,256), lhs, 'i8').residual_add_scaled(
        Tensor((256,256), rhs, 'i8'), a, b).data
    candidates = [1.0]
    common = max(1.0, abs(a), abs(b))
    if common != 1.0:
        candidates.append(common)
    best = None
    for readout in candidates:
        load_a,load_b = _f32(a/readout),_f32(b/readout)
        la = {x: _sat(round(_f32(x*load_a))) for x in range(-128,128)}
        rb = {x: _sat(round(_f32(x*load_b))) for x in range(-128,128)}
        maximum = 0
        for x,y,expected in zip(lhs,rhs,reference):
            actual = round(_f32((la[x]+rb[y])*readout))
            if relu:
                actual, expected = max(0,actual), max(0,expected)
            maximum = max(maximum, abs(_sat(actual)-_sat(expected)))
        proof = {'pairs_checked':65536, 'max_error_lsb':maximum, 'bound_lsb':bound_lsb,
                 'lhs_load_f32':load_a, 'rhs_load_f32':load_b,
                 'readout_scale_f32':readout}
        if best is None or maximum < best['max_error_lsb']:
            best = proof
        if maximum <= bound_lsb:
            return proof
    raise ValueError('operand-sum numerical proof gap: direct and common-factor '
                     f'load/add/readout bindings have minimum maximum error '
                     f'{best["max_error_lsb"]} LSB, declared bound is {bound_lsb}')


def _matrix(shape):
    shape = tuple(int(v) for v in shape)
    if not shape or any(v <= 0 for v in shape):
        raise ValueError('elementwise route requires positive static shape')
    return prod(shape[:-1]), shape[-1]


def _tiles(enc, shape, callback):
    rows, cols = _matrix(shape)
    # Peel each dimension's tail while retaining all full-tile loops.
    for row_start,row_stop,row_step,height in ((0, rows//16*16,16,16),
                                              (rows//16*16,rows,16,rows%16)):
        if row_start == row_stop:
            continue
        def row_body(r):
            for col_start,col_stop,col_step,width in ((0,cols//16*16,16,16),
                                                      (cols//16*16,cols,16,cols%16)):
                if col_start == col_stop:
                    continue
                def column_body(c):
                    offset = enc.add(enc.mul(r,cols),c)
                    callback(offset,c,height,width,cols)
                enc.fn.for_loop(col_start,col_stop,col_step,column_body,retain_loop=True)
        enc.fn.for_loop(row_start,row_stop,row_step,row_body,retain_loop=True)


def _scaled_tiles(enc, shape, attrs, callback, default_scale):
    """Split only equal-scale column regions; preserve source and result pitch."""
    rows,columns=_matrix(shape)
    vector=attrs.get('acc_scale_by_column')
    if vector is None:
        return _tiles(enc,shape,lambda off,c,h,w,n:callback(off,c,h,w,n,default_scale))
    if len(vector)!=columns:
        raise ValueError('per-column scales must match source trailing dimension')
    regions=[]
    for column,scale in enumerate(vector):
        scale=_f32(scale)
        if not isfinite(scale):
            raise ValueError('readout column scales must be finite f32')
        if regions and regions[-1][2]==scale:
            first,count,value=regions[-1]; regions[-1]=(first,count+1,value)
        else:
            regions.append((column,1,scale))
    for first,count,scale in regions:
        def region_tile(offset,c,h,w,n):
            row=enc.fn.udiv_i(enc.fn.sub_i(offset,c),enc.val(count))
            column=enc.add(c,first)
            actual=enc.add(enc.mul(row,columns),column)
            callback(actual,column,h,w,columns,scale)
        _tiles(enc,(rows,count),region_tile)


def _output(attrs, default='i8'):
    dtype = attrs.get('output_dtype',attrs.get('out_dtype',default))
    if dtype not in ('i8','i32'):
        raise ValueError(f'unsupported accelerator elementwise output type {dtype}')
    epilogue = attrs.get('epilogue') or []
    unknown = set(epilogue)-{'relu','acc_scale','bias_add','bias','maxpool','requant'}
    if unknown:
        raise ValueError(f'unsupported standalone readout stages {sorted(unknown)}')
    relu = bool(attrs.get('relu',False) or 'relu' in epilogue)
    if 'requant' in epilogue and 'requant_shift' not in attrs:
        raise ValueError('requant stage requires explicitly declared requant_shift')
    if 'requant' in epilogue and int(attrs['requant_shift']) > 0:
        raise ValueError('positive requant_shift requires half-up integer rounding; selected '
                         'accumulator scale readout uses half-even rounding, so no equivalence is proven')
    if 'acc_scale_by_column' in attrs:
        scale = 1.0  # readout is split into exact equal-scale column runs by caller
    elif 'acc_scale' in epilogue:
        from merlin.runtime.commandbuffer import declared_acc_scale
        scale = declared_acc_scale(attrs,op='standalone readout')
    else:
        scale = attrs.get('acc_scale',1.0)
    if dtype == 'i32' and (relu or scale != 1.0 or
                          any(float(v)!=1.0 for v in attrs.get('acc_scale_by_column',[]))):
        raise ValueError('full-width raw accumulator store cannot apply scale or activation')
    return dtype, float(scale), relu


def emit_movement(enc, src, out, shape, out_dtype='i32', **attrs):
    if 'maxpool' in (attrs.get('epilogue') or []):
        raise ValueError('movement with pooled readout must use emit_maxpool with its geometry')
    dtype,scale,relu = _output(attrs,out_dtype)
    src_dtype = attrs.get('src_dtype',attrs.get('input_dtype','i8'))
    if src_dtype not in ('i8','i32'):
        raise ValueError('movement accumulator load requires signed i8 or i32 input')
    size = 1 if src_dtype == 'i8' else 4
    def tile(offset,c,h,w,n,readout_scale):
        enc.load(enc.ptr_offset(src,enc.mul(offset,size)),ACC,h,w,n*size,
                 shrunk=src_dtype=='i8')
        enc.fence()
        enc.store(enc.ptr_offset(out,enc.mul(offset,4 if dtype=='i32' else 1)),0,h,w,
                  n*(4 if dtype=='i32' else 1),dtype=dtype,scale=readout_scale,relu=relu)
        enc.fence()
    _scaled_tiles(enc,shape,attrs,tile,scale)


def emit_residual(enc, lhs, rhs, out, shape, **attrs):
    from merlin.runtime.commandbuffer import residual_add_params
    a,b,bound = residual_add_params(attrs,op='RESIDUAL_ADD')
    dtype,scale,relu = _output(attrs)
    if dtype != 'i8' or scale != 1.0:
        raise ValueError('operand-sum contract requires signed i8 output with unit readout scale')
    proof = residual_screen(a,b,bound,relu)
    def tile(offset,c,h,w,n):
        enc.load(enc.ptr_offset(lhs,offset),ACC,h,w,n,scale=proof['lhs_load_f32'],shrunk=True)
        enc.fence()
        enc.load(enc.ptr_offset(rhs,offset),ACC|ADD,h,w,n,stream=1,scale=proof['rhs_load_f32'],shrunk=True)
        enc.fence()
        enc.store(enc.ptr_offset(out,offset),0,h,w,n,dtype='i8',
                  scale=proof['readout_scale_f32'],relu=relu)
        enc.fence()
    from elementwise_schedule import choice, plan_residual, disjoint_condition, emit_residual_epochs
    mode=choice(attrs)
    if mode=='sync':
        _tiles(enc,shape,tile)
    else:
        try:
            plan,receipt=plan_residual(tuple(map(int,shape)),proof,relu,mode,attrs)
        except ValueError as error:
            # Missing selected target premises preserve the original complete
            # schedule; a malformed knob is rejected by choice before here.
            _tiles(enc,shape,tile)
            proof=dict(proof,schedule=dict(requested=mode,realized='sync',refusal=str(error)))
        else:
            def epochs():
                receipt['emission']=emit_residual_epochs(enc,(lhs,rhs,out),plan,proof,relu)
            enc.index_builder.branch(disjoint_condition(enc,(lhs,rhs,out),prod(shape)),
                epochs,lambda:_tiles(enc,shape,tile))
            enc.fence()
            proof=dict(proof,schedule=dict(requested=mode,realized='runtime_disjoint_epochs_or_sync',
                checked_plan=receipt))
    return proof


def emit_bias_add(enc, src, bias, out, shape, **attrs):
    if 'maxpool' in (attrs.get('epilogue') or []):
        raise ValueError('bias with pooled readout requires i32 staging then emit_pool_from_i32')
    if attrs.get('bias_dtype','i32') != 'i32':
        raise ValueError('standalone bias requires i32 accumulator-grid bias')
    dtype,scale,relu = _output(attrs,'i32')
    src_dtype = attrs.get('src_dtype',attrs.get('input_dtype','i32'))
    if src_dtype not in ('i8','i32'):
        raise ValueError('bias accumulator add requires signed i8 or i32 source')
    size = 1 if src_dtype=='i8' else 4
    def tile(offset,c,h,w,n,readout_scale):
        enc.load(enc.ptr_offset(bias,enc.mul(c,4)),ACC,h,w,0,shrunk=False)
        enc.fence()
        enc.load(enc.ptr_offset(src,enc.mul(offset,size)),ACC|ADD,h,w,n*size,
                 stream=1,shrunk=src_dtype=='i8')
        enc.fence()
        enc.store(enc.ptr_offset(out,enc.mul(offset,4 if dtype=='i32' else 1)),0,h,w,
                  n*(4 if dtype=='i32' else 1),dtype=dtype,scale=readout_scale,relu=relu)
        enc.fence()
    _scaled_tiles(enc,shape,attrs,tile,scale)


def emit_maxpool(enc, src, out, shape, **attrs):
    """Pool ABI [B*H*W,C] or NHWC input on accelerator accumulator readout.

    The private gathered window occupies at most nine accumulator rows. Each
    source scalar is read out with its original f32 scale and activation before
    the signed maximum. Zero padding uses the hardware zero-fill DMA address;
    minimum-i8 padding uses the library's proven border-duplication rule.
    """
    from merlin.compile.logical_abi import pool_params, pool_out_dims
    if len(shape) == 2:
        rows,channels = map(int,shape)
        params=pool_params(attrs,op='MAXPOOL2D')
        h,w=params['pool_in_dims']
        if h<=0 or w<=0 or rows%(h*w):
            raise ValueError('pool_in_dims must exactly divide flattened pool source rows')
        batch=rows//(h*w)
    elif len(shape) == 4:
        batch,h,w,channels=map(int,shape)
        normalized=dict(attrs)
        normalized.setdefault('pool_in_dims',(h,w))
        normalized.setdefault('pool_size',attrs.get('kernel',attrs.get('kernel_size',(2,2))))
        normalized.setdefault('pool_stride',attrs.get('stride',attrs.get('strides',(2,2))))
        if isinstance(normalized['pool_size'],int):
            normalized['pool_size']=(normalized['pool_size'],)*2
        if isinstance(normalized['pool_stride'],int):
            normalized['pool_stride']=(normalized['pool_stride'],)*2
        if 'pool_padding' not in normalized:
            pad=attrs.get('padding',0)
            normalized['pool_padding']=(pad,)*4 if isinstance(pad,int) else pad
        params=pool_params(normalized,op='MAXPOOL2D')
    else:
        raise ValueError('pool requires rank-two [B*H*W,C] with pool_in_dims, or NHWC rank four')
    kh,kw=params['pool_size']
    sh,sw=params['pool_stride']
    pt,pl,pb,pr=params['pool_padding']
    pad_value=params['pad_value']
    if min(batch,h,w,channels,kh,kw,sh,sw)<=0 or min(pt,pl,pb,pr)<0:
        raise ValueError('pool requires positive dimensions/strides and nonnegative padding')
    if kh != kw or kh not in (1,2,3):
        raise ValueError('StoreController maximum requires square window size in [1,3]')
    oh,ow=pool_out_dims(h,w,(kh,kw),(sh,sw),(pt,pl,pb,pr))
    if min(oh,ow)<=0:
        raise ValueError('pool geometry has no output positions')
    padded=any((pt,pl,pb,pr))
    if padded and pad_value not in (0,-128):
        raise ValueError('pool padding must explicitly declare 0 or -128 for available hardware forms')
    if padded and pad_value == -128 and (kh<=pt or kw<=pl or (oh-1)*sh-pt>=h or (ow-1)*sw-pl>=w):
        raise ValueError('minimum-i8 border duplication requires every window to intersect input')
    dtype,scale,relu=_output(attrs)
    if dtype != 'i8':
        raise ValueError('StoreController pooling requires narrowing signed i8 readout')
    src_dtype=attrs.get('src_dtype',attrs.get('input_dtype','i8'))
    if src_dtype not in ('i8','i32'):
        raise ValueError('pool accumulator staging requires signed i8 or i32 source')
    itemsize=1 if src_dtype=='i8' else 4
    scales=attrs.get('acc_scale_by_column')
    if scales is None:
        runs=[(0,channels,scale)]
    else:
        if len(scales)!=channels:
            raise ValueError('pool channel scales must match complete source channel count')
        runs=[]
        for c,value in enumerate(scales):
            value=float(value)
            if runs and runs[-1][2]==value:
                first,count,v=runs[-1]; runs[-1]=(first,count+1,v)
            else:
                runs.append((c,1,value))
    control=enc.index_builder
    def batch_body(b):
        def y_body(y):
            def x_body(x):
                def channel_body(c0,width,readout_scale):
                    for ky in range(kh):
                        iy=enc.add(enc.mul(y,sh),ky-pt)
                        for kx in range(kw):
                            ix=enc.add(enc.mul(x,sw),kx-pl)
                            def source_load():
                                # Clamped source coordinates occur only in the
                                # minimum-padding border-duplication route.
                                yy=enc.fn.clamp(iy,0,h-1)
                                xx=enc.fn.clamp(ix,0,w-1)
                                pix=enc.add(enc.mul(b,h*w),enc.add(enc.mul(yy,w),xx))
                                off=enc.mul(enc.add(enc.mul(pix,channels),c0),itemsize)
                                enc.load(enc.ptr_offset(src,off),ACC|(ky*kw+kx),1,width,
                                         channels*itemsize,shrunk=src_dtype=='i8')
                            if padded and pad_value == 0:
                                valid=control.op('and',control.cmp('ult',iy,h),
                                                 control.cmp('ult',ix,w),dtype='i1')
                                def zero_load():
                                    enc.load(enc.ptr_zero(),ACC|(ky*kw+kx),1,width,0,
                                             shrunk=src_dtype=='i8')
                                control.branch(valid,source_load,zero_load)
                            else:
                                source_load()
                    enc.fence()
                    pix=enc.add(enc.mul(b,oh*ow),enc.add(enc.mul(y,ow),x))
                    off=enc.add(enc.mul(pix,channels),c0)
                    enc.store(enc.ptr_offset(out,off),0,1,width,channels,dtype='i8',
                              scale=readout_scale,relu=relu,pool_out_dim=1,
                              pool_size=kh,pool_stride=1,porows=1,pocols=1,
                              orows=kh,ocols=kw,upad=0,lpad=0)
                    enc.fence()
                for first,count,readout_scale in runs:
                    stop=first+count//16*16
                    enc.fn.for_loop(first,stop,16,
                        lambda c:channel_body(c,16,readout_scale),retain_loop=True)
                    if count%16:
                        channel_body(enc.val(stop),count%16,readout_scale)
            enc.fn.for_loop(0,ow,1,x_body,retain_loop=True)
        enc.fn.for_loop(0,oh,1,y_body,retain_loop=True)
    enc.fn.for_loop(0,batch,1,batch_body,retain_loop=True)
    return (batch*oh*ow,channels) if len(shape)==2 else (batch,oh,ow,channels)


def emit_pool_from_i32(enc, src_i32, out, shape, **attrs):
    """Consume raw contraction+bias staging with exact scaled/ReLU pool readout.

    The caller owns the i32 temporary and completes its producer before entry.
    Bias must already be present in that temporary; this helper never touches
    the caller's other live outputs.
    """
    attrs=dict(attrs)
    attrs['src_dtype']='i32'
    attrs.setdefault('output_dtype','i8')
    return emit_maxpool(enc,src_i32,out,shape,**attrs)
