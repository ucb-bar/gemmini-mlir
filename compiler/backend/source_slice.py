"""Exact static integer source slices through checked strided accumulator DMA.

The input and result remain distinct typed tensor boundaries. Positive static
strides use row tiles, with one-column tiles for a strided innermost axis. Every source
address is proved against the original multidimensional slice expression;
output injectivity and cardinality prove complete result coverage.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from math import prod

from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa


class SourceSliceError(ValueError):
    pass


def _strides(shape):
    return tuple(prod(shape[i+1:]) for i in range(len(shape)))


def _call(name, **fields):
    return Call(name, tuple((k, as_expr(v)) for k,v in fields.items()))


def _geometry(shape, offsets, sizes, strides, dtype):
    shape, offsets, sizes, strides = map(tuple, (shape, offsets, sizes, strides))
    rank=len(shape)
    if (not 1 <= rank <= 6 or any(len(v)!=rank for v in (offsets,sizes,strides))
            or any(type(v) is not int for row in (shape,offsets,sizes,strides) for v in row)
            or any(v<=0 for row in (shape,sizes,strides) for v in row)
            or any(v<0 for v in offsets) or dtype not in ('i8','i32')):
        raise SourceSliceError('static integer slice requires positive sizes and strides')
    if any(o+(n-1)*s>=d for d,o,n,s in zip(shape,offsets,sizes,strides,strict=True)):
        raise SourceSliceError('static slice exceeds its declared source shape')
    width=1 if dtype=='i8' else 4
    if prod(shape)*width>=1<<63:
        raise SourceSliceError('static slice exceeds signed address arithmetic')
    ss=_strides(shape)
    sp=ss[-2]*strides[-2] if rank>1 else sizes[-1]
    dp=sizes[-1]
    if max(sp,dp)*width>=1<<32:
        raise SourceSliceError('slice row stride exceeds selected DMA configuration field')
    return shape,offsets,sizes,strides,ss,sp,dp,width


def _kernel(shape, offsets, sizes, strides, dtype):
    shape,offsets,sizes,strides,ss,sp,dp,width=_geometry(shape,offsets,sizes,strides,dtype)
    rank=len(shape);ds=_strides(sizes);prefix=range(max(0,rank-2))
    sb=Const(sum(o*s for o,s in zip(offsets,ss,strict=True)))
    db=Const(0)
    for i in prefix:
        sb=sb+Var(f'p{i}')*(ss[i]*strides[i]);db=db+Var(f'p{i}')*ds[i]
    rows=sizes[-2] if rank>1 else 1
    def tile(r,nr,c,nc):
        source_column=c if strides[-1]==1 else c*strides[-1]
        return (_call('slice_mvin',offset=sb+r*sp+source_column,addr=isa.acc_addr(0),rows=nr,cols=nc),
                _call('slice_drain'),
                _call('slice_mvout',offset=db+r*dp+c,addr=isa.acc_addr(0,full=dtype=='i32'),rows=nr,cols=nc),
                _call('slice_drain'))
    def columns(r,nr):
        if strides[-1]!=1:
            # DMA columns within one descriptor are contiguous. Enumerating
            # one-column descriptors applies the original last-axis stride
            # without gathering or changing any payload on the host.
            return (Loop('c',sizes[-1],tile(r,nr,Var('c'),1)),)
        full,tail=divmod(sizes[-1],isa.DIM);body=[]
        if full:body.append(Loop('c',full,tile(r,nr,Var('c')*isa.DIM,isa.DIM)))
        if tail:body.extend(tile(r,nr,Const(full*isa.DIM),tail))
        return tuple(body)
    full,tail=divmod(rows,isa.DIM);body=[]
    if full:body.append(Loop('r',full,columns(Var('r')*isa.DIM,isa.DIM)))
    if tail:body.extend(columns(Const(full*isa.DIM),tail))
    body=tuple(body)
    for i in reversed(tuple(prefix)):body=(Loop(f'p{i}',sizes[i],body),)
    prologue=(_call('slice_drain'),_call('slice_config_load',stride=sp*width),
              _call('slice_config_store',stride=dp*width))
    return Kernel('source_integer_slice',
        (TensorArg('src',shape,dtype,'read'),TensorArg('dst',sizes,dtype,'write')),prologue+body)


def check_slice(kernel, shape, offsets, sizes, strides, dtype):
    shape,offsets,sizes,strides,ss,sp,dp,width=_geometry(shape,offsets,sizes,strides,dtype)
    if kernel!=_kernel(shape,offsets,sizes,strides,dtype):
        raise SourceSliceError('slice counted offsets, DMA descriptors or completion changed')
    ds=_strides(sizes);sites=[]
    def walk(body,scope):
        at=0
        while at<len(body):
            op=body[at]
            if isinstance(op,Loop):walk(op.body,scope+((op.var,op.extent),));at+=1;continue
            load,_,store,_=body[at:at+4]
            nr,nc=load.arg('rows').value,load.arg('cols').value
            if not 0<nr<=min(isa.DIM,isa.ACC_ROWS) or not 0<nc<=isa.DIM:
                raise SourceSliceError('slice tile exceeds selected accumulator geometry')
            sites.append((load,store,scope,nr,nc));at+=4
    walk(kernel.body[3:],())
    if sum(prod(n for _,n in scope)*nr*nc for _,_,scope,nr,nc in sites)!=prod(sizes):
        raise SourceSliceError('slice DMA cardinality differs from source result')
    # The actual stream is structurally equal to the canonical constructor
    # above, including every affine coefficient, bound, tail descriptor and
    # drain.  divmod partitions each row/column axis into disjoint full tiles
    # and one tail; a strided last axis enumerates singleton column tiles.
    # Prefix loops enumerate the Cartesian product once. This
    # proves output injectivity constructively without a solver or enumeration.
    # For each output coordinate j, the source address is
    # sum((offset[i] + j[i]*stride[i]) * source_pitch[i]); all terms are
    # nonnegative and the per-axis upper bounds were checked in _geometry.
    source_first=sum(o*s for o,s in zip(offsets,ss,strict=True))
    source_last=sum((o+(n-1)*step)*pitch
        for o,n,step,pitch in zip(offsets,sizes,strides,ss,strict=True))
    if not 0<=source_first<=source_last<prod(shape):
        raise SourceSliceError('slice linear address bounds do not follow from axis bounds')
    return dict(schema='submission.static_integer_slice.v1',kernel_sha256=kernel.digest(),
        source_shape=list(shape),output_shape=list(sizes),offsets=list(offsets),strides=list(strides),
        source_row_stride_bytes=sp*width,destination_row_stride_bytes=dp*width,
        output_elements=prod(sizes),source_address_bounds=[source_first,source_last],
        proof_method='exact canonical counted-stream equality and Cartesian tile partitions',
        requested_load_bytes=prod(sizes)*width,requested_store_bytes=prod(sizes)*width,
        coverage='bounded exact source projection; injective output and complete output cardinality',
        numeric='signed i8 identity readout' if dtype=='i8' else 'raw full-width i32 readout',
        completion='full drain between load and store, and before accumulator row reuse')


@dataclass(frozen=True)
class SourceSlicePlan:
    shape: tuple
    offsets: tuple
    sizes: tuple
    strides: tuple
    dtype: str
    kernel: Kernel
    proof: dict

    def revalidate(self):
        proof=check_slice(self.kernel,self.shape,self.offsets,self.sizes,self.strides,self.dtype)
        if proof!=self.proof:raise SourceSliceError('slice proof receipt changed')
        return proof


def plan_source_slice(shape,offsets,sizes,strides,dtype):
    shape,offsets,sizes,strides=map(tuple,(shape,offsets,sizes,strides))
    kernel=_kernel(shape,offsets,sizes,strides,dtype)
    return SourceSlicePlan(shape,offsets,sizes,strides,dtype,kernel,
        check_slice(kernel,shape,offsets,sizes,strides,dtype))


def lower_source_slice(ctx,operands,attributes):
    src,dst=operands['src'],operands['dst'];a,b=ctx.spec(src),ctx.spec(dst)
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    if src==dst or src in aliases or dst in aliases:
        raise SourceSliceError('source slice requires distinct materialized boundary buffers')
    if a['dtype']!=b['dtype'] or attributes.get('output_dtype',a['dtype'])!=a['dtype']:
        raise SourceSliceError('source slice cannot change its integer element type')
    if set(attributes)-{'offsets','sizes','strides','output_dtype'}:
        raise SourceSliceError('source slice has no numeric epilogue or dynamic addressing contract')
    if list(attributes.get('sizes',b['shape']))!=list(b['shape']):
        raise SourceSliceError('source slice sizes differ from its output tensor shape')
    plan=plan_source_slice(a['shape'],attributes['offsets'],b['shape'],attributes['strides'],a['dtype'])
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,arguments):
        if name=='slice_drain':ctx.fence();return
        if name=='slice_config_load':ctx.emit(*isa.config_ld(arguments['stride'],shrink=plan.dtype=='i8'));return
        if name=='slice_config_store':ctx.emit(*isa.config_st(arguments['stride']));return
        args={k:bridge.value(v) for k,v in arguments.items()}
        descriptor=isa.pack_matrix(args['addr'],args['rows'],args['cols'],fb=ctx.fb)
        if name=='slice_mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(src,args['offset']),descriptor)
        elif name=='slice_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,args['offset']),descriptor)
        else:raise SourceSliceError('unrecognized source slice instruction')
    return {'proof':plan.revalidate(),'emission':emit_index_program(plan.kernel,bridge,emit)}


def _insert_kernel(source_shape,base_shape,offsets,strides,dtype):
    """Invert a proved slice projection after an independent full base copy."""
    source_shape,base_shape,offsets,strides=map(tuple,(source_shape,base_shape,offsets,strides))
    full=_kernel(base_shape,(0,)*len(base_shape),base_shape,(1,)*len(base_shape),dtype)
    region=_kernel(base_shape,offsets,source_shape,strides,dtype)
    def base_body(body):
        return tuple(replace(o,body=base_body(o.body)) if isinstance(o,Loop) else
            replace(o,instr={'slice_mvin':'insert_base_mvin','slice_mvout':'insert_mvout',
                'slice_drain':'insert_drain','slice_config_load':'insert_config_load',
                'slice_config_store':'insert_config_store'}[o.instr]) for o in body)
    def inverse(body):
        result=[];at=0
        while at<len(body):
            o=body[at]
            if isinstance(o,Loop):result.append(replace(o,body=inverse(o.body)));at+=1;continue
            load,_,store,_=body[at:at+4]
            result.extend((Call('insert_src_mvin',tuple((k,Const(isa.acc_addr(0)) if k=='addr' else v) for k,v in store.args)),
                _call('insert_drain'),
                Call('insert_mvout',tuple((k,Const(isa.acc_addr(0,full=dtype=='i32')) if k=='addr' else v) for k,v in load.args)),
                _call('insert_drain')))
            at+=4
        return tuple(result)
    prologue=(_call('insert_drain'),
        _call('insert_config_load',stride=region.body[2].arg('stride')),
        _call('insert_config_store',stride=region.body[1].arg('stride')))
    return Kernel('source_integer_insert_slice',
        (TensorArg('src',source_shape,dtype,'read'),TensorArg('base',base_shape,dtype,'read'),
         TensorArg('dst',base_shape,dtype,'write')),base_body(full.body)+prologue+inverse(region.body[3:]))


def check_insert_slice(kernel,source_shape,base_shape,offsets,strides,dtype):
    source_shape,base_shape,offsets,strides=map(tuple,(source_shape,base_shape,offsets,strides))
    base=plan_source_slice(base_shape,(0,)*len(base_shape),base_shape,(1,)*len(base_shape),dtype)
    region=plan_source_slice(base_shape,offsets,source_shape,strides,dtype)
    if kernel!=_insert_kernel(source_shape,base_shape,offsets,strides,dtype):
        raise SourceSliceError('insert-slice copy, inverted projection or completion changed')
    return dict(schema='submission.static_integer_insert_slice.v1',kernel_sha256=kernel.digest(),
        source_shape=list(source_shape),base_shape=list(base_shape),offsets=list(offsets),strides=list(strides),
        base_copy=base.revalidate(),inverted_slice_projection=region.revalidate(),
        proof_method='actual stream equals complete base copy followed by inverted exact slice projection',
        coverage='all base elements retained; only the declared slice is overwritten by the source',
        numeric='unchanged signed integer payload; no arithmetic or dtype conversion',
        completion='full base copy completes before overwrite; each DMA tile completes before row reuse')


@dataclass(frozen=True)
class SourceInsertSlicePlan:
    source_shape: tuple
    base_shape: tuple
    offsets: tuple
    strides: tuple
    dtype: str
    kernel: Kernel
    proof: dict

    def revalidate(self):
        proof=check_insert_slice(self.kernel,self.source_shape,self.base_shape,self.offsets,self.strides,self.dtype)
        if proof!=self.proof:raise SourceSliceError('insert-slice proof receipt changed')
        return proof


def plan_source_insert_slice(source_shape,base_shape,offsets,strides,dtype):
    source_shape,base_shape,offsets,strides=map(tuple,(source_shape,base_shape,offsets,strides))
    kernel=_insert_kernel(source_shape,base_shape,offsets,strides,dtype)
    return SourceInsertSlicePlan(source_shape,base_shape,offsets,strides,dtype,kernel,
        check_insert_slice(kernel,source_shape,base_shape,offsets,strides,dtype))


def lower_source_insert_slice(ctx,operands,attributes):
    src,base,dst=(operands[k] for k in ('src','base','dst'))
    a,b,c=(ctx.spec(v) for v in (src,base,dst));aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    if dst in (src,base) or any(v in aliases for v in (src,base,dst)):
        raise SourceSliceError('insert-slice requires a distinct materialized output buffer')
    if (a['dtype']!=b['dtype'] or b['dtype']!=c['dtype'] or b['shape']!=c['shape']
            or attributes.get('output_dtype',a['dtype'])!=a['dtype']
            or list(attributes.get('sizes',a['shape']))!=list(a['shape'])
            or set(attributes)-{'offsets','sizes','strides','output_dtype'}):
        raise SourceSliceError('insert-slice must preserve declared source/base/output integer types and shapes')
    plan=plan_source_insert_slice(a['shape'],b['shape'],attributes['offsets'],attributes['strides'],a['dtype'])
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,arguments):
        if name=='insert_drain':ctx.fence();return
        if name=='insert_config_load':ctx.emit(*isa.config_ld(arguments['stride'],shrink=plan.dtype=='i8'));return
        if name=='insert_config_store':ctx.emit(*isa.config_st(arguments['stride']));return
        args={k:bridge.value(v) for k,v in arguments.items()}
        descriptor=isa.pack_matrix(args['addr'],args['rows'],args['cols'],fb=ctx.fb)
        if name in ('insert_src_mvin','insert_base_mvin'):
            ctx.emit(isa.FUNCT['mvin'],ctx.ptr(src if name=='insert_src_mvin' else base,args['offset']),descriptor)
        elif name=='insert_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,args['offset']),descriptor)
        else:raise SourceSliceError('unrecognized source insert-slice instruction')
    return {'proof':plan.revalidate(),'emission':emit_index_program(plan.kernel,bridge,emit)}
