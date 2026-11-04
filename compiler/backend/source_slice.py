"""Exact static integer slices carried by checked accumulator DMA programs.

The scalar host program issues transfers only. Address proofs cover the emitted
counted loops, tail descriptors, output coverage and accumulator completion.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Sum, TensorArg, Var, as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa


class SliceError(ValueError):
    pass


def _pitches(shape):
    return tuple(prod(shape[i+1:]) for i in range(len(shape)))


def _call(name, **args):
    return Call(name, tuple((k,as_expr(v)) for k,v in args.items()))


@dataclass(frozen=True)
class SlicePlan:
    kernel: Kernel
    proof: dict
    dtype: str


def plan_slice(shape, offsets, sizes, strides, dtype):
    shape,offsets,sizes,strides=map(tuple,(shape,offsets,sizes,strides))
    return _plan_slice(shape,offsets,sizes,strides,dtype)


@lru_cache(maxsize=512)
def _plan_slice(shape,offsets,sizes,strides,dtype):
    rank=len(shape)
    if (not 1<=rank<=4 or any(len(v)!=rank for v in (offsets,sizes,strides))
            or any(type(x) is not int for v in (shape,offsets,sizes,strides) for x in v)
            or any(n<=0 or z<=0 or s<=0 or o<0 or o+(z-1)*s>=n
                   for n,o,z,s in zip(shape,offsets,sizes,strides)) or dtype not in ('i8','i32')):
        raise SliceError('slice requires bounded positive static rank1-4 same-rank signed integer geometry')
    width=1 if dtype=='i8' else 4
    if max(prod(shape),prod(sizes))*width>=1<<63:raise SliceError('slice byte footprint exceeds i64')
    ss=_pitches(shape);ds=_pitches(sizes)
    # Columns must be contiguous in both source and result. A stepped final
    # axis instead becomes the row axis, whose input DMA pitch is explicit.
    run=sizes[-1] if strides[-1]==1 else 1
    row_axis=rank-2 if strides[-1]==1 else rank-1
    rows=sizes[row_axis] if row_axis>=0 else 1
    sp=ss[row_axis]*strides[row_axis] if row_axis>=0 else run
    dp=run
    if max(sp,dp)*width>=1<<32:raise SliceError('slice DMA pitch exceeds selected configuration field')
    prefix=range(max(0,row_axis))
    src_base=Const(sum(o*p for o,p in zip(offsets,ss)))+sum((Var(f'p{i}')*ss[i]*strides[i] for i in prefix),Const(0))
    dst_base=sum((Var(f'p{i}')*ds[i] for i in prefix),Const(0))
    def tile(r,nr,c,nc):
        return (_call('slice_mvin',offset=src_base+r*sp+c,addr=isa.acc_addr(0),rows=nr,cols=nc),
                _call('slice_drain'),
                _call('slice_mvout',offset=dst_base+r*dp+c,addr=isa.acc_addr(0,full=dtype=='i32'),rows=nr,cols=nc),
                _call('slice_drain'))
    def columns(r,nr):
        full,tail=divmod(run,isa.DIM);body=[]
        if full:body.append(Loop('c',full,tile(r,nr,Var('c')*isa.DIM,isa.DIM)))
        if tail:body.extend(tile(r,nr,Const(full*isa.DIM),tail))
        return tuple(body)
    full,tail=divmod(rows,isa.DIM);body=[]
    if full:body.append(Loop('r',full,columns(Var('r')*isa.DIM,isa.DIM)))
    if tail:body.extend(columns(Const(full*isa.DIM),tail))
    body=tuple(body)
    for i in reversed(tuple(prefix)):body=(Loop(f'p{i}',sizes[i],body),)
    configs=(_call('slice_drain'),_call('slice_config_load',stride=sp*width),_call('slice_config_store',stride=dp*width))
    kernel=Kernel('integer_slice',(TensorArg('src',shape,dtype,'read'),TensorArg('dst',sizes,dtype,'write')),configs+body)
    proof=check_slice(kernel,shape,offsets,sizes,strides,dtype)
    return SlicePlan(kernel,proof,dtype)


def check_slice(kernel,shape,offsets,sizes,strides,dtype):
    """Check actual addresses, in-bounds injective coverage and DMA order."""
    import z3
    width=1 if dtype=='i8' else 4
    if kernel.args!=(TensorArg('src',tuple(shape),dtype,'read'),TensorArg('dst',tuple(sizes),dtype,'write')):
        raise SliceError('slice ABI changed')
    first=kernel.body[:3]
    if any(not isinstance(o,Call) for o in first) or [o.instr for o in first]!=['slice_drain','slice_config_load','slice_config_store']:
        raise SliceError('slice configuration or completion prologue changed')
    def integer(call,key):
        v=call.arg(key)
        if not isinstance(v,Const):raise SliceError('slice descriptor is not static')
        return v.value
    sp,dp=integer(first[1],'stride'),integer(first[2],'stride')
    if min(sp,dp)<=0 or max(sp,dp)>=1<<32 or sp%width or dp%width:raise SliceError('invalid slice pitches')
    sites=[]
    def walk(body,scope):
        at=0
        while at<len(body):
            op=body[at]
            if isinstance(op,Loop):
                if op.var in dict(scope) or op.extent<=0 or op.iter_args or op.yields or op.results:raise SliceError('invalid slice loop')
                walk(op.body,scope+((op.var,op.extent),));at+=1;continue
            group=body[at:at+4]
            if len(group)!=4 or any(not isinstance(o,Call) for o in group) or [o.instr for o in group]!=['slice_mvin','slice_drain','slice_mvout','slice_drain']:
                raise SliceError('slice accumulator completion/order changed')
            a,_,b,_=group;r,c=integer(a,'rows'),integer(a,'cols')
            if (not 0<r<=min(isa.DIM,isa.ACC_ROWS) or not 0<c<=isa.DIM
                    or integer(b,'rows')!=r or integer(b,'cols')!=c
                    or integer(a,'addr')!=isa.acc_addr(0) or integer(b,'addr')!=isa.acc_addr(0,full=dtype=='i32')
                    or set(dict(a.args))!={'offset','rows','cols','addr'} or set(dict(b.args))!={'offset','rows','cols','addr'}):
                raise SliceError('slice descriptor or local footprint changed')
            sites.append((a,b,scope,r,c));at+=4
    walk(kernel.body[3:],())
    count=prod(sizes)
    if not sites or sum(prod(n for _,n in scope)*r*c for _,_,scope,r,c in sites)!=count:raise SliceError('slice cardinality differs')
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):return env[e.name]
        if isinstance(e,Sum):return sum(expr(x,env) for x in e.terms)
        if isinstance(e,Scaled):return e.factor*expr(e.expr,env)
        raise SliceError('unsupported slice expression')
    def event(site,label):
        a,b,scope,rows,cols=site
        env={v:z3.Int(label+'_'+v) for v,_ in scope};r=z3.Int(label+'_row');c=z3.Int(label+'_col')
        bounds=[b for v,n in scope for b in (env[v]>=0,env[v]<n)]+[r>=0,r<rows,c>=0,c<cols]
        return bounds,expr(a.arg('offset'),env)+r*(sp//width)+c,expr(b.arg('offset'),env)+r*(dp//width)+c,tuple(env[v] for v,_ in scope)+(r,c)
    queries=0
    def prove(bounds,bad):
        nonlocal queries
        solver=z3.Solver();solver.set(timeout=5000);solver.add(*bounds,bad);queries+=1
        result=solver.check()
        if result!=z3.unsat:raise SliceError('slice projection unproved: '+str(result))
    ss,ds=_pitches(shape),_pitches(sizes)
    for i,site in enumerate(sites):
        bounds,src,dst,ids=event(site,'a'+str(i))
        ref=sum((o+((dst/p)%n)*s)*pitch for o,p,n,s,pitch in zip(offsets,ds,sizes,strides,ss))
        prove(bounds,z3.Or(dst<0,dst>=count,src<0,src>=prod(shape),src!=ref))
        for j,other in enumerate(sites[:i+1]):
            rhs,_,out,other_ids=event(other,'b'+str(j))
            distinct=z3.Or(*(a!=b for a,b in zip(ids,other_ids))) if i==j else z3.BoolVal(True)
            prove(bounds+rhs,z3.And(dst==out,distinct))
    return dict(schema='submission.slice_address_projection.v1',kernel_sha256=kernel.digest(),solver_queries=queries,
        source_shape=list(shape),offsets=list(offsets),sizes=list(sizes),strides=list(strides),output_elements=count,
        dynamic_dma_pairs=sum(prod(n for _,n in scope) for _,_,scope,_,_ in sites),
        source_row_stride_bytes=sp,destination_row_stride_bytes=dp,requested_load_bytes=count*width,requested_store_bytes=count*width,
        coverage='in-bounds injective output addresses with exact source cardinality',
        semantics='signed i8 identity narrowing' if dtype=='i8' else 'raw full-width i32 readout',
        ownership='MVIN; full drain; MVOUT; full drain before row reuse')


def lower_slice(ctx,operands,attributes):
    src,dst=operands['src'],operands['dst'];a,b=ctx.spec(src),ctx.spec(dst)
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    if (src==dst or src in aliases or dst in aliases
            or a['dtype']!=b['dtype'] or attributes.get('output_dtype')!=a['dtype']
            or attributes.get('epilogue',[]) or attributes.get('relu')
            or attributes.get('acc_scale',1.0)!=1.0 or attributes.get('acc_scale_by_column') is not None
            or attributes.get('bias') or attributes.get('accumulator_init')):
        raise SliceError('slice cannot alias or change numeric values')
    plan=plan_slice(a['shape'],attributes['offsets'],attributes['sizes'],attributes['strides'],a['dtype'])
    if list(b['shape'])!=attributes['sizes']:raise SliceError('slice result shape changed')
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,args):
        if name=='slice_drain':ctx.fence();return
        if name=='slice_config_load':ctx.emit(*isa.config_ld(args['stride'],shrink=plan.dtype=='i8'));return
        if name=='slice_config_store':ctx.emit(*isa.config_st(args['stride']));return
        values={k:bridge.value(v) for k,v in args.items()}
        packed=isa.pack_matrix(values['addr'],values['rows'],values['cols'],fb=ctx.fb)
        if name=='slice_mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(src,values['offset']),packed)
        elif name=='slice_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,values['offset']),packed)
        else:raise SliceError('unrecognized checked slice operation')
    emitted=emit_index_program(plan.kernel,bridge,emit)
    receipt=dict(proof=plan.proof,emission=emitted,kernel_sha256=plan.kernel.digest())
    ctx.metadata.setdefault('source_slice',[]).append(receipt)
    return receipt
