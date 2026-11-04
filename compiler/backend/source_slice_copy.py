"""Source-bound materialized copies for static signed-i8 tensor slices.

The immutable counted DMA program is checked before that exact program reaches
common LLVM/ISA emission. This realizes a distinct result; it does not assert
that original tensor storage semantics required a runtime copy.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from merlin.common.symbolic_lock import serialized_symbolic_check
from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Sum, TensorArg, Var, as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa

class SliceCopyError(ValueError):
    pass

def _strides(shape):return tuple(prod(shape[i+1:]) for i in range(len(shape)))
def _call(name,**args):return Call(name,tuple((k,as_expr(v)) for k,v in args.items()))

@dataclass(frozen=True)
class SliceCopyPlan:
    source_shape: tuple
    result_shape: tuple
    offsets: tuple
    strides: tuple
    kernel: Kernel
    def revalidate(self):
        proof=check_projection(self.kernel,self.source_shape,self.result_shape,self.offsets,self.strides)
        ss=_strides(self.source_shape);ds=_strides(self.result_shape)
        index_strides=tuple(a*b for a,b in zip(ss,self.strides))
        return {**proof,'source_index_strides':list(index_strides),
                'is_dense_contiguous_result':all(n==1 or s==d for n,s,d in zip(self.result_shape,index_strides,ds)),
                'realization':'distinct materialized device result',
                'original_storage_alias':None,'original_physical_copy':None,
                'original_storage_realization':'unresolved'}

def validate_slice_copy(source_shape,result_shape,dtype,attrs):
    allowed={'source_shape','offsets','sizes','strides','output_dtype','semantic','epilogue','relu','acc_scale','bound_lsb'}
    if not isinstance(attrs,dict) or any(not isinstance(k,str) or (k not in allowed and not k.startswith('prov.') and k not in ('merlin.source_op_index','merlin.global_task')) for k in attrs):
        raise SliceCopyError('unknown slice attributes')
    if any(not isinstance(attrs.get(k),(list,tuple)) for k in ('source_shape','sizes','offsets','strides')):raise SliceCopyError('slice geometry must be static integer sequences')
    try:shape,output,offsets,strides=map(tuple,(source_shape,result_shape,attrs['offsets'],attrs['strides']))
    except (KeyError,TypeError):raise SliceCopyError('missing static slice geometry') from None
    if (not 1<=len(shape)<=4 or len(output)!=len(shape) or len(offsets)!=len(shape) or len(strides)!=len(shape)
        or any(type(n) is not int or n<=0 for n in shape+output+strides)
        or any(type(n) is not int or n<0 for n in offsets)
        or dtype!='i8' or attrs.get('output_dtype')!='i8'
        or attrs.get('semantic')!='slice_copy'
        or tuple(attrs.get('source_shape',()))!=shape or tuple(attrs.get('sizes',()))!=output
        or attrs.get('epilogue',[]) or attrs.get('relu',False) or attrs.get('acc_scale',1.0)!=1.0 or attrs.get('bound_lsb',0)!=0):
        raise SliceCopyError('requires exact static rank-preserving signed-i8 slice')
    if any(o+(n-1)*s>=d for d,n,o,s in zip(shape,output,offsets,strides)):
        raise SliceCopyError('slice coordinate exceeds source boundary')
    if max(prod(shape),prod(output))>=1<<63:raise SliceCopyError('slice byte footprint exceeds signed i64')
    return _plan(shape,output,offsets,strides)

@lru_cache(maxsize=512)
def _plan(shape,output,offsets,strides):
    ss=_strides(shape);ds=_strides(output)
    # A nonunit innermost stride is represented as one-column DMA rows.
    if strides[-1]==1:
        run=output[-1];row_axis=len(shape)-2
        rows=output[row_axis] if row_axis>=0 else 1
        sp=ss[row_axis]*strides[row_axis] if row_axis>=0 else run;dp=run
    else:
        run=1;row_axis=len(shape)-1;rows=output[-1];sp=strides[-1];dp=1
    if max(sp,dp)>=1<<32:raise SliceCopyError('slice DMA row pitch exceeds configuration field')
    prefix=range(max(0,row_axis))
    base=sum(o*s for o,s in zip(offsets,ss))
    src=sum((Var(f'p{i}')*ss[i]*strides[i] for i in prefix),Const(base))
    dst=sum((Var(f'p{i}')*ds[i] for i in prefix),Const(0))
    def tile(r,n,c,m):
        return (_call('slice_mvin',offset=src+r*sp+c,addr=isa.acc_addr(0),rows=n,cols=m),
                _call('slice_drain'),_call('slice_mvout',offset=dst+r*dp+c,addr=isa.acc_addr(0),rows=n,cols=m),_call('slice_drain'))
    def columns(r,n):
        full,tail=divmod(run,isa.DIM);body=[]
        if full:body.append(Loop('c',full,tile(r,n,Var('c')*isa.DIM,isa.DIM)))
        if tail:body.extend(tile(r,n,Const(full*isa.DIM),tail))
        return tuple(body)
    full,tail=divmod(rows,isa.DIM);body=[]
    if full:body.append(Loop('r',full,columns(Var('r')*isa.DIM,isa.DIM)))
    if tail:body.extend(columns(Const(full*isa.DIM),tail))
    body=tuple(body)
    for i in reversed(tuple(prefix)):body=(Loop(f'p{i}',output[i],body),)
    kernel=Kernel('source_slice_copy',(TensorArg('src',shape,'i8','read'),TensorArg('dst',output,'i8','write')),
        (_call('slice_drain'),_call('slice_config_load',stride=sp),_call('slice_config_store',stride=dp))+body)
    plan=SliceCopyPlan(shape,output,offsets,strides,kernel)
    plan.revalidate()
    return plan

@serialized_symbolic_check
def check_projection(kernel, shape, output, offsets, strides):
    """Check actual instruction fields, including tails, with bounded integer proofs.

    Injectivity plus the exact dynamic element count establishes full output
    coverage without enumerating the tensor or using a sampled input palette.
    """
    import z3
    shape,output,offsets,strides=map(tuple,(shape,output,offsets,strides))
    dtype='i8';width=1;ss=_strides(shape);ds=_strides(output);count=prod(output)
    expected=(TensorArg('src',shape,dtype,'read'),TensorArg('dst',output,dtype,'write'))
    if kernel.args!=expected or kernel.attrs:raise SliceCopyError('slice source boundary changed')
    first=kernel.body[:3]
    if len(first)!=3 or any(not isinstance(o,Call) for o in first) or [o.instr for o in first]!=['slice_drain','slice_config_load','slice_config_store']:
        raise SliceCopyError('slice configuration/completion prologue changed')
    def integer(call,key):
        x=call.arg(key)
        if not isinstance(x,Const):raise SliceCopyError('slice static descriptor became dynamic')
        return x.value
    if first[0].args or any(set(dict(o.args))!={'stride'} for o in first[1:]):raise SliceCopyError('unexpected configuration fields')
    sp,dp=integer(first[1],'stride'),integer(first[2],'stride')
    if min(sp,dp)<=0 or max(sp,dp)>=1<<32 or sp%width or dp%width:
        raise SliceCopyError('invalid slice DMA row pitches')
    sites=[]
    def walk(body,scope):
        at=0
        while at<len(body):
            op=body[at]
            if isinstance(op,Loop):
                if op.var in dict(scope) or op.extent<=0 or op.iter_args or op.yields or op.results:
                    raise SliceCopyError('invalid slice counted loop')
                walk(op.body,scope+((op.var,op.extent),));at+=1;continue
            group=body[at:at+4]
            if len(group)!=4 or any(not isinstance(o,Call) for o in group) or [o.instr for o in group]!=['slice_mvin','slice_drain','slice_mvout','slice_drain']:
                raise SliceCopyError('slice local ownership or command order changed')
            load,before,store,after=group
            if before.args or after.args:raise SliceCopyError('unexpected drain fields')
            rows,cols=integer(load,'rows'),integer(load,'cols')
            if (not 0<rows<=min(isa.DIM,isa.ACC_ROWS) or not 0<cols<=isa.DIM
                    or integer(store,'rows')!=rows or integer(store,'cols')!=cols
                    or integer(load,'addr')!=isa.acc_addr(0)
                    or integer(store,'addr')!=isa.acc_addr(0,full=dtype=='i32')):
                raise SliceCopyError('slice descriptor/local footprint mismatch')
            if set(dict(load.args))!={'offset','rows','cols','addr'} or set(dict(store.args))!={'offset','rows','cols','addr'}:
                raise SliceCopyError('unexpected slice DMA descriptor fields')
            sites.append((load,store,scope,rows,cols));at+=4
    walk(kernel.body[3:],())
    if not sites or sum(prod(n for _,n in s)*r*c for _,_,s,r,c in sites)!=count:
        raise SliceCopyError('slice transferred element count differs from source')
    queries=0
    def check(conditions,bad):
        nonlocal queries
        solver=z3.Solver();solver.set(timeout=5000);solver.add(*conditions,bad);queries+=1
        status=solver.check()
        if status!=z3.unsat:raise SliceCopyError('slice address projection not proved: '+str(status))
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):return env[e.name]
        if isinstance(e,Sum):return sum(expr(x,env) for x in e.terms)
        if isinstance(e,Scaled):return e.factor*expr(e.expr,env)
        raise SliceCopyError('unsupported slice address expression')
    def event(site,label):
        load,store,scope,rows,cols=site
        env={v:z3.Int(label+'_'+v) for v,_ in scope};r=z3.Int(label+'_row');c=z3.Int(label+'_col')
        bounds=[b for v,n in scope for b in (env[v]>=0,env[v]<n)]+[r>=0,r<rows,c>=0,c<cols]
        dst=expr(store.arg('offset'),env)+r*(dp//width)+c
        src=expr(load.arg('offset'),env)+r*(sp//width)+c
        ids=tuple(env[v] for v,_ in scope)+(r,c)
        return bounds,src,dst,ids
    for i,site in enumerate(sites):
        bounds,src,dst,_=event(site,'a'+str(i))
        reference=sum((offsets[i]+((dst/ds[i])%output[i])*strides[i])*ss[i] for i in range(len(shape)))
        check(bounds,z3.Or(dst<0,dst>=count,src<0,src>=prod(shape),src!=reference))
        for j,other in enumerate(sites[:i+1]):
            rhs,_,out,ids=event(other,'b'+str(j));lhs,_,_,myids=event(site,'a'+str(i))
            distinct=z3.Or(*(a!=b for a,b in zip(myids,ids))) if i==j else z3.BoolVal(True)
            check(lhs+rhs,z3.And(dst==out,distinct))
    return dict(schema='submission.slice_address_projection.v1',kernel_sha256=kernel.digest(),
                solver_queries=queries,source_shape=list(shape),output_shape=list(output),offsets=list(offsets),strides=list(strides),
                output_elements=count,actual_dma_sites=len(sites),
                dynamic_dma_pairs=sum(prod(n for _,n in scope) for _,_,scope,_,_ in sites),
                requested_load_bytes=count*width,requested_store_bytes=count*width,
                actual_tile_columns=sorted({cols for _,_,_,_,cols in sites}),
                source_row_stride_bytes=sp,destination_row_stride_bytes=dp,
                coverage='in-bounds injective output addresses with exact source cardinality',
                semantics='signed i8 identity narrowing' if dtype=='i8' else 'raw full-width i32 readout',
                ownership='MVIN then full drain then MVOUT then full drain before row reuse',
                alias_contract='distinct ordinary ABI source/output buffers')


def match_slice_copy(op):
    from merlin.common import mlir_query as mq
    from .source_region_diagnostics import describe_source_operation
    if mq.op_name(op)!='tensor.extract_slice':return None
    described=describe_source_operation(op)
    if described is None:return None
    _,metadata=described
    a=metadata['inputs'][0];b=metadata['outputs'][0];p=metadata['structural_proof']
    attrs=dict(source_shape=list(a['shape']),offsets=list(p['offsets']),sizes=list(p['sizes']),
               strides=list(p['strides']),output_dtype='i8',semantic='slice_copy')
    try:
        if b['dtype']!='i8':return None
        plan=validate_slice_copy(a['shape'],b['shape'],a['dtype'],attrs)
    except SliceCopyError:return None
    return op.operands[0],attrs,{**metadata,**plan.revalidate()}


def lower_slice_copy(ctx,operands,attrs):
    if set(operands)!={'src','dst'}:raise SliceCopyError('slice requires source and destination')
    src,dst=operands['src'],operands['dst'];a,b=ctx.spec(src),ctx.spec(dst)
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    if src==dst or src in aliases or dst in aliases:
        raise SliceCopyError('slice requires distinct materialized source/output ABI buffers')
    if b['dtype']!='i8':raise SliceCopyError('slice result must be signed i8')
    plan=validate_slice_copy(a['shape'],b['shape'],a['dtype'],attrs)
    proof=plan.revalidate()
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,arguments):
        if name=='slice_drain':ctx.fence();return
        if name=='slice_config_load':ctx.emit(*isa.config_ld(arguments['stride'],shrink=True));return
        if name=='slice_config_store':ctx.emit(*isa.config_st(arguments['stride']));return
        values={k:bridge.value(v) for k,v in arguments.items()}
        packed=isa.pack_matrix(values['addr'],values['rows'],values['cols'],fb=ctx.fb)
        if name=='slice_mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(src,values['offset']),packed)
        elif name=='slice_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,values['offset']),packed)
        else:raise SliceCopyError('unknown checked slice operation')
    emitted=emit_index_program(plan.kernel,bridge,emit)
    return dict(proof=proof,emission=emitted,kernel_sha256=plan.kernel.digest())
