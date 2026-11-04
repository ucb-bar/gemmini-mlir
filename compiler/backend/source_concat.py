"""Static typed concatenation with source-bound checked integer DMA effects."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Sum, TensorArg, Var, as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa


class ConcatError(ValueError):
    pass


def _call(name,**args):
    return Call(name,tuple((k,as_expr(v)) for k,v in args.items()))


@dataclass(frozen=True)
class ConcatPlan:
    kernel: Kernel
    proof: dict
    output: tuple
    dtype: str


def plan_concat(shapes,axis,dtype):
    return _plan_concat(tuple(tuple(s) for s in shapes),axis,dtype)


@lru_cache(maxsize=256)
def _plan_concat(shapes,axis,dtype):
    if not shapes or len(shapes)>64:raise ConcatError('concat requires 1..64 explicit inputs')
    rank=len(shapes[0])
    if (not 1<=rank<=8 or type(axis) is not int or not 0<=axis<rank or dtype not in ('i8','i32')
            or any(len(s)!=rank or any(type(n) is not int or n<=0 for n in s) for s in shapes)
            or any(s[j]!=shapes[0][j] for s in shapes for j in range(rank) if j!=axis)):
        raise ConcatError('concat requires static rank1-8 integer inputs with matching non-concatenated axes')
    output=list(shapes[0]);output[axis]=sum(s[axis] for s in shapes);output=tuple(output)
    width=1 if dtype=='i8' else 4
    if prod(output)*width>=1<<63:raise ConcatError('concat byte footprint exceeds i64')
    prefix=prod(output[:axis]);suffix=prod(output[axis:]);axis_step=prod(output[axis+1:])
    body=[];position=0
    for i,shape in enumerate(shapes):
        run=prod(shape[axis:]);base=position*axis_step
        def tile(offset,rows,cols):
            return (_call('concat_mvin',input=i,offset=Var('p')*run+offset,addr=isa.acc_addr(0),rows=rows,cols=cols),
                _call('concat_drain'),
                _call('concat_mvout',offset=Var('p')*suffix+base+offset,addr=isa.acc_addr(0,full=dtype=='i32'),rows=rows,cols=cols),
                _call('concat_drain'))
        full,tail=divmod(run,isa.DIM*isa.DIM);pieces=[]
        if full:pieces.append(Loop('t',full,tile(Var('t')*(isa.DIM*isa.DIM),isa.DIM,isa.DIM)))
        rows,cols=divmod(tail,isa.DIM)
        if rows:pieces.extend(tile(Const(full*isa.DIM*isa.DIM),rows,isa.DIM))
        if cols:pieces.extend(tile(Const(full*isa.DIM*isa.DIM+rows*isa.DIM),1,cols))
        body.append(Loop('p',prefix,tuple(pieces)));position+=shape[axis]
    prologue=(_call('concat_drain'),_call('concat_config_load',stride=isa.DIM*width),_call('concat_config_store',stride=isa.DIM*width))
    args=tuple(TensorArg(f'src{i}',s,dtype,'read') for i,s in enumerate(shapes))+(TensorArg('dst',output,dtype,'write'),)
    kernel=Kernel('integer_concat',args,prologue+tuple(body))
    proof=check_concat(kernel,shapes,axis,dtype)
    return ConcatPlan(kernel,proof,output,dtype)


def check_concat(kernel,shapes,axis,dtype):
    """Prove each emitted transfer equals one disjoint source concat segment."""
    import z3
    width=1 if dtype=='i8' else 4
    output=list(shapes[0]);output[axis]=sum(s[axis] for s in shapes);output=tuple(output)
    expected=tuple(TensorArg(f'src{i}',tuple(s),dtype,'read') for i,s in enumerate(shapes))+(TensorArg('dst',output,dtype,'write'),)
    if kernel.args!=expected:raise ConcatError('concat boundary changed')
    def integer(call,key):
        value=call.arg(key)
        if not isinstance(value,Const):raise ConcatError('concat static descriptor became dynamic')
        return value.value
    first=kernel.body[:3]
    if (any(not isinstance(o,Call) for o in first)
            or [o.instr for o in first]!=['concat_drain','concat_config_load','concat_config_store']
            or integer(first[1],'stride')!=isa.DIM*width or integer(first[2],'stride')!=isa.DIM*width):
        raise ConcatError('concat configuration/order changed')
    sites=[]
    def walk(body,scope):
        at=0
        while at<len(body):
            op=body[at]
            if isinstance(op,Loop):
                if op.var in dict(scope) or op.extent<=0 or op.iter_args or op.yields or op.results:raise ConcatError('invalid concat loop')
                walk(op.body,scope+((op.var,op.extent),));at+=1;continue
            group=body[at:at+4]
            if len(group)!=4 or any(not isinstance(o,Call) for o in group) or [o.instr for o in group]!=['concat_mvin','concat_drain','concat_mvout','concat_drain']:
                raise ConcatError('concat accumulator order/completion changed')
            a,_,b,_=group;r,c=integer(a,'rows'),integer(a,'cols');index=integer(a,'input')
            if (not 0<=index<len(shapes) or not 0<r<=min(isa.DIM,isa.ACC_ROWS) or not 0<c<=isa.DIM
                    or integer(b,'rows')!=r or integer(b,'cols')!=c
                    or integer(a,'addr')!=isa.acc_addr(0) or integer(b,'addr')!=isa.acc_addr(0,full=dtype=='i32')
                    or len(a.args)!=5 or set(dict(a.args))!={'input','offset','addr','rows','cols'}
                    or len(b.args)!=4 or set(dict(b.args))!={'offset','addr','rows','cols'}):raise ConcatError('concat descriptor mismatch')
            sites.append((index,a,b,scope,r,c));at+=4
    walk(kernel.body[3:],())
    for i,shape in enumerate(shapes):
        if sum(prod(n for _,n in scope)*r*c for index,_,_,scope,r,c in sites if index==i)!=prod(shape):raise ConcatError('concat source cardinality differs')
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):return env[e.name]
        if isinstance(e,Sum):return sum(expr(x,env) for x in e.terms)
        if isinstance(e,Scaled):return e.factor*expr(e.expr,env)
        raise ConcatError('unsupported concat address')
    def event(site,label):
        _,a,b,scope,r,c=site
        env={v:z3.Int(label+'_'+v) for v,_ in scope};row=z3.Int(label+'_r');col=z3.Int(label+'_c')
        bounds=[b for v,n in scope for b in (env[v]>=0,env[v]<n)]+[row>=0,row<r,col>=0,col<c]
        return bounds,expr(a.arg('offset'),env)+row*isa.DIM+col,expr(b.arg('offset'),env)+row*isa.DIM+col,tuple(env[v] for v,_ in scope)+(row,col)
    queries=0
    def prove(bounds,bad):
        nonlocal queries
        solver=z3.Solver();solver.set(timeout=5000);solver.add(*bounds,bad);queries+=1
        status=solver.check()
        if status!=z3.unsat:raise ConcatError('concat address projection unproved: '+str(status))
    suffix=prod(output[axis:]);axis_step=prod(output[axis+1:])
    for i,site in enumerate(sites):
        index=site[0];bounds,src,dst,ids=event(site,'a'+str(i));run=prod(shapes[index][axis:])
        ref=(src/run)*suffix+sum(s[axis] for s in shapes[:index])*axis_step+src%run
        prove(bounds,z3.Or(src<0,src>=prod(shapes[index]),dst<0,dst>=prod(output),dst!=ref))
        # Other input segments are disjoint by the source shape/axis partition;
        # within this segment prove the actual DMA source coverage injective.
        for j,other in enumerate(sites[:i+1]):
            if other[0]!=index:continue
            rhs,other_src,_,other_ids=event(other,'b'+str(j))
            distinct=z3.Or(*(a!=b for a,b in zip(ids,other_ids))) if i==j else z3.BoolVal(True)
            prove(bounds+rhs,z3.And(src==other_src,distinct))
    count=prod(output)
    return dict(schema='submission.concat_address_projection.v1',kernel_sha256=kernel.digest(),solver_queries=queries,
        input_shapes=[list(s) for s in shapes],axis=axis,output_shape=list(output),output_elements=count,
        dynamic_dma_pairs=sum(prod(n for _,n in scope) for _,_,_,scope,_,_ in sites),
        requested_load_bytes=count*width,requested_store_bytes=count*width,
        coverage='in-bounds injective complete source coverage; exact source-to-output concat partition',
        ownership='MVIN; full drain; MVOUT; full drain before row reuse',
        semantics='signed i8 identity narrowing' if dtype=='i8' else 'raw full-width i32 readout')


def lower_concat(ctx,operands,attributes):
    dst=operands['dst'];sources=[operands[f'src{i}'] for i in range(attributes['inputs'])]
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    dtype=ctx.spec(dst)['dtype']
    if (dst in sources or any(s in aliases for s in [*sources,dst])
            or any(ctx.spec(s)['dtype']!=dtype for s in sources) or attributes.get('output_dtype')!=dtype
            or attributes.get('epilogue',[]) or attributes.get('relu') or attributes.get('acc_scale',1.0)!=1.0):raise ConcatError('concat cannot alias output or change numeric values')
    plan=plan_concat([ctx.spec(s)['shape'] for s in sources],attributes['axis'],dtype)
    if tuple(ctx.spec(dst)['shape'])!=plan.output:raise ConcatError('concat destination shape changed')
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,args):
        if name=='concat_drain':ctx.fence();return
        if name=='concat_config_load':ctx.emit(*isa.config_ld(args['stride'],shrink=dtype=='i8'));return
        if name=='concat_config_store':ctx.emit(*isa.config_st(args['stride']));return
        if name=='concat_mvin':
            index=args.pop('input')
            if not isinstance(index,int):raise ConcatError('concat input selector is not static')
        values={k:bridge.value(v) for k,v in args.items()}
        packed=isa.pack_matrix(values['addr'],values['rows'],values['cols'],fb=ctx.fb)
        if name=='concat_mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(sources[index],values['offset']),packed)
        elif name=='concat_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,values['offset']),packed)
        else:raise ConcatError('unrecognized checked concat operation')
    receipt={'proof':plan.proof,'emission':emit_index_program(plan.kernel,bridge,emit),'kernel_sha256':plan.kernel.digest()}
    ctx.metadata.setdefault('source_concat',[]).append(receipt)
    return receipt
