"""Exact static f32 concatenation through raw full-width accumulator DMA.

Each input owns a disjoint interval of each output outer row. Checked counted
DMA tiles copy those intervals without reading or computing payloads on CPU.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import prod
from merlin.sched.ir import Call, Const, Kernel, Loop, Sum, Scaled, TensorArg, Var, as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa

class BitConcatError(ValueError):
    pass

def _call(name, **args):
    return Call(name, tuple((k,as_expr(v)) for k,v in args.items()))

@dataclass(frozen=True)
class BitConcatPlan:
    shapes: tuple
    axis: int
    output_shape: tuple
    kernel: Kernel
    proof: dict
    def revalidate(self):
        return check_concat(self.kernel,self.shapes,self.axis)

def plan_concat(shapes,axis,dtype='f32'):
    shapes=tuple(tuple(s) for s in shapes)
    if (dtype!='f32' or not 2<=len(shapes)<=16 or not shapes
            or not 2<=len(shapes[0])<=4 or type(axis) is not int
            or not 0<=axis<len(shapes[0])
            or any(len(s)!=len(shapes[0]) or any(type(n) is not int or n<=0 for n in s) for s in shapes)
            or any(s[j]!=shapes[0][j] for s in shapes for j in range(len(s)) if j!=axis)):
        raise BitConcatError('concat requires two to sixteen static dense rank2-4 f32 inputs with exact non-concat axes')
    output=list(shapes[0]);output[axis]=sum(s[axis] for s in shapes);output=tuple(output)
    if prod(output)*4>=1<<63:raise BitConcatError('concat byte footprint exceeds signed i64')
    inner=prod(output[axis+1:]);outer=prod(output[:axis]);output_run=output[axis]*inner
    body=[_call('concat_drain'),_call('concat_config')];base=0
    for member,shape in enumerate(shapes):
        run=shape[axis]*inner
        def tile(offset,rows,cols):
            args={'member':member,'offset':Var('outer')*run+offset,'addr':isa.acc_addr(0),'rows':rows,'cols':cols}
            return (_call('concat_mvin',**args),_call('concat_drain'),
                    _call('concat_mvout',offset=Var('outer')*output_run+base+offset,
                          addr=isa.acc_addr(0,full=True),rows=rows,cols=cols),_call('concat_drain'))
        full,tail=divmod(run,isa.DIM*isa.DIM);piece=[]
        if full:piece.append(Loop('tile',full,tile(Var('tile')*(isa.DIM*isa.DIM),isa.DIM,isa.DIM)))
        rows,cols=divmod(tail,isa.DIM)
        if rows:piece.extend(tile(Const(full*isa.DIM*isa.DIM),rows,isa.DIM))
        if cols:piece.extend(tile(Const(full*isa.DIM*isa.DIM+rows*isa.DIM),1,cols))
        body.append(Loop('outer',outer,tuple(piece)));base+=run
    args=tuple(TensorArg('src'+str(i),s,'f32','read') for i,s in enumerate(shapes))+(TensorArg('dst',output,'f32','write'),)
    kernel=Kernel('raw_bit_concat',args,tuple(body))
    proof=check_concat(kernel,shapes,axis)
    return BitConcatPlan(shapes,axis,output,kernel,proof)

def check_concat(kernel,shapes,axis):
    """Prove actual DMA offsets and disjoint complete output ownership."""
    import z3
    shapes=tuple(tuple(s) for s in shapes);output=list(shapes[0]);output[axis]=sum(s[axis] for s in shapes);output=tuple(output)
    expected=tuple(TensorArg('src'+str(i),s,'f32','read') for i,s in enumerate(shapes))+(TensorArg('dst',output,'f32','write'),)
    if kernel.args!=expected:raise BitConcatError('concat tensor boundary changed')
    if kernel.body[:2]!=(_call('concat_drain'),_call('concat_config')) or len(kernel.body)!=2+len(shapes):
        raise BitConcatError('concat configuration or member partition changed')
    inner=prod(output[axis+1:]);outer=prod(output[:axis]);output_run=output[axis]*inner;base=0;queries=0;sites_total=0;intervals=[]
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):return env[e.name]
        if isinstance(e,Sum):return sum(expr(x,env) for x in e.terms)
        if isinstance(e,Scaled):return e.factor*expr(e.expr,env)
        raise BitConcatError('unsupported concat address expression')
    def integer(call,key):
        e=call.arg(key)
        if not isinstance(e,Const):raise BitConcatError('concat descriptor must be static')
        return e.value
    for member,(shape,loop) in enumerate(zip(shapes,kernel.body[2:])):
        run=shape[axis]*inner;sites=[]
        if not isinstance(loop,Loop) or loop.var!='outer' or loop.extent!=outer or loop.iter_args or loop.yields or loop.results:raise BitConcatError('concat outer ownership changed')
        def walk(body,scope):
            at=0
            while at<len(body):
                op=body[at]
                if isinstance(op,Loop):
                    if op.var in dict(scope) or op.extent<=0 or op.iter_args or op.yields or op.results:raise BitConcatError('invalid concat counted loop')
                    walk(op.body,scope+((op.var,op.extent),));at+=1;continue
                group=body[at:at+4]
                if len(group)!=4 or any(not isinstance(o,Call) for o in group) or [o.instr for o in group]!=['concat_mvin','concat_drain','concat_mvout','concat_drain']:raise BitConcatError('concat completion or row ownership changed')
                load,drain1,store,drain2=group
                if drain1.args or drain2.args:raise BitConcatError('concat completion descriptor changed')
                r,c=integer(load,'rows'),integer(load,'cols')
                if (not 0<r<=min(isa.DIM,isa.ACC_ROWS) or not 0<c<=isa.DIM
                        or integer(store,'rows')!=r or integer(store,'cols')!=c
                        or integer(load,'member')!=member or integer(load,'addr')!=isa.acc_addr(0)
                        or integer(store,'addr')!=isa.acc_addr(0,full=True)
                        or set(dict(load.args))!={'member','offset','addr','rows','cols'}
                        or set(dict(store.args))!={'offset','addr','rows','cols'}):raise BitConcatError('concat DMA descriptor changed')
                sites.append((load,store,scope,r,c));at+=4
        walk(loop.body,(('outer',outer),))
        if sum(prod(n for _,n in scope)*r*c for _,_,scope,r,c in sites)!=outer*run:raise BitConcatError('concat member cardinality differs')
        def event(site,label):
            load,store,scope,r,c=site;env={v:z3.Int(label+'_'+v) for v,_ in scope};rr=z3.Int(label+'_r');cc=z3.Int(label+'_c')
            bounds=[b for v,n in scope for b in (env[v]>=0,env[v]<n)]+[rr>=0,rr<r,cc>=0,cc<c]
            src=expr(load.arg('offset'),env)+rr*isa.DIM+cc;dst=expr(store.arg('offset'),env)+rr*isa.DIM+cc
            return bounds,src,dst,tuple(env[v] for v,_ in scope)+(rr,cc)
        for i,site in enumerate(sites):
            bounds,src,dst,ids=event(site,'a'+str(member)+'_'+str(i));solver=z3.Solver();solver.set(timeout=5000)
            solver.add(*bounds,z3.Or(dst<0,dst>=prod(output),dst%output_run<base,dst%output_run>=base+run,src<0,src>=prod(shape),src!=(dst/output_run)*run+dst%output_run-base))
            queries+=1
            if solver.check()!=z3.unsat:raise BitConcatError('concat source address map not proved')
            for j,other in enumerate(sites[:i+1]):
                bb,_,dd,jds=event(other,'b'+str(member)+'_'+str(j));solver=z3.Solver();solver.set(timeout=5000)
                distinct=z3.Or(*(a!=b for a,b in zip(ids,jds))) if i==j else z3.BoolVal(True)
                solver.add(*(bounds+bb),dst==dd,distinct);queries+=1
                if solver.check()!=z3.unsat:raise BitConcatError('concat output uniqueness not proved')
        intervals.append({'member':member,'source_shape':list(shape),'outer_rows':outer,'source_elements_per_outer':run,'destination_interval_per_outer':[base,base+run]});base+=run;sites_total+=len(sites)
    if base!=output_run:raise BitConcatError('concat output partition incomplete')
    return {'schema':'submission.raw_bit_concat.v1','kernel_sha256':kernel.digest(),'axis':axis,'output_shape':list(output),'source_intervals':intervals,'actual_dma_sites':sites_total,'solver_queries':queries,'coverage':'disjoint member intervals partition each output row; injective actual DMA addresses and exact cardinality','source_map':'src = floor(dst/output_run)*member_run + dst%output_run - member_base','transport':'raw i32 preload and full-width readout; all f32 IEEE bits preserved without arithmetic','ownership':'MVIN; full drain; MVOUT; full drain before staging rows reused','requested_load_bytes':prod(output)*4,'requested_store_bytes':prod(output)*4}

def lower_concat(ctx,operands,attributes):
    if (attributes.get('output_dtype')!='f32' or attributes.get('storage_dtype')!='i32'
            or attributes.get('semantic')!='bit_preserving_concat'
            or set(attributes)-{'axis','output_dtype','storage_dtype','semantic'}):raise BitConcatError('concat requires exact raw-bit source contract without numeric attributes')
    keys={'dst'}|{'src'+str(i) for i in range(len(operands)-1)}
    if set(operands)!=keys:raise BitConcatError('concat inputs must be a complete ordered src0..srcN list')
    srcs=[operands['src'+str(i)] for i in range(len(operands)-1)];dst=operands['dst'];aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    if dst in srcs or any(n in aliases for n in srcs+[dst]):raise BitConcatError('concat requires distinct materialized destination without alias views')
    specs=[ctx.spec(n) for n in srcs+[dst]]
    if any(s['dtype']!='f32' or s.get('strides') or s.get('offset',0) for s in specs):raise BitConcatError('concat requires ordinary dense f32 storage')
    plan=plan_concat([s['shape'] for s in specs[:-1]],attributes['axis'])
    if list(plan.output_shape)!=specs[-1]['shape']:raise BitConcatError('concat result shape differs')
    from xdsl.dialects.builtin import IntegerType
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,arguments):
        if name=='concat_drain':ctx.fence();return
        if name=='concat_config':
            ctx.emit(*isa.config_ld(isa.DIM*4,shrink=False));ctx.emit(*isa.config_st(isa.DIM*4));return
        a={k:bridge.value(v) for k,v in arguments.items() if k!='member'}
        tensor=srcs[arguments['member']] if name=='concat_mvin' else dst
        ptr=ctx.fb.gep(ctx.buffer(tensor),a['offset'],IntegerType(32));packed=isa.pack_matrix(a['addr'],a['rows'],a['cols'],fb=ctx.fb)
        ctx.emit(isa.FUNCT['mvin' if name=='concat_mvin' else 'mvout'],ptr,packed)
    emitted=emit_index_program(plan.kernel,bridge,emit)
    return {'proof':plan.proof,'emission':emitted,'kernel_sha256':plan.kernel.digest(),'storage_view':{'logical_dtype':'f32','transfer_dtype':'i32','element_bytes':4,'mapping':'same pointer and byte offsets; no CPU payload access'}}
