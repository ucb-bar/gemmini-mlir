"""Checked materialization of static signed-i8 concatenations by ACC DMA.

Repeated read-only inputs remain separate source positions. Original storage
realization is unknown; this adapter constructs a distinct device result.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from merlin.common.symbolic_lock import serialized_symbolic_check
from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Sum, TensorArg, Var, as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa

class ConcatCopyError(ValueError):pass

def _strides(shape):return tuple(prod(shape[i+1:]) for i in range(len(shape)))
def _call(name,**args):return Call(name,tuple((k,as_expr(v)) for k,v in args.items()))

@dataclass(frozen=True)
class ConcatCopyPlan:
    input_shapes: tuple
    result_shape: tuple
    axis: int
    kernel: Kernel
    def revalidate(self):return check_projection(self.kernel,self.input_shapes,self.result_shape,self.axis)

def validate_concat_copy(input_shapes,result_shape,dtypes,attrs):
    allowed={'input_shapes','axis','output_dtype','semantic','epilogue','relu','acc_scale','bound_lsb'}
    if not isinstance(attrs,dict) or any(not isinstance(k,str) or (k not in allowed and not k.startswith('prov.') and k not in ('merlin.source_op_index','merlin.global_task')) for k in attrs):
        raise ConcatCopyError('unknown concat attributes')
    try:
        shapes=tuple(tuple(s) for s in input_shapes);output=tuple(result_shape);types=tuple(dtypes)
        bound=tuple(tuple(s) for s in attrs['input_shapes']);axis=attrs['axis']
    except (TypeError,KeyError):raise ConcatCopyError('missing static concat geometry') from None
    if (not shapes or not 1<=len(output)<=4 or type(axis) is not int or not 0<=axis<len(output)
        or any(len(s)!=len(output) for s in shapes) or len(types)!=len(shapes) or any(t!='i8' for t in types)
        or any(type(n) is not int or n<=0 for s in shapes+(output,) for n in s)
        or shapes!=bound or attrs.get('output_dtype')!='i8' or attrs.get('semantic')!='concat_copy'
        or attrs.get('epilogue',[]) or attrs.get('relu',False) or attrs.get('acc_scale',1.0)!=1.0 or attrs.get('bound_lsb',0)!=0):
        raise ConcatCopyError('requires exact static rank1-4 signed-i8 concat')
    if any(s[i]!=output[i] for s in shapes for i in range(len(output)) if i!=axis) or sum(s[axis] for s in shapes)!=output[axis]:
        raise ConcatCopyError('concat result does not match ordered input extents')
    if max(prod(s) for s in shapes+(output,))>=1<<63:raise ConcatCopyError('concat byte footprint exceeds signed i64')
    if max(s[-1] for s in shapes+(output,))>=1<<32:raise ConcatCopyError('concat DMA pitch exceeds selected field')
    return _plan(shapes,output,axis)

@lru_cache(maxsize=256)
def _plan(shapes,output,axis):
    ds=_strides(output);body=[_call('concat_drain'),_call('concat_config_store',stride=output[-1])];start=0
    for member,shape in enumerate(shapes):
        ss=_strides(shape);run=shape[-1];row_axis=len(shape)-2;rows=shape[row_axis] if row_axis>=0 else 1
        prefix=range(max(0,row_axis));src=sum((Var(f'p{i}')*ss[i] for i in prefix),Const(0))
        dst=sum((Var(f'p{i}')*ds[i] for i in prefix),Const(start*ds[axis]))
        def tile(r,n,c,m):
            return (_call('concat_mvin',source=member,offset=src+r*run+c,addr=isa.acc_addr(0),rows=n,cols=m),_call('concat_drain'),
                    _call('concat_mvout',offset=dst+r*output[-1]+c,addr=isa.acc_addr(0),rows=n,cols=m),_call('concat_drain'))
        def columns(r,n):
            full,tail=divmod(run,isa.DIM);b=[]
            if full:b.append(Loop('c',full,tile(r,n,Var('c')*isa.DIM,isa.DIM)))
            if tail:b.extend(tile(r,n,Const(full*isa.DIM),tail))
            return tuple(b)
        full,tail=divmod(rows,isa.DIM);b=[]
        if full:b.append(Loop('r',full,columns(Var('r')*isa.DIM,isa.DIM)))
        if tail:b.extend(columns(Const(full*isa.DIM),tail))
        b=tuple(b)
        for i in reversed(tuple(prefix)):b=(Loop(f'p{i}',shape[i],b),)
        body.append(_call('concat_config_load',source=member,stride=run));body.extend(b);start+=shape[axis]
    args=tuple(TensorArg(f'src_{i}',s,'i8','read') for i,s in enumerate(shapes))+(TensorArg('dst',output,'i8','write'),)
    plan=ConcatCopyPlan(shapes,output,axis,Kernel('source_concat_copy',args,tuple(body)))
    plan.revalidate();return plan

@serialized_symbolic_check
def check_projection(kernel,shapes,output,axis):
    """Prove the actual DMA program, including input selection and completions."""
    import z3
    shapes=tuple(map(tuple,shapes));output=tuple(output);ds=_strides(output);count=prod(output)
    expected=tuple(TensorArg(f'src_{i}',s,'i8','read') for i,s in enumerate(shapes))+(TensorArg('dst',output,'i8','write'),)
    if kernel.args!=expected or kernel.attrs:raise ConcatCopyError('concat source boundary changed')
    def integer(call,key):
        try:x=call.arg(key)
        except (KeyError,ValueError):raise ConcatCopyError('missing descriptor field') from None
        if not isinstance(x,Const) or type(x.value) is not int:raise ConcatCopyError('descriptor must be static integer')
        return x.value
    first=kernel.body[:2]
    if (len(first)!=2 or any(not isinstance(o,Call) for o in first) or first[0]!=_call('concat_drain')
        or first[1]!=_call('concat_config_store',stride=output[-1])):raise ConcatCopyError('concat prologue changed')
    sites=[];configured=[]
    def walk(body,scope,member):
        at=0
        while at<len(body):
            op=body[at]
            if isinstance(op,Loop):
                if op.var in dict(scope) or type(op.extent) is not int or op.extent<=0 or op.iter_args or op.yields or op.results:
                    raise ConcatCopyError('invalid counted concat loop')
                walk(op.body,scope+((op.var,op.extent),),member);at+=1;continue
            if isinstance(op,Call) and op.instr=='concat_config_load':
                if scope or set(dict(op.args))!={'source','stride'}:raise ConcatCopyError('configuration inside DMA loop')
                member=integer(op,'source')
                if member!=len(configured) or member>=len(shapes) or integer(op,'stride')!=shapes[member][-1]:raise ConcatCopyError('source/load configuration changed')
                configured.append(member);at+=1;continue
            group=body[at:at+4]
            if member is None or len(group)!=4 or any(not isinstance(o,Call) for o in group) or [o.instr for o in group]!=['concat_mvin','concat_drain','concat_mvout','concat_drain']:
                raise ConcatCopyError('concat local ownership/order changed')
            load,before,store,after=group
            if before.args or after.args or set(dict(load.args))!={'source','offset','addr','rows','cols'} or set(dict(store.args))!={'offset','addr','rows','cols'}:
                raise ConcatCopyError('unexpected concat descriptor fields')
            r,c=integer(load,'rows'),integer(load,'cols')
            if (integer(load,'source')!=member or not 0<r<=min(isa.DIM,isa.ACC_ROWS) or not 0<c<=isa.DIM
                or integer(store,'rows')!=r or integer(store,'cols')!=c or integer(load,'addr')!=isa.acc_addr(0) or integer(store,'addr')!=isa.acc_addr(0)):
                raise ConcatCopyError('concat local descriptor mismatch')
            sites.append((member,load,store,scope,r,c));at+=4
    walk(kernel.body[2:],(),None)
    if configured!=list(range(len(shapes))) or sum(prod(n for _,n in scope)*r*c for _,_,_,scope,r,c in sites)!=count:
        raise ConcatCopyError('concat transferred cardinality differs from result')
    queries=0
    def prove(bounds,bad):
        nonlocal queries
        solver=z3.Solver();solver.set(timeout=5000);solver.add(*bounds,bad);queries+=1
        status=solver.check()
        if status!=z3.unsat:raise ConcatCopyError('concat address projection not proved: '+str(status))
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):
            if e.name not in env:raise ConcatCopyError('unbound address loop variable')
            return env[e.name]
        if isinstance(e,Sum):return sum(expr(t,env) for t in e.terms)
        if isinstance(e,Scaled):return e.factor*expr(e.expr,env)
        raise ConcatCopyError('unsupported concat index expression')
    def event(site,label):
        member,load,store,scope,rows,cols=site;env={v:z3.Int(label+'_'+v) for v,_ in scope};r=z3.Int(label+'_row');c=z3.Int(label+'_col')
        bounds=[b for v,n in scope for b in (env[v]>=0,env[v]<n)]+[r>=0,r<rows,c>=0,c<cols]
        src=expr(load.arg('offset'),env)+r*shapes[member][-1]+c
        dst=expr(store.arg('offset'),env)+r*output[-1]+c
        return bounds,src,dst,tuple(env[v] for v,_ in scope)+(r,c)
    for i,site in enumerate(sites):
        member=site[0];shape=shapes[member];ss=_strides(shape);bounds,src,dst,ids=event(site,'a'+str(i))
        start=sum(s[axis] for s in shapes[:member]);reference=start*ds[axis]+sum(((src/ss[j])%shape[j])*ds[j] for j in range(len(shape)))
        prove(bounds,z3.Or(src<0,src>=prod(shape),dst<0,dst>=count,dst!=reference))
        # Different members occupy disjoint source-proved concat axis intervals.
        # Only sites from the same member need an event injectivity query.
        for j,other in enumerate(sites[:i+1]):
            if other[0]!=member:continue
            rhs,_,out,otherids=event(other,'b'+str(j));distinct=z3.Or(*(a!=b for a,b in zip(ids,otherids))) if i==j else z3.BoolVal(True)
            prove(bounds+rhs,z3.And(dst==out,distinct))
    return dict(schema='submission.concat_address_projection.v1',kernel_sha256=kernel.digest(),solver_queries=queries,
        input_shapes=[list(s) for s in shapes],output_shape=list(output),axis=axis,output_elements=count,
        actual_dma_sites=len(sites),dynamic_dma_pairs=sum(prod(n for _,n in scope) for _,_,_,scope,_,_ in sites),
        requested_load_bytes=count,requested_store_bytes=count,byte_basis='requested logical DMA bytes, not measured traffic',
        coverage='source-projected disjoint member intervals; injective output addresses and exact result cardinality',
        ownership='overwrite ACC MVIN; full drain; matched MVOUT; full drain before reuse',
        numeric_semantics='signed-i8 sign extension and exact scale1 half-even i8 identity readout',
        effects='each source position read only; distinct result written once per element',
        repeated_input_contract='read-only source positions may share an original pointer',
        realization='distinct materialized device result',original_storage_alias=None,original_physical_copy=None,original_storage_realization='unresolved')

def match_concat_copy(op):
    from merlin.common import mlir_query as mq
    from .source_region_diagnostics import describe_source_operation
    if mq.op_name(op)!='tensor.concat':return None
    observed=describe_source_operation(op)
    if observed is None:return None
    _,metadata=observed;a=metadata['inputs'];b=metadata['outputs'][0]
    attrs=dict(input_shapes=[x['shape'] for x in a],axis=metadata['structural_proof']['axis'],output_dtype='i8',semantic='concat_copy')
    try:
        if b['dtype']!='i8':return None
        plan=validate_concat_copy([x['shape'] for x in a],b['shape'],[x['dtype'] for x in a],attrs)
    except ConcatCopyError:return None
    return {f'src_{i}':value for i,value in enumerate(op.operands)},attrs,{**metadata,**plan.revalidate()}

def lower_concat_copy(ctx,operands,attrs):
    shapes=attrs.get('input_shapes')
    if not isinstance(shapes,(tuple,list)):raise ConcatCopyError('missing concat input shapes')
    expected={f'src_{i}' for i in range(len(shapes))}|{'dst'}
    if set(operands)!=expected:raise ConcatCopyError('concat source positions/order missing')
    dst=operands['dst'];names=[operands[f'src_{i}'] for i in range(len(shapes))];out=ctx.spec(dst);specs=[ctx.spec(n) for n in names]
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    if dst in names or dst in aliases or any(n in aliases for n in names):raise ConcatCopyError('concat requires distinct ordinary source/result ABI buffers')
    if out['dtype']!='i8':raise ConcatCopyError('concat result must be signed i8')
    plan=validate_concat_copy([s['shape'] for s in specs],out['shape'],[s['dtype'] for s in specs],attrs);proof=plan.revalidate()
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,arguments):
        if name=='concat_drain':ctx.fence();return
        if name=='concat_config_load':ctx.emit(*isa.config_ld(arguments['stride'],shrink=True));return
        if name=='concat_config_store':ctx.emit(*isa.config_st(arguments['stride']));return
        source=arguments.get('source');values={k:bridge.value(v) for k,v in arguments.items() if k!='source'}
        packed=isa.pack_matrix(values['addr'],values['rows'],values['cols'],fb=ctx.fb)
        if name=='concat_mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(names[source],values['offset']),packed)
        elif name=='concat_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,values['offset']),packed)
        else:raise ConcatCopyError('unknown checked concat operation')
    emitted=emit_index_program(plan.kernel,bridge,emit)
    return dict(proof=proof,emission=emitted,kernel_sha256=plan.kernel.digest())
