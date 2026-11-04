"""Opt-in exact integer permutations through bounded accumulator DMA tiles.

The common counted index program carries the actual DMA offsets, dimensions,
configuration and completion calls. Its checker proves address correspondence,
unique output coverage and local ownership before the same program is emitted.
There are no tensor loads/stores or tensor arithmetic in the CPU program.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from math import prod

from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Sum, TensorArg, Var, as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa


class TransposeError(ValueError):
    pass


def _strides(shape):
    return tuple(prod(shape[i+1:]) for i in range(len(shape)))


def _call(name, **args):
    return Call(name, tuple((k,as_expr(v)) for k,v in args.items()))


@dataclass(frozen=True)
class TransposePlan:
    shape: tuple
    permutation: tuple
    dtype: str
    kernel: Kernel
    proof: dict

    def revalidate(self):
        return check_projection(self.kernel,self.shape,self.permutation,self.dtype)


def _baseline_kernel(shape, permutation, dtype):
    """Pure canonical counted construction; no checker or optional dependency."""
    shape,permutation=tuple(shape),tuple(permutation)
    if (not 1<=len(shape)<=6 or any(type(n) is not int or n<=0 for n in shape)
            or any(type(i) is not int for i in permutation)
            or sorted(permutation)!=list(range(len(shape))) or dtype not in ('i8','i32')):
        raise TransposeError('transpose requires rank1-6 positive static signed integer tensors and a complete integer permutation')
    width=1 if dtype=='i8' else 4
    if prod(shape)*width>=1<<63:
        raise TransposeError('transpose byte footprint exceeds signed i64')
    output=tuple(shape[i] for i in permutation);ss=_strides(shape);ds=_strides(output)
    # Prefix loops preserve every independent output axis (up to rank six).
    # Rank one naturally has no prefix and a single contiguous output run.
    # Maximal trailing output run which is also contiguous in the input. Unit
    # axes carry no physical displacement and can move across either boundary.
    run=1;start=len(shape)
    for axis in range(len(shape)-1,-1,-1):
        if output[axis]!=1 and ss[permutation[axis]]!=run:break
        run*=output[axis];start=axis
    row_axis=start-1
    row_extent=output[row_axis] if row_axis>=0 else 1
    src_pitch=ss[permutation[row_axis]] if row_axis>=0 else run
    dst_pitch=run
    if max(src_pitch,dst_pitch)*width>=1<<32:
        raise TransposeError('transpose DMA stride exceeds selected configuration field')
    prefix=range(max(0,row_axis))
    src_base=sum((Var(f'p{i}')*ss[permutation[i]] for i in prefix),Const(0))
    dst_base=sum((Var(f'p{i}')*ds[i] for i in prefix),Const(0))
    def tile(r,rows,c,cols):
        load=_call('transpose_mvin',offset=src_base+r*src_pitch+c,
                   addr=isa.acc_addr(0),rows=rows,cols=cols)
        store=_call('transpose_mvout',offset=dst_base+r*dst_pitch+c,
                    addr=isa.acc_addr(0,full=dtype=='i32'),rows=rows,cols=cols)
        return load,_call('transpose_drain'),store,_call('transpose_drain')
    def columns(r,rows):
        full,tail=divmod(run,isa.DIM);body=[]
        if full:body.append(Loop('c',full,tile(r,rows,Var('c')*isa.DIM,isa.DIM)))
        if tail:body.extend(tile(r,rows,Const(full*isa.DIM),tail))
        return tuple(body)
    full,tail=divmod(row_extent,isa.DIM);body=[]
    if full:body.append(Loop('r',full,columns(Var('r')*isa.DIM,isa.DIM)))
    if tail:body.extend(columns(Const(full*isa.DIM),tail))
    body=tuple(body)
    for i in reversed(tuple(prefix)):body=(Loop(f'p{i}',output[i],body),)
    configs=(_call('transpose_drain'),_call('transpose_config_load',stride=src_pitch*width),
             _call('transpose_config_store',stride=dst_pitch*width))
    kernel=Kernel('integer_transpose',(TensorArg('src',shape,dtype,'read'),TensorArg('dst',output,dtype,'write')),configs+body)
    return kernel


def plan_transpose(shape, permutation, dtype, *, schedule='baseline'):
    if schedule not in ('baseline','batched'):raise TransposeError('unknown transpose schedule')
    shape,permutation=tuple(shape),tuple(permutation)
    kernel=_baseline_kernel(shape,permutation,dtype)
    proof=check_projection(kernel,shape,permutation,dtype)
    if schedule=='batched':
        try:
            candidate=_batched_kernel(kernel)
            candidate_proof=check_projection(candidate,shape,permutation,dtype)
        except (TransposeError,ModuleNotFoundError) as exc:
            if isinstance(exc,ModuleNotFoundError)and exc.name not in ('z3',None):raise
            proof={**proof,'batch_refusal':str(exc)}
        else:
            kernel,proof=candidate,candidate_proof
    return TransposePlan(shape,permutation,dtype,kernel,proof)


def check_projection(kernel, shape, permutation, dtype):
    """Check actual instruction fields, including tails, with bounded integer proofs.

    Injectivity plus the exact dynamic element count establishes full output
    coverage without enumerating the tensor or using a sampled input palette.
    """
    if dict(kernel.attrs).get('transpose_schedule')=='batched':
        if kernel.attrs!=(('transpose_schedule','batched'),):
            raise TransposeError('unexpected batched transpose attributes')
        return _check_batched_projection(kernel,shape,permutation,dtype)
    from .transpose_proof import check_baseline_projection
    return check_baseline_projection(kernel,shape,permutation,dtype)


def _check_symbolic_projection(kernel, shape, permutation, dtype):
    """Retained SMT projection law for the optional normalized batched stream."""
    import z3
    shape,permutation=tuple(shape),tuple(permutation)
    width=1 if dtype=='i8' else 4
    output=tuple(shape[i] for i in permutation);ss=_strides(shape);ds=_strides(output);count=prod(shape)
    expected=(TensorArg('src',shape,dtype,'read'),TensorArg('dst',output,dtype,'write'))
    if kernel.args!=expected:raise TransposeError('transpose source boundary changed')
    first=kernel.body[:3]
    if [o.instr for o in first]!=['transpose_drain','transpose_config_load','transpose_config_store']:
        raise TransposeError('transpose configuration/completion prologue changed')
    def integer(call,key):
        x=call.arg(key)
        if not isinstance(x,Const):raise TransposeError('transpose static descriptor became dynamic')
        return x.value
    sp,dp=integer(first[1],'stride'),integer(first[2],'stride')
    if min(sp,dp)<=0 or max(sp,dp)>=1<<32 or sp%width or dp%width:
        raise TransposeError('invalid transpose DMA row pitches')
    sites=[]
    def walk(body,scope):
        at=0
        while at<len(body):
            op=body[at]
            if isinstance(op,Loop):
                if op.var in dict(scope) or op.extent<=0 or op.iter_args or op.yields or op.results:
                    raise TransposeError('invalid transpose counted loop')
                walk(op.body,scope+((op.var,op.extent),));at+=1;continue
            group=body[at:at+4]
            if len(group)!=4 or [o.instr for o in group]!=['transpose_mvin','transpose_drain','transpose_mvout','transpose_drain']:
                raise TransposeError('transpose local ownership or command order changed')
            load,_,store,_=group
            rows,cols=integer(load,'rows'),integer(load,'cols')
            if (not 0<rows<=min(isa.DIM,isa.ACC_ROWS) or not 0<cols<=isa.DIM
                    or integer(store,'rows')!=rows or integer(store,'cols')!=cols
                    or integer(load,'addr')!=isa.acc_addr(0)
                    or integer(store,'addr')!=isa.acc_addr(0,full=dtype=='i32')):
                raise TransposeError('transpose descriptor/local footprint mismatch')
            if set(dict(load.args))!={'offset','rows','cols','addr'} or set(dict(store.args))!={'offset','rows','cols','addr'}:
                raise TransposeError('unexpected transpose DMA descriptor fields')
            sites.append((load,store,scope,rows,cols));at+=4
    walk(kernel.body[3:],())
    if not sites or sum(prod(n for _,n in s)*r*c for _,_,s,r,c in sites)!=count:
        raise TransposeError('transpose transferred element count differs from source')
    queries=0
    def check(conditions,bad):
        nonlocal queries
        solver=z3.Solver();solver.set(timeout=5000);solver.add(*conditions,bad);queries+=1
        status=solver.check()
        if status!=z3.unsat:raise TransposeError('transpose address projection not proved: '+str(status))
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):return env[e.name]
        if isinstance(e,Sum):return sum(expr(x,env) for x in e.terms)
        if isinstance(e,Scaled):return e.factor*expr(e.expr,env)
        raise TransposeError('unsupported transpose address expression')
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
        reference=sum(((dst/stride)%extent)*ss[axis] for stride,extent,axis in zip(ds,output,permutation))
        check(bounds,z3.Or(dst<0,dst>=count,src<0,src>=count,src!=reference))
        for j,other in enumerate(sites[:i+1]):
            rhs,_,out,ids=event(other,'b'+str(j));lhs,_,_,myids=event(site,'a'+str(i))
            distinct=z3.Or(*(a!=b for a,b in zip(myids,ids))) if i==j else z3.BoolVal(True)
            check(lhs+rhs,z3.And(dst==out,distinct))
    return dict(schema='submission.transpose_address_projection.v1',kernel_sha256=kernel.digest(),
                solver_queries=queries,source_shape=list(shape),output_shape=list(output),permutation=list(permutation),
                output_elements=count,actual_dma_sites=len(sites),
                dynamic_dma_pairs=sum(prod(n for _,n in scope) for _,_,scope,_,_ in sites),
                requested_load_bytes=count*width,requested_store_bytes=count*width,
                actual_tile_columns=sorted({cols for _,_,_,_,cols in sites}),
                source_row_stride_bytes=sp,destination_row_stride_bytes=dp,
                coverage='in-bounds injective output addresses with exact source cardinality',
                semantics='signed i8 identity narrowing' if dtype=='i8' else 'raw full-width i32 readout',
                ownership='MVIN then full drain then MVOUT then full drain before row reuse',
                alias_contract='distinct ordinary ABI source/output buffers')


def _substitute(body, values):
    def expr(e):
        if isinstance(e,Var):return values.get(e.name,e)
        if isinstance(e,Const):return e
        if isinstance(e,Sum):return sum((expr(t) for t in e.terms),Const(0))
        if isinstance(e,Scaled):return expr(e.expr)*e.factor
        raise TransposeError('unsupported batched transpose expression')
    return tuple(replace(o,body=_substitute(o.body,values)) if isinstance(o,Loop)
                 else replace(o,args=tuple((k,expr(v)) for k,v in o.args)) for o in body)


def _pair_count(body):
    return sum(o.extent*_pair_count(o.body) if isinstance(o,Loop) else int(o.instr=='transpose_mvin') for o in body)


def _drain_count(body):
    return sum(o.extent*_drain_count(o.body) if isinstance(o,Loop) else int(o.instr=='transpose_drain') for o in body)


def _batched_kernel(kernel):
    """Capacity-bounded staging transform on the actual checked DMA program."""
    def calls(body):
        for o in body:
            if isinstance(o,Loop):yield from calls(o.body)
            else:yield o
    spacing=max(o.arg('rows').value for o in calls(kernel.body) if o.instr=='transpose_mvin')
    slots=isa.ACC_ROWS//spacing
    if slots<2:return kernel
    serial=0
    def phase(body,store,ordinal=Const(0)):
        result=[];used=0;at=0
        while at<len(body):
            o=body[at]
            if isinstance(o,Loop):
                count=_pair_count(o.body)
                result.append(replace(o,body=phase(o.body,store,ordinal+used+Var(o.var)*count)))
                used+=count*o.extent;at+=1
            else:
                pair=body[at:at+4]
                if [p.instr for p in pair]!=['transpose_mvin','transpose_drain','transpose_mvout','transpose_drain']:
                    raise TransposeError('batching requires checked serial DMA pairs')
                actual=pair[2 if store else 0]
                result.append(replace(actual,args=tuple((k,v+(ordinal+used)*spacing if k=='addr' else v) for k,v in actual.args)))
                used+=1;at+=4
        return tuple(result)
    def epoch(body):
        return (*phase(body,False),_call('transpose_drain'),*phase(body,True),_call('transpose_drain'))
    def rewrite(body):
        nonlocal serial
        if _pair_count(body)<=slots:return epoch(body)
        result=[];at=0
        while at<len(body):
            o=body[at]
            if not isinstance(o,Loop):
                result.extend(epoch(body[at:at+4]));at+=4;continue
            count=_pair_count(o.body)
            if count>slots:
                result.append(replace(o,body=rewrite(o.body)))
            else:
                batch=min(o.extent,slots//count);full,tail=divmod(o.extent,batch)
                name=f'transpose_batch_{serial}';serial+=1
                if full:
                    part=_substitute(o.body,{o.var:Var(name)*batch+Var(o.var)})
                    result.append(Loop(name,full,epoch((Loop(o.var,batch,part),))))
                if tail:
                    part=_substitute(o.body,{o.var:Const(full*batch)+Var(o.var)})
                    result.extend(epoch((Loop(o.var,tail,part),)))
            at+=1
        return tuple(result)
    candidate=replace(kernel,body=kernel.body[:3]+rewrite(kernel.body[3:]),attrs=(('transpose_schedule','batched'),))
    return candidate if _drain_count(candidate.body)<_drain_count(kernel.body) else kernel


def _check_batched_projection(kernel,shape,permutation,dtype):
    """Prove physical epoch ownership before projecting its actual logical DMA."""
    import z3
    queries=0;epochs=0;max_rows=0
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):return env[e.name]
        if isinstance(e,Sum):return sum(expr(t,env) for t in e.terms)
        if isinstance(e,Scaled):return expr(e.expr,env)*e.factor
        raise TransposeError('unsupported physical transpose address')
    def prove(bounds,bad):
        nonlocal queries
        solver=z3.Solver();solver.set(timeout=5000);solver.add(*bounds,bad);queries+=1
        status=solver.check()
        if status!=z3.unsat:raise TransposeError('batched accumulator ownership not proved: '+str(status))
    def has_drain(o):
        return any(has_drain(x) for x in o.body) if isinstance(o,Loop) else o.instr=='transpose_drain'
    def zip_phase(loads,stores,inner,outer,sites):
        if len(loads)!=len(stores):raise TransposeError('batched DMA phase structures differ')
        result=[]
        for load,store in zip(loads,stores,strict=True):
            if isinstance(load,Loop):
                if not isinstance(store,Loop) or (load.var,load.extent)!=(store.var,store.extent):
                    raise TransposeError('batched DMA counted domains differ')
                if (load.var in dict(inner+outer) or load.extent<=0
                        or load.iter_args or load.yields or load.results
                        or store.iter_args or store.yields or store.results):
                    raise TransposeError('invalid batched DMA loop domain')
                result.append(replace(load,body=zip_phase(load.body,store.body,inner+((load.var,load.extent),),outer,sites)))
            else:
                if (not isinstance(store,Call) or load.instr!='transpose_mvin' or store.instr!='transpose_mvout'
                        or load.arg('rows')!=store.arg('rows') or load.arg('cols')!=store.arg('cols')):
                    raise TransposeError('batched DMA descriptors do not match')
                rows=load.arg('rows')
                if not isinstance(rows,Const) or not 0<rows.value<=isa.DIM:raise TransposeError('invalid batch footprint')
                sites.append((load,store,inner,rows.value))
                # Physical equivalence is proved separately below. This exact
                # projection keeps all external offsets/config/rows/columns.
                a=replace(load,args=tuple((k,Const(isa.acc_addr(0)) if k=='addr' else v) for k,v in load.args))
                b=replace(store,args=tuple((k,Const(isa.acc_addr(0,full=dtype=='i32')) if k=='addr' else v) for k,v in store.args))
                result.extend((a,_call('transpose_drain'),b,_call('transpose_drain')))
        return tuple(result)
    def walk(body,outer):
        nonlocal epochs,max_rows
        result=[];at=0
        while at<len(body):
            o=body[at]
            if isinstance(o,Loop) and has_drain(o):
                if o.var in dict(outer) or o.extent<=0 or o.iter_args or o.yields or o.results:raise TransposeError('invalid batch outer loop')
                result.append(replace(o,body=walk(o.body,outer+((o.var,o.extent),))));at+=1;continue
            stop=at
            while stop<len(body) and not has_drain(body[stop]):stop+=1
            end=stop+1
            while end<len(body) and not has_drain(body[end]):end+=1
            if stop==at or end>=len(body) or not isinstance(body[stop],Call) or not isinstance(body[end],Call):
                raise TransposeError('batch must complete staging and readout before reuse')
            sites=[];normalized=zip_phase(body[at:stop],body[stop+1:end],(),outer,sites)
            epochs+=1
            live=sum(prod(n for _,n in scope)*rows for _,_,scope,rows in sites)
            if live>isa.ACC_ROWS:raise TransposeError('batch live rows exceed selected accumulator capacity')
            max_rows=max(max_rows,live)
            def event(site,label):
                load,store,scope,rows=site
                env={v:z3.Int('outer_'+v) for v,_ in outer}
                env.update({v:z3.Int(label+'_'+v) for v,_ in scope})
                bounds=[b for v,n in outer+scope for b in (env[v]>=0,env[v]<n)]
                local=expr(load.arg('addr'),env)-isa.acc_addr(0)
                target=expr(store.arg('addr'),env)-isa.acc_addr(0,full=dtype=='i32')
                return bounds,local,target,tuple(env[v] for v,_ in scope),rows
            for i,site in enumerate(sites):
                bounds,local,target,ids,rows=event(site,'a')
                prove(bounds,z3.Or(local<0,local+rows>isa.ACC_ROWS,target!=local))
                for j,other in enumerate(sites[:i+1]):
                    bs,loc,_,other_ids,other_rows=event(other,'b')
                    different=z3.Or(*(a!=b for a,b in zip(ids,other_ids))) if i==j else z3.BoolVal(True)
                    prove(bounds+bs,z3.And(different,local<loc+other_rows,loc<local+rows))
            result.extend(normalized);at=end+1
        return tuple(result)
    normalized=replace(kernel,body=kernel.body[:3]+walk(kernel.body[3:],()),attrs=())
    proof=_check_symbolic_projection(normalized,shape,permutation,dtype)
    return {**proof,'kernel_sha256':kernel.digest(),'logical_projection_sha256':normalized.digest(),
            'schedule':'batched','physical_ownership_solver_queries':queries,'static_batch_epochs':epochs,
            'max_simultaneous_accumulator_rows':max_rows,'selected_accumulator_rows':isa.ACC_ROWS,
            'dynamic_completion_drains':_drain_count(kernel.body),
            'ownership':'pairwise disjoint staging rows; full drain; matched readouts; full drain before reuse'}


def lower_transpose(ctx, operands, attributes):
    """Emit an internal STRIDED_MOVEMENT; return its actual projection receipt."""
    src,dst=operands['src'],operands['dst'];a,b=ctx.spec(src),ctx.spec(dst)
    if src==dst or src in ctx.cb.get('params',{}).get('tensor_aliases',{}) or dst in ctx.cb.get('params',{}).get('tensor_aliases',{}):
        raise TransposeError('transpose requires distinct materialized source/output buffers')
    if (a['dtype']!=b['dtype'] or attributes.get('output_dtype',a['dtype'])!=a['dtype']
            or attributes.get('epilogue',[]) or attributes.get('relu')
            or attributes.get('acc_scale',1.0)!=1.0 or attributes.get('acc_scale_by_column') is not None
            or attributes.get('bias') or attributes.get('accumulator_init')):
        raise TransposeError('transpose cannot change dtype or apply a numeric epilogue')
    schedule=attributes.get('transpose_schedule')
    if schedule is None:
        from pathlib import Path
        from merlin.compile.tuning_knobs import knob
        schedule=knob('TRANSPOSE_SCHEDULE',default='baseline',values=('baseline','batched'),root=Path(__file__).resolve().parent.parent)
    plan=plan_transpose(a['shape'],attributes['permutation'],a['dtype'],schedule=schedule)
    if b['shape']!=[a['shape'][i] for i in plan.permutation]:raise TransposeError('transpose result axes differ')
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,arguments):
        if name=='transpose_drain':ctx.fence();return
        if name=='transpose_config_load':
            ctx.emit(*isa.config_ld(arguments['stride'],shrink=plan.dtype=='i8'));return
        if name=='transpose_config_store':
            ctx.emit(*isa.config_st(arguments['stride']));return
        a={k:bridge.value(v) for k,v in arguments.items()}
        packed=isa.pack_matrix(a['addr'],a['rows'],a['cols'],fb=ctx.fb)
        if name=='transpose_mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(src,a['offset']),packed)
        elif name=='transpose_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,a['offset']),packed)
        else:raise TransposeError('unknown checked transpose operation')
    emitted=emit_index_program(plan.kernel,bridge,emit)
    return dict(proof=plan.proof,emission=emitted,kernel_sha256=plan.kernel.digest(),
                requested_schedule=schedule,consumed_schedule=dict(plan.kernel.attrs).get('transpose_schedule','baseline'))
