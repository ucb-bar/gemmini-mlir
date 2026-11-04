"""Exact materialized signed-i8 insert_slice through disjoint counted DMA boxes.

Each output is loaded once, either from the source rectangle or from the runtime
initializer. The checked immutable Kernel is the program handed to LLVM/ISA.
Original tensor storage realization remains unresolved.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import prod
from merlin.common.symbolic_lock import serialized_symbolic_check
from merlin.sched.ir import Call,Const,Kernel,Loop,Scaled,Sum,TensorArg,Var,as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
from . import isa

class InsertCopyError(ValueError):pass

def _strides(shape):return tuple(prod(shape[i+1:])for i in range(len(shape)))
def _call(name,**args):return Call(name,tuple((k,as_expr(v))for k,v in args.items()))

@dataclass(frozen=True)
class InsertBox:
    lower:tuple
    upper:tuple
    source:int  # 0=inserted source; 1=runtime destination_initial
    @property
    def shape(self):return tuple(b-a for a,b in zip(self.lower,self.upper))

@dataclass(frozen=True)
class InsertCopyPlan:
    source_shape:tuple
    initial_shape:tuple
    result_shape:tuple
    offsets:tuple
    boxes:tuple
    kernel:Kernel
    def revalidate(self):
        return check_projection(self.kernel,self.source_shape,self.initial_shape,self.result_shape,self.offsets,self.boxes)


def _boxes(shape,output,offsets):
    intervals=tuple(tuple((a,b,inside)for a,b,inside in ((0,o,False),(o,o+s,True),(o+s,d,False))if a<b)
                    for s,d,o in zip(shape,output,offsets))
    return tuple(InsertBox(tuple(x[0]for x in axes),tuple(x[1]for x in axes),
                           0 if all(x[2]for x in axes)else 1)for axes in product(*intervals))


def validate_insert_copy(source_shape,initial_shape,result_shape,dtypes,attrs):
    allowed={'source_shape','initial_shape','offsets','sizes','strides','output_dtype','semantic',
             'epilogue','relu','acc_scale','bound_lsb'}
    if not isinstance(attrs,dict)or any(not isinstance(k,str)or(k not in allowed and not k.startswith('prov.')and k not in ('merlin.source_op_index','merlin.global_task'))for k in attrs):
        raise InsertCopyError('unknown insert attributes')
    seq=(source_shape,initial_shape,result_shape,dtypes,*(attrs.get(k)for k in ('source_shape','initial_shape','offsets','sizes','strides')))
    if any(not isinstance(x,(tuple,list))for x in seq):raise InsertCopyError('insert geometry and dtypes must be static sequences')
    shape,initial,output=map(tuple,(source_shape,initial_shape,result_shape));offsets=tuple(attrs['offsets'])
    rank=len(shape)
    if (not 1<=rank<=4 or len(initial)!=rank or output!=initial or len(offsets)!=rank
        or any(type(n)is not int or n<=0 for n in shape+initial+output)
        or any(type(o)is not int or o<0 for o in offsets)
        or tuple(dtypes)!=('i8','i8','i8')or attrs.get('output_dtype')!='i8'or attrs.get('semantic')!='insert_copy'
        or tuple(attrs['source_shape'])!=shape or tuple(attrs['initial_shape'])!=initial
        or tuple(attrs['sizes'])!=shape or len(attrs['strides'])!=rank
        or any(type(s)is not int or s!=1 for s in attrs['strides'])
        or attrs.get('epilogue',[])or attrs.get('relu',False)or attrs.get('acc_scale',1.0)!=1.0 or attrs.get('bound_lsb',0)!=0):
        raise InsertCopyError('requires static rank-preserving unit-stride signed-i8 insert')
    if any(o+s>d for o,s,d in zip(offsets,shape,initial)):raise InsertCopyError('insert rectangle exceeds initializer')
    if max(prod(shape),prod(initial))>=1<<63:raise InsertCopyError('insert byte footprint exceeds signed i64')
    if max(shape[-1],initial[-1])>=1<<32:raise InsertCopyError('insert row pitch exceeds DMA configuration')
    return _plan(shape,initial,offsets)


def _box_body(box,shape,output,offsets):
    original=shape if box.source==0 else output;ss=_strides(original);ds=_strides(output)
    lengths=box.shape;rank=len(shape);row_axis=rank-2
    sp=original[-1];dp=output[-1]
    source_lower=tuple(x-o if box.source==0 else x for x,o in zip(box.lower,offsets))
    src=sum((Var('p'+str(i))*ss[i]for i in range(max(0,row_axis))),Const(sum(x*s for x,s in zip(source_lower,ss))))
    dst=sum((Var('p'+str(i))*ds[i]for i in range(max(0,row_axis))),Const(sum(x*s for x,s in zip(box.lower,ds))))
    rows=lengths[row_axis]if row_axis>=0 else 1;columns=lengths[-1]
    def tile(r,n,c,m):
        return (_call('insert_mvin',source=box.source,offset=src+r*sp+c,addr=isa.acc_addr(0),rows=n,cols=m),
                _call('insert_drain'),_call('insert_mvout',offset=dst+r*dp+c,addr=isa.acc_addr(0),rows=n,cols=m),_call('insert_drain'))
    def cols(r,n):
        full,tail=divmod(columns,isa.DIM);body=[]
        if full:body.append(Loop('c',full,tile(r,n,Var('c')*isa.DIM,isa.DIM)))
        if tail:body.extend(tile(r,n,Const(full*isa.DIM),tail))
        return tuple(body)
    full,tail=divmod(rows,isa.DIM);body=[]
    if full:body.append(Loop('r',full,cols(Var('r')*isa.DIM,isa.DIM)))
    if tail:body.extend(cols(Const(full*isa.DIM),tail))
    body=tuple(body)
    for i in reversed(range(max(0,row_axis))):body=(Loop('p'+str(i),lengths[i],body),)
    return (_call('insert_config_load',stride=sp),_call('insert_config_store',stride=dp))+body

@lru_cache(maxsize=512)
def _plan(shape,initial,offsets):
    boxes=_boxes(shape,initial,offsets)
    kernel=Kernel('source_insert_copy',(TensorArg('src',shape,'i8','read'),TensorArg('initial',initial,'i8','read'),TensorArg('dst',initial,'i8','write')),
        (_call('insert_drain'),)+tuple(Loop('box'+str(i),1,_box_body(box,shape,initial,offsets))for i,box in enumerate(boxes)))
    plan=InsertCopyPlan(shape,initial,initial,offsets,boxes,kernel);plan.revalidate();return plan


@serialized_symbolic_check
def check_projection(kernel,shape,initial,output,offsets,boxes):
    """Prove actual DMA fields, per-box injectivity and exact branch semantics.

    The canonical Cartesian interval partition is disjoint and exhaustive.
    Each box must independently have in-box injective destination addresses and
    exactly its cardinality; this proves full coverage without tensor sampling.
    """
    import z3
    shape,initial,output,offsets,boxes=map(tuple,(shape,initial,output,offsets,boxes))
    if output!=initial or boxes!=_boxes(shape,output,offsets):raise InsertCopyError('insert partition changed')
    expected=(TensorArg('src',shape,'i8','read'),TensorArg('initial',initial,'i8','read'),TensorArg('dst',output,'i8','write'))
    if kernel.args!=expected or kernel.attrs:raise InsertCopyError('insert source boundary changed')
    if len(kernel.body)!=len(boxes)+1 or kernel.body[0]!=_call('insert_drain'):raise InsertCopyError('insert opening ownership drain changed')
    ds=_strides(output);ss=_strides(shape);queries=0;site_count=pair_count=0;roles=[0,0]
    def integer(call,key):
        v=call.arg(key)
        if not isinstance(v,Const):raise InsertCopyError('insert static descriptor became dynamic')
        return v.value
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):
            if e.name not in env:raise InsertCopyError('insert address uses an unbound index')
            return env[e.name]
        if isinstance(e,Sum):return sum(expr(x,env)for x in e.terms)
        if isinstance(e,Scaled):return e.factor*expr(e.expr,env)
        raise InsertCopyError('unsupported insert address expression')
    def check(bounds,bad):
        nonlocal queries
        solver=z3.Solver();solver.set(timeout=5000);solver.add(*bounds,bad);queries+=1
        status=solver.check()
        if status!=z3.unsat:raise InsertCopyError('insert address projection unproved: '+str(status))
    for index,(box,outer)in enumerate(zip(boxes,kernel.body[1:])):
        if (not isinstance(outer,Loop)or outer.var!='box'+str(index)or outer.extent!=1 or outer.iter_args or outer.yields or outer.results):
            raise InsertCopyError('insert box domain changed')
        prologue=outer.body[:2]
        expected_sp=(shape if box.source==0 else output)[-1];dp=output[-1]
        if prologue!=(_call('insert_config_load',stride=expected_sp),_call('insert_config_store',stride=dp)):
            raise InsertCopyError('insert row configuration changed')
        sites=[]
        def walk(body,scope):
            at=0
            while at<len(body):
                op=body[at]
                if isinstance(op,Loop):
                    if op.var in dict(scope)or op.extent<=0 or op.iter_args or op.yields or op.results:raise InsertCopyError('invalid insert counted loop')
                    walk(op.body,scope+((op.var,op.extent),));at+=1;continue
                group=body[at:at+4]
                if len(group)!=4 or any(not isinstance(x,Call)for x in group)or [x.instr for x in group]!=['insert_mvin','insert_drain','insert_mvout','insert_drain']:
                    raise InsertCopyError('insert DMA order or completion changed')
                load,before,store,after=group
                if any(any((x.unit,x.produces,x.awaits,x.stages,x.sets,x.assumes))for x in group):
                    raise InsertCopyError('insert annotated effects require a different emission contract')
                if before.args or after.args:raise InsertCopyError('insert unexpected drain operands')
                if set(dict(load.args))!={'source','offset','addr','rows','cols'}or set(dict(store.args))!={'offset','addr','rows','cols'}:raise InsertCopyError('insert descriptor fields changed')
                rows,cols=integer(load,'rows'),integer(load,'cols')
                if (integer(load,'source')!=box.source or not 0<rows<=min(isa.DIM,isa.ACC_ROWS)or not 0<cols<=isa.DIM
                    or integer(store,'rows')!=rows or integer(store,'cols')!=cols
                    or integer(load,'addr')!=isa.acc_addr(0)or integer(store,'addr')!=isa.acc_addr(0)):
                    raise InsertCopyError('insert source/local descriptor changed')
                sites.append((load,store,scope,rows,cols));at+=4
        walk(outer.body[2:],())
        count=sum(prod(n for _,n in scope)*r*c for _,_,scope,r,c in sites)
        if count!=prod(box.shape):raise InsertCopyError('insert box coverage cardinality changed')
        roles[box.source]+=count;site_count+=len(sites);pair_count+=sum(prod(n for _,n in scope)for _,_,scope,_,_ in sites)
        def event(site,label):
            load,store,scope,rows,cols=site;env={v:z3.Int(label+'_'+v)for v,_ in scope}
            r,c=z3.Int(label+'_row'),z3.Int(label+'_col')
            bounds=[b for v,n in scope for b in (env[v]>=0,env[v]<n)]+[r>=0,r<rows,c>=0,c<cols]
            dst=expr(store.arg('offset'),env)+r*dp+c;src=expr(load.arg('offset'),env)+r*expected_sp+c
            return bounds,src,dst,tuple(env[v]for v,_ in scope)+(r,c)
        for i,site in enumerate(sites):
            bounds,src,dst,_=event(site,'a'+str(i));coords=tuple((dst/s)%n for s,n in zip(ds,output))
            inside=z3.And(*(z3.And(o<=x,x<o+n)for x,o,n in zip(coords,offsets,shape)))
            reference=z3.If(inside,sum((x-o)*s for x,o,s in zip(coords,offsets,ss)),dst)
            bad=z3.Or(dst<0,dst>=prod(output),*(z3.Or(x<lo,x>=hi)for x,lo,hi in zip(coords,box.lower,box.upper)),
                      src<0,src>=(prod(shape)if box.source==0 else prod(output)),src!=reference,inside!=(box.source==0))
            check(bounds,bad)
            for j,other in enumerate(sites[:i+1]):
                rhs,_,out,ids=event(other,'b'+str(j));lhs,_,_,myids=event(site,'a'+str(i))
                distinct=z3.Or(*(a!=b for a,b in zip(myids,ids)))if i==j else z3.BoolVal(True)
                check(lhs+rhs,z3.And(dst==out,distinct))
    if sum(roles)!=prod(output)or roles[0]!=prod(shape):raise InsertCopyError('insert global partition coverage changed')
    return dict(schema='submission.insert_address_projection.v1',kernel_sha256=kernel.digest(),solver_queries=queries,
        source_shape=list(shape),initial_shape=list(initial),result_shape=list(output),offsets=list(offsets),
        partition_boxes=len(boxes),actual_dma_sites=site_count,dynamic_dma_pairs=pair_count,
        output_elements=prod(output),source_elements=roles[0],preserved_initializer_elements=roles[1],
        requested_load_bytes=sum(roles),requested_store_bytes=sum(roles),
        coverage='canonical Cartesian below/inside/above partition; actual in-box injective writes and exact cardinality',
        semantics='original runtime initializer outside insertion; shifted source inside; signed i8 identity readout',
        ownership='MVIN -> real full drain -> MVOUT -> real full drain before ACC reuse',
        realization='distinct materialized device result',original_storage_alias=None,original_physical_copy=None,
        original_storage_realization='unresolved')


def match_insert_copy(op):
    from merlin.common import mlir_query as mq
    from .source_region_diagnostics import describe_source_operation
    if mq.op_name(op)!='tensor.insert_slice':return None
    described=describe_source_operation(op)
    if described is None:return None
    _,metadata=described;source,initial=metadata['inputs'];result=metadata['outputs'][0];p=metadata['structural_proof']
    attrs=dict(source_shape=list(source['shape']),initial_shape=list(initial['shape']),offsets=list(p['offsets']),
        sizes=list(p['sizes']),strides=list(p['strides']),output_dtype='i8',semantic='insert_copy')
    try:plan=validate_insert_copy(source['shape'],initial['shape'],result['shape'],(source['dtype'],initial['dtype'],result['dtype']),attrs)
    except InsertCopyError:return None
    return dict(src=op.operands[0],initial=op.operands[1]),attrs,{**metadata,**plan.revalidate()}


def lower_insert_copy(ctx,operands,attrs):
    if set(operands)!={'src','initial','dst'}:raise InsertCopyError('insert requires source, runtime initializer and destination')
    src,initial,dst=(operands[k]for k in ('src','initial','dst'))
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    # Inputs may coincide or share an immutable ordinary ABI buffer. A declared
    # alias is supported only by the existing zero-offset shape/dtype alias ABI.
    if dst in (src,initial)or dst in aliases or any(isinstance(v,dict)and v.get('source')==dst for v in aliases.values()):
        raise InsertCopyError('insert result must be distinct from inputs and declared aliases')
    for name in (src,initial):
        seen=set();current=name
        while current in aliases:
            if current in seen:raise InsertCopyError('cyclic insert input alias')
            seen.add(current);a=aliases[current]
            if not isinstance(a,dict)or set(a)-{'source','shape','dtype'}or not isinstance(a.get('source'),str):raise InsertCopyError('unsupported insert input alias')
            if (set(a)!={'source','shape','dtype'} or a['dtype']!='i8'
                or tuple(a['shape'])!=tuple(ctx.spec(current)['shape'])
                or ctx.spec(a['source'])['dtype']!='i8'
                or prod(ctx.spec(current)['shape'])!=prod(ctx.spec(a['source'])['shape'])):
                raise InsertCopyError('insert input alias byte extent or dtype changed')
            current=a['source']
        if current==dst:raise InsertCopyError('insert output aliases a readonly input')
    specs=[ctx.spec(n)for n in (src,initial,dst)]
    buffers=getattr(ctx,'buffers',{})
    if dst in buffers and any(n in buffers and buffers[n]is buffers[dst]for n in (src,initial)):
        raise InsertCopyError('insert output shares an input buffer')
    plan=validate_insert_copy(*(x['shape']for x in specs),tuple(x['dtype']for x in specs),attrs)
    proof=plan.revalidate();bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,args):
        if name=='insert_drain':ctx.fence();return
        if name=='insert_config_load':ctx.emit(*isa.config_ld(args['stride'],shrink=True));return
        if name=='insert_config_store':ctx.emit(*isa.config_st(args['stride']));return
        values={k:bridge.value(v)for k,v in args.items()if k!='source'}
        packed=isa.pack_matrix(values['addr'],values['rows'],values['cols'],fb=ctx.fb)
        if name=='insert_mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(src if args['source']==0 else initial,values['offset']),packed)
        elif name=='insert_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,values['offset']),packed)
        else:raise InsertCopyError('unknown checked insert instruction')
    emitted=emit_index_program(plan.kernel,bridge,emit)
    return dict(proof=proof,emission=emitted,kernel_sha256=plan.kernel.digest())
