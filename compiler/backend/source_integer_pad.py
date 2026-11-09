"""Exact immutable integer insert_slice through base-copy and overwrite DMA.

Captured padding is represented as insert_slice into an initialized tensor. The
base remains an actual ABI input: no fill value or allocation adjacency is inferred.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from functools import lru_cache
from merlin.sched.ir import Call, Kernel, Loop, TensorArg
from merlin.compiler.schedule.llvm_program import FnBuilderIndexAdapter, emit_index_program
from .source_integer_slice import plan_slice, check_slice_projection, IntegerSliceError
from . import isa


class IntegerPadError(ValueError):
    pass


def _rename(body, prefix):
    return tuple(replace(o, body=_rename(o.body,prefix)) if isinstance(o,Loop)
                 else replace(o,instr=prefix+o.instr.removeprefix('slice_')) for o in body)


def _reverse(body, dtype):
    """Reverse every checked transfer pair, preserving full completion order."""
    if len(body)<3 or any(isinstance(o,Loop) for o in body[:3]) or [o.instr for o in body[:3]] != ['slice_drain','slice_config_load','slice_config_store']:
        raise IntegerPadError('insert DMA prologue changed')
    result=[body[0],replace(body[1],args=body[2].args),replace(body[2],args=body[1].args)]
    def walk(ops):
        out=[];at=0
        while at<len(ops):
            op=ops[at]
            if isinstance(op,Loop):out.append(replace(op,body=walk(op.body)));at+=1;continue
            group=ops[at:at+4]
            if len(group)!=4 or any(isinstance(o,Loop) for o in group) or [o.instr for o in group]!=['slice_mvin','slice_drain','slice_mvout','slice_drain']:
                raise IntegerPadError('insert ownership/completion sequence changed')
            load,d1,store,d2=group
            # Args are Expr values; replace local address with an Expr as well.
            from merlin.sched.ir import Const
            def swapped(args,full):return tuple((k,Const(isa.acc_addr(0,full=full)) if k=='addr' else v) for k,v in args)
            if load.arg('addr').value!=isa.acc_addr(0) or store.arg('addr').value!=isa.acc_addr(0,full=dtype=='i32'):
                raise IntegerPadError('insert local accumulator address changed')
            out.extend((replace(load,args=swapped(store.args,False)),d1,
                        replace(store,args=swapped(load.args,dtype=='i32')),d2));at+=4
        return tuple(out)
    return tuple(result)+walk(body[3:])


@dataclass(frozen=True)
class PadPlan:
    source_shape: tuple
    base_shape: tuple
    offsets: tuple
    sizes: tuple
    strides: tuple
    dtype: str
    kernel: Kernel
    proof: dict
    def revalidate(self):
        return check_pad_projection(self.kernel,self.source_shape,self.base_shape,
                                    self.offsets,self.sizes,self.strides,self.dtype)


def plan_pad(source_shape,base_shape,offsets,sizes,strides,dtype):
    return _plan_pad(*(tuple(x) for x in (source_shape,base_shape,offsets,sizes,strides)),dtype)


@lru_cache(maxsize=128)
def _plan_pad(source_shape,base_shape,offsets,sizes,strides,dtype):
    if source_shape!=sizes:raise IntegerPadError('insert must preserve rank and all source coordinates')
    try:
        projection=plan_slice(base_shape,offsets,sizes,strides,dtype)
        identity=plan_slice(base_shape,(0,)*len(base_shape),base_shape,(1,)*len(base_shape),dtype)
    except IntegerSliceError as e:raise IntegerPadError(str(e)) from e
    args=(TensorArg('src',source_shape,dtype,'read'),TensorArg('base',base_shape,dtype,'read'),TensorArg('dst',base_shape,dtype,'write'))
    body=_rename(identity.kernel.body,'pad_base_')+_rename(_reverse(projection.kernel.body,dtype),'pad_insert_')
    kernel=Kernel('integer_insert_slice',args,body)
    proof=check_pad_projection(kernel,source_shape,base_shape,offsets,sizes,strides,dtype)
    return PadPlan(source_shape,base_shape,offsets,sizes,strides,dtype,kernel,proof)


def check_pad_projection(kernel,source_shape,base_shape,offsets,sizes,strides,dtype):
    source_shape,base_shape,offsets,sizes,strides=map(tuple,(source_shape,base_shape,offsets,sizes,strides))
    if source_shape!=sizes:raise IntegerPadError('insert source axes or shape changed')
    expected=(TensorArg('src',source_shape,dtype,'read'),TensorArg('base',base_shape,dtype,'read'),TensorArg('dst',base_shape,dtype,'write'))
    if kernel.args!=expected:raise IntegerPadError('insert ABI changed')
    # The phase boundary is explicit in actual command names. The first phase
    # completes the immutable base copy before any overwrite can begin.
    split=next((i for i,o in enumerate(kernel.body) if isinstance(o,Call) and o.instr=='pad_insert_drain'),None)
    if split is None:raise IntegerPadError('missing ordered insertion phase')
    def phase(body,prefix):
        def walk(ops):
            out=[]
            for o in ops:
                if isinstance(o,Loop):out.append(replace(o,body=walk(o.body)))
                elif isinstance(o,Call) and o.instr.startswith(prefix):out.append(replace(o,instr='slice_'+o.instr[len(prefix):]))
                else:raise IntegerPadError('insert phases interleaved or unknown effect')
            return tuple(out)
        return walk(body)
    base=phase(kernel.body[:split],'pad_base_');insert=phase(kernel.body[split:],'pad_insert_')
    try:
        args=(TensorArg('src',base_shape,dtype,'read'),TensorArg('dst',base_shape,dtype,'write'))
        bp=check_slice_projection(Kernel('base_copy',args,base),base_shape,(0,)*len(base_shape),base_shape,(1,)*len(base_shape),dtype)
        args=(TensorArg('src',base_shape,dtype,'read'),TensorArg('dst',source_shape,dtype,'write'))
        ip=check_slice_projection(Kernel('reverse_insert',args,_reverse(insert,dtype)),base_shape,offsets,sizes,strides,dtype)
    except (IntegerSliceError,AttributeError,TypeError) as e:raise IntegerPadError('insert DMA proof failed: '+str(e)) from e
    return {'schema':'submission.integer_insert_projection.v1','kernel_sha256':kernel.digest(),
            'source_shape':list(source_shape),'base_shape':list(base_shape),'offsets':list(offsets),'sizes':list(sizes),'strides':list(strides),
            'base_copy':bp,'insertion_reverse_projection':ip,
            'coverage':'complete base copy followed by injective source overwrite at original insert coordinates; all other base words retained',
            'numeric':bp['numeric'],'completion':'full base completion before overwrite; each local row read/write/reuse fully drained',
            'alias_contract':'dst distinct from both immutable ABI inputs; src and base may share a pointer'}


def integer_insert_candidate(op):
    from merlin.common.mlir_query import op_name
    from .quantized_boundary import _type
    from .source_groups import _Standalone,_standalone_capability
    if (op_name(op)!='tensor.insert_slice' or len(op.operands)!=2 or len(op.results)!=1 or op.regions
            or getattr(getattr(op,'parent',None),'parent_op',lambda:None)() is None
            or op.parent.parent_op().name!='func.func'):return None
    a,b,c=map(_type,(*op.operands,op.results[0]))
    if not all(t['dense'] for t in (a,b,c)) or len({t['dtype'] for t in (a,b,c)})!=1 or b['shape']!=c['shape']:return None
    try:
        def values(k):return tuple(op.properties.get(k,op.attributes.get(k)).get_values())
        offsets,sizes,strides=map(values,('static_offsets','static_sizes','static_strides'))
        p=plan_pad(a['shape'],b['shape'],offsets,sizes,strides,a['dtype'])
        attrs={'offsets':list(offsets),'sizes':list(sizes),'strides':list(strides),'output_dtype':a['dtype']}
        proof={'form':'static_integer_insert_slice','address_projection':p.proof}
        member=_Standalone(op,(op,),tuple(dict.fromkeys(op.operands)),op.results[0],'INTEGER_INSERT_SLICE',{'src':op.operands[0],'base':op.operands[1]},attrs,proof)
        if member is not None:_standalone_capability(member)
        return member
    except (ValueError,TypeError,AttributeError,OverflowError):return None


def lower_pad(ctx,operands,attributes):
    src,base,dst=(operands[k] for k in ('src','base','dst'))
    specs=[ctx.spec(x) for x in (src,base,dst)]
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    alias_sources={v.get('source') for v in aliases.values() if isinstance(v,dict)}
    if dst in (src,base) or {src,base,dst}&(set(aliases)|alias_sources):raise IntegerPadError('insert requires explicit materialized immutable inputs and distinct output')
    if (set(attributes)-{'offsets','sizes','strides','output_dtype'} or len({s['dtype'] for s in specs})!=1
            or specs[1]['shape']!=specs[2]['shape'] or attributes.get('output_dtype')!=specs[0]['dtype']):
        raise IntegerPadError('insert cannot change dtype, base shape, or apply arithmetic')
    p=plan_pad(specs[0]['shape'],specs[1]['shape'],attributes['offsets'],attributes['sizes'],attributes['strides'],specs[0]['dtype']);p.revalidate()
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,args):
        phase='base' if name.startswith('pad_base_') else 'insert'
        action=name.removeprefix('pad_'+phase+'_')
        if action=='drain':ctx.fence();return
        if action=='config_load':ctx.emit(*isa.config_ld(args['stride'],shrink=p.dtype=='i8'));return
        if action=='config_store':ctx.emit(*isa.config_st(args['stride'],scale=1.0,activation=0));return
        fields={k:bridge.value(v) for k,v in args.items()};packed=isa.pack_matrix(fields['addr'],fields['rows'],fields['cols'],fb=ctx.fb)
        if action=='mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(base if phase=='base' else src,fields['offset']),packed)
        elif action=='mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,fields['offset']),packed)
        else:raise IntegerPadError('unknown checked insert effect')
    emitted=emit_index_program(p.kernel,bridge,emit)
    receipt={'proof':p.proof,'emission':emitted,'kernel_sha256':p.kernel.digest()}
    ctx.metadata.setdefault('integer_insert_slice',[]).append(receipt)
    return receipt
