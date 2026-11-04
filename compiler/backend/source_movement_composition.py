"""Source-bound composition of single-use static integer slice chains.

Only original slice results whose sole use belongs to this chain disappear.
The existing checked slice DMA implementation consumes the composed descriptor;
no intermediate is represented as an offset/strided ABI alias.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from math import prod

from .source_integer_slice import plan_slice, lower_slice, IntegerSliceError


class MovementCompositionError(ValueError):
    pass


def _normalize(stages):
    result=[]
    for stage in stages:
        if set(stage) != {'offsets','sizes','strides'}:
            raise MovementCompositionError('slice composition requires original offsets/sizes/strides only')
        result.append(tuple(tuple(stage[k]) for k in ('offsets','sizes','strides')))
    if not 2 <= len(result) <= 32:
        raise MovementCompositionError('slice composition requires 2-32 source stages')
    return tuple(result)


@dataclass(frozen=True)
class ComposedSlicePlan:
    source_shape: tuple
    stages: tuple
    dtype: str
    output_shape: tuple
    slice_plan: object
    proof: dict

    @property
    def kernel(self):
        return self.slice_plan.kernel

    def revalidate(self):
        self.slice_plan.revalidate()
        return _check_composition(self.source_shape,self.stages,self.dtype,self.slice_plan)


def plan_composed_slice(shape, stages, dtype):
    return _plan_composed_slice(tuple(shape),_normalize(stages),dtype)


@lru_cache(maxsize=128)
def _plan_composed_slice(shape, stages, dtype):
    current=shape; offsets=(0,)*len(shape); strides=(1,)*len(shape)
    for off,size,step in stages:
        stage=plan_slice(current,off,size,step,dtype)
        offsets=tuple(a+b*s for a,b,s in zip(offsets,off,strides,strict=True))
        strides=tuple(a*b for a,b in zip(strides,step,strict=True))
        current=stage.sizes
    final=plan_slice(shape,offsets,current,strides,dtype)
    proof=_check_composition(shape,stages,dtype,final)
    return ComposedSlicePlan(shape,stages,dtype,current,final,proof)


def _check_composition(shape,stages,dtype,final):
    """Prove original axis recurrence against the actual final DMA mapping."""
    import z3
    current=shape; stage_shapes=[]; intermediate_elements=0
    for off,size,step in stages:
        stage=plan_slice(current,off,size,step,dtype)
        stage_shapes.append(current); current=stage.sizes
        intermediate_elements+=prod(current)
    if final.shape != shape or final.sizes != current or final.dtype != dtype:
        raise MovementCompositionError('composed slice boundary changed')
    descriptor_proof=final.revalidate()
    index=[z3.Int('output_axis_'+str(i)) for i in range(len(shape))]
    coordinate=list(index)
    for off,_,step in reversed(stages):
        coordinate=[o+x*s for o,x,s in zip(off,coordinate,step,strict=True)]
    solver=z3.Solver();solver.set(timeout=5000)
    solver.add(*(b for x,n in zip(index,current,strict=True) for b in (x>=0,x<n)))
    solver.add(z3.Or(*(a != o+x*s for a,o,x,s in zip(coordinate,final.offsets,index,final.strides,strict=True))))
    status=solver.check()
    if status != z3.unsat:
        raise MovementCompositionError('original slice axis recurrence not proved: '+str(status))
    width=1 if dtype=='i8' else 4
    return {'schema':'submission.source_slice_composition.v1',
            'source_shape':list(shape),'output_shape':list(current),'dtype':dtype,
            'stages':[{'input_shape':list(sh),'offsets':list(o),'sizes':list(n),'strides':list(s)}
                      for sh,(o,n,s) in zip(stage_shapes,stages,strict=True)],
            'composed_offsets':list(final.offsets),'composed_strides':list(final.strides),
            'address_projection':descriptor_proof,'kernel_sha256':final.kernel.digest(),
            'removed_intermediate_elements':intermediate_elements-prod(current),
            'removed_intermediate_materialization_bytes':(intermediate_elements-prod(current))*width,
            'source_membership':'original slice chain only; every erased intermediate has exactly its next slice as sole user',
            'coverage':'per-axis original recurrence equals actual final checked DMA source address; all final output words retained',
            'completion':descriptor_proof['completion']}


def _slice(op):
    from merlin.common.mlir_query import op_name
    from .quantized_boundary import _type
    if (getattr(op,'name',None) is None or op_name(op)!='tensor.extract_slice'
            or len(op.operands)!=1 or len(op.results)!=1 or op.regions
            or getattr(getattr(op,'parent',None),'parent_op',lambda:None)() is None
            or op.parent.parent_op().name!='func.func'):
        return None
    src,dst=_type(op.operands[0]),_type(op.results[0])
    if not src['dense'] or not dst['dense'] or src['dtype']!=dst['dtype'] or src['dtype'] not in ('i8','i32'):
        return None
    try:
        def values(key):return tuple(op.properties.get(key,op.attributes.get(key)).get_values())
        stage={k:values('static_'+k) for k in ('offsets','sizes','strides')}
        plan=plan_slice(src['shape'],stage['offsets'],stage['sizes'],stage['strides'],src['dtype'])
        if list(plan.sizes)!=dst['shape']:return None
        return stage,src,dst
    except (ValueError,TypeError,AttributeError,OverflowError):
        return None


def composed_integer_slice_candidate(op):
    """Admit maximal source chains without erasing any observed intermediate."""
    from .source_groups import _standalone_member,_standalone_capability
    root=_slice(op)
    if root is None:return None
    uses=tuple(op.results[0].uses)
    if len(uses)==1 and _slice(uses[0].operation) is not None:
        # Let the outermost eligible slice own the entire single-use chain.
        return None
    chain=[op];current=op
    while len(chain)<32:
        source=current.operands[0];producer=source.owner
        if (_slice(producer) is None or producer.parent is not op.parent
                or len(tuple(source.uses))!=1 or next(iter(source.uses)).operation is not current):
            break
        chain.append(producer);current=producer
    if len(chain)<2:return None
    chain.reverse();parsed=[_slice(stage) for stage in chain]
    try:
        stages=[item[0] for item in parsed];dtype=parsed[0][1]['dtype']
        plan=plan_composed_slice(parsed[0][1]['shape'],stages,dtype)
        if list(plan.output_shape)!=root[2]['shape']:return None
        attrs={'source_slice_stages':[{k:list(v) for k,v in stage.items()} for stage in stages],
               'output_dtype':dtype}
        proof={'form':'single_use_integer_slice_chain','composition':plan.proof,
               'source_members':[{'operation':'tensor.extract_slice',
                                  'source_ids':{k:str(v) for k,v in stage.attributes.items()
                                                if k.startswith(('prov.','merlin.source'))}}
                                 for stage in chain]}
        candidate=_standalone_member(op,chain,'COMPOSED_INTEGER_SLICE',{'src':chain[0].operands[0]},attrs,proof)
        if candidate is not None:
            # Dependency closure may include original literal initializers.
            # Those values obey the same observation boundary as slice results.
            owned=set(candidate.operations)
            if any(value is not candidate.output and any(use.operation not in owned for use in value.uses)
                   for member in candidate.operations for value in member.results):
                return None
            _standalone_capability(candidate)
        return candidate
    except (ValueError,TypeError,AttributeError,OverflowError):
        return None


def lower_composed_slice(ctx,operands,attributes):
    """Emit exactly the original-chain projection using shared slice lowering."""
    if set(operands)!={'src','dst'} or set(attributes)!={'source_slice_stages','output_dtype'}:
        raise MovementCompositionError('composed slice operands or attributes changed')
    src,dst=ctx.spec(operands['src']),ctx.spec(operands['dst'])
    dtype=attributes['output_dtype']
    if src['dtype']!=dtype or dst['dtype']!=dtype:
        raise MovementCompositionError('composed slice dtype changed')
    plan=plan_composed_slice(src['shape'],attributes['source_slice_stages'],dtype)
    if list(plan.output_shape)!=dst['shape']:
        raise MovementCompositionError('composed slice final output shape changed')
    plan.revalidate()
    final=plan.slice_plan
    receipt=lower_slice(ctx,operands,{'source_offsets':list(final.offsets),'source_sizes':list(final.sizes),
                                   'source_strides':list(final.strides),'output_dtype':dtype})
    result={'composition':plan.proof,'slice_emission':receipt,'kernel_sha256':plan.kernel.digest()}
    ctx.metadata.setdefault('source_movement_composition',[]).append(result)
    return result
