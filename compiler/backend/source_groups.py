"""Exact source-bound dense groups and typed standalone device regions.

Selected output facts live beside this module. Unknown float values remain host
seams; no scale or bias is guessed. All external uses are retained at boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json


class SourceGroupError(ValueError):
    pass


@dataclass
class _Plan:
    root: object
    operations: tuple
    inputs: tuple
    output: object
    attributes: dict = field(default_factory=dict)
    constants: dict = field(default_factory=dict)
    diagnostic: dict = field(default_factory=dict)
    storage_shapes: dict = field(default_factory=dict)


def _path():
    from merlin.targetgen.output_path import validate
    return validate(json.loads(Path(__file__).with_name('source_groups_output_path.json').read_text()))


def _spec(value):
    from xdsl.dialects.builtin import TensorType
    if not isinstance(value.type, TensorType) or any(d <= 0 for d in value.type.get_shape()):
        raise SourceGroupError(f'device group needs a static ranked tensor: {value.type}')
    return {'shape': list(value.type.get_shape()), 'dtype': str(value.type.element_type)}


def _dependencies(root, members):
    """Close initializer/constant dependencies without absorbing another tensor computation."""
    from xdsl.ir import Operation
    block = root.parent
    owned = set(members)
    inputs = []
    def visit(value):
        owner = value.owner
        if owner in owned:
            return
        if isinstance(owner, Operation) and owner.parent is block and owner.name in ('arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill'):
            owned.add(owner)
            for operand in owner.operands:
                visit(operand)
            return
        _spec(value)
        if value not in inputs:
            inputs.append(value)
    # Captured scalar constants may occur inside a linalg region.
    for member in members:
        local = set(member.walk())
        for nested in member.walk():
            for value in nested.operands:
                if value.owner in local or getattr(value.owner, 'parent_op', lambda: None)() in local:
                    continue
                # Region block arguments are local, not function boundaries.
                from xdsl.ir import BlockArgument
                if isinstance(value, BlockArgument) and value.owner is not block:
                    continue
                visit(value)
    operations = tuple(op for op in block.ops if op in owned)
    # Function-argument order makes callback ABI stable after source cloning.
    external = [value for value in block.args if value in inputs]
    external += [value for value in inputs if value not in external]
    return operations, tuple(external)


def _root_inputs(root):
    from merlin.frontends.linalg_patterns import recognize_signed_i8_i32_matmul
    if root.name == 'linalg.generic':
        return recognize_signed_i8_i32_matmul(root)
    if root.name == 'linalg.matmul' and len(root.operands) == 3:
        lhs, rhs, init = root.operands
        a,b,c = map(_spec,(lhs,rhs,init))
        if a['dtype']=='i8' and b['dtype']=='i8' and c['dtype']=='i32' and len(a['shape'])==len(b['shape'])==2 and a['shape'][1]==b['shape'][0] and c['shape']==[a['shape'][0],b['shape'][1]]:
            return lhs,rhs,init
    raise SourceGroupError('integer source contraction has no supported dense scalar-body proof')


def _contraction_plans(module,*,prefer_quantized_readout=False):
    from merlin.compile.analysis.contraction_shapes import observe_contractions, zero_initialised
    from merlin.xdsl_dialects.lowering import output_fusion as OF
    from merlin.llvmlower.requantization_match import match
    plans=[]
    exact_quantizers={}
    if prefer_quantized_readout:
        from .quantized_boundary import prove_accumulator_quantize,QuantizedBoundaryError
        from merlin.common.mlir_query import op_name
        for op in module.walk():
            if op_name(op) != 'quant_ext.quantize_per_tensor':continue
            try:
                proof=prove_accumulator_quantize(op,allow_shape_views=True)
            except QuantizedBoundaryError:
                continue
            exact_quantizers.setdefault(proof.source,[]).append(proof)
    for root, shape in observe_contractions(module):
        if shape.dtypes and all(t.startswith(('f','bf')) for t in shape.dtypes):
            continue
        try:
            lhs,rhs,init = _root_inputs(root)
        except ValueError as exc:
            raise SourceGroupError(f'integer contraction compiler gap: {exc}') from exc
        a,b=_spec(lhs),_spec(rhs)
        if len(a['shape']) != 2 or len(b['shape']) != 2:
            raise SourceGroupError('source group adapter currently requires rank-two dense contractions')
        m,k=a['shape']; _,n=b['shape']
        zero=zero_initialised(root)
        attrs={'epilogue':[],'output_dtype':'i32'}
        members=[root]; out=root.results[0]; constants={}
        diagnostic={'cut':'bare','refusals':[]}
        if zero:
            values=OF.Values()
            chain=OF.source_chain(root,values)
            decision=OF.decide(chain,_path(),values,{id(root):(m,k,n,2)})
            diagnostic=decision.record()
            # Layout/pool/residual realization needs a different device schedule.
            supported=(decision.cut=='requant' and not any(op.name=='linalg.transpose' for op in decision.members)
                       and not any(s.kind in ('pool','pad','residual') for s in chain.stages))
            if supported:
                candidate=decision.members[-1].results[0]
                supported=_spec(candidate)=={'shape':[m,n],'dtype':'i8'}
            if supported:
                members += list(decision.members); out=candidate
                scales=decision.proof['device_scale']
                attrs={'epilogue':['acc_scale']+(['relu'] if 'relu' in decision.stages else []),'output_dtype':'i8'}
                if len(set(scales))==1:attrs['acc_scale']=scales[0]
                else:attrs['acc_scale_by_column']=scales
                offsets=decision.proof.get('offsets',[])
                if any(offsets):
                    constants['_proved_bias']={'shape':[n],'dtype':'i32','values':offsets}
                    attrs['bias']='_proved_bias'; attrs['epilogue'].insert(0,'bias_add')
            else:
                uses=list(out.uses)
                if len(uses)==1:
                    from .frontend import _saturating_cast
                    narrow = uses[0].operation
                    if _saturating_cast(narrow) is out:
                        members.append(narrow); out=narrow.results[0]
                        attrs={'epilogue':[], 'output_dtype':'i8'}
                        diagnostic={'cut':'requant','matcher':'exact integer saturating cast'}
                    try:
                        rq=match(uses[0].operation)
                    except ValueError:
                        rq=None
                    if rq is not None:
                        members.append(rq.epilogue); out=rq.epilogue.results[0]
                        attrs={'epilogue':['acc_scale']+(['relu'] if rq.relu else []),'output_dtype':'i8','acc_scale':rq.proof['scale']}
                        if rq.integer_bias is not None:
                            constants['_proved_bias']={'shape':[n],'dtype':'i32','values':rq.integer_bias}
                            attrs['bias']='_proved_bias';attrs['epilogue'].insert(0,'bias_add')
                        diagnostic={'cut':'requant','proof':rq.proof,'matcher':'requantization_match'}
        storage={}
        # The explicit quant_ext contract can cross dense source reshapes. Its
        # full-i32 proof admits runtime initializers too; no contraction bound
        # or guessed quantization parameter is required.
        if prefer_quantized_readout and diagnostic.get('cut')=='bare':
            for proof in exact_quantizers.get(root.results[0],()):
                if any(member.parent is not root.parent for member in proof.operations):continue
                proposed=[root,*proof.operations]
                owned=set(proposed)
                candidate=proof.quantizer.results[0]
                if any(value is not candidate and any(use.operation not in owned for use in value.uses)
                       for member in proposed for value in member.results):continue
                # Source arithmetic proves a candidate, not a target capability.
                # Reuse the selected integer DMA/readout admission before the
                # candidate can change a device group's exit.
                probe=_Standalone(root,(),(),candidate,'ACCUMULATOR_READOUT',
                                  {'src':root.results[0]},proof.attributes,{})
                try:
                    _standalone_capability(probe)
                except SourceGroupError:
                    continue
                proof.revalidate()
                members=proposed;out=candidate;attrs=dict(proof.attributes);constants={}
                diagnostic={'cut':'requant','matcher':'exact_source_quant_ext_readout',
                            'proof':proof.record(),'source_accumulator_retained':False}
                # Dense lowering writes the original [M,N] row-major matrix;
                # shape-only source views have the identical flattened ABI.
                if _spec(out)['shape'] != [m,n]:storage[out]=[m,n]
                break
        # A fused region must publish every live value. Fall back to its raw
        # contraction when fanout requires an additional output ABI.
        owned=set(members)
        if any(v is not out and any(u.operation not in owned for u in v.uses) for op in members for v in op.results):
            members=[root];out=root.results[0];constants={};attrs={'epilogue':[],'output_dtype':'i32'}
            diagnostic={'cut':'bare','reason':'fusion has another live intermediate output'}
            storage={}
        operations,inputs=_dependencies(root,members)
        # Root operands precede auxiliary inputs for stable explicit addressing.
        if not zero:
            attrs['_source_initializer']=init
            diagnostic['initializer']='runtime accumulator matrix retained exactly'
        plans.append(_Plan(root,operations,inputs,out,attrs,constants,diagnostic,storage))
    return plans


@dataclass
class _Standalone:
    root: object
    operations: tuple
    inputs: tuple
    output: object
    opcode: str
    operands: dict
    attributes: dict
    diagnostic: dict
    constants: dict = field(default_factory=dict)
    storage_shapes: dict = field(default_factory=dict)


def _scalar_constant(value):
    owner=value.owner
    if getattr(owner,'name','')=='tensor.splat':
        return _scalar_constant(owner.operands[0])
    if getattr(owner,'name','')=='arith.constant':
        attr=owner.properties.get('value',owner.attributes.get('value'))
        return getattr(getattr(attr,'value',None),'data',None)
    return None


def _integer_tensor_literal(value):
    """Decode only source-declared integer literals; never evaluate tensor math."""
    from math import prod
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    spec=_spec(value)
    if spec['dtype'] not in ('i8','i32'):
        return None
    owner=value.owner
    name=getattr(owner,'name','')
    count=prod(spec['shape'])
    values=None
    if name=='arith.constant':
        attr=owner.properties.get('value',owner.attributes.get('value'))
        if isinstance(attr,DenseIntOrFPElementsAttr):
            values=list(attr.get_values())
    elif name in ('tensor.splat','linalg.fill'):
        scalar=_scalar_constant(owner.operands[0])
        if isinstance(scalar,int):values=[scalar]*count
    if values is None or len(values)!=count or any(not isinstance(v,int) for v in values):
        return None
    return {**spec,'values':values}


def _identity_maps(op,arity,rank):
    from xdsl.ir.affine import AffineMap
    maps=op.properties.get('indexing_maps')
    iters=op.properties.get('iterator_types')
    return (maps is not None and tuple(m.data for m in maps)==(AffineMap.identity(rank),)*arity
            and iters is not None and [a.data.value for a in iters]==['parallel']*rank)


def _standalone_member(root,members,opcode,operands,attrs,diagnostic,storage_shapes=None):
    out=root.results[0]
    owned=set(members)
    if any(v is not out and any(u.operation not in owned for u in v.uses) for op in members for v in op.results):
        return None
    operations,inputs=_dependencies(root,members)
    return _Standalone(root,operations,inputs,out,opcode,operands,attrs,diagnostic,storage_shapes=storage_shapes or {})


def _typed_movement(op):
    if op.name=='linalg.copy' and len(op.operands)==2 and len(op.results)==1:
        src=op.operands[0]
        if _spec(src)==_spec(op.results[0]) and _spec(src)['dtype'] in ('i8','i32'):
            return src,[]
        return None
    if op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1:
        return None
    src=op.operands[0]
    a,c=_spec(src),_spec(op.results[0])
    if a['shape']!=c['shape'] or a['dtype'] not in ('i8','i32') or c['dtype'] not in ('i8','i32'):
        return None
    if not _identity_maps(op,2,len(a['shape'])):
        return None
    body=op.regions[0].block
    ops=[o for o in body.ops if o.name!='arith.constant']
    if len(ops)==1 and ops[0].name=='linalg.yield' and list(ops[0].operands)==[body.args[0]] and a==c:
        return src,[]
    if (len(ops)==2 and ops[0].name=='arith.extsi' and list(ops[0].operands)==[body.args[0]]
            and str(ops[0].results[0].type)=='i32' and a['dtype']=='i8' and c['dtype']=='i32'
            and ops[1].name=='linalg.yield' and list(ops[1].operands)==list(ops[0].results)):
        return src,[]
    if (a==c and a['dtype']=='i8' and len(ops)==2 and ops[0].name=='arith.maxsi'
            and body.args[0] in ops[0].operands and any(_scalar_constant(v)==0 for v in ops[0].operands)
            and ops[1].name=='linalg.yield' and list(ops[1].operands)==list(ops[0].results)):
        return src,['relu']
    return None


def _contiguous_transpose(op):
    """Prove a transpose preserves contiguous element order after unit axes.

    Moving a nonunit axis across another nonunit axis requires actual data
    permutation, which this movement adapter deliberately does not represent.
    """
    from math import prod
    if op.name!='linalg.transpose' or len(op.operands)!=2 or len(op.results)!=1:
        return None
    src=op.operands[0];a,c=_spec(src),_spec(op.results[0])
    from xdsl.dialects.builtin import NoneAttr
    if any(not isinstance(v.type.encoding,NoneAttr) for v in (src,op.results[0])):return None
    if a['dtype'] not in ('i8','i32') or c['dtype']!=a['dtype']:
        return None
    attr=op.properties.get('permutation')
    if attr is None or not hasattr(attr,'get_values'):return None
    perm=list(attr.get_values());shape=a['shape']
    if sorted(perm)!=list(range(len(shape))) or c['shape']!=[shape[i] for i in perm]:return None
    if [i for i in perm if shape[i]!=1]!=[i for i,n in enumerate(shape) if n!=1]:return None
    if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return None
    body=op.regions[0].block;ops=list(body.ops)
    if len(body.args)!=2 or len(ops)!=1 or ops[0].name!='linalg.yield' or list(ops[0].operands)!=[body.args[0]]:return None
    return src,{src:[prod(shape)],op.results[0]:[prod(shape)]},dict(
        form='contiguous_unit_axis_transpose',permutation=perm,
        source_shape=shape,result_shape=c['shape'],
        preserved_nonunit_axis_order=[i for i,n in enumerate(shape) if n!=1])


def _integer_transpose(op):
    """Bind a true static integer permutation to checked strided DMA effects."""
    if op.name!='linalg.transpose' or len(op.operands)!=2 or len(op.results)!=1:return None
    src=op.operands[0];a,c=_spec(src),_spec(op.results[0])
    from xdsl.dialects.builtin import NoneAttr
    if any(not isinstance(v.type.encoding,NoneAttr) for v in (src,op.results[0])):return None
    if a['dtype'] not in ('i8','i32') or c['dtype']!=a['dtype'] or not 2<=len(a['shape'])<=4:return None
    attr=op.properties.get('permutation')
    if attr is None or not hasattr(attr,'get_values'):return None
    perm=list(attr.get_values())
    if sorted(perm)!=list(range(len(a['shape']))) or c['shape']!=[a['shape'][i] for i in perm]:return None
    if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return None
    body=op.regions[0].block;ops=list(body.ops)
    if len(body.args)!=2 or len(ops)!=1 or ops[0].name!='linalg.yield' or list(ops[0].operands)!=[body.args[0]]:return None
    from .transpose import plan_transpose,TransposeError
    try:plan=plan_transpose(tuple(a['shape']),tuple(perm),a['dtype'])
    except TransposeError:return None
    return src,{'permutation':perm,'output_dtype':a['dtype']},dict(
        form='checked_strided_integer_transpose',address_projection=plan.proof)


def _accumulator_readout(op):
    """Prove one pointwise i32 readout from its complete scalar SSA body.

    Integer clamps never round before saturation. Floating forms explicitly
    convert to f32, optionally multiply once, round to even, then clamp. No
    scale composition, fastmath, dynamic bounds or unsigned casts are inferred.
    """
    import math
    if op.name != 'linalg.generic' or len(op.operands) != 2 or len(op.results) != 1:
        return None
    src = op.operands[0]
    a,c = _spec(src),_spec(op.results[0])
    if a['dtype'] != 'i32' or c != {'shape':a['shape'],'dtype':'i8'} or not _identity_maps(op,2,len(a['shape'])):
        return None
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return None
    body=op.regions[0].block
    ops=[o for o in body.ops if o.name!='arith.constant']
    if len(body.args)!=2 or not ops:
        return None
    if any(str(o.properties.get('fastmath','#arith.fastmath<none>'))!='#arith.fastmath<none>' for o in ops):
        return None
    value=body.args[0]; scale=1.0; floating=False; multiplied=False
    if ops[0].name=='arith.sitofp':
        cast=ops.pop(0)
        if list(cast.operands)!=[value] or str(cast.results[0].type)!='f32':return None
        value=cast.results[0];floating=True
        if ops and ops[0].name=='arith.mulf':
            mul=ops.pop(0);multiplied=True
            if value not in mul.operands or str(mul.results[0].type)!='f32':return None
            literal=mul.operands[1] if mul.operands[0] is value else mul.operands[0]
            scale=_scalar_constant(literal)
            if str(literal.type)!='f32' or not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<0:return None
            value=mul.results[0]
        if not ops or ops[0].name!='math.roundeven' or list(ops[0].operands)!=[value]:return None
        value=ops.pop(0).results[0]
    low_name,high_name,cast_name=('arith.maximumf','arith.minimumf','arith.fptosi') if floating else ('arith.maxsi','arith.minsi','arith.trunci')
    relu=False
    # Accept max(x,0) followed by the full signed clamp, or lower clamp zero.
    if len(ops)==5 and ops[0].name==low_name and value in ops[0].operands:
        other=ops[0].operands[1] if ops[0].operands[0] is value else ops[0].operands[0]
        if _scalar_constant(other)!=0:return None
        value=ops.pop(0).results[0];relu=True
    if len(ops)!=4 or [o.name for o in ops]!=[low_name,high_name,cast_name,'linalg.yield']:return None
    lo,hi,cast,yld=ops
    if value not in lo.operands:return None
    bound=lo.operands[1] if lo.operands[0] is value else lo.operands[0]
    lower=_scalar_constant(bound)
    if lower not in (-128,0):return None
    relu=relu or lower==0
    if lo.results[0] not in hi.operands:return None
    bound=hi.operands[1] if hi.operands[0] is lo.results[0] else hi.operands[0]
    if _scalar_constant(bound)!=127:return None
    if list(cast.operands)!=[hi.results[0]] or str(cast.results[0].type)!='i8' or list(yld.operands)!=list(cast.results):return None
    ep=(['acc_scale'] if floating else [])+(['relu'] if relu else [])
    attrs={'output_dtype':'i8','acc_scale':float(scale),'epilogue':ep}
    proof={'form':'standalone_f32_roundeven_readout' if floating else 'standalone_integer_saturating_readout',
           'input_dtype':'i32','clamp':[-128,127],'source_clamp':[lower,127],'relu':relu,'scale_f32':float(scale),
           'source_rounding':'half_even' if floating else 'none_integer_identity',
           'source_order':(['sitofp_f32']+(['multiply_f32'] if multiplied else [])+['roundeven','clamp','fptosi_i8']) if floating else ['signed_clamp','trunci_i8']}
    return src,attrs,proof


def _integer_saturating_add(op):
    if op.name!='linalg.generic' or len(op.operands)!=3 or len(op.results)!=1:
        return None
    lhs,rhs=op.operands[:2]
    spec=_spec(lhs)
    if spec['dtype']!='i8' or _spec(rhs)!=spec or _spec(op.results[0])!=spec or not _identity_maps(op,3,len(spec['shape'])):
        return None
    body=op.regions[0].block
    ops=[o for o in body.ops if o.name!='arith.constant']
    if [o.name for o in ops]!=['arith.extsi','arith.extsi','arith.addi','arith.maxsi','arith.minsi','arith.trunci','linalg.yield']:
        return None
    a,b,add,lo,hi,cast,yld=ops
    if (list(a.operands)!=[body.args[0]] or list(b.operands)!=[body.args[1]]
            or str(a.results[0].type)!='i32' or str(b.results[0].type)!='i32'
            or set(add.operands)!={a.results[0],b.results[0]} or add.overflow_flags.data
            or add.results[0] not in lo.operands or lo.results[0] not in hi.operands
            or list(cast.operands)!=[hi.results[0]] or str(cast.results[0].type)!='i8'
            or list(yld.operands)!=list(cast.results)):
        return None
    low=next((_scalar_constant(v) for v in lo.operands if v is not add.results[0]),None)
    high=next((_scalar_constant(v) for v in hi.operands if v is not lo.results[0]),None)
    if low not in (-128,0) or high!=127:
        return None
    return lhs,rhs,low==0


def _quantized_residual(op):
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_numerics as GN
    from .elementwise import residual_numeric_screen
    if mq.op_name(op)!='quant_ext.quantize_per_tensor' or not op.results or _spec(op.results[0])['dtype']!='i8':
        return None
    tail=op.operands[0].owner
    from xdsl.ir import Operation
    if not isinstance(tail,Operation):return None
    stage=CG.classify(tail)
    relu=stage is not None and stage.kind==CG.RELU
    members=[op]
    if relu:
        members.append(tail)
        tail=tail.operands[0].owner
    if not isinstance(tail,Operation):return None
    stage=CG.classify(tail)
    if stage is None or stage.kind!=CG.RESIDUAL_ADD or len(tail.operands)<2:
        return None
    # Classification locates an opportunity; scalar SSA topology proves it.
    if tail.name!='linalg.generic' or len(tail.operands)!=3 or not _identity_maps(tail,3,len(_spec(tail.results[0])['shape'])):
        return None
    body=tail.regions[0].block
    scalar=[o for o in body.ops if o.name!='arith.constant']
    if (len(scalar)!=2 or scalar[0].name!='arith.addf'
            or set(scalar[0].operands)!={body.args[0],body.args[1]}
            or scalar[1].name!='linalg.yield' or list(scalar[1].operands)!=list(scalar[0].results)):
        return None
    if relu:
        activation=members[1]
        if activation.name!='linalg.generic' or len(activation.operands)!=2 or not _identity_maps(activation,2,len(_spec(activation.results[0])['shape'])):
            return None
        bb=activation.regions[0].block
        scalar=[o for o in bb.ops if o.name!='arith.constant']
        if (len(scalar)!=2 or scalar[0].name not in ('arith.maximumf','arith.maxnumf')
                or bb.args[0] not in scalar[0].operands or not any(_scalar_constant(v)==0 for v in scalar[0].operands)
                or scalar[1].name!='linalg.yield' or list(scalar[1].operands)!=list(scalar[0].results)):
            return None
    sources=[v.owner for v in tail.operands[:2]]
    if any(not isinstance(src,Operation) or mq.op_name(src)!='quant_ext.dequantize_per_tensor' for src in sources):
        return None
    lhs,rhs=[src.operands[0] for src in sources]
    if _spec(lhs)!=_spec(rhs) or _spec(lhs)!=_spec(op.results[0]) or _spec(lhs)['dtype']!='i8':
        return None
    if any(_spec(v)['dtype']!='f32' for v in tail.operands[:2]):
        return None
    try:
        scales=[GN._scale_source(src) for src in [*sources,op]]
    except GN.GroupNumericsError:
        return None
    from math import isfinite
    if any(s.value is None or not isfinite(s.value) or s.zero_point!=0 for s in scales) or scales[-1].value<=0:
        return None
    if GN._int_attr(op,'quant_min')!=-128 or GN._int_attr(op,'quant_max')!=127:
        return None
    # A target's bounded approximation is not a licence to alter an exact source.
    # Only the typed source's explicit numeric bound can grant nonzero error.
    bound=0
    for key in ('merlin.bound_lsb','bound_lsb'):
        attr=op.attributes.get(key,op.properties.get(key))
        val=getattr(getattr(attr,'value',None),'data',None)
        if val is not None:bound=int(val)
    ls,rs=scales[0].value/scales[2].value,scales[1].value/scales[2].value
    try:
        proof=residual_numeric_screen(float(ls),float(rs),bound,relu)
    except ValueError:
        return None
    # Prove the original dequantize/add/quantize chain too: normalization of
    # float scales is not assumed to preserve its separately rounded f32 stages.
    import numpy as np
    a=np.arange(-128,128,dtype=np.float32)[:,None]
    b=np.arange(-128,128,dtype=np.float32)[None,:]
    source=np.add(np.multiply(a,np.float32(scales[0].value),dtype=np.float32),np.multiply(b,np.float32(scales[1].value),dtype=np.float32),dtype=np.float32)
    if relu:source=np.maximum(source,np.float32(0))
    source=np.clip(np.rint(np.multiply(source,np.float32(1.0/np.float32(scales[2].value)),dtype=np.float32)),-128,127)
    av=np.clip(np.rint(np.multiply(a,np.float32(proof['lhs_load_f32']),dtype=np.float32)),-128,127)
    bv=np.clip(np.rint(np.multiply(b,np.float32(proof['rhs_load_f32']),dtype=np.float32)),-128,127)
    dev=av+bv
    if relu:dev=np.maximum(dev,np.float32(0))
    dev=np.clip(np.rint(np.multiply(dev,np.float32(proof['readout_scale_f32']),dtype=np.float32)),-128,127)
    actual=int(np.max(np.abs(source-dev)))
    if actual>bound:return None
    members += [tail,*sources]
    attrs={'lhs_scale':float(ls),'rhs_scale':float(rs),'bound_lsb':bound,'source_scales':[s.value for s in scales],'epilogue':['relu'] if relu else [],'output_dtype':'i8'}
    return members,{'lhs':lhs,'rhs':rhs},attrs,{'form':'standalone_operand_sum','pairs_checked':65536,'max_error_lsb':actual,'source_bound_lsb':bound}


def _integer_pool(op):
    """Prove an unpadded NHWC signed max reduction, including its initializer."""
    from xdsl.ir.affine import AffineMap,AffineExpr
    if op.name!='linalg.generic' or len(op.operands)!=3 or len(op.results)!=1:
        return None
    src,window,init=op.operands
    a,c=_spec(src),_spec(op.results[0])
    w=_spec(window)
    if a['dtype']!='i8' or c['dtype']!='i8' or len(a['shape'])!=4 or len(c['shape'])!=4 or len(w['shape'])!=2:
        return None
    batch,h,width,channels=a['shape'];nb,oh,ow,nc=c['shape'];ph,pw=w['shape']
    if nb!=batch or nc!=channels or ph!=pw or not 1<=ph<=3:
        return None
    if _scalar_constant(init)!=-128:
        # linalg.fill is another exact initialization spelling.
        own=init.owner
        if getattr(own,'name','')!='linalg.fill' or _scalar_constant(own.operands[0])!=-128:return None
    maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
    if maps is None or len(maps)!=3 or iters is None or [a.data.value for a in iters]!=['parallel']*4+['reduction']*2:
        return None
    maps=[m.data for m in maps]
    d=[AffineExpr.dimension(i) for i in range(6)]
    if maps[1]!=AffineMap(6,0,(d[4],d[5])) or maps[2]!=AffineMap(6,0,tuple(d[:4])):
        return None
    sh=sw=None
    for stride in range(1,max(h,width)+1):
        if maps[0]==AffineMap(6,0,(d[0],d[1]*stride+d[4],d[2]*stride+d[5],d[3])):
            sh=sw=stride;break
    if sh is None or oh!=(h-ph)//sh+1 or ow!=(width-pw)//sw+1:
        return None
    body=op.regions[0].block;ops=list(body.ops)
    if (len(ops)!=2 or ops[0].name!='arith.maxsi' or set(ops[0].operands)!={body.args[0],body.args[2]}
            or ops[1].name!='linalg.yield' or list(ops[1].operands)!=list(ops[0].results)):
        return None
    attrs={'epilogue':['acc_scale','maxpool'],'acc_scale':1.0,'output_dtype':'i8','pool_in_dims':[h,width],'pool_size':[ph,pw],'pool_stride':[sh,sw],'pool_padding':[0,0,0,0]}
    return src,attrs,{}


def _standalone_capability(candidate):
    """Require the selected output stages used by the concrete DMA schedule."""
    path=_path()
    acc={s['kind'] for s in path.get('accumulator_stages',()) if s.get('dtype')=='i32'}
    readout=path.get('readout',())
    raw=path.get('exits',{}).get('accumulator')
    narrow=any(s.get('kind')=='requant' and s.get('to')=='i8'
               and s.get('rounding')=='half_even' and s.get('clamp')==[-128,127] for s in readout)
    need_raw=_spec(candidate.output)['dtype']=='i32'
    if path.get('accumulator',{}).get('dtype')!='i32' or 'preload' not in acc or (need_raw and not raw) or (not need_raw and not narrow):
        raise SourceGroupError('selected output path does not admit this standalone DMA/readout form')
    if candidate.opcode in ('BIAS_ADD','RESIDUAL_ADD') and 'accumulate' not in acc:
        raise SourceGroupError('selected output path does not admit accumulator operand addition')
    if 'relu' in candidate.attributes.get('epilogue',()) and not any(s.get('kind')=='activation' and s.get('function')=='relu' and s.get('domain')=='accumulator' for s in readout):
        raise SourceGroupError('selected output path does not admit accumulator ReLU')
    if candidate.opcode=='MAXPOOL2D' and not any(s.get('kind')=='pool' and 'max' in s.get('reductions',()) for s in readout):
        raise SourceGroupError('selected output path does not admit max pooling')


def _standalone_plans(module,taken,*,transpose_transfers=False,prefer_quantized_readout=False):
    from .frontend import _bias_add
    from merlin.common.mlir_query import op_name
    plans=[]
    for op in module.walk():
        if op in taken or not op.results or op_name(op) not in ('linalg.generic','linalg.copy','linalg.transpose','quant_ext.quantize_per_tensor'):
            continue
        from xdsl.dialects.builtin import TensorType
        ty=op.results[0].type
        if not isinstance(ty,TensorType) or str(ty.element_type) not in ('i8','i32') or not ty.get_shape():
            continue
        # Only block-level tensor operations can become one device call.
        if getattr(op.parent,'parent_op',lambda:None)().name!='func.func':
            continue
        candidate=None
        transpose=_contiguous_transpose(op)
        if transpose is not None:
            src,storage,proof=transpose
            candidate=_standalone_member(op,[op],'MOVEMENT',{'src':src},
                {'semantic':'mvin_mvout','output_dtype':_spec(op.results[0])['dtype']},proof,storage)
        if candidate is None and transpose_transfers:
            transpose=_integer_transpose(op)
            if transpose is not None:
                src,attrs,proof=transpose
                candidate=_standalone_member(op,[op],'STRIDED_MOVEMENT',{'src':src},attrs,proof)
        bias=_bias_add(op) if op.name=='linalg.generic' else None
        if bias is not None:
            candidate=_standalone_member(op,[op],'BIAS_ADD',{'src':bias[0],'bias':bias[1]}, {'output_dtype':'i32'}, {'form':'standalone_i32_column_bias','arithmetic':'exact accumulator addition'})
        if candidate is None:
            movement=_typed_movement(op)
            if movement is not None:
                src,ep=movement
                if ep:
                    # MOVEMENT preserves values. Express i8 ReLU as the exact
                    # legal residual x + 0 followed by its activation instead.
                    candidate=_standalone_member(op,[op],'RESIDUAL_ADD',{'lhs':src,'rhs':'_relu_zero'},
                        {'lhs_scale':1.0,'rhs_scale':1.0,'bound_lsb':0,'output_dtype':'i8','epilogue':['relu']},
                        {'form':'exact_standalone_i8_relu','source_bound_lsb':0})
                    if candidate is not None:
                        from math import prod
                        spec=_spec(src)
                        candidate.constants['_relu_zero']={**spec,'values':[0]*prod(spec['shape'])}
                else:
                    candidate=_standalone_member(op,[op],'MOVEMENT',{'src':src},
                        {'semantic':'mvin_mvout','output_dtype':_spec(op.results[0])['dtype']},
                        {'form':'value_preserving_movement'})
        if candidate is None:
            readout=_accumulator_readout(op)
            if readout is not None:
                src,attrs,proof=readout
                candidate=_standalone_member(op,[op],'ACCUMULATOR_READOUT',{'src':src},attrs,proof)
        if candidate is None and op_name(op)=='quant_ext.quantize_per_tensor':
            from .quantized_boundary import prove_accumulator_quantize,QuantizedBoundaryError
            try:
                readout=prove_accumulator_quantize(op,allow_shape_views=prefer_quantized_readout)
            except QuantizedBoundaryError:
                readout=None
            if readout is not None and not any(member in taken for member in readout.operations):
                readout.revalidate()
                storage={}
                if _spec(readout.source)['shape'] != _spec(op.results[0])['shape']:
                    from math import prod
                    shape=[prod(_spec(readout.source)['shape'])]
                    storage={readout.source:shape,op.results[0]:shape}
                candidate=_standalone_member(op,readout.operations,'ACCUMULATOR_READOUT',
                    {'src':readout.source},readout.attributes,readout.record(),storage)
        if candidate is None:
            add=_integer_saturating_add(op)
            if add is not None:
                lhs,rhs,relu=add
                candidate=_standalone_member(op,[op],'RESIDUAL_ADD',{'lhs':lhs,'rhs':rhs},{'lhs_scale':1.0,'rhs_scale':1.0,'bound_lsb':0,'output_dtype':'i8','epilogue':['relu'] if relu else []},{'form':'exact_standalone_saturating_sum','source_bound_lsb':0})
        if candidate is None:
            residual=_quantized_residual(op)
            if residual is not None:
                members,operands,attrs,diag=residual
                if not any(member in taken for member in members):
                    candidate=_standalone_member(op,members,'RESIDUAL_ADD',operands,attrs,diag)
        if candidate is None:
            pool=_integer_pool(op)
            if pool is not None:
                src,attrs,storage=pool
                candidate=_standalone_member(op,[op],'MAXPOOL2D',{'src':src},attrs,{'form':'standalone_i8_nhwc_maxpool','padding':'none','source_initializer':-128},storage)
        if candidate is not None:
            _standalone_capability(candidate)
            plans.append(candidate);taken.update(candidate.operations)
    return plans


def _plans(module,*,transpose_transfers=False,prefer_quantized_readout=False):
    contractions=_contraction_plans(module,prefer_quantized_readout=prefer_quantized_readout)
    taken={op for p in contractions for op in p.operations}
    return contractions+_standalone_plans(module,taken,transpose_transfers=transpose_transfers,
                                       prefer_quantized_readout=prefer_quantized_readout)


def discover_regions(module,*,transpose_transfers=False,prefer_quantized_readout=False):
    """Return exact regions; transpose transfers and source quantizer fusion are opt-in."""
    from .host_runtime import DeviceRegion
    return tuple(DeviceRegion(p.operations,p.inputs,(p.output,),reason=json.dumps(p.diagnostic,default=str))
                 for p in _plans(module,transpose_transfers=transpose_transfers,
                                 prefer_quantized_readout=prefer_quantized_readout))


def placement_diagnostics(module,*,transpose_transfers=False,prefer_quantized_readout=False):
    """Source-bound device decisions and explicit misses, without guessing eligibility.

    This is a static placement report, not measured host time or transfer traffic.
    A miss remains distinct from a proved hardware unsupported floating seam.
    """
    from math import prod
    from xdsl.dialects.builtin import TensorType
    from .frontend import _saturating_cast
    from merlin.common.mlir_query import op_name
    from .quantized_boundary import analyze_quantized_boundary,analyze_scalar_boundary
    plans=_plans(module,transpose_transfers=transpose_transfers,prefer_quantized_readout=prefer_quantized_readout)
    by_op={op:p for p in plans for op in p.operations}
    records=[]
    for index,op in enumerate(module.walk()):
        name=op_name(op)
        if not op.results or not (name.startswith('linalg.') or name.startswith('quant_ext.')) or name in ('linalg.fill','linalg.yield'):
            continue
        ty=op.results[0].type
        if not isinstance(ty,TensorType):continue
        p=by_op.get(op)
        quantized_boundary=analyze_quantized_boundary(op)
        scalar_boundary=analyze_scalar_boundary(op)
        rec={'source_index':index,'operation':name,'source_ids':{k:str(v) for k,v in op.attributes.items() if k.startswith(('prov.','merlin.source'))},'dtype':str(ty.element_type),'shape':list(ty.get_shape())}
        if p is not None:
            rec.update(placement='device',device_form=p.opcode if isinstance(p,_Standalone) else 'dense_accumulator_readout',proof=p.diagnostic)
        elif quantized_boundary is not None:
            remaining=quantized_boundary.get('remaining_proof',())
            rec.update(placement='host',status=quantized_boundary['status'],
                reason=('standalone: '+quantized_boundary.get('standalone',{}).get('reason','typed source admission is unresolved')
                        +('; source composition: '+str(remaining[0]) if remaining else '')),
                standalone=quantized_boundary.get('standalone',{}),
                fused={'status':quantized_boundary.get('fused',{}).get('status','unresolved_proof')},
                quantized_boundary=quantized_boundary)
        elif scalar_boundary is not None:
            rec.update(placement='host',status=scalar_boundary['status'],
                reason=(scalar_boundary['standalone']['reason']
                        +('; source composition: '+scalar_boundary['fused']['reason']
                          if scalar_boundary['status']=='unresolved_proof' else '')),
                standalone=scalar_boundary['standalone'],
                fused={'status':scalar_boundary['fused']['status']},scalar_boundary=scalar_boundary)
        elif op.name=='linalg.transpose' and _integer_transpose(op) is not None:
            rec.update(placement='host',status='eligible_but_host',
                standalone={'status':'eligible','device_form':'STRIDED_MOVEMENT'},
                fused={'status':'not_required'},
                reason='exact typed strided DMA permutation is available; transpose transfer preference is disabled')
        elif str(ty.element_type).startswith(('f','bf')):
            rec.update(placement='host',status='hardware_unsupported',reason='typed floating tensor boundary is preserved; selected device exits are i32 and i8')
        elif op.name=='linalg.generic' and _saturating_cast(op) is not None:
            rec.update(placement='host',status='compiler_adapter_gap',reason='recognized saturating body failed standalone boundary or selected output-path admission')
        else:
            rec.update(placement='host',status='unresolved_proof',reason='no supported scalar-body, layout and numeric-order proof for a standalone device form')
        records.append(rec)
    sizes={'i8':1,'i32':4}
    boundary_bytes=sum(sum(prod(_spec(v)['shape'])*sizes.get(_spec(v)['dtype'],0) for v in (*p.inputs,p.output)) for p in plans)
    return {'regions':records,'device_regions':len(plans),
            'covered_regions':sum(r['placement']=='device' for r in records),
            'eligible_but_host_regions':sum(r.get('status')=='eligible_but_host' for r in records),
            'hardware_unsupported_regions':sum(r.get('status')=='hardware_unsupported' for r in records),
            'unknown_regions':sum(r.get('status') in ('unresolved_proof','compiler_adapter_gap') for r in records),
            'static_device_boundary_bytes':boundary_bytes,'measured_host_time':None,'measured_transfer_bytes':None}


def source_to_buffer(source_text,*,transpose_transfers=False,prefer_quantized_readout=False):
    """Compile an extracted region into its typed device-callback buffer.

    Strided transpose extraction is opt-in. Internal opcodes occur only in
    nested whole-program device diagnostics, outside the public command stream.

    ``constants`` holds proved immutable tensor contents for compiler-owned
    storage initialization. Runtime initializer matrices use ``accumulator_init``.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    module=parse_mlir_text(source_text)
    functions=[op for op in module.walk() if op.name=='func.func' and op.body.blocks]
    if len(functions)!=1 or len(functions[0].body.blocks)!=1:
        raise SourceGroupError('one extracted single-block function is required')
    block=functions[0].body.block
    plans=_plans(module,transpose_transfers=transpose_transfers,prefer_quantized_readout=prefer_quantized_readout)
    if len(plans)!=1 or len(block.last_op.operands)!=1 or plans[0].output is not block.last_op.operands[0]:
        raise SourceGroupError('extracted function is not exactly one supported source group')
    p=plans[0]
    names={v:f'arg{i}' for i,v in enumerate(block.args)}
    tensors={names[v]:{**_spec(v),'role':'input'} for v in block.args}
    for name,spec in p.constants.items():tensors[name]={k:v for k,v in spec.items() if k!='values'}|{'role':'intermediate'}
    dst='Y0';tensors[dst]={**_spec(p.output),'role':'output'}
    if isinstance(p,_Standalone):
        operands={}
        literal_names={}
        for key,value in p.operands.items():
            if isinstance(value,str) and value in p.constants:
                name=value
            elif value in names:
                name=names[value]
            elif value in literal_names:
                name=literal_names[value]
            else:
                literal=_integer_tensor_literal(value)
                if literal is None:
                    raise SourceGroupError(f'standalone {p.opcode} operand {key} is neither a boundary input nor a declared integer tensor literal')
                name=f'_source_literal_{len(literal_names)}'
                while name in tensors:name+='_' 
                literal_names[value]=name
                p.constants[name]=literal
                tensors[name]={k:v for k,v in literal.items() if k!='values'}|{'role':'intermediate'}
            operands[key]=name
        operands['dst']=dst
        views={}
        for value,shape in p.storage_shapes.items():
            name=dst if value is p.output else names.get(value)
            if name is None:raise SourceGroupError('device storage view is not a boundary tensor')
            views[name]={'source_shape':tensors[name]['shape'],'device_shape':list(shape)}
            tensors[name]['shape']=list(shape)
        internal={}
        if p.opcode in ('ACCUMULATOR_READOUT','STRIDED_MOVEMENT'):
            # An internal typed callback, never an extension to the public CB
            # command schema. The outer whole-program CB has commands=[] and
            # carries this buffer only as nested device-region diagnostics.
            internal={'params':{'source_internal_opcodes':[p.opcode]},
                      'kernel_abi':{'kind':'whole_program',
                          'args':[{'tensor':names[v],'access':'read'} for v in block.args]+[{'tensor':dst,'access':'write'}],
                          'outputs':[dst]}}
        return {**internal,'abi_version':'0.1','target':'gemmini','tensors':tensors,
                'commands':[{'opcode':p.opcode,'operands':operands,'attributes':dict(p.attributes)}],
                'constants':p.constants,'source_group_proof':p.diagnostic,'source_storage_views':views}
    lhs,rhs,init=_root_inputs(p.root)
    if lhs not in names or rhs not in names:
        raise SourceGroupError('device contraction operands must be explicit boundary tensor arguments')
    attrs=dict(p.attributes)
    if '_source_initializer' in attrs:
        value=attrs.pop('_source_initializer')
        if value not in names:raise SourceGroupError('runtime initializer must be a boundary tensor')
        attrs['accumulator_init']=names[value]
    commands=[{'opcode':'RES_PACK','operands':{'src':names[rhs],'dst':'w_res'},'attributes':{'layout':'packed_rhs'}},
              {'opcode':'MATMUL_RESIDENT','operands':{'lhs':names[lhs],'rhs':'w_res','dst':'acc'}},
              {'opcode':'COMMIT','operands':{'src':'acc','dst':dst},'attributes':attrs},
              {'opcode':'EVICT','operands':{'handle':'w_res'}}]
    views={}
    if p.output in p.storage_shapes:
        shape=list(p.storage_shapes[p.output])
        views[dst]={'source_shape':tensors[dst]['shape'],'device_shape':shape}
        tensors[dst]['shape']=shape
    return {'abi_version':'0.1','target':'gemmini','tensors':tensors,'commands':commands,
            'constants':p.constants,'source_group_proof':p.diagnostic,
            **({'source_storage_views':views} if views else {})}
