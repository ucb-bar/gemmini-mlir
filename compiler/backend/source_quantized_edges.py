"""Original-source Q1/residual-Q2 composition through a private i8 tensor.

The library's command-source checker validates the actual emitted ordinary
statements. This adapter retains original SSA custody separately; it does not
manufacture an ordinary DeviceRewrite or its borrowed declarations.
"""
from pathlib import Path


def enabled():
    from merlin.compile.tuning_knobs import knob
    return knob('MIXED_QUANTIZED_RESIDUAL_EDGES',default='off',values=('off','on'),
                root=Path(__file__).resolve().parent.parent)=='on'


def scalar_residual(op):
    """Prove a literal-scaled f32 sum with exactly one source rounding stage."""
    from .source_groups import _spec,_identity_maps,_scalar_constant
    if op.name!='linalg.generic' or len(op.operands)!=3 or len(op.results)!=1:return None
    lhs,rhs=op.operands[:2];spec=_spec(lhs)
    if (spec['dtype']!='i8' or _spec(rhs)!=spec or _spec(op.results[0])!=spec
            or _spec(op.operands[-1])!=spec or not _identity_maps(op,3,len(spec['shape']))
            or len(op.regions)!=1 or len(op.regions[0].blocks)!=1):return None
    body=op.regions[0].block
    if len(body.args)!=3 or any(str(a.type)!='i8' for a in body.args):return None
    ops=[o for o in body.ops if o.name!='arith.constant'];index=0
    for scalar in ops:
        flags=scalar.properties.get('fastmath')
        if flags is not None and str(flags)!='#arith.fastmath<none>':return None
    scales=[];values=[]
    for argument in body.args[:2]:
        if index+1>=len(ops):return None
        cast,mul=ops[index:index+2];index+=2
        if (cast.name!='arith.sitofp' or list(cast.operands)!=[argument]
                or str(cast.results[0].type)!='f32' or mul.name!='arith.mulf'
                or cast.results[0] not in mul.operands or str(mul.results[0].type)!='f32'):return None
        other=mul.operands[1] if mul.operands[0] is cast.results[0] else mul.operands[0]
        scale=_scalar_constant(other)
        if str(other.type)!='f32' or not isinstance(scale,(int,float)):return None
        scales.append(float(scale));values.append(mul.results[0])
    if index>=len(ops):return None
    add=ops[index];index+=1
    if add.name!='arith.addf' or set(add.operands)!=set(values) or str(add.results[0].type)!='f32':return None
    current=add.results[0];relu=False
    if index<len(ops) and ops[index].name in ('arith.maximumf','arith.maxnumf'):
        activation=ops[index]
        if current not in activation.operands:return None
        other=activation.operands[1] if activation.operands[0] is current else activation.operands[0]
        if _scalar_constant(other)!=0:return None
        current=activation.results[0];relu=True;index+=1
    tail=ops[index:]
    if len(tail)!=5:return None
    rounding,low,high,cast,yielding=tail
    if (rounding.name!='math.roundeven' or list(rounding.operands)!=[current]
            or low.name not in ('arith.maximumf','arith.maxnumf')
            or high.name not in ('arith.minimumf','arith.minnumf')
            or rounding.results[0] not in low.operands or low.results[0] not in high.operands
            or cast.name!='arith.fptosi' or list(cast.operands)!=list(high.results)
            or str(cast.results[0].type)!='i8' or yielding.name!='linalg.yield'
            or list(yielding.operands)!=list(cast.results)):return None
    lower=low.operands[1] if low.operands[0] is rounding.results[0] else low.operands[0]
    upper=high.operands[1] if high.operands[0] is low.results[0] else high.operands[0]
    if _scalar_constant(lower)!=-128 or _scalar_constant(upper)!=127:return None
    bound=0
    for key in ('merlin.bound_lsb','bound_lsb'):
        attr=op.attributes.get(key,op.properties.get(key))
        value=getattr(getattr(attr,'value',None),'data',None)
        if value is not None:
            if type(value)is not int or value<0:return None
            bound=value
    from .elementwise import residual_numeric_screen
    try:proof=residual_numeric_screen(*scales,bound,relu)
    except (ValueError,OverflowError):return None
    return {'lhs':lhs,'rhs':rhs},dict(lhs_scale=scales[0],rhs_scale=scales[1],
        bound_lsb=bound,epilogue=['relu'] if relu else [],output_dtype='i8'),dict(
        form='standalone_scalar_f32_operand_sum',source_order=['sitofp_f32','multiply_f32',
            'add_f32',*(['relu_f32'] if relu else []),'round_half_even_once','clamp_signed_i8','fptosi_i8'],
        source_scales=scales,source_bound_lsb=bound,**proof)


def discover(plans):
    """Group sole-use original Q1 edges, leaving all other source plans intact."""
    from .source_groups import _Plan,_Standalone,_spec
    from .host_runtime import DeviceRegion
    import json
    by_output={p.output:p for p in plans};pairs={};consumed=set()
    for consumer in plans:
        if not isinstance(consumer,_Standalone) or consumer.opcode!='RESIDUAL_ADD':continue
        if consumer.diagnostic.get('form')!='standalone_operand_sum':continue
        if consumer.attributes.get('output_dtype')!='i8':continue
        candidates=[]
        for value in consumer.operands.values():
            producer=by_output.get(value)
            if not isinstance(producer,_Plan) or any(producer is p for p in candidates):continue
            if (producer.attributes.get('output_dtype')!='i8'
                    or producer.attributes.get('epilogue') not in
                    (['acc_scale'],['acc_scale','relu'],['bias_add','acc_scale'],['bias_add','acc_scale','relu'])):continue
            spec=_spec(value)
            if len(spec['shape'])!=2 or spec!=_spec(consumer.output):continue
            uses=list(value.uses)
            if len(uses)!=1 or uses[0].operation not in consumer.operations:continue
            if producer.root.parent is not consumer.root.parent:continue
            candidates.append(producer)
        if len(candidates)!=1:continue
        producer=candidates[0]
        if id(producer) in consumed or id(consumer) in consumed:continue
        block=producer.root.parent;positions={op:i for i,op in enumerate(block.ops)}
        owned=set(producer.operations)|set(consumer.operations)
        low=positions[producer.root];high=positions[consumer.output.owner]
        if low>=high or any(op not in owned and op.name not in ('arith.constant','tensor.empty')
                for i,op in enumerate(block.ops) if low<=i<=high):continue
        if any(use.operation not in owned for op in owned for result in op.results
               if result is not consumer.output for use in result.uses):continue
        inputs=list(dict.fromkeys(v for p in (producer,consumer) for v in p.inputs if v.owner not in owned))
        inputs=[v for v in block.args if v in inputs]+[v for v in inputs if v not in block.args]
        region=DeviceRegion(tuple(op for op in block.ops if op in owned),tuple(inputs),(consumer.output,),
            reason=json.dumps(dict(form='original_quantized_producer_residual_edge',
                quantization='original producer Q1 then original residual sum/Q2/activation',
                intermediate_uses=1,intermediate_observers=[],storage='private typed i8 matrix',
                source_numeric_proof=consumer.diagnostic)))
        pairs[id(producer)]=region;consumed.update((id(producer),id(consumer)))
    return tuple(pairs[id(p)] if id(p) in pairs else
                 DeviceRegion(p.operations,p.inputs,(p.output,),reason=json.dumps(p.diagnostic,default=str))
                 for p in plans if id(p) not in consumed or id(p) in pairs)


def validate_emitted_commands(buffer):
    """Recheck the actual target request, keeping both numerical entries intact."""
    commands=buffer.get('commands',())
    if [c['opcode'] for c in commands]!=['RES_PACK','MATMUL_RESIDENT','COMMIT','EVICT','RESIDUAL_ADD']:
        return
    if commands[2].get('attributes',{}).get('output_dtype')!='i8':return
    if not commands[-1].get('attributes',{}).get('source_scales'):return
    edge=commands[2]['operands']['dst']
    if (buffer['tensors'][edge].get('role')!='intermediate'
            or edge in buffer.get('kernel_abi',{}).get('outputs',())):return
    from merlin.llvmlower.quantized_sum_source import plan_quantized_sum_commands
    from copy import deepcopy
    source=plan_quantized_sum_commands({key:deepcopy(buffer[key])
        for key in ('abi_version','target','tensors','commands')})
    source.revalidate()
    buffer['source_group_proof']['original_quantized_command_source']=source.receipt
    buffer['source_group_proof']['original_quantized_command_identity']=source.identity
