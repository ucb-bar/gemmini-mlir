"""Exact dense f32 permutations using checked raw full-width accumulator DMA."""
from math import prod
from .float_movement import _spec, FloatMovementError


def match(op):
    if op.name=='linalg.generic':
        from .float_generic_permutation import match as match_generic
        return match_generic(op)
    if op.name!='linalg.transpose' or len(op.operands)!=2 or len(op.results)!=1:return None
    src=op.operands[0];a,b=_spec(src),_spec(op.results[0])
    if a is None or b is None or _spec(op.operands[1])!=b or not 2<=len(a['shape'])<=4:return None
    attr=op.properties.get('permutation')
    if attr is None or not hasattr(attr,'get_values'):return None
    perm=list(attr.get_values())
    if sorted(perm)!=list(range(len(a['shape']))) or b['shape']!=[a['shape'][i] for i in perm]:return None
    if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return None
    body=op.regions[0].block;ops=list(body.ops)
    if (len(body.args)!=2 or any(str(v.type)!='f32' for v in body.args) or len(ops)!=1
            or ops[0].name!='linalg.yield' or list(ops[0].operands)!=[body.args[0]]):return None
    from .transpose import plan_transpose,TransposeError
    try:plan=plan_transpose(a['shape'],perm,'i32')
    except TransposeError:return None
    return src,{'permutation':perm,'output_dtype':'i32'},dict(
        form='exact_f32_bit_transpose',source_operation=op.name,logical_dtype='f32',
        transport_dtype='i32',source_shape=a['shape'],result_shape=b['shape'],
        representation='all IEEE32 bits unchanged including NaN payloads and signed zero',
        numeric_operations=[],address_projection=plan.proof)


def bind_buffer(buffer):
    tensors=buffer['tensors']
    if not any(c['opcode']=='STRIDED_MOVEMENT' and any(
            tensors.get(c['operands'].get(k),{}).get('dtype')=='f32' for k in ('src','dst'))
            for c in buffer.get('commands',())):return buffer
    params=buffer.setdefault('params',{})
    for index,command in enumerate(buffer.get('commands',())):
        if command['opcode']!='STRIDED_MOVEMENT':continue
        src,dst=(command['operands'][k] for k in ('src','dst'))
        a,b=tensors.get(src,{}),tensors.get(dst,{})
        if a.get('dtype')!='f32' and b.get('dtype')!='f32':continue
        aliases=params.setdefault('tensor_aliases',{})
        attrs=command.get('attributes',{});perm=attrs.get('permutation',[])
        if (a.get('dtype')!='f32' or b.get('dtype')!='f32' or src==dst or src in aliases or dst in aliases
                or set(attrs)!={'permutation','output_dtype'} or attrs['output_dtype']!='i32'):
            raise FloatMovementError('f32 transpose requires distinct original dense f32 buffers')
        from .transpose import plan_transpose
        plan=plan_transpose(a['shape'],perm,'i32')
        if b['shape']!=[a['shape'][i] for i in perm]:raise FloatMovementError('f32 transpose output shape changed')
        names=[f'_f32_transpose_{index}_{label}' for label in ('src','dst')]
        if any(n in aliases or n in tensors for n in names):raise FloatMovementError('f32 transpose alias collision')
        for name,source,spec in zip(names,(src,dst),(a,b),strict=True):
            aliases[name]={'source':source,'shape':list(spec['shape']),'dtype':'i32'}
        command['operands']={'src':names[0],'dst':names[1]}
        params.setdefault('float_transpose_alias_proof',[]).append(dict(
            command_index=index,source=src,destination=dst,source_alias=names[0],destination_alias=names[1],
            source_shape=list(a['shape']),result_shape=list(b['shape']),permutation=list(perm),
            storage_bytes=prod(a['shape'])*4,address_projection=plan.proof))
    return buffer


def validate_aliases(buffer,src,dst,attributes):
    """Admit only command-local aliases of distinct same-size f32 ABI storage."""
    aliases=buffer.get('params',{}).get('tensor_aliases',{});tensors=buffer['tensors']
    for record in buffer.get('params',{}).get('float_transpose_alias_proof',()):
        if (record.get('source_alias'),record.get('destination_alias'))!=(src,dst):continue
        source,destination=record['source'],record['destination']
        if source==destination or source in aliases or destination in aliases:return False
        a,b=tensors.get(source,{}),tensors.get(destination,{})
        if a.get('dtype')!='f32' or b.get('dtype')!='f32':return False
        if (aliases.get(src)!={'source':source,'shape':a['shape'],'dtype':'i32'}
                or aliases.get(dst)!={'source':destination,'shape':b['shape'],'dtype':'i32'}):return False
        if (record['source_shape']!=a['shape'] or record['result_shape']!=b['shape']
                or record['permutation']!=attributes.get('permutation')
                or record['storage_bytes']!=prod(a['shape'])*4 or prod(a['shape'])!=prod(b['shape'])):return False
        index=record['command_index'];commands=buffer.get('commands',())
        return (0<=index<len(commands) and commands[index]['opcode']=='STRIDED_MOVEMENT'
            and commands[index]['operands']=={'src':src,'dst':dst} and commands[index].get('attributes')==attributes)
    return False
