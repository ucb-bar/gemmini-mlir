"""Exact dense integer identity movement with original logical ABI custody."""
from math import prod


def spec(value):
    from xdsl.dialects.builtin import NoneAttr,TensorType
    t=value.type
    if (not isinstance(t,TensorType) or not isinstance(t.encoding,NoneAttr)
            or str(t.element_type) not in ('i8','i32') or not t.get_shape()
            or any(d<=0 for d in t.get_shape())):return None
    a={'shape':list(t.get_shape()),'dtype':str(t.element_type)}
    if prod(a['shape'])*(1 if a['dtype']=='i8' else 4)>=1<<63:return None
    return a


def match(op,*,transpose_transfers=False):
    if len(op.results)!=1 or not op.operands:return None
    src=op.operands[0];a,b=spec(src),spec(op.results[0])
    if a is None or b is None or a['dtype']!=b['dtype'] or prod(a['shape'])!=prod(b['shape']):return None
    proof=dict(form='exact_integer_movement',source_operation=op.name,
        source_shape=a['shape'],result_shape=b['shape'],dtype=a['dtype'],
        numeric_operations=[],original_logical_shapes_preserved=True)
    if op.name in ('tensor.collapse_shape','tensor.expand_shape'):
        from .quantized_boundary import _contiguous_view,QuantizedBoundaryError
        try:
            if _contiguous_view(op,op.results[0],dtypes=('i8','i32')) is not src:return None
        except QuantizedBoundaryError:return None
        proof.update(source_form='contiguous_reassociation',address_projection='identical row-major linear index')
        return src,'MOVEMENT',{'semantic':'mvin_mvout','output_dtype':a['dtype']},proof
    if op.name not in ('linalg.copy','linalg.generic','linalg.transpose') or len(op.operands)!=2 or spec(op.operands[1])!=b:return None
    if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return None
    body=op.regions[0].block;ops=list(body.ops)
    if (len(body.args)!=2 or any(str(v.type)!=a['dtype'] for v in body.args)
            or len(ops)!=1 or ops[0].name!='linalg.yield' or list(ops[0].operands)!=[body.args[0]]):return None
    rank=len(a['shape']);perm=list(range(rank))
    if op.name=='linalg.copy':
        if a!=b:return None
    elif op.name=='linalg.transpose':
        attr=op.properties.get('permutation')
        if attr is None or not hasattr(attr,'get_values'):return None
        perm=list(attr.get_values())
    else:
        maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
        if maps is None or len(maps)!=2 or iters is None or [v.data.value for v in iters]!=['parallel']*rank:return None
        from xdsl.ir.affine import AffineDimExpr
        axes=[]
        for attr in maps:
            m=attr.data
            if (m.num_dims!=rank or m.num_symbols or len(m.results)!=rank
                    or any(not isinstance(e,AffineDimExpr) for e in m.results)):return None
            indices=[e.position for e in m.results]
            if sorted(indices)!=list(range(rank)):return None
            axes.append(indices)
        perm=[axes[0].index(axis) for axis in axes[1]]
        proof.update(input_loop_axes=axes[0],output_loop_axes=axes[1],map_composition='inverse(input_map) applied to output_map')
    if sorted(perm)!=list(range(rank)) or b['shape']!=[a['shape'][i] for i in perm]:return None
    proof.update(source_form='literal_scalar_identity',permutation=perm,
        destination_initializer='external original value; ignored by scalar identity; every output overwritten')
    if [i for i in perm if a['shape'][i]!=1]==[i for i,n in enumerate(a['shape']) if n!=1]:
        proof['address_projection']='identical row-major linear index'
        return src,'MOVEMENT',{'semantic':'mvin_mvout','output_dtype':a['dtype']},proof
    if not transpose_transfers or not 2<=rank<=4:return None
    from .transpose import plan_transpose,TransposeError
    try:plan=plan_transpose(a['shape'],perm,a['dtype'])
    except TransposeError:return None
    proof['address_projection']=plan.proof
    return src,'STRIDED_MOVEMENT',{'permutation':perm,'output_dtype':a['dtype']},proof


def bind_buffer(buffer):
    """Equal-size same-dtype aliases flatten only addresses, never logical ABI."""
    tensors=buffer['tensors'];records=[]
    for index,command in enumerate(buffer.get('commands',())):
        if command['opcode']!='MOVEMENT':continue
        src,dst=(command['operands'][k]for k in ('src','dst'))
        a,b=tensors.get(src,{}),tensors.get(dst,{})
        if a.get('dtype')not in ('i8','i32') or a.get('shape')==b.get('shape'):continue
        if (a.get('dtype')!=b.get('dtype') or prod(a['shape'])!=prod(b['shape'])
                or command.get('attributes')!={'semantic':'mvin_mvout','output_dtype':a['dtype']}):
            raise ValueError('integer view movement lost exact same-dtype dense extent proof')
        aliases=buffer.setdefault('params',{}).setdefault('tensor_aliases',{})
        if src==dst or src in aliases or dst in aliases:raise ValueError('integer movement requires distinct physical source/output buffers')
        names=[f'_integer_bits_{index}_{label}'for label in ('src','dst')]
        if any(n in aliases or n in tensors for n in names):raise ValueError('integer movement alias collision')
        count=prod(a['shape'])
        for name,source in zip(names,(src,dst),strict=True):aliases[name]={'source':source,'shape':[count],'dtype':a['dtype']}
        command['operands']={'src':names[0],'dst':names[1]}
        records.append(dict(command_index=index,source=src,destination=dst,source_shape=a['shape'],result_shape=b['shape'],dtype=a['dtype'],element_count=count))
    if records:buffer['params']['integer_movement_alias_proof']=records
    return buffer
