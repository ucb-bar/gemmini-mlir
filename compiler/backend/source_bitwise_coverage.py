"""Checked f32 storage transport, with original typed source ABI retained.

No floating computation is issued. The selected target's bias_preload and
full_width_readout laws copy 32-bit lanes without arithmetic. Runtime words
therefore retain every bit, including signaling NaN payloads.
"""
import math
import struct


RAW32_LAW = {
    'source': 'target_contract.yaml',
    'source_sha256': '4944bd51cb43ce2d4c23c3f7ecee983c0027dd882679cc7c935007b092674039',
    'line_ranges': [[190,210],[214,223]],
    'byte_order': 'little (target_contract.yaml harness_abi.byte_order)',
    'load': 'output_epilogue.bias_preload: derived; 16 32-bit lanes copied with no arithmetic; rounding none',
    'store': 'output_epilogue.full_width_readout: derived; accumulator copied unmodified; no arithmetic fan-in',
    'commands': 'config_ld(shrink=False); nonaccumulating MVIN; drain; MVOUT(full=True); drain',
    'abi': 'original f32 ranked source callback; opaque pointers; unchanged 4-byte element stride',
}


def constant_f32_pad(op, spec):
    """Prove a static f32 constant pad as a raw storage partition.

    NaN *fill literals* refuse: the frontend FloatAttr stores a Python float,
    which cannot certify an original signaling NaN payload. Runtime source
    values undergo no host floating load, conversion or comparison.
    """
    from xdsl.dialects.builtin import NoneAttr, FloatAttr, TensorType
    from .source_integer_coverage import pad_stages
    if op.name != 'tensor.insert_slice' or len(op.operands) != 2 or len(op.results) != 1:
        return None
    src, base = op.operands
    if any(not isinstance(v.type, TensorType) or any(n <= 0 for n in v.type.get_shape())
           for v in (src, base, op.results[0])):
        return None
    a, b, out = map(spec, (src, base, op.results[0]))
    rank = len(a['shape'])
    if (not 1 <= rank <= 4 or len(b['shape']) != rank or b != out
            or a['dtype'] != 'f32' or b['dtype'] != 'f32'
            or any(not isinstance(v.type.encoding, NoneAttr) for v in (src, base, op.results[0]))):
        return None
    splat = base.owner
    if getattr(splat, 'name', '') != 'tensor.splat' or len(splat.operands) != 1:
        return None
    scalar = splat.operands[0]
    owner = scalar.owner
    if str(scalar.type) != 'f32' or getattr(owner, 'name', '') != 'arith.constant':
        return None
    attr = owner.properties.get('value', owner.attributes.get('value'))
    if not isinstance(attr, FloatAttr) or str(attr.type) != 'f32' or math.isnan(attr.value.data):
        return None
    raw = attr.type.pack((attr.value.data,))
    word = struct.unpack('<i', raw)[0]
    geometry = []
    for key in ('static_offsets', 'static_sizes', 'static_strides'):
        attr = op.properties.get(key)
        if attr is None or not hasattr(attr, 'get_values'):
            return None
        geometry.append(list(attr.get_values()))
    offsets, sizes, strides = geometry
    if (sizes != a['shape'] or len(offsets) != rank or strides != [1]*rank
            or any(o < 0 or o+n > d for o,n,d in zip(offsets,sizes,b['shape']))):
        return None
    stages = pad_stages(sizes,b['shape'],offsets,'i32')
    if stages is None:
        return None
    attrs = dict(stages=stages,fill=word,output_dtype='i32')
    proof = dict(form='source_constant_f32_bitwise_pad',source_dtype='f32',storage_dtype='i32',
        element_bytes=4,conversion='none',fill_bits=f'{word & 0xffffffff:08x}',
        offsets=offsets,sizes=sizes,strides=strides,stages=stages,transport_law=RAW32_LAW,
        semantics='disjoint constant border word slabs around bit-identical source words')
    return src,attrs,proof


def f32_movement(op, spec):
    """Bind source-declared pure f32 movement to a checked raw32 DMA form.

    Source tensors keep their original types and axes. Unit-axis transposes
    are metadata permutations and deliberately retain the source view route.
    """
    from xdsl.dialects.builtin import NoneAttr, TensorType
    from .source_slice import plan_slice, SliceError
    from .source_concat import plan_concat, ConcatError
    from .transpose import plan_transpose, TransposeError
    if op.name not in ('tensor.extract_slice','tensor.concat','linalg.transpose') or len(op.results)!=1:
        return None
    values=(*op.operands,*op.results)
    if any(not isinstance(v.type,TensorType) or str(v.type.element_type)!='f32'
           or not isinstance(v.type.encoding,NoneAttr)
           or not v.type.get_shape() or any(n<=0 for n in v.type.get_shape()) for v in values):
        return None
    out=spec(op.results[0]);proof=dict(form='source_f32_bitwise_movement',
        source_dtype='f32',storage_dtype='i32',element_bytes=4,conversion='none',
        transport_law=RAW32_LAW)
    try:
        if op.name=='tensor.extract_slice':
            if len(op.operands)!=1:return None
            geometry=[]
            for key in ('static_offsets','static_sizes','static_strides'):
                attr=op.properties.get(key)
                if attr is None or not hasattr(attr,'get_values'):return None
                geometry.append(list(attr.get_values()))
            offsets,sizes,strides=geometry
            if sizes!=out['shape']:return None
            plan=plan_slice(spec(op.operands[0])['shape'],offsets,sizes,strides,'i32')
            opcode='STRIDED_SLICE';operands={'src':op.operands[0]}
            attrs=dict(offsets=offsets,sizes=sizes,strides=strides,output_dtype='i32')
        elif op.name=='tensor.concat':
            if not op.operands:return None
            attr=op.properties.get('dim',op.attributes.get('dim'))
            axis=getattr(getattr(attr,'value',None),'data',None)
            plan=plan_concat([spec(v)['shape'] for v in op.operands],axis,'i32')
            if list(plan.output)!=out['shape']:return None
            opcode='INTEGER_CONCAT';operands={f'src{i}':v for i,v in enumerate(op.operands)}
            attrs=dict(axis=axis,inputs=len(op.operands),output_dtype='i32')
        else:
            if len(op.operands)!=2 or len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return None
            attr=op.properties.get('permutation')
            if attr is None or not hasattr(attr,'get_values'):return None
            perm=list(attr.get_values());shape=spec(op.operands[0])['shape']
            if sorted(perm)!=list(range(len(shape))) or out['shape']!=[shape[i] for i in perm]:return None
            body=op.regions[0].block;ops=list(body.ops)
            if len(body.args)!=2 or len(ops)!=1 or ops[0].name!='linalg.yield' or list(ops[0].operands)!=[body.args[0]]:return None
            if [i for i in perm if shape[i]!=1]==[i for i,n in enumerate(shape) if n!=1]:return None
            plan=plan_transpose(shape,perm,'i32')
            opcode='STRIDED_MOVEMENT';operands={'src':op.operands[0]}
            attrs=dict(permutation=perm,output_dtype='i32')
    except (SliceError,ConcatError,TransposeError):
        return None
    proof.update(source_operation=op.name,address_projection=plan.proof,
                 source_shapes=[spec(v)['shape'] for v in operands.values()],output_shape=out['shape'])
    return opcode,operands,attrs,proof


def validate_raw32_buffer(buffer):
    """Refuse arithmetic or nonraw readout at the source/storage boundary."""
    allowed={'INTEGER_CONCAT':{'axis','inputs','output_dtype'},
             'STRIDED_SLICE':{'offsets','sizes','strides','output_dtype'},
             'STRIDED_MOVEMENT':{'permutation','output_dtype'}}
    views=buffer.get('source_storage_views',{})
    if not views or any(v.get('source_dtype')!='f32' or v.get('device_dtype')!='i32'
                        or v.get('conversion')!='none' or not _valid_raw32_shape(v)
                        or v.get('transport_law')!=RAW32_LAW for v in views.values()):
        raise ValueError('raw32 storage view must preserve source f32 shape and bits without conversion')
    _validate_broadcast_commands(buffer)
    _validate_multi_broadcast_commands(buffer)
    for command in buffer['commands']:
        attrs=command.get('attributes',{});opcode=command['opcode']
        if (opcode not in allowed or set(attrs)!=allowed[opcode] or attrs['output_dtype']!='i32'
                or any(buffer['tensors'][name]['dtype']!='i32' for name in command['operands'].values())):
            raise ValueError('source f32 bit view admits checked raw32 movement only; arithmetic and narrow readout are forbidden')
    return True


def _broadcast_projection(source_shape, output_shape, axis):
    """Construct and check a bijective row-major view, never a tensor cast.

    The inverse of each prefix/suffix mixed-radix flattening is repeated
    divmod by its dimensions (last axis first). Each step is a unique Euclidean
    quotient/remainder; the checked stride recurrence reconstructs the original
    linear offset. Thus this also proves element and four-byte lane bijections.
    """
    from math import prod
    if (not 1 <= len(source_shape) <= 8 or len(output_shape) != len(source_shape)
            or type(axis) is not int or not 0 <= axis < len(source_shape)
            or any(type(n) is not int or n <= 0 for n in (*source_shape,*output_shape))
            or source_shape[axis] != 1 or not 2 <= output_shape[axis] <= 64
            or any(a != b for i,(a,b) in enumerate(zip(source_shape,output_shape)) if i != axis)):
        raise ValueError('raw32 broadcast requires one expanded unit axis, static rank1-8 and 2..64 copies')
    prefix=prod(source_shape[:axis]);suffix=prod(source_shape[axis+1:]);copies=output_shape[axis]
    if prod(output_shape)*4 >= 1 << 63:
        raise ValueError('raw32 broadcast byte footprint exceeds i64')
    def strides(shape):return [prod(shape[i+1:]) for i in range(len(shape))]
    source_strides=strides(source_shape);output_strides=strides(output_shape)
    # Compare every coefficient of the source/output coordinate polynomials.
    # Physical indices are (prefix_index, axis_index, suffix_index).
    pstrides=strides(source_shape[:axis]);sstrides=strides(source_shape[axis+1:])
    assert source_strides == [n*suffix for n in pstrides]+[suffix]+sstrides
    assert output_strides == [n*copies*suffix for n in pstrides]+[suffix]+sstrides
    for shape in (source_shape,output_shape,source_shape[:axis],source_shape[axis+1:]):
        ss=strides(shape)
        assert sum((n-1)*s for n,s in zip(shape,ss)) == prod(shape)-1
        assert all(ss[i] == shape[i+1]*ss[i+1] for i in range(len(shape)-1))
    return dict(kind='row_major_single_unit_axis_repeat',axis=axis,
        source_shape=list(source_shape),output_shape=list(output_shape),source_rank=len(source_shape),
        output_rank=len(output_shape),source_physical_shape=[prefix,1,suffix],
        output_physical_shape=[prefix,copies,suffix],prefix=prefix,suffix=suffix,copies=copies,
        source_strides=source_strides,output_strides=output_strides,
        prefix_strides=pstrides,suffix_strides=sstrides,element_bytes=4,
        source_offset='prefix_index*suffix + suffix_index (source axis is zero)',
        output_offset='(prefix_index*copies + axis_index)*suffix + suffix_index',
        inverse='mixed-radix divmod, last original axis first; quotient/remainder unique for positive radix',
        proof={'coordinate_coefficients_equal':True,'positive_mixed_radix_bijection':True,
               'linear_range_exact':True,'byte_offset_equal':True,'conversion':'none'})


def f32_broadcast(op,spec):
    """Recognize only an exact, all-parallel, single-unit-axis source repeat."""
    from xdsl.dialects.builtin import NoneAttr,TensorType
    from xdsl.ir.affine import AffineMap,AffineExpr
    from .source_concat import plan_concat,ConcatError
    if (op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1
            or len(op.regions)!=1 or len(op.regions[0].blocks)!=1):return None
    src,init=op.operands;out=op.results[0]
    if any(not isinstance(v.type,TensorType) or str(v.type.element_type)!='f32'
           or not isinstance(v.type.encoding,NoneAttr) or not v.type.get_shape()
           or any(n<=0 for n in v.type.get_shape()) for v in (src,init,out)):return None
    a,b=spec(src),spec(out);rank=len(a['shape'])
    if spec(init)!=b:return None
    body=op.regions[0].block;ops=list(body.ops)
    if (len(body.args)!=2 or any(str(v.type)!='f32' for v in body.args)
            or len(ops)!=1 or ops[0].name!='linalg.yield'
            or list(ops[0].operands)!=[body.args[0]]):return None
    maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
    if (maps is None or len(maps)!=2 or iters is None
            or [v.data.value for v in iters]!=['parallel']*rank):return None
    maps=tuple(maps)
    axes=[i for i,(x,y) in enumerate(zip(a['shape'],b['shape'])) if x!=y]
    if len(axes)!=1:return None
    axis=axes[0]
    expected=AffineMap(rank,0,tuple(AffineExpr.constant(0) if i==axis else AffineExpr.dimension(i) for i in range(rank)))
    if maps[0].data!=expected or maps[1].data!=AffineMap.identity(rank):return None
    try:
        projection=_broadcast_projection(a['shape'],b['shape'],axis)
        copies=projection['copies'];physical=projection['source_physical_shape']
        plan=plan_concat([physical]*copies,1,'i32')
        if list(plan.output)!=projection['output_physical_shape']:return None
    except (ValueError,ConcatError):return None
    proof=dict(form='source_f32_bitwise_broadcast',source_operation=op.name,
        source_dtype='f32',storage_dtype='i32',element_bytes=4,conversion='none',transport_law=RAW32_LAW,
        storage_projection=projection,indexing_maps=[str(v.data) for v in maps],
        iterators=['parallel']*rank,address_projection=plan.proof,
        semantics='each output word reads the same source word at the declared zero-axis affine projection; init is unobserved')
    if getattr(init.owner,'name','')!='tensor.empty' or init.owner.operands:
        from copy import deepcopy
        proof=deepcopy(proof)
        proof['unobserved_init']=_unobserved_init_proof(b['shape'])
    return ({f'src{i}':src for i in range(copies)},dict(axis=1,inputs=copies,output_dtype='i32'),proof,
            {src:physical,out:projection['output_physical_shape']})


def _unobserved_init_proof(shape):
    return dict(shape=list(shape),source_dtype='f32',storage_dtype='i32',
        element_bytes=4,bytes=math.prod(shape)*4,conversion='none',access='unobserved')


def _valid_raw32_shape(view):
    projection=view.get('storage_projection')
    if projection is None:return view['source_shape']==view['device_shape']
    try:
        expected=_broadcast_projection(projection['source_shape'],projection['output_shape'],projection['axis'])
        role=view['projection_role']
        return (projection==expected and role in ('source','output')
            and view['source_shape']==expected[role+'_shape']
            and view['device_shape']==expected[role+'_physical_shape']
            and view.get('source_rank')==len(view['source_shape'])
            and view.get('device_rank')==3)
    except (KeyError,TypeError,ValueError,AssertionError):return False


def _validate_broadcast_commands(buffer):
    views=buffer['source_storage_views']
    projected={k:v for k,v in views.items() if 'storage_projection' in v}
    if not projected:return
    proof=buffer.get('source_group_proof',{});runtime_init='unobserved_init' in proof
    if len(projected)!=2 or len(views)!=(3 if runtime_init else 2) or len(buffer['commands'])!=1:
        raise ValueError('raw32 broadcast requires exactly one source, one output and one concat')
    sources=[k for k,v in projected.items() if v['projection_role']=='source']
    outputs=[k for k,v in projected.items() if v['projection_role']=='output']
    if len(sources)!=1 or len(outputs)!=1:
        raise ValueError('raw32 broadcast source/output roles differ')
    src,dst=sources[0],outputs[0];projection=projected[src]['storage_projection']
    command=buffer['commands'][0];copies=projection['copies']
    operands={f'src{i}':src for i in range(copies)}|{'dst':dst}
    if (projected[dst]['storage_projection']!=projection or src==dst
            or command!={'opcode':'INTEGER_CONCAT','operands':operands,
                'attributes':dict(axis=1,inputs=copies,output_dtype='i32')}
            or any(buffer['tensors'][name]['shape']!=view['device_shape'] for name,view in projected.items())
            or buffer.get('source_group_proof',{}).get('storage_projection')!=projection):
        raise ValueError('raw32 broadcast must repeat the exact source into checked disjoint concat segments')
    if runtime_init:
        _validate_single_broadcast_init(buffer,src,dst,projection)


def _validate_single_broadcast_init(buffer,src,dst,projection):
    """Preserve one unread logical init without changing the concat command."""
    from xdsl.ir.affine import AffineMap,AffineExpr
    from .source_concat import plan_concat
    views=buffer['source_storage_views'];tensors=buffer['tensors'];proof=buffer['source_group_proof']
    unused=set(views)-{src,dst}
    if len(unused)!=1:raise ValueError('single-axis raw32 broadcast requires exactly one unobserved init')
    init=next(iter(unused));shape=projection['output_shape'];rank=len(shape);axis=projection['axis']
    expected_view=dict(source_shape=list(shape),device_shape=list(shape),source_dtype='f32',device_dtype='i32',
        conversion='none',transport_law=RAW32_LAW)
    expected_tensors={src:dict(shape=projection['source_physical_shape'],dtype='i32',role='input'),
        dst:dict(shape=projection['output_physical_shape'],dtype='i32',role='output'),
        init:dict(shape=list(shape),dtype='i32',role='input')}
    expected_map=AffineMap(rank,0,tuple(AffineExpr.constant(0) if i==axis else AffineExpr.dimension(i) for i in range(rank)))
    concat=plan_concat([projection['source_physical_shape']]*projection['copies'],1,'i32')
    expected_proof=dict(form='source_f32_bitwise_broadcast',source_operation='linalg.generic',
        source_dtype='f32',storage_dtype='i32',element_bytes=4,conversion='none',transport_law=RAW32_LAW,
        storage_projection=projection,indexing_maps=[str(expected_map),str(AffineMap.identity(rank))],
        iterators=['parallel']*rank,address_projection=concat.proof,
        semantics='each output word reads the same source word at the declared zero-axis affine projection; init is unobserved',
        unobserved_init=_unobserved_init_proof(shape))
    if (views[init]!=expected_view or tensors!=expected_tensors or proof!=expected_proof
            or buffer.get('constants') or buffer.get('params',{}).get('tensor_aliases')
            or buffer.get('tensor_aliases')):
        raise ValueError('single-axis raw32 broadcast init must remain an unread same-byte logical f32 boundary')


def multi_broadcast_stages(source_shape, output_shape):
    """Check a rank2..8 repeat as successive logical-axis integer concats.

    Every intermediate keeps its logical axes and has distinct contiguous
    storage. Induction on stages replaces exactly the processed coordinates
    by zero. Each concat independently proves complete disjoint output
    ownership, fresh nonaccumulating MVIN and drains before accumulator reuse.
    """
    from .source_concat import plan_concat
    from copy import deepcopy
    if (not 2 <= len(source_shape) <= 8 or len(output_shape) != len(source_shape)
            or any(type(n) is not int or n <= 0 for n in (*source_shape,*output_shape))):
        raise ValueError('multi-axis raw32 broadcast requires positive static rank2..8 matching shapes')
    axes=[i for i,(a,b) in enumerate(zip(source_shape,output_shape)) if a != b]
    if len(axes)<2 or any(source_shape[i]!=1 or not 2<=output_shape[i]<=64 for i in axes):
        raise ValueError('multi-axis raw32 broadcast requires at least two expanded unit axes, each 2..64 copies')
    if math.prod(output_shape)*4 >= 1 << 63:
        raise ValueError('raw32 broadcast byte footprint exceeds i64')
    shape=list(source_shape);stages=[]
    for axis in reversed(axes):
        copies=output_shape[axis]
        concat=plan_concat([shape]*copies,axis,'i32')
        stages.append(dict(axis=axis,copies=copies,source_shape=list(shape),
            output_shape=list(concat.output),address_projection=deepcopy(concat.proof)))
        shape=list(concat.output)
    if shape!=list(output_shape):raise ValueError('raw32 broadcast final shape differs')
    return stages


def _multi_broadcast_proof(source_shape,output_shape,stages,unobserved_init=False):
    from copy import deepcopy
    from xdsl.ir.affine import AffineMap,AffineExpr
    rank=len(source_shape)
    axes=[i for i,(a,b) in enumerate(zip(source_shape,output_shape)) if a!=b]
    input_map=AffineMap(rank,0,tuple(AffineExpr.constant(0) if i in axes
                                   else AffineExpr.dimension(i) for i in range(rank)))
    proof=dict(form='source_f32_bitwise_multi_axis_broadcast',source_operation='linalg.generic',
        source_dtype='f32',storage_dtype='i32',element_bytes=4,conversion='none',
        source_shape=list(source_shape),output_shape=list(output_shape),axes=axes,
        source_bytes=math.prod(source_shape)*4,output_bytes=math.prod(output_shape)*4,
        transport_law=deepcopy(RAW32_LAW),indexing_maps=[str(input_map),str(AffineMap.identity(rank))],
        iterators=['parallel']*rank,stages=stages,
        semantics='stage induction replaces precisely the expanded source coordinates by zero; init is unobserved')
    if unobserved_init:
        proof['unobserved_init']=_unobserved_init_proof(output_shape)
    return proof


def f32_multi_broadcast(op,spec):
    """Recognize an exact pure repeat on >=2 axes, preserving one-axis ABI.

    Returns (source SSA value, stage attributes, source proof). Concat uses
    checked flattened addresses while every stage retains its logical axes.
    Init is either static tensor.empty or a tensor retained as an explicit
    callback boundary. The latter keeps its original slot and is never read;
    its producer and all source effects remain outside this device callback.
    """
    from xdsl.dialects.builtin import NoneAttr,TensorType
    from xdsl.ir.affine import AffineMap,AffineExpr
    from .source_concat import ConcatError
    if (op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1
            or len(op.regions)!=1 or len(op.regions[0].blocks)!=1):return None
    src,init=op.operands;out=op.results[0]
    empty_init=getattr(init.owner,'name','')=='tensor.empty' and not init.owner.operands
    runtime_init=not empty_init
    if any(not isinstance(v.type,TensorType) or str(v.type.element_type)!='f32'
           or not isinstance(v.type.encoding,NoneAttr) or not v.type.get_shape()
           or any(n<=0 for n in v.type.get_shape()) for v in (src,init,out)):return None
    a,b=spec(src),spec(out)
    if spec(init)!=b:return None
    body=op.regions[0].block;ops=list(body.ops)
    if (len(body.args)!=2 or any(str(v.type)!='f32' for v in body.args)
            or len(ops)!=1 or ops[0].name!='linalg.yield'
            or list(ops[0].operands)!=[body.args[0]]):return None
    try:
        stages=multi_broadcast_stages(a['shape'],b['shape'])
    except (ValueError,ConcatError):return None
    proof=_multi_broadcast_proof(a['shape'],b['shape'],stages,runtime_init)
    maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
    rank=len(a['shape'])
    expected=AffineMap(rank,0,tuple(AffineExpr.constant(0) if i in proof['axes']
                                   else AffineExpr.dimension(i) for i in range(rank)))
    if (maps is None or len(maps)!=2 or iters is None
            or tuple(maps)[0].data!=expected or tuple(maps)[1].data!=AffineMap.identity(rank)
            or [v.data.value for v in iters]!=proof['iterators']):return None
    attrs=dict(stages=stages,output_dtype='i32')
    if runtime_init:attrs['unobserved_init']=True
    return src,attrs,proof


def emit_broadcast_buffer(tensors,constants,src,dst,attrs):
    """Expand a checked multi-axis broadcast into existing concat commands.

    Adds only distinct i32 intermediates. Source and output keep their logical
    shapes and original f32 callback ABI; no storage reshapes or aliases occur.
    """
    unused=set(tensors)-{src,dst};runtime_init=attrs.get('unobserved_init',False)
    expected_attrs={'stages','output_dtype'}|({'unobserved_init'} if runtime_init else set())
    if (src==dst or constants or type(runtime_init) is not bool
            or len(unused)!=(1 if runtime_init else 0)
            or set(attrs)!=expected_attrs or attrs['output_dtype']!='i32'
            or any(tensors[n]['dtype']!='i32' for n in (src,dst))):
        raise ValueError('multi-axis raw32 broadcast requires two distinct endpoints and no constants')
    if any(tensors[n]!=dict(shape=list(tensors[dst]['shape']),dtype='i32',role='input') for n in unused):
        raise ValueError('multi-axis raw32 broadcast extra input must be one unobserved output-shaped init')
    stages=multi_broadcast_stages(tensors[src]['shape'],tensors[dst]['shape'])
    if attrs['stages']!=stages:raise ValueError('multi-axis raw32 broadcast stages differ from checked geometry')
    if any(f'_broadcast_stage_{i}' in tensors for i in range(len(stages)-1)):
        raise ValueError('multi-axis raw32 broadcast scratch collides with endpoint')
    commands=[];current=src
    for index,stage in enumerate(stages):
        target=dst if index==len(stages)-1 else f'_broadcast_stage_{index}'
        if target==src or (target in tensors and target!=dst):
            raise ValueError('multi-axis raw32 broadcast scratch collides with endpoint')
        if target!=dst:tensors[target]=dict(shape=list(stage['output_shape']),dtype='i32',role='intermediate')
        commands.append(dict(opcode='INTEGER_CONCAT',
            operands={f'src{i}':current for i in range(stage['copies'])}|{'dst':target},
            attributes=dict(axis=stage['axis'],inputs=stage['copies'],output_dtype='i32')))
        current=target
    return commands


def _validate_multi_broadcast_commands(buffer):
    proof=buffer.get('source_group_proof',{})
    if (proof.get('form')!='source_f32_bitwise_multi_axis_broadcast'
            and not any(n.startswith('_broadcast_stage_') for n in buffer['tensors'])):return
    views=buffer['source_storage_views'];tensors=buffer['tensors']
    try:
        sources=[n for n,t in tensors.items() if t.get('role')=='input']
        outputs=[n for n,t in tensors.items() if t.get('role')=='output']
        runtime_init='unobserved_init' in proof
        if (len(sources)!=(2 if runtime_init else 1) or len(outputs)!=1 or set(views)!=set(sources+outputs)
                or buffer.get('params',{}).get('tensor_aliases') or buffer.get('tensor_aliases')
                or buffer.get('constants')):
            raise ValueError('multi-axis raw32 broadcast endpoints/constants/aliases changed')
        src=buffer['commands'][0]['operands']['src0'];dst=outputs[0]
        if src not in sources:raise ValueError('multi-axis raw32 broadcast used source is not an input')
        unused=set(sources)-{src}
        for name in (*sources,dst):
            view=views[name]
            if (set(view)!={'source_shape','device_shape','source_dtype','device_dtype','conversion','transport_law'}
                    or view['source_shape']!=tensors[name]['shape']
                    or view['device_shape']!=tensors[name]['shape']):
                raise ValueError('multi-axis raw32 broadcast must preserve logical source axes and bytes')
        stages=multi_broadcast_stages(tensors[src]['shape'],tensors[dst]['shape'])
        if proof!=_multi_broadcast_proof(tensors[src]['shape'],tensors[dst]['shape'],stages,runtime_init):
            raise ValueError('multi-axis raw32 broadcast source/map/byte proof differs')
        expected={src:dict(shape=list(tensors[src]['shape']),dtype='i32',role='input'),
                  dst:dict(shape=list(tensors[dst]['shape']),dtype='i32',role='output')}
        expected.update({n:dict(shape=list(tensors[dst]['shape']),dtype='i32',role='input') for n in unused})
        attrs=dict(stages=stages,output_dtype='i32')
        if runtime_init:attrs['unobserved_init']=True
        commands=emit_broadcast_buffer(expected,{},src,dst,attrs)
        if tensors!=expected or buffer['commands']!=commands:
            raise ValueError('multi-axis raw32 broadcast must use the exact distinct-storage concat sequence')
    except (KeyError,TypeError,IndexError,AssertionError) as error:
        raise ValueError('malformed multi-axis raw32 broadcast buffer') from error
