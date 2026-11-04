"""Source proofs for f32 payload permutations without floating arithmetic."""
from __future__ import annotations


def prove_float_bit_permutation(op):
    """Prove an exact named transpose or parallel affine permutation copy.

    f32 is the unchanged source/output ABI type. The selected transpose adapter
    transports each 32-bit word through raw i32 preload/full-width readout, with
    no cast, scale, clamp or other arithmetic on its IEEE payload.
    """
    from xdsl.dialects.builtin import TensorType, NoneAttr
    from xdsl.ir.affine import AffineMap, AffineDimExpr
    from xdsl.utils.exceptions import VerifyException
    from .transpose import plan_transpose, TransposeError

    if (op.name not in ('linalg.transpose', 'linalg.generic')
            or len(op.operands) != 2 or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or any(not key.startswith('prov.') for key in op.attributes)):
        return None
    src, init = op.operands
    typ, out = src.type, op.results[0].type
    if (not isinstance(typ, TensorType) or not isinstance(out, TensorType)
            or any(not isinstance(t.encoding, NoneAttr) for t in (typ, out))
            or str(typ.element_type) != 'f32' or out.element_type != typ.element_type
            or init.type != out or not 2 <= len(typ.get_shape()) <= 4
            or any(n <= 0 for n in (*typ.get_shape(), *out.get_shape()))):
        return None
    body = op.regions[0].block
    ops = list(body.ops)
    if (len(body.args) != 2 or len(ops) != 1 or ops[0].name != 'linalg.yield'
            or list(ops[0].operands) != [body.args[0]] or body.args[1].uses):
        return None
    rank = len(typ.get_shape())
    if op.name == 'linalg.transpose':
        if set(op.properties) != {'permutation'}:
            return None
        perm = list(op.permutation.get_values())
    else:
        if set(op.properties) - {'indexing_maps', 'iterator_types', 'operandSegmentSizes'}:
            return None
        maps = tuple(a.data for a in op.indexing_maps)
        if (len(maps) != 2 or maps[1] != AffineMap.identity(rank)
                or len(op.iterator_types) != rank
                or any(a.data.value != 'parallel' for a in op.iterator_types)
                or maps[0].num_dims != rank or maps[0].num_symbols
                or len(maps[0].results) != rank
                or any(not isinstance(e, AffineDimExpr) for e in maps[0].results)):
            return None
        source_axes = [e.position for e in maps[0].results]
        if sorted(source_axes) != list(range(rank)):
            return None
        perm = [source_axes.index(axis) for axis in range(rank)]
    if (sorted(perm) != list(range(rank))
            or list(out.get_shape()) != [typ.get_shape()[axis] for axis in perm]):
        return None
    try:
        op.verify()
        plan = plan_transpose(tuple(typ.get_shape()), tuple(perm), 'f32')
    except (ValueError, TransposeError, VerifyException):
        return None
    return src, {'permutation': perm, 'output_dtype': 'f32', 'storage_dtype': 'i32',
                 'semantic': 'bit_preserving_permutation'}, {
        'form': 'checked_f32_payload_permutation', 'source_dtype': 'f32',
        'output_dtype': 'f32', 'transport_dtype': 'i32', 'payload_bits': 32,
        'arithmetic': 'none; raw preload and full-width readout preserve every IEEE bit',
        'address_projection': plan.proof}


def prove_float_bit_concat(op):
    """Prove static tensor concatenation with unchanged f32 payloads and axes."""
    from xdsl.dialects.builtin import TensorType, NoneAttr, IntegerAttr
    from xdsl.utils.exceptions import VerifyException
    from .bit_concat import plan_concat, BitConcatError

    if (op.name != 'tensor.concat' or len(op.operands) < 2
            or len(op.results) != 1 or op.regions
            or set(op.properties) != {'dim'}
            or any(not key.startswith('prov.') for key in op.attributes)):
        return None
    types = [value.type for value in (*op.operands, op.results[0])]
    if any(not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr)
           or str(t.element_type) != 'f32' or any(d <= 0 for d in t.get_shape())
           for t in types):
        return None
    rank = len(types[-1].get_shape())
    axis_attr = op.properties['dim']
    if (not 2 <= rank <= 4 or any(len(t.get_shape()) != rank for t in types)
            or not isinstance(axis_attr, IntegerAttr)):
        return None
    axis = axis_attr.value.data
    if not 0 <= axis < rank:
        return None
    shapes = [tuple(t.get_shape()) for t in types]
    expected = list(shapes[0])
    expected[axis] = sum(shape[axis] for shape in shapes[:-1])
    if (tuple(expected) != shapes[-1] or any(
            any(shape[d] != shapes[-1][d] for d in range(rank) if d != axis)
            for shape in shapes[:-1])):
        return None
    try:
        op.verify()
        plan = plan_concat(shapes[:-1], axis, dtype='f32')
        plan.revalidate()
    except (ValueError, BitConcatError, VerifyException):
        return None
    if tuple(plan.output_shape) != shapes[-1]:
        return None
    return tuple(op.operands), {
        'axis': axis, 'output_dtype': 'f32', 'storage_dtype': 'i32',
        'semantic': 'bit_preserving_concat'}, {
        'form': 'checked_f32_payload_concat', 'axis': axis,
        'source_shapes': [list(shape) for shape in shapes[:-1]],
        'output_shape': list(shapes[-1]), 'source_dtype': 'f32',
        'output_dtype': 'f32', 'transport_dtype': 'i32', 'payload_bits': 32,
        'arithmetic': 'none; each source IEEE word is copied once to its concatenated output coordinate',
        'address_projection': plan.proof}


def prove_float_bit_slice(op):
    """Prove a static rank-preserving, unit-stride f32 extract_slice copy."""
    from xdsl.dialects.builtin import TensorType, NoneAttr, DenseArrayBase
    from xdsl.utils.exceptions import VerifyException
    from .bit_slice import plan_slice, BitSliceError

    if (op.name != 'tensor.extract_slice' or len(op.operands) != 1
            or len(op.results) != 1 or op.regions
            or set(op.properties) != {'static_offsets', 'static_sizes', 'static_strides', 'operandSegmentSizes'}
            or any(not key.startswith('prov.') for key in op.attributes)
            or any(not isinstance(v, DenseArrayBase) for v in op.properties.values())):
        return None
    source, out = op.operands[0].type, op.results[0].type
    if any(not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr)
           or str(t.element_type) != 'f32' for t in (source, out)):
        return None
    offsets = list(op.properties['static_offsets'].get_values())
    sizes = list(op.properties['static_sizes'].get_values())
    strides = list(op.properties['static_strides'].get_values())
    if (list(op.properties['operandSegmentSizes'].get_values()) != [1, 0, 0, 0]
            or strides != [1] * len(source.get_shape()) or list(out.get_shape()) != sizes):
        return None
    try:
        op.verify()
        plan = plan_slice(source.get_shape(), offsets, sizes)
        plan.revalidate()
    except (ValueError, BitSliceError, VerifyException):
        return None
    return op.operands[0], {
        'offsets': offsets, 'sizes': sizes, 'strides': strides,
        'output_dtype': 'f32', 'storage_dtype': 'i32', 'semantic': 'bit_preserving_slice'}, {
        'form': 'checked_static_f32_payload_slice', 'source_dtype': 'f32',
        'output_dtype': 'f32', 'transport_dtype': 'i32', 'payload_bits': 32,
        'arithmetic': 'none; exact source coordinates copied to every dense destination element',
        'address_projection': plan.proof}


def prove_float_bit_repeat(op):
    """Yield-only dense broadcast along exactly one static singleton axis."""
    from xdsl.dialects.builtin import TensorType, NoneAttr
    from xdsl.ir.affine import AffineMap, AffineDimExpr, AffineConstantExpr
    from xdsl.utils.exceptions import VerifyException
    from .bit_repeat import plan_repeat
    if (op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1
        or len(op.regions)!=1 or len(op.regions[0].blocks)!=1
        or any(not k.startswith('prov.') for k in op.attributes)
        or set(op.properties)-{'indexing_maps','iterator_types','operandSegmentSizes'}):return None
    src,init=op.operands;types=(src.type,op.results[0].type)
    if any(not isinstance(t,TensorType) or not isinstance(t.encoding,NoneAttr)
           or str(t.element_type)!='f32' or any(d<=0 for d in t.get_shape()) for t in types):return None
    shape,out=(tuple(t.get_shape()) for t in types);rank=len(shape)
    if not 2<=rank<=6 or len(out)!=rank or init.type!=types[1]:return None
    body=op.regions[0].block;ops=list(body.ops)
    if (len(body.args)!=2 or len(ops)!=1 or ops[0].name!='linalg.yield'
        or list(ops[0].operands)!=[body.args[0]] or body.args[1].uses):return None
    maps=tuple(a.data for a in op.indexing_maps)
    if (len(maps)!=2 or maps[1]!=AffineMap.identity(rank)
        or maps[0].num_dims!=rank or maps[0].num_symbols or len(maps[0].results)!=rank
        or len(op.iterator_types)!=rank or any(a.data.value!='parallel' for a in op.iterator_types)):return None
    axes=[]
    for i,e in enumerate(maps[0].results):
        if isinstance(e,AffineDimExpr) and e.position==i:
            if shape[i]!=out[i]:return None
        elif isinstance(e,AffineConstantExpr) and e.value==0 and shape[i]==1 and out[i]>1:axes.append(i)
        else:return None
    if len(axes)!=1:return None
    axis=axes[0]
    try:
        op.verify();plan=plan_repeat(shape,axis,out[axis]);plan.revalidate()
    except (ValueError,VerifyException):return None
    return src,{'axis':axis,'repeats':out[axis],'output_dtype':'f32','storage_dtype':'i32',
                'semantic':'bit_preserving_repeat'}, {'form':'checked_static_f32_repeated_blocks',
                'arithmetic':'none; source input yielded unchanged','address_projection':plan.proof,
                'capability':'standalone','eligible':True,'device_form':'RAW_BIT_REPEAT',
                'complete_call_cycles':'UNKNOWN','placement_policy':'required device coverage; performance does not determine eligibility'}


def analyze_float_bit_repeat(op):
    """Diagnose genuine copy/broadcast candidates without inventing eligibility.

    A refusal of this adapter is a proof/compiler gap, not evidence that the
    target cannot transport f32 bits. Identity copies and unrelated arithmetic
    stay with their own source-semantic analysis.
    """
    from xdsl.dialects.builtin import TensorType, NoneAttr
    from xdsl.ir.affine import AffineConstantExpr
    admitted=prove_float_bit_repeat(op)
    if admitted is not None:
        return {'eligible':True,'status':'eligible_device','capability':'standalone',
                'device_form':'RAW_BIT_REPEAT','proof':admitted[2],
                'complete_call_cycles':'UNKNOWN'}
    if (op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1
        or len(op.regions)!=1 or len(op.regions[0].blocks)!=1):return None
    src,out=op.operands[0].type,op.results[0].type
    if not all(isinstance(t,TensorType) and str(t.element_type)=='f32' for t in (src,out)):return None
    shape,target=tuple(src.get_shape()),tuple(out.get_shape())
    if len(shape)!=len(target):return None
    body=op.regions[0].block;ops=list(body.ops)
    if (len(body.args)!=2 or len(ops)!=1 or ops[0].name!='linalg.yield'
        or list(ops[0].operands)!=[body.args[0]]):return None
    maps=tuple(getattr(op,'indexing_maps',()))
    if len(maps)!=2:return None
    source_map=maps[0].data
    if len(source_map.results)!=len(shape):return None
    expanded=[i for i,e in enumerate(source_map.results)
              if isinstance(e,AffineConstantExpr) and e.value==0 and shape[i]==1 and target[i]!=1]
    if not expanded:return None
    status='unresolved_proof'
    if any(not k.startswith('prov.') for k in op.attributes):
        reason='unknown source attribute/effect prevents proving a pure broadcast materialization'
    elif any(not isinstance(t.encoding,NoneAttr) for t in (src,out)):
        reason='encoded/strided source layout has no ordinary dense address proof'
    elif any(d<=0 for d in (*shape,*target)):
        reason='dynamic or empty broadcast extents lack a static address and output-coverage proof'
    elif not 2<=len(shape)<=6:
        status='compiler_adapter_gap';reason='raw repeat adapter supports rank2-6; this pure broadcast rank is outside its implemented domain'
    elif len(expanded)!=1:
        status='compiler_adapter_gap';reason='multiple expanded axes require an additional broadcast address/coverage adapter'
    elif max(shape[-1],target[-1])*4>=1<<32 or __import__('math').prod(target)*4>=1<<63:
        reason='dense broadcast row pitch or byte footprint exceeds the proved descriptor/address range'
    else:
        reason='source affine maps, iterator/body types, effects or operation verification do not establish exact singleton-axis repetition'
    return {'eligible':None,'status':status,'capability':'unresolved standalone',
            'candidate_device_form':'RAW_BIT_REPEAT','reason':reason,
            'complete_call_cycles':'UNKNOWN','hardware_support':'not inferred from adapter refusal'}
