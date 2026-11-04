"""Prove exact f32 identity generics whose two maps are full axis permutations."""
from .float_movement import _spec


def match(op):
    if op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1:return None
    src,initial=op.operands;a,b=_spec(src),_spec(op.results[0])
    if a is None or b is None or _spec(initial)!=b or not 2<=len(a['shape'])<=4:return None
    rank=len(a['shape'])
    if len(b['shape'])!=rank or len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return None
    body=op.regions[0].block;ops=list(body.ops)
    if (len(body.args)!=2 or any(str(v.type)!='f32' for v in body.args)
            or len(ops)!=1 or ops[0].name!='linalg.yield'
            or list(ops[0].operands)!=[body.args[0]]):return None
    maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
    if maps is None or len(maps)!=2 or iters is None or [v.data.value for v in iters]!=['parallel']*rank:return None
    from xdsl.ir.affine import AffineDimExpr
    permutations=[]
    for attr in maps:
        m=attr.data
        if (m.num_dims!=rank or m.num_symbols!=0 or len(m.results)!=rank
                or any(not isinstance(e,AffineDimExpr) for e in m.results)):return None
        axes=[e.position for e in m.results]
        if sorted(axes)!=list(range(rank)):return None
        permutations.append(axes)
    input_axes,output_axes=permutations
    # source[j] = loop[input_axes[j]]; destination[k] = loop[output_axes[k]].
    # Thus destination axis k names source axis input_axes.index(output_axes[k]).
    perm=[input_axes.index(axis) for axis in output_axes]
    if b['shape']!=[a['shape'][i] for i in perm]:return None
    from .transpose import plan_transpose,TransposeError
    try:plan=plan_transpose(a['shape'],perm,'i32')
    except TransposeError:return None
    return src,{'permutation':perm,'output_dtype':'i32'},dict(
        form='exact_f32_bit_transpose',source_operation=op.name,
        source_form='scalar_identity_with_bijective_permutation_maps',
        input_loop_axes=input_axes,output_loop_axes=output_axes,permutation=perm,
        map_composition='inverse(input_map) applied to output_map',
        logical_dtype='f32',transport_dtype='i32',source_shape=a['shape'],result_shape=b['shape'],
        numeric_operations=[],destination_initializer='ignored by scalar identity; every output address overwritten exactly once',
        representation='all IEEE32 bits unchanged including NaN payloads and signed zero',
        address_projection=plan.proof)
