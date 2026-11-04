"""Exact contiguous source-slice views consumed by the shared dense emitter.

Source grouping owns slice/view operations only when every result has one use.
The alternative changes addresses, never contraction arithmetic or readout.
Noncontiguous views are refused before any target effect is emitted.
"""
from __future__ import annotations
from dataclasses import replace
from math import prod

from .source_slice import plan_source_slice,SourceSliceError


def contiguous_slice_view(source_shape,offsets,sizes,strides,logical_shape,dtype):
    source_shape,offsets,sizes,strides,logical_shape=map(tuple,
        (source_shape,offsets,sizes,strides,logical_shape))
    if (dtype!='i8' or len(logical_shape)!=2 or any(type(n) is not int or n<=0 for n in logical_shape)
            or prod(logical_shape)!=prod(sizes)):
        raise SourceSliceError('dense source view needs a static signed-i8 matrix with unchanged element count')
    plan=plan_source_slice(source_shape,offsets,sizes,strides,dtype)
    first,last=plan.proof['source_address_bounds']
    # Positive static slice maps are injective and lexicographically increasing.
    # If their span equals their cardinality, their image is exactly the dense
    # interval [first,last], in output order: no gap, permutation or packing.
    if last-first+1!=prod(sizes):
        raise SourceSliceError('source slice is not one contiguous row-major interval')
    return dict(schema='submission.contiguous_slice_dense_view.v1',
        source_shape=list(source_shape),offsets=list(offsets),sizes=list(sizes),strides=list(strides),
        logical_shape=list(logical_shape),dtype=dtype,source_element_offset=first,
        source_element_end=last+1,source_slice=plan.revalidate(),
        address_law='logical matrix flat index j maps to original source offset + j',
        materialization='none; original source pointer plus checked element offset')


def _reshape(op):
    from .source_groups import _spec
    from xdsl.dialects.builtin import NoneAttr
    if (op.name not in ('tensor.collapse_shape','tensor.expand_shape') or len(op.operands)!=1
            or len(op.results)!=1 or not all(isinstance(v.type.encoding,NoneAttr) for v in (*op.operands,*op.results))):
        return False
    a,b=map(_spec,(op.operands[0],op.results[0]))
    if a['dtype']!='i8' or b['dtype']!='i8' or prod(a['shape'])!=prod(b['shape']):return False
    attr=op.properties.get('reassociation')
    if attr is None:return False
    groups=[[x.value.data for x in group] for group in attr]
    wide,narrow=(a['shape'],b['shape']) if op.name=='tensor.collapse_shape' else (b['shape'],a['shape'])
    if [i for group in groups for i in group]!=list(range(len(wide))) or len(groups)!=len(narrow):return False
    return all(group and prod(wide[i] for i in group)==size for group,size in zip(groups,narrow,strict=True))


def _trace(value,consumer):
    from .source_groups import _spec,_integer_slice
    logical=_spec(value);current=value;members=[];next_op=consumer
    while getattr(current.owner,'name','') in ('tensor.collapse_shape','tensor.expand_shape'):
        op=current.owner
        if (op.parent is not consumer.parent or len(list(current.uses))!=1
                or next(iter(current.uses)).operation is not next_op or not _reshape(op)):
            return None
        members.append(op);next_op=op;current=op.operands[0]
    op=current.owner
    if (getattr(op,'name','')!='tensor.extract_slice' or op.parent is not consumer.parent or len(list(current.uses))!=1
            or next(iter(current.uses)).operation is not next_op):return None
    sliced=_integer_slice(op)
    if sliced is None:return None
    source,attrs,_=sliced
    try:
        proof=contiguous_slice_view(_spec(source)['shape'],attrs['offsets'],attrs['sizes'],attrs['strides'],
            logical['shape'],logical['dtype'])
    except SourceSliceError:return None
    members.append(op)
    return source,tuple(reversed(members)),proof


def compose_slice_dense(plan):
    """Return a source-bound alternative or the unchanged independent plan."""
    from .source_groups import _root_inputs,_dependencies
    lhs,rhs,_=_root_inputs(plan.root);views={};members=list(plan.operations)
    for key,value in (('lhs',lhs),('rhs',rhs)):
        result=_trace(value,plan.root)
        if result is None:continue
        source,operations,proof=result
        views[key]={'source':source,'proof':proof};members.extend(operations)
    if not views:return plan
    operations,inputs=_dependencies(plan.root,members)
    if any(row['source'] not in inputs for row in views.values()):return plan
    attrs={**plan.attributes,'_source_dense_views':views}
    diagnostic={**plan.diagnostic,'source_slice_dense':{
        'selection':'on','consumed':'original contiguous source subview',
        'views':{key:row['proof'] for key,row in views.items()},
        'numeric':'unchanged shared dense recurrence, initializer, bias and readout',
        'live_outputs':'all absorbed slice/reshape results have exactly one source use'}}
    return replace(plan,operations=operations,inputs=inputs,attributes=attrs,diagnostic=diagnostic)


def lower_source_slice_dense(ctx,operands,attributes):
    """Register proved original pointers, then invoke unchanged lower_dense."""
    from .dense import lower_dense
    lhs,rhs,dst=(operands[k] for k in ('lhs','rhs','dst'))
    shapes=attributes['matrix_shapes'];views=attributes.get('views',{});names={};proofs={}
    for key,name in (('lhs',lhs),('rhs',rhs)):
        spec=ctx.spec(name);shape=shapes[key]
        if spec['dtype']!='i8' or len(shape)!=2:
            raise SourceSliceError('slice-dense operands must remain signed-i8 matrices')
        if key not in views:
            if list(spec['shape'])!=list(shape):raise SourceSliceError('unsliced dense operand shape changed')
            names[key]=name;continue
        row=views[key]
        proof=contiguous_slice_view(spec['shape'],row['offsets'],row['sizes'],row['strides'],shape,spec['dtype'])
        if proof!=row:raise SourceSliceError('source-slice dense address proof changed')
        alias=f'_source_slice_dense_{key}_{len(ctx.tensors)}'
        ctx.register(alias,ctx.ptr(name,proof['source_element_offset']),shape,'i8')
        names[key]=alias;proofs[key]=proof
    m,k=shapes['lhs'];kr,n=shapes['rhs'];output=ctx.spec(dst)
    if (k!=kr or output['shape']!=[m,n] or dst in (lhs,rhs)
            or output['dtype']!=attributes['dense_attributes'].get('output_dtype','i32')):
        raise SourceSliceError('slice-dense result or contraction geometry changed')
    receipt=lower_dense(ctx,names['lhs'],names['rhs'],dst,m=m,k=k,n=n,
        attributes=dict(attributes['dense_attributes']))
    report={'schema':'submission.source_slice_dense.v1','consumed':'original source pointer subviews',
        'views':proofs,'dense_leaf':receipt,'materialized_slice_bytes':0,
        'saved_slice_output_bytes':sum(prod(p['sizes']) for p in proofs.values())}
    ctx.metadata.setdefault('source_slice_dense',[]).append(report)
    return report
