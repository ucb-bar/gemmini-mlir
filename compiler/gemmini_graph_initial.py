"""Additional source-bound dot admission: multiple ordered reduction axes.

Live initial tensors already flow through the dense backend's D matrix load.
This reader extends the checked permutation path to products whose one logical K
is represented by several source axes. It performs no arithmetic reassociation:
K coordinates follow source iterator order, last reduction axis varying fastest.
"""
from math import prod


def recognize(op):
    from xdsl.dialects import builtin
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.ir import Block
    from xdsl.ir.affine import AffineDimExpr as D, AffineMap
    from gemmini_graph_layouts import recognize as signed_dot
    if type(op) is not GenericOp or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    values = tuple(op.operands) + tuple(op.results)
    if any(not isinstance(v.type, builtin.TensorType) or not isinstance(v.type.encoding, builtin.NoneAttr)
           or any(d <= 0 for d in v.type.get_shape()) for v in values):
        return None
    if [str(v.type.element_type) for v in values] != ['i8','i8','i32','i32'] or values[2].type != values[3].type:
        return None
    if set(op.properties) - {'indexing_maps','iterator_types','operandSegmentSizes'}:
        return None
    iters = tuple(x.data for x in op.iterator_types)
    red = tuple(i for i,t in enumerate(iters) if t == IteratorType.REDUCTION)
    par = tuple(i for i,t in enumerate(iters) if t == IteratorType.PARALLEL)
    if len(red) < 2 or len(red) + len(par) != len(iters):
        return None
    maps = tuple(x.data for x in op.indexing_maps)
    if len(maps) != 3 or any(mp.num_dims != len(iters) or mp.num_symbols or any(not isinstance(x,D) for x in mp.results) for mp in maps):
        return None
    axes = tuple(tuple(x.position for x in mp.results) for mp in maps)
    if any(len(set(ax)) != len(ax) for ax in axes):
        return None
    a,b,c = axes
    if set(c) != set(par) or not set(red) <= set(a) & set(b) or set(a) | set(b) != set(c) | set(red):
        return None
    extents = {}
    for value,ax in zip(values,axes):
        if len(ax) != len(value.type.get_shape()):
            return None
        for dim,size in zip(ax,value.type.get_shape()):
            if dim in extents and extents[dim] != size:
                return None
            extents[dim] = size
    batch = tuple(x for x in c if x in a and x in b)
    ma = tuple(x for x in c if x in a and x not in b)
    na = tuple(x for x in c if x in b and x not in a)
    canonical = (batch + ma + red, batch + red + na, batch + ma + na)
    if any(set(x) != set(y) for x,y in zip(axes,canonical)):
        return None
    m,k,n = prod(extents[x] for x in ma),prod(extents[x] for x in red),prod(extents[x] for x in na)
    fake = Block(arg_types=[builtin.TensorType(builtin.i8,(m,k)),builtin.TensorType(builtin.i8,(k,n)),builtin.TensorType(builtin.i32,(m,n))])
    shell = GenericOp(inputs=fake.args[:2], outputs=fake.args[2:], body=op.regions[0].clone(),
        indexing_maps=builtin.ArrayAttr(builtin.AffineMapAttr(mp) for mp in (
            AffineMap(3,0,(D(0),D(2))),AffineMap(3,0,(D(2),D(1))),AffineMap(3,0,(D(0),D(1))))),
        iterator_types=builtin.ArrayAttr(IteratorTypeAttr(t) for t in (IteratorType.PARALLEL,IteratorType.PARALLEL,IteratorType.REDUCTION)),
        result_types=[fake.args[2].type])
    shell.attributes = dict(op.attributes)
    # Existing checker proves exact ordered signed widening, modular i32
    # multiply/add recurrence, attributes, body ownership and yield.
    if signed_dot(shell) is None:
        return None
    return dict(m=m,k=k,n=n,batch_shape=tuple(extents[x] for x in batch),
        lhs_batch_axes=tuple(range(len(batch))),rhs_batch_axes=tuple(range(len(batch))),
        out_batch_axes=tuple(range(len(batch))),permutation_preparation=True,
        source_axes=axes,canonical_axes=canonical,extents=extents,
        source_reduction_axes=red)


def emit(*args, **kwargs):
    """Use the existing checked bounded-copy and dense D-load implementation."""
    from gemmini_graph_layouts import emit as emit_layout
    return emit_layout(*args, **kwargs)
