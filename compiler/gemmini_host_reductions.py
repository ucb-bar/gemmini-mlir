"""Source-bound ordered expansion around the shared exact generic lowering."""
from __future__ import annotations

import hashlib
from math import prod


def describe_ordered_generic(op):
    """Prove static, in-bounds float compare/select reductions with <=16 steps.

    Provenance is descriptive. Unknown semantic attributes, affine forms,
    encodings, scalar flags and nontrailing reductions retain ordinary lowering.
    """
    from xdsl.dialects.builtin import TensorType
    from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpExpr, AffineBinaryOpKind
    if op.name != 'linalg.generic' or len(op.regions) != 1:
        return None
    if any(not k.startswith('prov.') for k in op.attributes):
        return None
    if set(op.properties) != {'indexing_maps', 'iterator_types', 'operandSegmentSizes'}:
        return None
    kinds = [str(k.data) for k in op.iterator_types]
    if 'reduction' not in kinds:
        return None
    split = kinds.index('reduction')
    if not split or kinds != ['parallel'] * split + ['reduction'] * (len(kinds)-split):
        return None
    values = (*op.inputs, *op.outputs, *op.results)
    if any(not isinstance(v.type, TensorType) or str(v.type.element_type) != 'f32'
           or v.type.encoding is not None and str(v.type.encoding) != 'none'
           or any(n <= 0 for n in v.type.get_shape()) for v in values):
        return None
    if len(op.outputs) != len(op.results) or any(a.type != b.type for a,b in zip(op.outputs, op.results)):
        return None
    allowed = {'arith.cmpf': {'predicate', 'fastmath'}, 'arith.ori': set(),
               'arith.select': set(), 'linalg.yield': set()}
    body = list(op.body.block.ops)
    if not body or len(body) > 64:
        return None
    for inner in body:
        if (inner.regions or inner.name not in allowed
                or set(inner.properties) != allowed[inner.name]
                or any(not k.startswith('prov.') for k in inner.attributes)):
            return None
        if inner.name == 'arith.cmpf' and inner.properties['fastmath'].data:
            return None
    maps = [m.data for m in op.indexing_maps]
    if len(maps) != len(op.inputs) + len(op.outputs) or any(m.num_dims != len(kinds) or m.num_symbols for m in maps):
        return None
    bounds = [0]*len(kinds)
    for mapping, value in zip(maps, (*op.inputs, *op.outputs)):
        if len(mapping.results) != len(value.type.get_shape()):
            return None
        for expr, extent in zip(mapping.results, value.type.get_shape()):
            if isinstance(expr, AffineDimExpr):
                if bounds[expr.position] not in (0, extent):
                    return None
                bounds[expr.position] = extent
    if any(n <= 0 for n in bounds) or not 2 <= prod(bounds[split:]) <= 16:
        return None

    def interval(expr):
        if isinstance(expr, AffineDimExpr):
            return 0, bounds[expr.position]-1
        if isinstance(expr, AffineConstantExpr):
            return expr.value, expr.value
        if isinstance(expr, AffineBinaryOpExpr):
            left, right = interval(expr.lhs), interval(expr.rhs)
            if left is None or right is None:
                return None
            if expr.kind == AffineBinaryOpKind.Add:
                return left[0]+right[0], left[1]+right[1]
            if expr.kind == AffineBinaryOpKind.Mul and (left[0] == left[1] or right[0] == right[1]):
                products = [a*b for a in left for b in right]
                return min(products), max(products)
        return None

    for mapping, value in zip(maps, (*op.inputs, *op.outputs)):
        for expr, extent in zip(mapping.results, value.type.get_shape()):
            span = interval(expr)
            if span is None or span[0] < 0 or span[1] >= extent:
                return None
    for mapping in maps[len(op.inputs):]:
        if (len(mapping.results) != split
                or any(not isinstance(e, AffineDimExpr) for e in mapping.results)
                or sorted(e.position for e in mapping.results) != list(range(split))):
            return None
    return dict(recipe='shared_ordered_trailing_reduction_expansion',
                source_operation_sha256=hashlib.sha256(str(op).encode()).hexdigest(),
                bounds=bounds, reduction_start=split,
                reduction_iterations=prod(bounds[split:]),
                arithmetic='unchanged original _body, _round and each-step store',
                reduction_order='unchanged source lexicographic order')


def lower_ordered_generic(host, op, *, ordinary_lowerer):
    """Call the original bound lowerer, temporarily adapting this instance only.

    Returns precisely the callback result. Exceptions restore the instance's
    original attribute state. A refusal calls the callback without any mutation.
    Selection policy belongs to the caller.
    """
    proof = describe_ordered_generic(op)
    if proof is None:
        return ordinary_lowerer(op)
    old = host._loops
    had_override = '_loops' in host.__dict__
    previous = host.__dict__.get('_loops')

    def loops(shape, body):
        if tuple(shape) != tuple(proof['bounds']):
            return old(shape, body)
        def axis(dim, indices):
            if dim == len(shape):
                return body(tuple(indices))
            if dim >= proof['reduction_start']:
                host.fb.for_range(shape[dim], lambda iv: axis(dim+1, [*indices, iv]))
            else:
                host.fb.for_loop(0, shape[dim], 1,
                    lambda iv: axis(dim+1, [*indices, iv]), retain_loop=True)
        axis(0, [])
    host._loops = loops
    try:
        result = ordinary_lowerer(op)
    finally:
        if had_override:
            host._loops = previous
        else:
            del host._loops
    if not hasattr(host, 'host_ordered_reduction_reports'):
        host.host_ordered_reduction_reports = []
    host.host_ordered_reduction_reports.append(proof)
    return result
