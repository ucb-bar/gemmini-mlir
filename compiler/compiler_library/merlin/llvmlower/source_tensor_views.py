"""Compose static tensor slices with a following permutation copy.

The source scalar yield and its exact address expression are retained.  Shared
slice results remain available to every original observer; only dead slices are
removed.  This preparation operates on tensor SSA before bufferization and
never changes the public argument or return bindings.
"""
from __future__ import annotations


def _static_tensor(value):
    from xdsl.dialects.builtin import TensorType, NoneAttr
    typ = value.type
    if (not isinstance(typ, TensorType) or not isinstance(typ.encoding, NoneAttr)
            or not typ.get_shape() or any(d <= 0 for d in typ.get_shape())):
        return None
    return tuple(typ.get_shape())


def _array(op, key):
    attr = op.properties.get(key, op.attributes.get(key))
    if attr is None or not hasattr(attr, 'get_values'):
        return None
    return tuple(attr.get_values())


def _slice(value):
    """Admit only complete-rank static in-bounds slice address maps."""
    op = value.owner
    if getattr(op, 'name', '') != 'tensor.extract_slice' or len(op.operands) != 1 or len(op.results) != 1:
        return None
    shape, result = _static_tensor(op.operands[0]), _static_tensor(value)
    if shape is None or result is None or len(shape) != len(result):
        return None
    offsets, sizes, strides = (_array(op, k) for k in ('static_offsets', 'static_sizes', 'static_strides'))
    if (offsets is None or sizes != result or strides is None
            or not len(offsets) == len(strides) == len(shape)
            or op.operands[0].type.element_type != value.type.element_type
            or any(a < 0 or s <= 0 or a + (n-1)*s >= d
                   for a, n, s, d in zip(offsets, sizes, strides, shape, strict=True))):
        return None
    # No unproved attributes with semantic consequences are discarded.
    allowed = {'static_offsets', 'static_sizes', 'static_strides', 'operandSegmentSizes'}
    if any(k not in allowed and not k.startswith('prov.') for k in (*op.properties, *op.attributes)):
        return None
    return op, offsets, strides


def _permutation(op):
    from xdsl.dialects import tensor
    if (op.name != 'linalg.transpose' or len(op.operands) != 2 or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or not isinstance(op.operands[1].owner, tensor.EmptyOp)):
        return None
    source, output = _static_tensor(op.operands[0]), _static_tensor(op.results[0])
    permutation = _array(op, 'permutation')
    if (source is None or output is None or permutation is None
            or sorted(permutation) != list(range(len(source)))
            or tuple(source[i] for i in permutation) != output
            or op.operands[1].type != op.results[0].type
            or op.operands[0].type.element_type != op.results[0].type.element_type):
        return None
    block = op.regions[0].block
    body = list(block.ops)
    if (len(block.args) != 2 or len(body) != 1 or body[0].name != 'linalg.yield'
            or tuple(body[0].operands) != (block.args[0],)):
        return None
    if any(k != 'permutation' and not k.startswith('prov.') for k in (*op.properties, *op.attributes)):
        return None
    return permutation


def prepare_source_tensor_views(module):
    """Fuse slice/permutation copies without changing arithmetic or output ABI.

    Run after device routing and before host tensor lowering: device calls are
    opaque and stay untouched. Every output element is copied once using the
    composed original source address. Original live slice values are retained.
    """
    from math import prod
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DictionaryAttr
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap

    module.verify()
    records = []
    for op in tuple(module.walk()):
        permutation = _permutation(op)
        if permutation is None:
            continue
        current = op.operands[0]
        rank = len(permutation)
        inverse = tuple(permutation.index(i) for i in range(rank))
        expression = [AffineExpr.dimension(inverse[i]) for i in range(rank)]
        slices = []
        admitted = None
        while (entry := _slice(current)) is not None:
            view, offsets, strides = entry
            # Stop at block boundaries; no dominance or effects are inferred.
            if view.parent is not op.parent:
                break
            expression = [expr * stride + offset for expr, stride, offset
                          in zip(expression, strides, offsets, strict=True)]
            slices.append(view)
            current = view.operands[0]
            # Linalg derives loop sizes from every pure dimension map. A
            # cropped stride-one/offset-zero dimension cannot be represented
            # by that map without an explicit subview, so retain the original
            # slice rather than inventing shape or bounds facts.
            shape = _static_tensor(current)
            output_shape = _static_tensor(op.results[0])
            if all(not isinstance(expr, AffineDimExpr) or
                   shape[i] == output_shape[expr.position]
                   for i, expr in enumerate(expression)):
                admitted = (current, tuple(expression), tuple(slices))
        if admitted is None:
            continue
        current, expression, slices = admitted
        replacement = GenericOp([current], [op.operands[1]], op.regions[0].clone(),
            [AffineMapAttr(AffineMap(rank, 0, tuple(expression))),
             AffineMapAttr(AffineMap.identity(rank))],
            [IteratorTypeAttr.parallel() for _ in range(rank)],
            result_types=[op.results[0].type])
        replacement.attributes.update(op.attributes)
        metadata = [DictionaryAttr({k: v for k, v in view.attributes.items()
                                    if k.startswith('prov.')}) for view in slices]
        if any(metadata):
            replacement.attributes['prov.composed_slice_sources'] = ArrayAttr(metadata)
        op.parent.insert_op_before(replacement, op)
        op.results[0].replace_all_uses_with(replacement.results[0])
        op.parent.erase_op(op)
        removed = []
        for view in slices:
            if not view.results[0].uses:
                removed.append(dict(shape=list(view.results[0].type.get_shape()),
                                    dtype=str(view.results[0].type.element_type)))
                view.parent.erase_op(view)
        records.append(dict(slice_chain_length=len(slices), removed_slices=removed,
            source_shape=list(current.type.get_shape()),
            output_shape=list(replacement.results[0].type.get_shape()),
            input_indexing_map=str(replacement.indexing_maps.data[0]),
            copied_elements=prod(replacement.results[0].type.get_shape())))
    module.verify()
    return dict(schema='submission.source_tensor_views.v1', transformed_copies=len(records),
        removed_slice_buffers=sum(len(r['removed_slices']) for r in records), copies=records,
        semantics='original scalar yield with exact composed static slice/permutation addresses',
        public_abi='unchanged; no argument or return aliases introduced')


def rewrite_text(source):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    module = parse_mlir_text(source)
    report = prepare_source_tensor_views(module)
    return (text(module, generic=True) if report['transformed_copies'] else source), report
