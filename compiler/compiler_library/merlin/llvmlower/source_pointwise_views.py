"""Evaluate a private pointwise producer in its final permutation layout.

Only complete row-major flatten/expand views and a permutation are crossed.
The scalar program remains unchanged and executes once per original element;
the removed producer and views must have no independent live observers.
"""
from __future__ import annotations


def prepare_source_pointwise_views(module):
    from math import prod
    from xdsl.dialects.builtin import AffineMapAttr, DictionaryAttr
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir.affine import AffineExpr, AffineMap
    from .host_pointwise_fusion import _pointwise, _transparent
    from .source_tensor_views import _static_tensor, _permutation

    module.verify()
    records = []
    for transpose in tuple(module.walk()):
        if getattr(transpose, 'name', '') != 'linalg.transpose':
            continue
        permutation = _permutation(transpose)
        if permutation is None:
            continue
        expand = transpose.operands[0].owner
        if (getattr(expand, 'name', '') != 'tensor.expand_shape'
                or len(expand.operands) != 1 or len(expand.results) != 1):
            continue
        collapse = expand.operands[0].owner
        if (getattr(collapse, 'name', '') != 'tensor.collapse_shape'
                or len(collapse.operands) != 1 or len(collapse.results) != 1
                or len(_static_tensor(collapse.results[0]) or ()) != 1):
            continue
        producer = collapse.operands[0].owner
        if not isinstance(producer, GenericOp) or not _pointwise(producer):
            continue
        shape = _static_tensor(producer.results[0])
        expanded = _static_tensor(expand.results[0])
        if shape is None or expanded is None or prod(shape) != prod(expanded):
            continue
        if (producer.indexing_maps.data[-1].data != AffineMap.identity(len(shape))
                or len(producer.iterator_types) != len(shape)
                or any(member.parent is not transpose.parent for member in (producer, collapse, expand))):
            continue
        members = (producer, collapse, expand)
        if any(len(tuple(member.results[0].uses)) != 1 for member in members):
            continue
        allowed = {'tensor.expand_shape': {'reassociation', 'static_output_shape'},
                   'tensor.collapse_shape': {'reassociation'}}
        if any(key not in allowed[view.name] and not key.startswith('prov.')
               for view in (collapse, expand) for key in (*view.attributes, *view.properties)):
            continue
        ops = tuple(transpose.parent.ops)
        crossed = ops[ops.index(producer)+1:ops.index(transpose)]
        if any(op not in members and not _transparent(op) for op in crossed):
            continue
        # Inverse permutation recovers the expanded row-major coordinate.
        rank = len(permutation)
        coordinates = [AffineExpr.dimension(permutation.index(i)) for i in range(rank)]
        flat = AffineExpr.constant(0)
        for coordinate, extent in zip(coordinates, expanded, strict=True):
            flat = flat * extent + coordinate
        source_coordinates = []
        for index, extent in enumerate(shape):
            stride = prod(shape[index+1:])
            coordinate = flat // stride if stride != 1 else flat
            if index:
                coordinate = coordinate % extent
            source_coordinates.append(coordinate)
        view_map = AffineMap(rank, 0, tuple(source_coordinates))
        maps = [AffineMapAttr(mapping.data.compose(view_map))
                for mapping in producer.indexing_maps.data[:-1]]
        maps.append(AffineMapAttr(AffineMap.identity(rank)))
        replacement = GenericOp(list(producer.inputs), [transpose.operands[1]],
            producer.body.clone(), maps, [IteratorTypeAttr.parallel()]*rank,
            result_types=[transpose.results[0].type])
        replacement.attributes.update(producer.attributes)
        replacement.attributes['prov.composed_layout_sources'] = DictionaryAttr({
            'collapse': DictionaryAttr(dict(collapse.attributes)),
            'expand': DictionaryAttr(dict(expand.attributes)),
            'transpose': DictionaryAttr(dict(transpose.attributes))})
        block = transpose.parent
        block.insert_op_before(replacement, transpose)
        transpose.results[0].replace_all_uses_with(replacement.results[0])
        block.erase_op(transpose)
        for member in reversed(members):
            block.erase_op(member)
        records.append(dict(source_shape=list(shape), output_shape=list(_static_tensor(replacement.results[0])),
            elements=prod(shape), scalar_program='unchanged; one evaluation per original element',
            input_indexing_maps=[str(mapping) for mapping in maps[:-1]]))
    module.verify()
    return dict(schema='merlin.source_pointwise_views.v1', transformed_producers=len(records),
                producers=records, public_abi='unchanged; all independent live observers retained')


def rewrite_text(source):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    module = parse_mlir_text(source)
    report = prepare_source_pointwise_views(module)
    return (text(module, generic=True) if report['transformed_producers'] else source), report
