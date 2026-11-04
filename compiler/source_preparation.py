"""Reuse exact immutable source layouts before ordinary region lowering.

No physical cache, pointer alias, arithmetic reassociation, or call purity is
inferred here. The consumed product is a changed tensor SSA graph with every
original observer redirected to a dominating identical tensor computation.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import prod

from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, NoneAttr, TensorType, UnitAttr
from xdsl.rewriter import Rewriter


@dataclass(frozen=True)
class LayoutPreparation:
    module: object
    witnesses: tuple[dict, ...]
    source_ordinals: tuple[int, ...]
    aliases: tuple[dict, ...] = ()


def _owner(module, entry='forward'):
    owners = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)
              and op.sym_name.data == entry and len(op.body.blocks) == 1]
    if len(owners) != 1:
        raise ValueError(f'source preparation requires one closed {entry} block')
    return owners[0]


def _static(value):
    ty = value.type
    return (isinstance(ty, TensorType) and isinstance(ty.encoding, NoneAttr)
            and all(dim > 0 for dim in ty.get_shape()))


def _layout_key(op):
    from merlin.llvmlower.tensor_preparation_identity import canonical_tensor_read_view, validate_tensor_read_view
    if any(not name.startswith('prov.') for name in op.attributes):
        return None
    if isinstance(op, tensor.ExtractSliceOp):
        if len(op.operands) != 1 or len(op.results) != 1 or not _static(op.results[0]):
            return None
        try:
            view = canonical_tensor_read_view(op.results[0])
            validate_tensor_read_view(view)
        except ValueError:
            return None
        return ('slice', view.identity)
    if op.name == 'linalg.transpose':
        if (len(op.operands) != 2 or len(op.results) != 1
                or not all(_static(v) for v in (*op.operands, *op.results))
                or set(op.properties) != {'permutation'} or len(op.regions) != 1):
            return None
        body = op.regions[0].block
        operations = tuple(body.ops)
        if (len(body.args) != 2 or len(operations) != 1 or operations[0].name != 'linalg.yield'
                or tuple(operations[0].operands) != (body.args[0],)):
            return None
        # Named transpose overwrites its entire result; the init tensor does not
        # contribute values. The actual source and permutation establish identity.
        return ('transpose', op.operands[0], tuple(op.result_types), op.properties['permutation'])
    return None


def _reshape_source(op):
    """Exact row-major static reshape domain; no encoded/dynamic views."""
    properties = {
        tensor.CollapseShapeOp: {'reassociation'},
        tensor.ExpandShapeOp: {'reassociation', 'static_output_shape'},
        tensor.CastOp: set(),
    }
    if (type(op) not in properties or len(op.operands) != 1 or len(op.results) != 1
            or set(op.properties) != properties[type(op)] or op.regions or op.successors
            or any(not key.startswith('prov.') for key in op.attributes)
            or not all(_static(v) for v in (*op.operands, *op.results))):
        return None
    before, after = op.operands[0].type, op.results[0].type
    if before.element_type != after.element_type or prod(before.get_shape()) != prod(after.get_shape()):
        return None
    if type(op) is tensor.CastOp and before != after:
        return None
    if type(op) is not tensor.CastOp:
        rank = len(before.get_shape()) if type(op) is tensor.CollapseShapeOp else len(after.get_shape())
        axes = tuple(axis.value.data for group in op.properties['reassociation'] for axis in group)
        if axes != tuple(range(rank)):
            return None
    return op.operands[0]


def prepare_source_layouts(module, *, entry='forward', protected_ordinals=()):
    """Return a verified clone, exact reuse witnesses and original op mapping.

    ``source_ordinals[i]`` identifies the original direct forward operation for
    transformed operation i. Consumers of attribution must use that mapping;
    witnesses retain all removed original operation ordinals and source attrs.
    Layouts are immutable tensors only. No memrefs or external helpers qualify.
    Original writers of returned values are always retained. Callers may retain
    additional writers with ``protected_ordinals``, using original direct forward
    operation indices. Their operands can still consume reused internal layouts.
    """
    from merlin.transforms.tensor_views import canonicalize_tensor_views
    module.verify()
    prepared = module.clone()
    original_ops = tuple(_owner(module, entry).body.block.ops)
    block = _owner(prepared, entry).body.block
    originals = dict(zip(tuple(block.ops), range(len(original_ops)), strict=True))
    protected = set(protected_ordinals)
    if any(type(index) is not int or not 0 <= index < len(original_ops)
           for index in protected):
        raise ValueError('protected ordinals must index original forward operations')
    protected.update(originals[value.owner] for op in block.ops
                     if isinstance(op, func.ReturnOp) for value in op.operands
                     if value.owner in originals)
    # Both the public view pass and our layout matcher deliberately exclude
    # operations with semantic/unknown attributes. Use that conservative public
    # boundary to protect writers, without modifying the shared pass or leaving
    # metadata in the consumed graph. Select a fresh attribute name so existing
    # source attributes are preserved exactly.
    marker = 'merlin.preparation_protected'
    while any(marker in op.attributes for op in prepared.walk()):
        marker += '_'
    protected_ops = tuple(op for op, index in originals.items() if index in protected)
    for op in protected_ops:
        op.attributes[marker] = UnitAttr()
    witnesses = []
    while True:
        before = tuple(block.ops)
        # Static collapse/expand preserve the element sequence. A chain ending
        # at an earlier identical tensor type therefore computes the earlier
        # value exactly, even when the two endpoint views have different keys.
        # Restrict replacements to original operation results so existing source
        # ownership/alias consumers need no invented argument-writer identity.
        for op in tuple(block.ops):
            value = _reshape_source(op)
            path = []
            while value is not None and value.owner in originals:
                if value.type == op.results[0].type:
                    retained = value.owner
                    if len(retained.results) == 1:
                        lineage = tuple(retained.attributes.get('prov.preparation_sources',
                            ArrayAttr([DictionaryAttr({key: attr for key, attr in retained.attributes.items()
                                                      if key != marker})])))
                        lineage = (*lineage, DictionaryAttr(dict(op.attributes)))
                        retained.attributes['prov.preparation_sources'] = ArrayAttr(lineage)
                        for name in ('prov.source_node_ids', 'prov.origin_node_ids'):
                            attributes = [record.data.get(name) for record in lineage]
                            if all(v is None or isinstance(v, ArrayAttr) for v in attributes):
                                union = tuple(dict.fromkeys(item for values in attributes
                                    if values is not None for item in values))
                                if union:
                                    retained.attributes[name] = ArrayAttr(union)
                        witnesses.append({'operation': op.name,
                            'source_ordinal': originals[op], 'retained_ordinal': originals[retained],
                            'result_type': str(op.results[0].type), 'inverse_static_reshape': True,
                            'through_ordinals': path,
                            'source_attributes': {key: str(v) for key, v in op.attributes.items()}})
                        Rewriter.replace_op(op, [], new_results=[value])
                    break
                path.append(originals[value.owner])
                value = _reshape_source(value.owner)
        view_positions = dict(enumerate(prepared.walk()))
        view_witnesses = canonicalize_tensor_views(prepared)
        view_retained = {view_positions[w.operation_index]: view_positions[w.retained_index]
                         for w in view_witnesses}
        known = {}
        for op in tuple(block.ops):
            key = _layout_key(op)
            if key is None:
                continue
            retained = known.get(key)
            if retained is None:
                known[key] = op
                continue
            lineage = tuple(retained.attributes.get('prov.preparation_sources',
                ArrayAttr([DictionaryAttr(dict(retained.attributes))])))
            lineage += (DictionaryAttr(dict(op.attributes)),)
            retained.attributes['prov.preparation_sources'] = ArrayAttr(lineage)
            for name in ('prov.source_node_ids', 'prov.origin_node_ids'):
                attributes = [record.data.get(name) for record in lineage]
                if all(value is None or isinstance(value, ArrayAttr) for value in attributes):
                    union = tuple(dict.fromkeys(item for value in attributes if value is not None for item in value))
                    if union:
                        retained.attributes[name] = ArrayAttr(union)
            witnesses.append({'operation': op.name, 'source_ordinal': originals[op],
                              'retained_ordinal': originals[retained], 'result_type': str(op.results[0].type),
                              'source_attributes': {key: str(value) for key, value in op.attributes.items()}})
            Rewriter.replace_op(op, [], new_results=list(retained.results))
        after = tuple(block.ops)
        erased_views = set(before) - set(after) - {op for op in before if originals[op] in {
            w['source_ordinal'] for w in witnesses}}
        for op in before:
            if op in erased_views:
                witnesses.append({'operation': op.name, 'source_ordinal': originals[op],
                                  'source_attributes': {key: str(value) for key, value in op.attributes.items()},
                                  'retained_ordinal': originals[view_retained[op]], 'public_view_cse': True})
        if after == before:
            break
    for op in protected_ops:
        del op.attributes[marker]
    prepared.verify()
    replacements = {w['source_ordinal']: w['retained_ordinal'] for w in witnesses
                    if 'retained_ordinal' in w}
    aliases = []
    for source_ordinal, retained in sorted(replacements.items()):
        while retained in replacements:
            retained = replacements[retained]
        aliases.append({'source_ordinal': source_ordinal, 'result_index': 0,
                        'retained_ordinal': retained, 'retained_result_index': 0})
    return LayoutPreparation(prepared, tuple(witnesses), tuple(originals[op] for op in block.ops), tuple(aliases))
