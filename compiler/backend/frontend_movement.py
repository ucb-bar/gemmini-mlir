"""Prove typed tensor permutations before selecting checked movement emission."""
from __future__ import annotations


def _fresh_destination(init, consumer):
    """Prove an unobserved pure constant fill of fresh static tensor storage.

    A filled initializer can be discarded only when its entire tensor chain is
    private to this full overwrite. Observer and alias uses are refused rather
    than relying on a later dead-code or buffer-alias analysis.
    """
    if getattr(init.owner, 'name', '') == 'tensor.empty':
        return True
    fill = init.owner
    if getattr(fill, 'name', '') != 'linalg.fill':
        return False
    from .frontend import _type
    if len(fill.inputs) != 1 or len(fill.outputs) != 1 or len(fill.results) != 1:
        return False
    scalar, empty = fill.inputs[0], fill.outputs[0]
    if getattr(empty.owner, 'name', '') != 'tensor.empty':
        return False
    if _type(empty) != _type(init) or str(scalar.type) != _type(init)['dtype']:
        return False
    # Constants have no reads/effects; arbitrary scalar producers are outside
    # this deliberately narrow extension, even if their result looks unused.
    if getattr(scalar.owner, 'name', '') != 'arith.constant':
        return False
    if (len(list(init.uses)) != 1 or next(iter(init.uses)).operation is not consumer
            or len(list(empty.uses)) != 1 or next(iter(empty.uses)).operation is not fill):
        return False
    if len(fill.regions) != 1 or len(fill.regions[0].blocks) != 1:
        return False
    body = fill.regions[0].block
    ops = list(body.ops)
    if (len(body.args) != 2 or len(ops) != 1 or ops[0].name != 'linalg.yield'
            or list(ops[0].operands) != [body.args[0]]):
        return False
    scalar.owner.verify()
    empty.owner.verify()
    fill.verify()
    return True


def recognize_permutation(op):
    """Return original input and output-axis permutation, or refuse the operation.

    Only ordinary, static rank 2--4 integer tensors with a fresh tensor.empty
    destination (optionally privately constant-filled) and one exact scalar input yield are admitted. No source view,
    buffer alias, arithmetic, broadcast, reduction or initializer read is inferred.
    """
    if op.name not in ('linalg.transpose', 'linalg.generic'):
        return None
    from .frontend import _type, FrontendError
    from xdsl.ir.affine import AffineDimExpr
    from xdsl.utils.exceptions import VerifyException
    try:
        if len(op.operands) != 2 or len(op.results) != 1:
            return None
        src, init = op.operands
        if src is init or not _fresh_destination(init, op):
            return None
        a, b, c = _type(src), _type(init), _type(op.results[0])
        rank = len(a['shape'])
        if not 2 <= rank <= 4 or a['dtype'] not in ('i8', 'i32') or c['dtype'] != a['dtype'] or b != c:
            return None
        if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
            return None
        body = op.regions[0].block
        ops = list(body.ops)
        if (len(body.args) != 2 or str(body.args[0].type) != a['dtype']
                or str(body.args[1].type) != c['dtype'] or len(ops) != 1
                or ops[0].name != 'linalg.yield' or list(ops[0].operands) != [body.args[0]]):
            return None
        # Include the named op's dialect verifier; do not use its spelling as
        # proof that permutation, result shape or scalar body is well formed.
        op.verify()
        if op.name == 'linalg.transpose':
            attr = op.properties.get('permutation')
            if attr is None or not hasattr(attr, 'get_values'):
                return None
            perm = list(attr.get_values())
        else:
            if len(op.inputs) != 1 or len(op.outputs) != 1:
                return None
            if len(op.iterator_types) != rank or any(a.data.value != 'parallel' for a in op.iterator_types):
                return None
            maps = tuple(a.data for a in op.indexing_maps)
            if len(maps) != 2:
                return None
            axes = []
            for m in maps:
                if m.num_dims != rank or m.num_symbols != 0 or len(m.results) != rank:
                    return None
                if any(not isinstance(e, AffineDimExpr) for e in m.results):
                    return None
                order = [e.position for e in m.results]
                if sorted(order) != list(range(rank)):
                    return None
                axes.append(order)
            # Maps send iteration coordinates to input/output coordinates.
            # Output axis j uses input axis input_map.index(output_map[j]).
            perm = [axes[0].index(axis) for axis in axes[1]]
        if any(type(i) is not int for i in perm) or sorted(perm) != list(range(rank)):
            return None
        if c['shape'] != [a['shape'][i] for i in perm]:
            return None
        from .transpose import plan_transpose
        plan_transpose(a['shape'], perm, a['dtype'])
        return src, perm
    except (ValueError, FrontendError, VerifyException):
        return None
