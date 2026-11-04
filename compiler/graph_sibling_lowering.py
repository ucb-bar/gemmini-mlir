"""Proven independent single-row MACs packed into a shared device matrix.

Only source SSA identity and typed scalar semantics determine grouping. Tensor
values move exclusively through ordinary Gemmini DMA; host code forms addresses.
"""
from xdsl.dialects.builtin import TensorType, StringAttr
from merlin.targetgen.contract import linalg_iface as li
from isa_encoder import ACC_ADDR, ACC_FULL, config_load, config_store
from matmul_lowering import emit_matmul, packed, extent

MAX_PRIVATE_BYTES = 64 * 1024 * 1024


def _candidate(op, max_rows=1):
    if li._op_name(op) != 'linalg.generic' or len(op.results) != 1:
        return None
    ins, outs = li._ins_outs(op)
    if len(ins) != 2 or len(outs) != 1:
        return None
    values = [*ins, *outs, op.results[0]]
    if any(not isinstance(v.type, TensorType) or len(v.type.get_shape()) != 2 for v in values):
        return None
    if [str(v.type.element_type) for v in values] != ['i8', 'i8', 'i32', 'i32']:
        return None
    m, k = ins[0].type.get_shape(); rk, n = ins[1].type.get_shape()
    if not 1 <= m <= max_rows or k <= 0 or n <= 0 or rk != k:
        return None
    if any(tuple(v.type.get_shape()) != (m, n) for v in [outs[0], op.results[0]]):
        return None
    initial = outs[0].owner
    if getattr(initial, 'name', None) not in ('linalg.fill', 'tensor.splat'):
        return None
    if li._scalar_const(initial.operands[0]) != 0:
        return None
    maps = [str(x).replace(' ', '') for x in op.indexing_maps]
    if maps != ['affine_map<(d0,d1,d2)->(d0,d2)>',
                'affine_map<(d0,d1,d2)->(d2,d1)>',
                'affine_map<(d0,d1,d2)->(d0,d1)>']:
        return None
    if [getattr(x.data, 'value', None) for x in op.iterator_types] != ['parallel', 'parallel', 'reduction']:
        return None
    body = op.regions[0].blocks[0]; scalar = list(body.ops)
    if [li._op_name(x) for x in scalar] != ['arith.extsi', 'arith.extsi', 'arith.muli', 'arith.addi', 'linalg.yield']:
        return None
    x, y, mul, add, yielded = scalar
    if list(x.operands) != [body.args[0]] or list(y.operands) != [body.args[1]]:
        return None
    if any(str(v.type) != 'i32' for v in [x.results[0], y.results[0], mul.results[0], add.results[0]]):
        return None
    if list(mul.operands) != [x.results[0], y.results[0]]:
        return None
    if list(add.operands) not in ([body.args[2], mul.results[0]], [mul.results[0], body.args[2]]):
        return None
    if list(yielded.operands) != [add.results[0]]:
        return None
    return {'lhs': ins[0], 'rhs': ins[1], 'm': m, 'k': k, 'n': n}


def sibling_groups(operations, *, forbidden_indices=(), max_rows=1):
    """Return disjoint tail-indexed groups of 2..16 proven sibling MACs.

    Members share M in [1,max_rows] (bounded at 16), K, N and RHS identity.

    Pass indices participating in any other fusion as forbidden. Inputs must
    dominate the first producer, and all result uses must remain at/after tail.
    Unknown/effectful crossed operations prevent grouping rather than guessing.
    """
    if not 1 <= max_rows <= 16:
        raise ValueError('sibling max_rows must lie in [1,16]')
    operations = list(operations)
    indices = {op: i for i, op in enumerate(operations)}
    forbidden = set(forbidden_indices)
    candidates = {i: c for i, op in enumerate(operations)
                  if i not in forbidden and (c := _candidate(op, max_rows=max_rows)) is not None}
    families = {}
    for i, c in candidates.items():
        families.setdefault((c['rhs'], c['m'], c['k'], c['n']), []).append(i)

    def input_dominates(value, first):
        owner = value.owner
        if owner in indices:
            return indices[owner] < first
        # Entry-block arguments dominate every operation; captures from an
        # unrecognized enclosing region are excluded.
        return owner is operations[first].parent_block()

    def legal(members):
        first, tail = members[0], members[-1]
        c = candidates[first]; h = len(members)
        if h > 16 or h * c['m'] * (c['k'] + 4 * c['n']) > MAX_PRIVATE_BYTES:
            return False
        for i in members:
            if not all(input_dominates(candidates[i][key], first) for key in ('lhs', 'rhs')):
                return False
            for use in operations[i].results[0].uses:
                user = use.operation
                while user is not None and user not in indices:
                    user = user.parent_op()
                if user is None or indices[user] < tail:
                    return False
        for i in range(first, tail + 1):
            if i in forbidden:
                return False
            op = operations[i]; kind = li._op_name(op)
            if i in candidates or kind in ('tensor.empty', 'arith.constant', 'tensor.splat'):
                continue
            if kind == 'linalg.fill' and len(op.operands) == 2 and li._scalar_const(op.operands[0]) is not None:
                continue
            return False
        return True

    groups = {}
    def finish(members):
        if len(members) < 2 or not legal(members):
            return
        c = candidates[members[0]]
        groups[members[-1]] = {'members': [(i, operations[i]) for i in members],
                              'lhs': [candidates[i]['lhs'] for i in members],
                              'rhs': c['rhs'], 'm': c['m'], 'k': c['k'], 'n': c['n']}
    for family in families.values():
        pending = []
        for index in family:
            proposed = pending + [index]
            if pending and (len(proposed) > 16 or not legal(proposed)):
                finish(pending); pending = [index]
            else:
                pending = proposed
        finish(pending)
    return groups


def emit_sibling_group(e, host, group):
    """DMA-pack rows, emit one shared MAC, and DMA-scatter all full i32 rows."""
    members = group['members']; h = len(members); k = group['k']; n = group['n']
    m = group.get('m', 1)
    if not 2 <= h <= 16 or not 1 <= m <= 16 or h * m * (k + 4 * n) > MAX_PRIVATE_BYTES:
        raise ValueError('sibling group exceeds private storage bounds')
    # Use graph's private allocator so byte accounting includes these buffers;
    # anonymous temporaries are intentionally excluded from graph slot reuse.
    a = host.allocate(h * m * k, 'i8'); c = host.allocate(h * m * n, 'i32')
    b = host.values[group['rhs']].ptr
    for row, ((index, op), lhs) in enumerate(zip(members, group['lhs'], strict=True)):
        region = li._prov(op).get('region_id')
        with e.task(0, source_op_index=index, region_id=region):
            source = host.values[lhs].ptr
            e.insn('config', *config_load(k, state_id=0))
            e.insn('config', *config_store(k))
            def pack(col):
                width = extent(e, k, col)
                src = e.ptr(source, col)
                dst = e.ptr(a, e.b.add_i(e.value(row * m * k), col))
                e.insn('mvin', e.address(src), packed(e, 0, m, width))
                e.insn('mvout', e.address(dst), packed(e, 0, m, width))
            e.for_loop(0, k, 16, pack)
    e.fence()
    source_indices = [index for index, _ in members]
    first_index, first_op = members[0]
    original_insn = e.insn
    def tagged_insn(name, *args, **kwargs):
        command = original_insn(name, *args, **kwargs)
        if name in ('preload', 'compute', 'compute_and_stay'):
            command.attributes['merlin.sibling_source_op_indices'] = StringAttr(','.join(map(str, source_indices)))
            command.attributes['merlin.schedule_transform'] = StringAttr('shared_rhs_sibling_rows')
        return command
    e.insn = tagged_insn
    try:
        with e.task(0, source_op_index=first_index, region_id=li._prov(first_op).get('region_id')):
            emit_matmul(e, a, b, c, h * m, k, n, attrs={'output_dtype': 'i32'},
                        compute_provenance={'task_index': 0, 'source_op_index': first_index,
                                            'region_id': li._prov(first_op).get('region_id')})
    finally:
        e.insn = original_insn
    e.fence()
    for row, (index, op) in enumerate(members):
        with e.task(0, source_op_index=index, region_id=li._prov(op).get('region_id')):
            result = host.result(op.results[0])
            e.insn('config', *config_load(n * 4, state_id=0))
            e.insn('config', *config_store(n * 4))
            def scatter(col):
                width = extent(e, n, col)
                src = e.ptr(c, e.b.add_i(e.value(row * m * n), col), 'i32')
                dst = e.ptr(result.ptr, col, 'i32')
                e.insn('mvin', e.address(src), packed(e, ACC_ADDR, m, width))
                e.insn('mvout', e.address(dst), packed(e, ACC_ADDR | ACC_FULL, m, width))
            e.for_loop(0, n, 16, scatter)
            e.fence()
    metadata = {'source_op_indices': source_indices, 'tail_source_op_index': members[-1][0],
            'rows': h * m, 'k': k, 'n': n, 'private_bytes': h * m * (k + 4 * n),
            'proof': 'exact signed zero-init i8 MAC; shared RHS SSA; dominating inputs; independent results; pure crossed operations',
            'packing': 'ordinary scratchpad DMA; full-i32 accumulator DMA scatter',
            'preserved_output_count': h}

    if m > 1:
        metadata.update(member_rows=m, sibling_count=h)
    return metadata
