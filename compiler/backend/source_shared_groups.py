"""Conservative plural source boundaries for independent raw signed products.

Grouping changes only the callback boundary. Each original scalar recurrence,
initializer and result survives extraction; physical reuse is a later decision.
"""
from __future__ import annotations

import json


def _raw_operands(plan):
    from .source_groups import _Plan, _root_inputs, _spec
    from .frontend import _zero, _batched_matmul
    if type(plan) is not _Plan or plan.root.name != 'linalg.generic':
        return None
    if (plan.output is not plan.root.results[0] or len(plan.root.results) != 1
            or plan.attributes.get('output_dtype') != 'i32'
            or plan.attributes.get('epilogue') != [] or plan.constants
            or set(plan.attributes) - {'output_dtype', 'epilogue', '_source_initializer', '_source_batched'}):
        return None
    # Provenance carries no arithmetic semantics. Unknown root annotations must
    # keep their original single-region path instead of disappearing in grouping.
    if any(not key.startswith('prov.') for key in plan.root.attributes):
        return None
    try:
        if plan.attributes.get('_source_batched'):
            operands = _batched_matmul(plan.root)
            if operands is None:
                return None
            lhs, rhs, init = operands
            rank = len(_spec(lhs)['shape'])
            # The actual counted batch emitter admits 1..6 identical leading
            # batch axes. Its shape proof owns all matrix offsets and tails.
            if not 3 <= rank <= 8 or any(len(_spec(value)['shape']) != rank for value in (rhs, init)):
                return None
        else:
            lhs, rhs, init = _root_inputs(plan.root)
            if any(len(_spec(value)['shape']) != 2 for value in (lhs, rhs, init)):
                return None
    except ValueError:
        return None
    if not _zero(init) and init not in plan.inputs:
        return None
    return lhs, rhs, init


def _enclosing(operation, block):
    while operation is not None and operation.parent is not block:
        operation = operation.parent_op()
    return operation


def _combine(plans):
    """Return a closed region only if its final original operation is an anchor."""
    from .host_runtime import DeviceRegion
    from .source_groups import _spec
    roots = tuple(plan.root for plan in plans)
    block = roots[0].parent
    if block is None or any(root.parent is not block for root in roots):
        return None
    members = {op for plan in plans for op in plan.operations}
    if any(op.parent is not block for op in members):
        return None
    order = tuple(block.ops)
    positions = {op: index for index, op in enumerate(order)}
    anchor = max(positions[op] for op in members)
    start = min(positions[op] for op in members)
    # Do not move products across arbitrary host effects or a consumer. Shared
    # scalar/initializer producers outside this interval remain exact captures.
    if any(op not in members for op in order[start:anchor + 1]):
        return None
    outputs = tuple(plan.output for plan in plans)
    inputs = []
    for plan in plans:
        for value in plan.inputs:
            if value.owner not in members and value not in inputs:
                inputs.append(value)
    # Products must be independent, including runtime accumulator initializers.
    from .frontend import _zero
    if any(lhs.owner in members or rhs.owner in members or (not _zero(init) and init.owner in members)
           for lhs, rhs, init in map(_raw_operands, plans)):
        return None
    # Stable original function argument order followed by producer/result order.
    def input_key(value):
        if value in block.args:
            return (0, tuple(block.args).index(value), 0)
        owner = value.owner
        return (1, positions.get(owner, -1), tuple(owner.results).index(value))
    from xdsl.ir import Operation
    if any(value not in block.args and (not isinstance(value.owner, Operation)
                                      or value.owner.parent is not block) for value in inputs):
        return None
    inputs.sort(key=input_key)
    try:
        for value in (*inputs, *outputs):
            _spec(value)
    except ValueError:
        return None
    for op in members:
        local = set(op.walk())
        for nested in op.walk():
            for operand in nested.operands:
                # Region block arguments belong to the scalar recurrence.
                from xdsl.ir import BlockArgument
                if isinstance(operand, BlockArgument) and operand.owner is not block:
                    continue
                if operand.owner not in members and operand.owner not in local and operand not in inputs:
                    return None
        for value in op.results:
            for use in value.uses:
                user = _enclosing(use.operation, block)
                if user in members:
                    continue
                if value not in outputs:
                    # Shared scalar constants may remain in the original host.
                    # An extra live tensor requires its own complete output ABI.
                    from xdsl.dialects.builtin import TensorType
                    if isinstance(value.type, TensorType) or op.name != 'arith.constant':
                        return None
                elif user is None or positions[user] < anchor:
                    return None
    return DeviceRegion(tuple(op for op in order if op in members), tuple(inputs), outputs,
                        reason=json.dumps({'form': 'independent_raw_shared_ssa_products',
                                           'members': len(plans), 'result_boundary': 'all original raw results'}))


def coalesce_raw_regions(module, plans):
    """Coalesce adjacent raw plans sharing exact LHS or RHS SSA; keep singles.

    No names, shapes, producer similarity or quantization values establish
    sharing. The library's stricter zero-splat shared-LHS proposals are consumed
    when their complete source contract applies; other forms use the exact
    typed-plan/closure proof above. Canonical rank-three through rank-eight
    products additionally require frontend._batched_matmul's exact identical
    leading axes/maps/scalar recurrence proof and the counted batch emitter's
    admitted rank. No broadcast or axis permutation is inferred. Runtime initializers remain explicit boundary
    tensors; complete emitted CFG certification is a separate callback obligation.
    No source operations are rewritten here.
    """
    from .host_runtime import DeviceRegion
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
    # This recognizer is a proposal, not target authority; it intentionally
    # returns no proposal for mixed shells, fill seeds and runtime initializers.
    try:
        proposals = find_shared_input_groups(module)
    except ValueError:
        # A proposal budget/refusal does not invalidate the independent raw
        # source proof, and must not break a formerly legal singleton.
        proposals = ()
    functions = [op for op in module.body.block.ops if op.name == 'func.func' and op.body.blocks]
    proposal_operations = tuple(functions[0].body.block.ops) if len(functions) == 1 else ()
    plans = tuple(plans)
    result = []
    index = 0
    while index < len(plans):
        first = plans[index]
        operands = _raw_operands(first)
        best, count = None, 1
        if operands is not None:
            shared_lhs, shared_rhs = operands[:2]
            for end in range(index + 1, len(plans)):
                candidate = _raw_operands(plans[end])
                if candidate is None:
                    break
                shared_lhs = shared_lhs if candidate[0] is shared_lhs else None
                shared_rhs = shared_rhs if candidate[1] is shared_rhs else None
                if shared_lhs is None and shared_rhs is None:
                    break
                region = _combine(plans[index:end + 1])
                if region is None:
                    break
                best, count = region, end - index + 1
        if best is None:
            best = DeviceRegion(first.operations, first.inputs, (first.output,),
                                reason=json.dumps(first.diagnostic, default=str))
        if count > 1:
            roots = {plan.root for plan in plans[index:index + count]}
            matching = [proposal.record() for proposal in proposals
                        if {proposal_operations[i] for i in proposal.contraction_ordinals} == roots]
            if matching:
                from dataclasses import replace
                proof = json.loads(best.reason)
                proof['library_source_proposals'] = matching
                best = replace(best, reason=json.dumps(proof))
        result.append(best)
        index += count
    return tuple(result)
