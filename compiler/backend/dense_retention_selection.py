"""Offline-frozen traffic hypothesis over legal retained operand choices.

The full-K scratchpad stores bound both candidate families. Reusing LHS across
N groups and reusing RHS across M groups trade repeated source bytes; the
shared planner counts the separate remainder families in both. Strictly less
requested operand traffic is the hypothesis, not a cycle estimate or an array
persistence permission. A tie keeps the exact incumbent LHS block and order.
"""
from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_choices


def select_retained_operand_by_traffic(problem, resources, storage, incumbent_block):
    """Return (role, actual legal block, detached receipt) without timing work.

    Compare the current LHS-retained block with the best legal RHS-retained
    candidate. All returned blocks originate in the same original planning
    domain and assigned physical stores. The normal source/epoch checker must
    subsequently consume the selected block, hoists and output-group order.
    """
    choices = dense_retained_choices(problem, resources, storage)
    incumbent = next((choice for choice in choices
                      if choice.role == 'lhs' and choice.block == incumbent_block), None)
    if incumbent is None:
        raise ValueError('traffic retention needs the exact admitted incumbent LHS block')
    weights = tuple(choice for choice in choices if choice.role == 'weight')
    candidate = min(weights, key=lambda choice: (
        choice.requested_operand_bytes, choice.lhs_loads + choice.rhs_loads,
        -choice.block.m*choice.block.n, choice.block.m, choice.block.n)) if weights else None
    improves = candidate is not None and candidate.requested_operand_bytes < incumbent.requested_operand_bytes
    selected = candidate if improves else incumbent

    def describe(choice):
        if choice is None:
            return None
        return dict(role=choice.role, block_tiles=[choice.block.m, choice.block.k, choice.block.n],
                    requested_operand_bytes=choice.requested_operand_bytes,
                    lhs_tile_loads=choice.lhs_loads, rhs_tile_loads=choice.rhs_loads)

    receipt = dict(schema='authored_dense_retention_traffic.v1',
        hypothesis='strictly fewer original-source operand bytes, including separate tail families',
        predicate='weight.requested_operand_bytes < incumbent_lhs.requested_operand_bytes',
        selection=selected.role, incumbent=describe(incumbent), candidate=describe(candidate),
        selected=describe(selected), strict_advantage=improves,
        legality='original full-K physical stores and dense_retained_choices; no new array or queue law',
        performance='offline hypothesis; no timing or profiling during compilation')
    return selected.role, selected.block, receipt
