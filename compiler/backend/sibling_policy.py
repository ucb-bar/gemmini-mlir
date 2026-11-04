"""Provisional sibling profitability policy, after source-semantic legality.

Version 1 is pinned before independent geometry validation. Three complete-call
modeled comparisons support two-member groups at M=16/17, K=17/32/33 and
N=15..24 (21..26 percent reductions). A large three-member decode comparison
regressed 5.42 percent, with 19.2 percent more modeled shared-memory activity.
The rounded bounds below are an explicit extrapolation, not a measured speedup
claim. Keep the compiler default disabled pending independent paired evidence.
"""

POLICY_VERSION = 'small_two_siblings_v1'
_TILE = 16
_ACCUMULATOR_ROWS = 512
_SCRATCHPAD_ROWS = 4096


def fusion_decision(members):
    """Return JSON-compatible policy facts for already legal i8/i8->i32 groups.

    This function does not establish source legality, tensor independence,
    quantization compatibility or ownership; graph.py establishes those first.
    No source identities, tensor names or timing feedback are consulted.
    """
    result = {'policy': POLICY_VERSION, 'profitable': False}
    if len(members) != 2:
        return {**result, 'reason': 'requires_two_members'}
    try:
        dims = [(x['m'], x['k'], x['n']) for x in members]
        if any(type(d) is not int or d <= 0 for shape in dims for d in shape):
            return {**result, 'reason': 'requires_static_positive_dimensions'}
        m, k, _ = dims[0]
        if any((mm, kk) != (m, k) for mm, kk, _ in dims):
            return {**result, 'reason': 'requires_shared_m_k'}
    except (KeyError, TypeError):
        return {**result, 'reason': 'requires_static_positive_dimensions'}
    ns = [n for _, _, n in dims]
    rows = (m + _TILE - 1) // _TILE
    # The footprint is an upper bound on unique i8 input bytes, not a claim
    # that the entire input resides in a particular cache or scratchpad.
    facts = {'m': m, 'k': k, 'ns': ns, 'members': len(members),
             'input_bytes': m * k + k * sum(ns),
             'accumulator_rows': len(members) * rows * _TILE,
             'scratchpad_rows': (rows + 1) * _TILE}
    result.update(facts)
    if m > 32 or k > 64 or max(ns) > 32:
        return {**result, 'reason': 'outside_small_geometry'}
    if (facts['input_bytes'] > 6144
            or facts['accumulator_rows'] > _ACCUMULATOR_ROWS
            or facts['scratchpad_rows'] > _SCRATCHPAD_ROWS):
        return {**result, 'reason': 'outside_resource_envelope'}
    return {**result, 'profitable': True, 'reason': 'within_provisional_small_envelope'}


def fusion_profitable(members):
    return fusion_decision(members)['profitable']
