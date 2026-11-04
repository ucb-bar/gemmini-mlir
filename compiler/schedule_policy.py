"""Geometry-only strategy requests; hardware storage legality stays in lowering.

This optional regional policy is provisional. Explicit nonincumbent schedule
knobs win as a pair, allowing independent baseline and alternative trials.
"""

REGIONAL_MIN_MACS = 1 << 20


def select_schedule(M, K, N, *, policy='incumbent',
                    matmul_order='output_tile', lhs_residency='stream'):
    """Return serializable requested choices and their geometry-only reason.

    Lowering may reject LHS residency for insufficient free scratch rows; the
    resulting instruction attributes record the actual path after that check.
    Pooling is handled before this function by its separate legal schedule.
    """
    if policy not in ('incumbent', 'regional'):
        raise ValueError(f'unknown schedule policy {policy}')
    if matmul_order not in ('output_tile', 'm_panel'):
        raise ValueError(f'unknown matmul order {matmul_order}')
    if lhs_residency not in ('stream', 'panel'):
        raise ValueError(f'unknown LHS residency {lhs_residency}')
    if min(M, K, N) < 0:
        raise ValueError('negative matrix geometry')
    work = M * K * N
    reason = 'incumbent'
    if policy == 'regional':
        if (matmul_order, lhs_residency) != ('output_tile', 'stream'):
            reason = 'explicit_schedule_override'
        elif K > 0 and N > 16 and work >= REGIONAL_MIN_MACS:
            lhs_residency = 'panel'
            matmul_order = 'm_panel' if M > 16 else 'output_tile'
            reason = 'regional_large_contraction'
        else:
            reason = 'regional_small_or_narrow'
    return {'policy': policy, 'matmul_order': matmul_order,
            'lhs_residency': lhs_residency, 'reason': reason,
            'geometry': [M, K, N], 'useful_macs': work,
            'regional_min_macs': REGIONAL_MIN_MACS}
