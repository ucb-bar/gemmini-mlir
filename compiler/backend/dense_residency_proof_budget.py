"""Finite proof resources for the optional checked boundary-RHS rewrite.

The incumbent's shorter budget remains its own policy. This optional rewrite
introduces additional counted continuation/residency domains, so it receives a
separate fixed limit. The limit changes no formula, source binding, storage
assignment, selected target premise or refusal behavior.
"""
BOUNDARY_RHS_PROOF_TIMEOUT_MS = 90_000
BOUNDARY_RHS_PROOF_POLICY = 'optional-boundary-rhs-fixed-90s-v1'


def boundary_rhs_proof_budget(incumbent_timeout_ms):
    if type(incumbent_timeout_ms) is not int or incumbent_timeout_ms <= 0:
        raise ValueError('incumbent proof timeout must be a positive integer')
    return BOUNDARY_RHS_PROOF_TIMEOUT_MS
