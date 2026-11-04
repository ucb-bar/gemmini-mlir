"""Keep unavailable optional schedule proofs separate from invalid programs.

The caller's ordinary dense dispatcher independently constructs and verifies a
primitive compact schedule after this noncapacity refusal. No unchecked kernel
is returned, no capacity or numerical admission is inferred, and invariant
failures (ordinary ValueError) must still propagate.
"""
from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable


def optional_verified_dense_plan(planner, *args, **kwargs):
    """Return a checked plan or a distinct reason to try the checked baseline."""
    try:
        return planner(*args, **kwargs)
    except (DenseVerificationUnavailable, ModuleNotFoundError) as error:
        # Only an absent top-level solver is an unavailable optional proof.
        # Missing z3 submodules, transitive dependencies and all other imports
        # remain implementation/environment failures and must propagate.
        if isinstance(error, ModuleNotFoundError) and error.name != 'z3':
            raise
        detail = ('symbolic dense checking requires unavailable top-level z3'
                  if isinstance(error, ModuleNotFoundError) else str(error))
        reason = 'optional dense schedule proof unavailable; use independently checked compact schedule: ' + detail
        return None, reason, {
            'schema': 'authored_dense_verification_fallback.v1',
            'selection': 'unavailable',
            'proof_status': 'unavailable',
            'reason': detail,
            'emitted_kernel': None,
            'fallback': 'independently_checked_compact',
            'admission_claimed': False,
        }
