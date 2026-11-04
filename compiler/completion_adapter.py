"""Gemmini binding for generic checked K-stream completion composition.

The public entry is the production entry: it returns the result the dense
emitter consumes. A plain premises dictionary is interpreted strictly (only
literal True proves a flag). Unproved contexts keep the original checked route.
"""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

_LIBRARY = str(Path(__file__).resolve().parent / 'compiler_library')
if _LIBRARY not in sys.path:
    sys.path.insert(0, _LIBRARY)

from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw, STREAM_FLAGS, EPOCH_FLAGS, _drains
from merlin.compile.scheduling.dense_completion_composition import (
    _admission, _storage, effective_completion_admission, prove_dense_completion_scope, schedule_dense_completion_epochs,
)


def _get(context, key, default=None):
    return context.get(key, default) if isinstance(context, dict) else getattr(context, key, default)


def _law(context):
    adapter = _get(context, 'adapter')
    premises = _get(context, 'premises', {})
    if type(premises) is not dict:
        raise ValueError('completion premises must be a plain dictionary')
    admission = effective_completion_admission(_get(context, 'admission'))
    if adapter is None or not all(hasattr(adapter, name) for name in ('configuration_identity','numerical_policy_identity')):
        raise ValueError('completion composition has no original adapter identity')
    return DensePanelEpochLaw(adapter.configuration_identity, adapter.numerical_policy_identity,
        admission.qualification_identity + ':checked-completion-composition',
        **{name: premises.get(name) is True for name in EPOCH_FLAGS})


def _fallback(original, reason, status='safe_unchanged'):
    original.revalidate()
    original.realization['completion_extension'] = dict(status=status, reason=reason,
        original_kernel_sha256=original.kernel.digest(), actual_kernel_sha256=original.kernel.digest(),
        removed_dynamic_drains=0, dynamic_drains=_drains(original.kernel.body))
    return original


def apply_completion(original, context):
    """Return the actual checked emission result, or the valid original route.

    Context fields: problem, block, resources, storage, adapter, admission,
    premises; optional issued_scope (exact source-bound count/progress proof),
    static_budget, timeout_ms, max_read_instances. No shape-name dispatch or
    inferred missing ordering flags. Unexpected implementation errors propagate.
    """
    original.revalidate()
    try:
        if _get(context, 'program_proof_refusal'):
            return _fallback(original, _get(context, 'program_proof_refusal'), 'proof_refused')
        law = _law(context)
        if law.unknown(STREAM_FLAGS):
            return _fallback(original, 'missing completion premises: ' + ', '.join(law.unknown(STREAM_FLAGS)))
        admission = effective_completion_admission(_get(context, 'admission'))
        scope = _get(context, 'issued_scope')
        base = _admission(original)
        if scope is None and admission != base:
            # Minimal contract callers supply an issued ceiling, not a private
            # certificate representation. Prove the exact scope here, within
            # that ceiling; never silently raise a caller's insufficient cap.
            bound, proof = prove_dense_completion_scope(original,
                *(_get(context, name) for name in ('problem','block','resources','storage','adapter')), law)
            if (replace(admission, max_issued_compute_pairs=base.max_issued_compute_pairs,
                        max_issued_loads=base.max_issued_loads) != base
                or admission.max_issued_compute_pairs < bound.max_issued_compute_pairs
                or admission.max_issued_loads < bound.max_issued_loads):
                raise ValueError('supplied issued budgets or pending/configuration contract do not cover the proved completion scope')
            admission, scope = bound, proof
        result = schedule_dense_completion_epochs(original,
            *(_get(context, name) for name in ('problem','block','resources','storage','adapter')),
            admission, law, scope=scope,
            static_budget=_get(context, 'static_budget',128),
            timeout_ms=_get(context, 'timeout_ms',30000),
            max_read_instances=_get(context, 'max_read_instances',4096))
        # Bind the executable owner of this adapter as well as the shared core.
        result.realization['completion_extension']['adapter_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
        return result
    except DenseVerificationUnavailable as exc:
        return _fallback(original, str(exc), 'proof_unavailable')
    except ValueError as exc:
        status = 'proof_unavailable' if any(term in str(exc) for term in ('proof budget exhausted', 'consumption proof budget exhausted', 'read-instance budget')) else 'proof_refused'
        return _fallback(original, str(exc), status)


def production_context(original, problem, block, resources, storage, adapter):
    """Bind supplied target laws to this invocation's actual owned program.

    The caller has no in-flight prior invocation: its prior output/final drain
    closed that lifetime. No host callbacks or foreign work appear in this
    checked kernel. Dense lowering uses one fixed configuration and the same
    full completion callback as the seed, with disjoint external buffers.
    Slot/role byte ownership is rechecked here and by the generic consumer.
    """
    root = Path(__file__).resolve().parent
    public = json.loads((root/'completion_premises.json').read_text())
    facts_sha = sha256((root/'rtl_facts/facts.json').read_bytes()).hexdigest()
    ordering_sha = sha256((root/'completion_ordering.json').read_bytes()).hexdigest()
    known = (facts_sha == public['inputs']['facts']['sha256']
             and ordering_sha == public['inputs']['ordering']['sha256'])
    flags = {name: known and record.get('status') == 'fact' and record.get('value') is True
             for name,record in public['premises'].items()}
    context = dict(problem=problem, block=block, resources=resources,
        storage=storage, adapter=adapter, premises=flags)
    try:
        from dense import dense_adapter
        if adapter != dense_adapter():
            raise ValueError('selected Gemmini premises are not bound to this encoder/decoder or completion mapping')
        context['admission'] = _admission(original)
        _storage(original, problem, block, resources, storage)
        # The source-checked original kernel consists solely of the adapter's
        # data commands plus explicit full drains; the emitter owns this entire
        # invocation and never overlaps another kernel's private allocation.
        flags['exclusive_private_state'] = True
        flags['complete_device_drain'] = adapter.synchronous_completion is True
        law = _law(context)
        if not law.unknown(STREAM_FLAGS):
            context['admission'], context['issued_scope'] = prove_dense_completion_scope(
                original, problem, block, resources, storage, adapter, law)
    except ValueError as exc:
        context['program_proof_refusal'] = str(exc)
        flags['exclusive_private_state'] = False
    return context
