"""Production completion entry. Unknown premises keep the checked construction."""
from merlin.compile.scheduling.dense_panel_epochs import (
    DensePanelEpochLaw, STREAM_FLAGS, schedule_dense_panel_epochs,
)


def apply_completion(original, context):
    """Remove only proved internal K-stream drains; retain all readout boundaries.

    Context is the original checked construction context. No resource budget,
    allocation, instruction, operand or issue order is changed here. A proof
    refusal returns the identical original object with its complete boundaries.
    """
    def get(key):
        return context[key] if isinstance(context, dict) else getattr(context,key)
    try:
        p,b,r,s,a,ad = (get(k) for k in ('problem','block','resources','storage','adapter','admission'))
        flags = get('premises')
        if not isinstance(flags,dict):
            original.realization['completion_refusal'] = 'plain selected premises dictionary required'
            return original
        values = {key: flags.get(key) is True for key in STREAM_FLAGS}
        if not all(values.values()):
            original.realization['completion_refusal'] = 'UNKNOWN: ' + ', '.join(k for k,v in values.items() if not v)
            return original
        law = DensePanelEpochLaw(a.configuration_identity,a.numerical_policy_identity,
            flags.get('ordering_identity','shared-selected-gemmini-ordering-v1'),
            **values,accumulator_result_visibility=False)
        checked = schedule_dense_panel_epochs(original,p,b,r,s,a,ad,law,
            static_budget=128,timeout_ms=15000)
        if checked.realization['panel_epochs']['removed_dynamic_drains'] <= 0:
            original.realization['completion_refusal'] = 'no internal K stream fits the unchanged issued budgets'
            return original
        return checked
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        # Preserve the safe result and make the missing obligation reviewable.
        # The input realization is a report, not the checked executable identity.
        original.realization['completion_refusal'] = str(exc)
        return original
