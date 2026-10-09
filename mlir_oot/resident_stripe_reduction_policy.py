"""Explicit command retention over an already admitted resident stripe layout."""

from .golden_resident_stripe_conv import GoldenResidentStripeConv
from .tables import rtl_facts as F


def retain_reduction_commands(control):
    """Revalidate typed numeric/resource facts before retaining ordinary K loops.

    This modifier does not select a residency family or infer profitability.
    Its source-semantic owner is the existing full-A/full-K-B stripe emitter.
    Unsupported families retain their exact current implementation.
    """
    decision = {
        "applied": False,
        "timing_claim": False,
        "basis": "explicit bounded ordinary tap/K/N/row/X command retention",
        "profitability": "UNKNOWN outside separately qualified paired scopes",
    }
    if type(control) is not GoldenResidentStripeConv:
        decision["refusal"] = (
            "selected schedule is not the complete resident stripe family"
        )
        return control, decision
    if getattr(control, "store_plan", None) is not None:
        decision["refusal"] = (
            "resident stripe emitter does not support the selected separate store plan"
        )
        return control, decision
    candidate = control.with_emission_options(
        compact_inner_commands=True, compact_reduction_commands=True
    )
    fields = (
        "plane",
        "kt",
        "xt",
        "input_rows",
        "weight_rows",
        "bbase",
        "stripe_rows",
        "shape",
    )
    if any(getattr(candidate, field) != getattr(control, field) for field in fields):
        raise ValueError(
            "resident stripe resources/numeric shape differ from their typed reconstruction"
        )
    if candidate.kt < 2:
        decision["refusal"] = "reduction has no remaining K tile loop to retain"
        return control, decision
    for name in ("resident_stripe_decision", "source_stride_decision"):
        if hasattr(control, name):
            setattr(candidate, name, getattr(control, name))
    decision.update(
        applied=True,
        refusal=None,
        resource_proof={
            "resident_a_rows": candidate.input_rows,
            "resident_b_rows": candidate.weight_rows,
            "b_base_row": candidate.bbase,
            "spad_rows": F.SPAD_ROWS,
            "accumulator_rows": candidate.stripe_rows
            * candidate.xt
            * candidate.conv.bn
            * F.DIM,
            "acc_rows": F.ACC_ROWS,
            "b_row_alignment": F.DIM,
            "lifetimes": "complete A and current N group's full-K B remain resident through every output stripe",
            "source_reduction_order": "tap/K/N/row/X unchanged; first K overwrite retained explicitly",
            "lowering_authority": "every dynamic A/B/C row is revalidated by complete actual SSA/CFG trace",
        },
    )
    return candidate, decision
