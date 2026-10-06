"""Explicit resource/layout admission for complete-A accumulator stripes.

This is a legal schedule-family choice, not a timing or profitability model.
The source route retains the original numeric proof and fresh output contract.
"""

from __future__ import annotations

from dataclasses import asdict, replace

from .golden_gemm import GoldenGemm, _ceil_div
from .golden_resident_stripe_gemm import GoldenResidentStripeGemm
from .tables import rtl_facts as F


def select_resident_accumulator_stripes(control):
    decision = {
        "applied": False,
        "basis": "Explicit typed complete-A/narrow-store schedule with bank and accumulator bounds",
        "performance": "UNKNOWN; opt-in does not infer profitability",
        "automatic_policy": False,
    }
    if type(control) is not GoldenGemm:
        return control, dict(
            decision,
            refusal="Existing schedule family has no dense complete-A contract",
        )
    s = control.shape
    if control.input_view is not None:
        return control, dict(
            decision,
            refusal="Segmented input requires a separate full-A load mapping proof",
        )
    if not s.cache_a or s.output_dtype != "i8" or not s.wide_store or s.bias:
        return control, dict(
            decision,
            refusal="Requires existing complete cached A, scaled byte wide stores and zero accumulator seed",
        )
    mt, nt, kt = (_ceil_div(x, F.DIM) for x in (s.m, s.n, s.k))
    # A four-tile output group uses the existing maximum proven DMA grouping.
    bn = 4
    if s.bn >= bn or nt < bn:
        return control, dict(
            decision, refusal="Control already has wide stores or too few output tiles"
        )
    plane = mt * F.DIM
    input_rows = plane * kt
    weight_rows = kt * bn * F.DIM
    if input_rows > 2 * F.SPAD_BANK_ROWS or weight_rows > F.SPAD_BANK_ROWS:
        return control, dict(
            decision,
            refusal="Complete A must fit lower two banks and full-K B one upper bank",
            input_rows=input_rows,
            weight_rows=weight_rows,
        )
    stripe_tiles = min(mt, F.ACC_ROWS // (bn * F.DIM))
    requested = replace(
        s,
        bm=stripe_tiles,
        bn=bn,
        cache_a=False,
        prefetch_b=False,
        wide_a=False,
        separate_b_bank=False,
    )
    selected = GoldenResidentStripeGemm(requested, stripe_tiles=stripe_tiles)
    if selected.input_rows != input_rows or selected.weight_rows != weight_rows:
        raise ValueError("Selected emitter disagrees with typed resource proof")
    if selected.bbase < F.SPAD_ROWS - F.SPAD_BANK_ROWS:
        raise ValueError("Selected full-K B overlaps undeclared bank placement")
    if selected.bbase < input_rows:
        raise ValueError("Selected live A/B storage aliases")
    return selected, dict(
        decision,
        applied=True,
        refusal=None,
        control_shape=asdict(s),
        selected_shape=asdict(selected.shape),
        input_rows=input_rows,
        weight_rows=weight_rows,
        input_reserved_interval=[0, input_rows],
        weight_reserved_interval=[selected.bbase, F.SPAD_ROWS],
        accumulator_rows=stripe_tiles * bn * F.DIM,
        accumulator_limit=F.ACC_ROWS,
        source_reduction_order="Increasing K per output; no reassociation",
        ABI="Same dense three-pointer kernel; caller owns immutable A/B and disjoint fully written C for whole call",
    )
