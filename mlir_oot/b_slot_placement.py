"""Opt-in next-K transfer overlap from complete operand extent proofs.

The existing cached-A schedule can retain more than two scratchpad banks.
Two disjoint B slots in its remaining rows admit the same ping-pong lifetime
ordering. Slots may share a bank: numerical/resource legality is proved here;
actual bank arbitration and overlap require measurement.
"""

from dataclasses import replace

from .golden_gemm import _ceil_div
from .tables import rtl_facts as F


def select_remaining_b_slots(control):
    """Derive placements using shape/resources only; preserve every fallback."""
    s=control.shape
    s.validate(prefetch_b_rows=control.prefetch_b_rows,
               cached_a_output_blocks=control.cached_a_output_blocks,
               cached_b_resource_capacity=control.cached_b_resource_capacity)
    decision=dict(applied=False,timing_claim=False,
                  policy='remaining_rows',selection='shape and complete SPAD extents')
    if s.prefetch_b:
        decision['refusal']='existing B prefetch schedule retained'
        return control,decision
    if not s.cache_a or s.k<=F.DIM or s.separate_b_bank:
        decision['refusal']='requires complete cached A, multiple K panels and no competing B placement'
        return control,decision
    a_end=control._a_storage_tiles()*_ceil_div(s.k,F.DIM)*F.DIM
    span=s.bn*F.DIM
    row=a_end
    bases=[]
    for _ in range(2):
        if row % F.SPAD_BANK_ROWS + span > F.SPAD_BANK_ROWS:
            row=_ceil_div(row,F.SPAD_BANK_ROWS)*F.SPAD_BANK_ROWS
        bases.append(row)
        row+=span
    placement=tuple(bases)
    candidate=replace(s,prefetch_b=True)
    try:
        candidate.validate(prefetch_b_rows=placement,
                           cached_a_output_blocks=control.cached_a_output_blocks,
                           cached_b_resource_capacity=control.cached_b_resource_capacity)
    except ValueError as failure:
        decision.update(refusal=str(failure),reserved_a_rows=a_end,
                        reserved_b_panel_rows=span,attempted_b_rows=bases)
        return control,decision
    decision.update(applied=True,refusal=None,reserved_a_rows=a_end,
                    reserved_b_panel_rows=span,prefetch_b_rows=bases,
                    b_banks=[base//F.SPAD_BANK_ROWS for base in bases],
                    ordering='initial two panels; load next free slot before current K compute; increasing source K order and exact tail drain',
                    emitted_delta='B panel command order and row placement change; primitive command counts, requested operand bytes and source arithmetic are unchanged')
    return control.with_emission_options(shape=candidate,prefetch_b_rows=placement),decision
