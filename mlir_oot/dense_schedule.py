"""Opt-in schedules selected from static Gemmini transfer/bank capabilities.

The banked-prefetch family has one full-width A transfer, one output store
per row tile, and cached B. Its measured anchor is M3136/N64/K64; other legal
members require their own performance evidence. Scalar epilogue parameters
are carried unchanged from the source proof.
"""
from dataclasses import replace
from .golden_gemm import GoldenGemm, _ceil_div
from .golden_tuning import estimate
from .tables import rtl_facts as F


def select_kernel(shape, *, banked_prefetch=False, grouped_b=False, separate_b_bank=False):
    shape.validate();selected=shape;policies=[]
    if grouped_b and not shape.wide_b:
        candidate=replace(shape,wide_b=True)
        candidate.validate()
        if estimate(candidate)['primitive_command_count'] < estimate(shape)['primitive_command_count']:
            selected=candidate;policies.append('grouped_b')
    if (banked_prefetch and shape.output_dtype=='i8' and not shape.bias
            and not shape.cache_a and shape.m>F.DIM
            and F.DIM<shape.k<=4*F.DIM and shape.k%F.DIM==0
            and 0<shape.n<=4*F.DIM and shape.n%F.DIM==0):
        nt,kt=_ceil_div(shape.n,F.DIM),_ceil_div(shape.k,F.DIM)
        candidate=replace(selected,bm=1,bn=nt,cache_b=True,cache_a=False,
            pipeline_m=True,prefetch_m=True,banked_m=True,separate_b_bank=False,
            wide_a=True,wide_b=True,wide_store=True,reuse_b=False)
        # Banked placement reserves scratch banks0/1 for alternating A,
        # banks2/3 for B, and one accumulator bank for each result slot.
        if (kt*F.DIM<=F.SPAD_BANK_ROWS and
                kt*nt*F.DIM<=F.SPAD_ROWS-2*F.SPAD_BANK_ROWS and
                nt*F.DIM<=F.ACC_BANK_ROWS):
            candidate.validate()
            if estimate(candidate)['primitive_command_count']<=estimate(selected)['primitive_command_count']:
                selected=candidate;policies.append('banked_prefetch_single_store')
    if separate_b_bank and not selected.banked_m and not selected.separate_b_bank:
        candidate=replace(selected,separate_b_bank=True)
        try:
            candidate.validate()
        except ValueError:
            pass  # A legal original schedule remains the fallback.
        else:
            selected=candidate;policies.append('separate_b_bank')
    return GoldenGemm(selected),'dense_gemm'+(':'+','.join(policies) if policies else '')
