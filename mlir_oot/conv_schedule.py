"""Explicit, recorded direct-convolution schedule selection.

Source binding proves the NHWC/HWIO contract before this policy runs. The
optional spatial schedule changes command ordering only. Shape and accumulator
capacity select eligibility; model names and region names are never consulted.
"""
from dataclasses import replace
from .golden_conv import GoldenConv
from .golden_flat_conv import GoldenFlatConv, eligible, choose_band_rows, spatial_runs, padding_segments
from .golden_gemm import _ceil_div
from .tables import rtl_facts as F


def _resident_stripe_choice(control):
    """Compare an admitted residency family using explicit issue/traffic costs.

    The score serializes padded DIM geometry and requested bytes at the declared
    16-byte DMA width. It is a search estimate, not a hardware timing claim;
    caches, shorter execute waves and DMA/execute overlap require measurement.
    """
    from .golden_resident_stripe_conv import GoldenResidentStripeConv, command_counts
    from .golden_flat_conv import command_counts as flat_counts
    s = control.conv
    decision = dict(applied=False, cost_unit='padded_DIM_issue_plus_requested_16B_transfer_estimate',
                    timing_claim=False)
    if s.w + 2 <= F.DIM:
        decision['refusal'] = 'narrow spatial planes retain their existing schedule family'
        return control, decision
    try:
        candidate = GoldenResidentStripeConv(s)
    except ValueError as failure:
        decision['refusal'] = str(failure)
        return control, decision
    counts = flat_counts(s,wide_a=control.wide_a,band_rows=control.band_rows,
                         virtual_padding=control.virtual_padding)
    nonzero_rows = 0
    for start in range(0,s.oh,control.band_rows):
        for _,_,y,x,rows in spatial_runs(s,min(control.band_rows,s.oh-start)):
            for kh in range(3):
                for kw in range(3):
                    nonzero_rows += sum(n for _,n,zero in padding_segments(s,start+y,x,rows,kh,kw) if not zero)
    channel_blocks = _ceil_div(_ceil_div(s.cout,F.DIM),s.bn)
    output_bytes = s.oh*s.ow*s.cout*(4 if s.output_dtype == 'i32' else 1)
    control_bytes = nonzero_rows*s.cin*channel_blocks + counts['weight_bytes'] + output_bytes
    new_counts = command_counts(s)
    candidate_bytes = new_counts['activation_dram_bytes']+new_counts['weight_dram_bytes']+output_bytes
    def score(padded,requested):
        return padded+_ceil_div(requested,16)
    decision.update(control=dict(padded_issue=counts['padded_array_issue_cycles'],
                                 requested_bytes=control_bytes,score=score(counts['padded_array_issue_cycles'],control_bytes)),
                    candidate=dict(padded_issue=new_counts['padded_array_issue_cycles'],
                                   requested_bytes=candidate_bytes,score=score(new_counts['padded_array_issue_cycles'],candidate_bytes)),
                    counts=new_counts)
    decision['applied'] = decision['candidate']['score'] < decision['control']['score']
    decision['refusal'] = None if decision['applied'] else 'admitted residency does not lower issue/traffic estimate'
    return (candidate if decision['applied'] else control), decision


def select_kernel(shape, *, flat_spatial=False, virtual_padding=False, resident_stripes=False):
    if type(resident_stripes) is not bool:
        raise ValueError('resident stripe policy selection must be boolean')
    if virtual_padding and (not flat_spatial or shape.explicit_halo):
        raise ValueError("virtual padding requires unpadded spatial schedule")
    if resident_stripes:
        if not flat_spatial or not virtual_padding:
            raise ValueError('resident stripe policy requires proved virtual padding and spatial scheduling')
        control,kind = select_kernel(shape,flat_spatial=flat_spatial,virtual_padding=virtual_padding)
        selected,decision = _resident_stripe_choice(control)
        selected.resident_stripe_decision = decision
        return selected, 'resident_full_k_stripes' if decision['applied'] else kind
    if flat_spatial and eligible(shape,virtual_padding=virtual_padding):
        # Keep at least the existing channel block. Prefer a multiple of four
        # tiles so all full B DMA commands transfer 64 adjacent channels.
        mt = _ceil_div(shape.oh * shape.ow, F.DIM)
        capacity = F.ACC_ROWS // (mt * F.DIM)
        nt = _ceil_div(shape.cout, F.DIM)
        bn = min(capacity, max(4, _ceil_div(nt, 4) * 4))
        if bn >= 4:
            bn = bn // 4 * 4
        shape = replace(shape, bn=bn, wide_b=True)
        return GoldenFlatConv(shape,wide_a=True,separate_b_bank=True,virtual_padding=virtual_padding), 'spatial_flat_wide_a_separate_b'
    if flat_spatial and (shape.explicit_halo or virtual_padding):
        rows = choose_band_rows(shape,virtual_padding=virtual_padding)
        return GoldenFlatConv(shape,wide_a=True,separate_b_bank=True,band_rows=rows,virtual_padding=virtual_padding), 'spatial_banded_wide_a_separate_b'
    return GoldenConv(shape), 'output_row'
