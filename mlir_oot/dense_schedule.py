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


def select_capacity_cached_b(control, *, row_tiles=1):
    """Expose a complete-weight family beyond the old static tile budget.

    Placement and lifetimes establish legality, not profitability. Aligned K
    panels use coalesced A and disjoint next-M slots; K tails retain the exact
    one-panel A path. All original numeric fields and increasing K are kept.
    The existing command policies and their compile-size cap remain unchanged.
    """
    if type(control) is not GoldenGemm:
        raise ValueError('complete-weight capacity choice requires dense GEMM')
    if type(row_tiles) is not int or row_tiles < 1:
        raise ValueError('row tile count must be a positive integer')
    if control.input_view is not None or control.shape.bias:
        raise ValueError('complete-weight capacity needs dense unseeded operands')
    s=control.shape
    mt,nt,kt=(_ceil_div(v,F.DIM) for v in (s.m,s.n,s.k))
    bm=min(row_tiles,mt)
    wide=s.k>F.DIM and s.k%F.DIM==0
    candidate=replace(s,bm=bm,bn=nt,cache_b=True,cache_a=False,
        wide_a=wide,wide_b=True,wide_store=s.output_dtype=='i8',reuse_b=False,
        pipeline_m=wide,prefetch_m=wide,banked_m=wide,
        separate_b_bank=not wide,prefetch_b=False)
    generator=GoldenGemm(candidate,cached_b_resource_capacity=True)
    input_span=bm*(kt if wide else 1)*F.DIM
    input_slots=([0,input_span],[F.SPAD_BANK_ROWS,F.SPAD_BANK_ROWS+input_span]) if wide else ([0,input_span],)
    weight_begin=2*F.SPAD_BANK_ROWS
    weight_end=weight_begin+kt*nt*F.DIM
    if any(end>weight_begin for begin,end in input_slots):
        raise ValueError('complete-weight live input and B storage overlap')
    decision=dict(applied=True,automatic_policy=False,performance='UNKNOWN',
        selection='explicit complete-weight scratchpad capacity proof',
        legacy_static_tile_budget=128,cached_weight_tiles=kt*nt,
        input_reserved_intervals=[list(slot) for slot in input_slots],
        weight_reserved_interval=[weight_begin,weight_end],
        weight_lifetime='All increasing K and N tiles retained through every M block',
        next_input_lifetime='Disjoint current/next scratch and accumulator slots' if wide else 'Single current K panel; K tail exact',
        accumulator_rows=bm*nt*F.DIM*(2 if wide else 1),
        scratchpad_rows=F.SPAD_ROWS,accumulator_limit=F.ACC_ROWS,
        source_reduction_order='Increasing K; no reassociation',
        requested_weight_bytes=s.k*s.n,
        requested_input_bytes=s.m*s.k,
        physical_DRAM_bytes='UNKNOWN',
        real_B_mesh_preloads=_ceil_div(mt,bm)*kt*nt,
        ABI='Same dense three-pointer kernel; immutable A/B and disjoint fully written C')
    generator.cached_b_capacity_decision=decision
    return generator,decision


def choose_capacity_cached_b(control):
    """Keep existing cached operands and admit one explicit larger B family."""
    refusal=dict(applied=False,automatic_policy=False,performance='UNKNOWN',
                 selection='explicit complete-weight scratchpad capacity proof')
    if type(control) is not GoldenGemm:
        return control,dict(refusal,refusal='Current family is not dense GEMM')
    s=control.shape
    if s.cache_a or s.cache_b:
        return control,dict(refusal,refusal='Existing cached operand lifetime retained')
    if _ceil_div(s.k,F.DIM)*_ceil_div(s.n,F.DIM)<=128:
        return control,dict(refusal,refusal='Within the existing cached B tile budget')
    try:
        return select_capacity_cached_b(control)
    except ValueError as failure:
        return control,dict(refusal,refusal=str(failure))


def _full_k_banked_candidate(shape):
    if (shape.output_dtype != 'i8' or shape.bias or shape.cache_a
            or shape.m <= F.DIM or shape.k <= F.DIM or shape.k % F.DIM
            or shape.n % F.DIM):
        raise ValueError('full-K banked schedule requires aligned unbiased i8 GEMM')
    candidate=replace(shape,bm=1,bn=_ceil_div(shape.n,F.DIM),
        cache_b=True,cache_a=False,pipeline_m=True,prefetch_m=True,
        banked_m=True,separate_b_bank=False,wide_a=True,wide_b=True,
        wide_store=True,reuse_b=False)
    candidate.validate()
    return candidate


def choose_banked_by_command_cost(shape):
    """Consider a resource-legal full-K panel without workload selectors.

    This explicit policy minimizes primitive command count, not predicted
    cycles. Equal/worse counts and resource refusals retain the legal input
    schedule. Numeric epilogue fields and source dimensions stay unchanged.
    """
    shape.validate()
    try:
        candidate=_full_k_banked_candidate(shape)
    except ValueError as failure:
        return shape,dict(applied=False,refusal=str(failure))
    control_cost=estimate(shape)
    candidate_cost=estimate(candidate)
    applied=candidate_cost['primitive_command_count'] < control_cost['primitive_command_count']
    return (candidate if applied else shape),dict(
        applied=applied,refusal=None if applied else 'primitive command count does not decrease',
        cost_unit='primitive_commands_not_cycles',
        control=control_cost,candidate=candidate_cost)


def choose_resident_a_by_command_cost(shape):
    """Cache a complete multirow M block when resources and command cost admit it."""
    shape.validate()
    mt,nt=_ceil_div(shape.m,F.DIM),_ceil_div(shape.n,F.DIM)
    if mt <= 1:
        return shape,dict(applied=False,refusal='policy requires multiple A row tiles')
    bn=min(nt,F.ACC_ROWS//(mt*F.DIM))
    if bn == 0:
        return shape,dict(applied=False,refusal='complete M block exceeds accumulator capacity')
    candidate=replace(shape,bm=mt,bn=bn,cache_a=True,cache_b=False,
        wide_a=False,wide_b=True,wide_store=shape.output_dtype=='i8',
        reuse_b=True,pipeline_m=False,prefetch_m=False,banked_m=False,
        separate_b_bank=False,prefetch_b=False)
    try:
        candidate.validate()
    except ValueError as failure:
        return shape,dict(applied=False,refusal=str(failure))
    prefetch_refusal=None
    try:
        prefetched=replace(candidate,prefetch_b=True)
        prefetched.validate()
    except ValueError as failure:
        prefetch_refusal=str(failure)
    else:
        candidate=prefetched
    control_cost,candidate_cost=estimate(shape),estimate(candidate)
    applied=candidate_cost['primitive_command_count'] < control_cost['primitive_command_count']
    return (candidate if applied else shape),dict(
        applied=applied,refusal=None if applied else 'primitive command count does not decrease',
        prefetch_b=candidate.prefetch_b,prefetch_b_refusal=prefetch_refusal,
        cost_unit='primitive_commands_not_cycles',control=control_cost,candidate=candidate_cost)


def choose_transfer_by_command_cost(shape):
    """Compare legal transfer families using one explicit geometric cost policy."""
    choices={}
    candidates=[]
    for name,choose in (('banked',choose_banked_by_command_cost),('resident_a',choose_resident_a_by_command_cost)):
        candidate,decision=choose(shape)
        choices[name]=decision
        if decision['applied']:
            cost=decision['candidate']
            key=(cost['primitive_command_count'],cost['dma_request_bytes_upper'],cost['output_blocks'],name)
            candidates.append((key,candidate,name))
    if not candidates:
        return shape,dict(applied=False,refusal='no legal transfer family reduces command count',families=choices)
    _,chosen,name=min(candidates,key=lambda c:c[0])
    return chosen,dict(applied=True,selected_family=name,cost_unit='primitive_commands_not_cycles',families=choices)


def resident_a_prefetch(shape):
    """Expose a resource-legal one-row-tile overlap alternative for measurement.

    This explicit choice does not require a command-count improvement. All A
    panels become resident before increasing-K B ping-pong. Tile sizes and
    numeric epilogues are preserved; actual calibrated latency decides whether
    this alternative should be selected, including a single N block.
    """
    shape.validate()
    if (_ceil_div(shape.m,F.DIM) != 1 or shape.cache_b or shape.separate_b_bank
            or shape.pipeline_m or shape.prefetch_m or shape.banked_m):
        raise ValueError('resident A prefetch requires one row tile and no competing cached/pipelined placement')
    candidate=replace(shape,cache_a=True,wide_a=False,prefetch_b=True)
    candidate.validate()
    return candidate


def select_coalesced_resident_a(control):
    """Group resident input DMA packets without changing allocation or work.

    A dense row-major source and complete cached A layout prove consecutive
    K tiles. CONFIG_LD.block_stride=DIM places each block at the established
    local address; the four-block target bound admits exact M/K tails.
    This ranks transfer command count only, never predicts cycle savings.
    """
    shape=control.shape
    shape.validate(prefetch_b_rows=control.prefetch_b_rows)
    decision=dict(applied=False,policy='resident_a_load_coalescing',
                  timing_claim=False,cost_unit='input_dma_commands_not_cycles')
    if not shape.cache_a:
        decision['refusal']='requires the complete resident A layout'
        return control,decision
    if control.resident_a_load_tiles != 1:
        decision['refusal']='existing explicit A grouping retained'
        return control,decision
    kt,mt=_ceil_div(shape.k,F.DIM),_ceil_div(shape.m,F.DIM)
    if kt <= 1:
        decision['refusal']='input DMA command count does not decrease'
        return control,decision
    candidate=GoldenGemm(shape,prefetch_b_rows=control.prefetch_b_rows,
                         resident_a_load_tiles=4,input_view=control.input_view)
    segments=(sum(len(tuple(control.input_view.split_rows(a*F.DIM,min(F.DIM,shape.m-a*F.DIM))))
                  for a in range(mt)) if control.input_view is not None else mt)
    decision.update(applied=True,refusal=None,resident_a_load_tiles=4,
                    control_input_dma_commands=segments*kt,
                    candidate_input_dma_commands=segments*_ceil_div(kt,4),
                    input_requested_bytes=shape.m*shape.k,
                    input_reserved_rows=shape.bm*kt*F.DIM,
                    block_stride_rows=F.DIM,
                    emitted_delta='Adjacent K input packets coalesce; source pointers, resident cells, B ordering, increasing K compute and stores are unchanged')
    return candidate,decision


def select_kernel(shape, *, banked_prefetch=False, grouped_b=False, separate_b_bank=False, full_k_banked=False, banked_command_policy=False, resident_a_command_policy=False, transfer_command_policy=False, resident_a_prefetch_policy=False,resident_a_load_coalescing=False):
    if type(resident_a_load_coalescing) is not bool:
        raise ValueError('resident A load coalescing requires a boolean selection')
    policies_selected=(banked_command_policy,resident_a_command_policy,transfer_command_policy,resident_a_prefetch_policy)
    if full_k_banked and any(policies_selected):
        raise ValueError('banked compiler policy cannot mix with explicit full-K selection')
    if sum(bool(p) for p in policies_selected)>1:
        raise ValueError('select one explicit dense command-cost policy')
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
    if full_k_banked:
        # Explicit per-source selection only: legality is general, performance
        # must be qualified separately. Scalar readout fields are unchanged.
        selected=_full_k_banked_candidate(selected)
        policies.append('full_k_banked_prefetch')
    if banked_command_policy:
        selected,decision=choose_banked_by_command_cost(selected)
        if decision['applied']:
            policies.append('full_k_banked_command_cost')
    if resident_a_command_policy:
        selected,decision=choose_resident_a_by_command_cost(selected)
        if decision['applied']:
            policies.append('resident_a_command_cost')
    if transfer_command_policy:
        selected,decision=choose_transfer_by_command_cost(selected)
        if decision['applied']:
            policies.append('transfer_command_cost_'+decision['selected_family'])
    if resident_a_prefetch_policy:
        selected=resident_a_prefetch(selected)
        policies.append('resident_a_prefetch')
    if separate_b_bank and not selected.banked_m and not selected.separate_b_bank:
        candidate=replace(selected,separate_b_bank=True)
        try:
            candidate.validate()
        except ValueError:
            pass  # A legal original schedule remains the fallback.
        else:
            selected=candidate;policies.append('separate_b_bank')
    generator=GoldenGemm(selected)
    if resident_a_load_coalescing:
        generator,decision=select_coalesced_resident_a(generator)
        generator.resident_a_load_decision=decision
        if decision['applied']:
            policies.append('resident_a_load_coalescing')
    return generator,'dense_gemm'+(':'+','.join(policies) if policies else '')
