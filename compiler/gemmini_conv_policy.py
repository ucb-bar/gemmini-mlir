"""Reusable convolution lifetime alternatives, admitted by storage and source effects.

No shape chooses a timing winner here. Declared knobs expose alternatives for
same-case whole-call A/B. Rolling input may use bounded complete-K RHS phases
when independent output columns and a real phase-last-use drain permit them.
"""
from __future__ import annotations
from dataclasses import replace

from merlin.compile.scheduling.window_transfer import ConvolutionInputRetentionLaw
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionRhsResidencyLaw, ConvolutionScheduleChoice,
    plan_convolution_schedule, plan_convolution_rhs_residency,
    ConvolutionRhsPhaseLaw, plan_convolution_rhs_phases,
)


def apply_retention(schedule, mode, *, input_rows=None, phased=False):
    """Return the actual checked schedule and optional complete resident RHS owner."""
    if type(phased) is not bool:
        raise ValueError('convolution phase admission must be explicit')
    if phased and input_rows is None:
        raise ValueError('bounded RHS phases require the original private rolling input owner')
    if mode == 'off':
        return schedule, None
    if mode not in ('input', 'input_rhs'):
        raise ValueError('unknown convolution retention alternative')
    source = schedule.source
    p = source.plan
    count = min(len(source.column_intervals),
                (p.accumulator_rows-p.accumulator_row)//p.row_tile)
    # The compiler owns the disjoint local rows for this whole invocation.
    # Input and RHS buffers are immutable/disjoint under the public convolution
    # source contract. Compute writes only ACC; RHS loads never touch LHS rows.
    ids = (schedule.issue_law.hardware_identity if schedule.issue_law else
           'gemmini-private-disjoint-local-rows-v1',
           schedule.issue_law.numerical_policy_identity if schedule.issue_law else
           'gemmini-signed-i8-i32-original-readout', source.identity,
           'original-channel-reductions-and-invocation-private-rows')
    retention = ConvolutionInputRetentionLaw(*ids, count, True, True, True)
    issue = schedule.issue_law
    if issue is not None:
        issue = replace(issue, load_issue_budget=max(issue.load_issue_budget,p.row_tile+1+count*int(p.bias)))
    transformed = plan_convolution_schedule(source,
        ConvolutionScheduleChoice(count, schedule.choice.load_issue, schedule.choice.compute_pairs),
        retention_law=retention, issue_law=issue, epoch_law=schedule.epoch_law)
    rhs = None
    if mode == 'input_rhs':
        law = ConvolutionRhsResidencyLaw(*ids, True, True, True, True, True)
        if phased:
            # Original NHWC output columns are distinct live-outs. The source
            # contract has immutable IFM/W and no observer until this invocation
            # returns. The shared consumer emits a full phase-last-use drain
            # before replacing any RHS rows and restages the exact input ring.
            phase_law = ConvolutionRhsPhaseLaw(source.identity,
                'original-disjoint-output-columns-no-intermediate-observers-v1',
                True, True, True, True)
            rhs = plan_convolution_rhs_phases(transformed, operand_rows=p.operand_rows,
                law=law, phase_law=phase_law, input_rows=input_rows)
        else:
            rhs = plan_convolution_rhs_residency(transformed, operand_rows=p.operand_rows,
                                                law=law)
    return transformed, rhs


def prepare_rolling_input(source):
    """Admit an invocation-private row ring, preserving original channel axes.

    Each source image row is staged as rectangular CI-strided DMA runs once per
    ring version; taps subsequently address those physical rows. Halos occupy
    disjoint cells and are filled exactly once. No source payload packing occurs.
    Insufficient private capacity retains the ordinary window transfer route.
    """
    from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
    from merlin.compile.scheduling.window_transfer import ConvolutionResidentInputRows
    p=source.plan
    window=p.window
    span=(window.kernel[0]-1)*window.dilation[0]+1
    pitch=window.input_shape[2]+window.padding[1]+window.padding[3]
    chunks=(window.input_shape[-1]+p.channel_tile-1)//p.channel_tile
    end=span*pitch*chunks
    rhs_base=(max(end,p.rhs_row)+p.channel_tile-1)//p.channel_tile*p.channel_tile
    if rhs_base+p.channel_tile>p.operand_rows:
        return source, None
    updated=ConvolutionExplorationSource(replace(p,rhs_row=rhs_base),
        source.column_intervals,source.source_identity)
    ring=ConvolutionResidentInputRows(window,updated.identity,p.channel_tile,
        0,pitch,span,0,end,min(p.primitive,pitch),halo_value=0,
        stage_channels=p.channel_tile,compute_row_step=window.stride[1])
    return updated,ring


def select_regional_schedule(source, *, uniform_readout, resource_identity):
    """A coarse supported region; capacity admits a plan, cycles motivate it.

    The large spatial region has paired whole-call model evidence for the
    stem and 3x3 training/held-out geometries. This opt-in coarse policy is
    consequently not a claim of an optimal per-shape choice. The narrow phased
    region has one paired public training point; a fresh selected geometric
    timing holdout and its crossover remain UNKNOWN. Its bounds derive from
    primitive width and complete-K storage, not empirically fitted cutoffs.
    The original source and shared planners still prove every selected plan.
    """
    from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
    p=source.plan;w=p.window
    candidate,ring=prepare_rolling_input(source)
    reduction_rows=w.kernel[0]*w.kernel[1]*((w.input_shape[-1]+p.channel_tile-1)//p.channel_tile)*p.channel_tile
    resources=dict(output_positions=w.output_spatial[0]*w.output_spatial[1],
        output_channels=p.output_channels,
        complete_rhs_intervals=0 if ring is None else
            (p.operand_rows-candidate.plan.rhs_row)//reduction_rows,
        narrow_output_rows=int(w.output_spatial[1]<=p.primitive and w.output_spatial[0]>=2),
        rhs_requires_phases=int(ring is not None and
            (p.operand_rows-candidate.plan.rhs_row)//reduction_rows<len(source.column_intervals)),
        supported_narrow_contract=int(w.kernel==(3,3) and w.stride in ((1,1),(2,2)) and
            w.dilation==(1,1) and uniform_readout),
        supported_spatial_contract=int(w.kernel in ((3,3),(7,7)) and
            w.stride in ((1,1),(2,2)) and w.dilation==(1,1) and uniform_readout))
    facts=PolicyFacts(AnalysisStamp(source.identity,
        PipelineContext(resources=resource_identity),'convolution_source'),
        resources=tuple(sorted(resources.items())))
    guards=(ResourcePredicate('output_positions',196),
        ResourcePredicate('output_channels',64),
        ResourcePredicate('complete_rhs_intervals',1),
        ResourcePredicate('supported_spatial_contract',1))
    large_selected=all(g.match(facts) is True for g in guards)
    narrow_guards=(ResourcePredicate('narrow_output_rows',1),
        ResourcePredicate('rhs_requires_phases',1),
        ResourcePredicate('complete_rhs_intervals',1),
        ResourcePredicate('supported_narrow_contract',1))
    narrow_selected=all(g.match(facts) is True for g in narrow_guards)
    selected=large_selected or narrow_selected
    return ('rolling_rhs',8) if selected else ('off',1), dict(
        policy='regional-v2',source_identity=source.identity,
        resource_identity=resource_identity,resources=resources,
        guards=[dict(resource=g.resource,at_least=g.at_least) for g in guards],
        narrow_guards=[dict(resource=g.resource,at_least=g.at_least) for g in narrow_guards],
        selected_region='large' if large_selected else 'narrow_phased' if narrow_selected else 'fallback',
        selected='rolling_rhs8' if selected else 'off1',
        evidence=('stem and training/held-out3x3 paired provisional whole-call model; '
            'narrow phased: one paired high-channel short-row training point, '
            'fresh selected geometric timing holdout and crossover UNKNOWN; '
            'no RTL timing or empirical cutoff claim'))
