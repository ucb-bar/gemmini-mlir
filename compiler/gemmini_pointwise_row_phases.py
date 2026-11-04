"""Optional original-row/RHS-phase tradeoff through checked shared storage.

The incumbent proves source/ARRAY/readout laws before this adapter runs. Each
phase retains complete original K and disjoint original output columns. Storage
planning uses the expanded actual row union, never a fictitious capacity limit.
"""
from dataclasses import replace, asdict
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionRhsPhases, ConvolutionRhsPhaseLaw, plan_convolution_schedule,
    plan_convolution_storage_alternative)


def plan_pointwise_row_phases(incumbent, *, input_rows, output_rows):
    if type(output_rows) is not int or output_rows not in (2, 4, 8):
        raise ValueError('pointwise row phases require an explicit bounded row choice')
    schedule, rhs, options, diagnostic = incumbent
    p = schedule.source.plan
    w = p.window
    original_input = options.get('input_rows', input_rows)
    # The one-row incumbent often has no input_rows override. Its actual ring
    # is passed separately by the wrapper, rather than reconstructed here.
    if (not diagnostic.get('selected') or w.kernel != (1, 1)
            or w.stride not in ((1, 1), (2, 2)) or any(w.padding)
            or w.dilation != (1, 1) or rhs is None
            or isinstance(rhs, ConvolutionRhsPhases)):
        return incumbent
    if original_input is None:
        raise ValueError('pointwise row phases need the incumbent actual input ring')
    rows = min(output_rows, w.output_spatial[0])
    m_blocks = (w.output_spatial[1] + p.row_tile - 1) // p.row_tile
    if rows < 2 or rows*m_blocks > 16:
        return incumbent
    columns = min(len(schedule.source.column_intervals),
        (p.accumulator_rows-p.accumulator_row)//(rows*m_blocks*p.primitive))
    if columns < 1:
        return incumbent
    slots = 1 + (rows-1)*w.stride[0]
    end = original_input.base + slots*original_input.rows_per_source_row
    if end+p.channel_tile > p.operand_rows:
        return incumbent
    ring = replace(original_input, slots=slots,
                   private_rows=end-original_input.private_first)
    issue = replace(schedule.issue_law, load_issue_budget=max(
        schedule.issue_law.load_issue_budget,
        ring.stage_command_bound(ring.slots),
        p.row_tile+1+rows*m_blocks*columns*int(p.bias)))
    updated = plan_convolution_schedule(schedule.source, schedule.choice,
        retention_law=schedule.retention_law, issue_law=issue,
        epoch_law=schedule.epoch_law)
    phase_law = ConvolutionRhsPhaseLaw(schedule.source.identity,
        'original-disjoint-output-columns-no-intermediate-observers-v1',
        True, True, True, True)
    storage = plan_convolution_storage_alternative(updated, ring,
        operand_rows=p.operand_rows, rhs_law=rhs.law,
        rhs_phase_law=phase_law, max_row_blocks=16, max_output_rows=rows,
        rhs_column_blocks=columns)
    group = storage.output_group
    groups = group if isinstance(group, tuple) else (group,)
    if any(g is None for g in groups):
        return incumbent
    maximum = max(g.row_blocks*g.panel.column_blocks for g in groups)
    readout = replace(options['readout_law'], max_issued_readouts=maximum)
    updated_options = dict(options, output_group=group, input_rows=ring,
                           readout_law=readout)
    return updated, storage.rhs, updated_options, dict(diagnostic,
        pointwise_row_phases=dict(requested_rows=output_rows,
            selected_rows=rows, rhs_column_blocks=columns,
            storage=storage.summary(), input_before=original_input.record(),
            input_after=ring.record(), readout_law=asdict(readout),
            timing='UNKNOWN; complete-call comparison required'))
