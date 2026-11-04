"""Opt-in shared pointwise lifetime selection for the ordinary conv callback.

The caller supplies off/resident/array explicitly. There is no automatic timing
selection or experiment-file dependency. Original Window2D source coordinates,
complete K order, exact readout and target ordering remain with shared owners.
"""


def select_pointwise(source, retention, pairs, selection, *, mode='off'):
    """Return a proposal and pure source/resource diagnostics; preserve fallback.

    Eligibility does not imply a cycle win. A caller may further restrict the
    source/readout region before opting in. Shared preparation and schedule
    consumers still prove actual layouts, capacities and ordering.
    """
    if mode not in ('off', 'resident', 'array'):
        raise ValueError('pointwise mode must be off, resident or array')
    p = source.plan
    w = p.window
    supported = (w.kernel == (1, 1) and w.stride in ((1, 1), (2, 2))
                 and w.dilation == (1, 1) and not any(w.padding))
    diagnostics = dict(mode=mode, admitted=False, supported_source=supported,
        kernel=w.kernel, stride=w.stride, input_shape=w.input_shape,
        source_identity=source.identity, original_window_projection=True,
        original_k_order=True, timing='UNKNOWN',
        contract='capacity proposal; shared owners must revalidate and consume actual effects')
    # Unrelated source families never enter the rolling planner. In particular
    # no attempt to construct an unsupported layout can perturb their fallback.
    if mode == 'off' or not supported:
        diagnostics['reason'] = 'off' if mode == 'off' else 'unsupported source geometry'
        return retention, pairs, dict(selection, pointwise_experiment=diagnostics)
    from gemmini_conv_policy import prepare_rolling_input
    prepared, ring = prepare_rolling_input(source)
    chunks = (w.input_shape[-1] + p.channel_tile - 1) // p.channel_tile
    rhs_rows = chunks * p.channel_tile * len(source.column_intervals)
    admitted = (ring is not None and
                prepared.plan.rhs_row + rhs_rows <= p.operand_rows)
    diagnostics.update(admitted=admitted,
        prepared_source_identity=prepared.identity, complete_rhs_rows=rhs_rows,
        operand_rows=p.operand_rows, rhs_base=prepared.plan.rhs_row,
        reason='complete RHS and original input ring fit' if admitted else
               'complete RHS and original input ring do not fit')
    if not admitted:
        return retention, pairs, dict(selection, pointwise_experiment=diagnostics)
    return 'rolling_rhs', 8, dict(selection, pointwise_experiment=diagnostics)


def select_array_mode(source, selection, original_mode):
    diagnostics = selection.get('pointwise_experiment', {})
    return ('on' if diagnostics.get('admitted') and diagnostics.get('mode') == 'array'
            else original_mode)


def plan_pointwise_array_retention(schedule, rhs, input_rows, *, multirow=False):
    """Bind actual group bias-load count, then consume the normal shared owner.

    The logical issue budget bounds authored commands. Finite physical queue
    capacity, backpressure and ordering keep their unchanged selected laws.
    """
    from dataclasses import replace
    from gemmini_conv_array import plan_array_retention
    from merlin.compile.scheduling.convolution_exploration import plan_convolution_schedule
    p = schedule.source.plan
    if p.window.kernel == (1, 1) and schedule.issue_law is not None:
        m_blocks = (p.window.output_spatial[1] + p.row_tile - 1) // p.row_tile
        columns = len(schedule.source.column_intervals)
        budget = p.row_tile + 1 + m_blocks * columns * int(p.bias)
        issue = replace(schedule.issue_law,
            load_issue_budget=max(schedule.issue_law.load_issue_budget, budget))
        schedule = plan_convolution_schedule(schedule.source, schedule.choice,
            retention_law=schedule.retention_law, issue_law=issue,
            epoch_law=schedule.epoch_law)
    result=plan_array_retention(schedule, rhs, input_rows, multirow=multirow)
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    rows=knob('conv_pointwise_output_rows',default=1,values=(1,2,4,8),
              root=Path(__file__).resolve().parent)
    if rows==1:
        return result
    from gemmini_pointwise_row_phases import plan_pointwise_row_phases
    return plan_pointwise_row_phases(result,input_rows=input_rows,output_rows=rows)
