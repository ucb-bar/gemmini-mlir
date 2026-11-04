"""Isolated pointwise lifetime proposal consumed by the ordinary conv emitter.

This does not flatten a stride-two source to a contiguous dense matrix. The
original Window2D remains authoritative for source M projection and K order.
The shared rolling-input, RHS, epoch and optional output-group owners perform
the actual transformations and retain their normal target effect consumer.
"""
import json
from pathlib import Path
from gemmini_conv_policy import prepare_rolling_input


def select_pointwise(source, retention, pairs, selection):
    mode = json.loads((Path(__file__).resolve().parent /
                       'POINTWISE_EXPERIMENT.json').read_text())['mode']
    if mode not in ('resident', 'array'):
        raise ValueError('unknown isolated pointwise alternative')
    p = source.plan
    w = p.window
    prepared, ring = prepare_rolling_input(source)
    chunks = (w.input_shape[-1] + p.channel_tile - 1) // p.channel_tile
    rhs_rows = chunks * p.channel_tile * len(source.column_intervals)
    legal = (w.kernel == (1, 1) and w.stride in ((1, 1), (2, 2))
             and w.dilation == (1, 1) and not any(w.padding)
             and ring is not None
             and prepared.plan.rhs_row + rhs_rows <= p.operand_rows)
    evidence = dict(mode=mode, admitted=legal, kernel=w.kernel,
        stride=w.stride, input_shape=w.input_shape,
        source_identity=source.identity, prepared_source_identity=prepared.identity,
        complete_rhs_rows=rhs_rows, operand_rows=p.operand_rows,
        rhs_base=prepared.plan.rhs_row, original_window_projection=True,
        original_k_order=True, timing='UNKNOWN',
        contract='capacity proposal; shared owners must revalidate and consume actual effects')
    if not legal:
        return retention, pairs, dict(selection, pointwise_experiment=evidence)
    return 'rolling_rhs', 8, dict(selection, pointwise_experiment=evidence)


def select_array_mode(source, selection, original_mode):
    evidence = selection.get('pointwise_experiment', {})
    return ('on' if evidence.get('admitted') and evidence.get('mode') == 'array'
            else original_mode)


def plan_pointwise_array_retention(schedule, rhs, input_rows, *, multirow=False):
    from dataclasses import replace
    from gemmini_conv_array import plan_array_retention
    from merlin.compile.scheduling.convolution_exploration import plan_convolution_schedule
    p = schedule.source.plan
    # The ordinary one-row shared group delays all of its bias seeds until
    # initialization completes. Its logical issue budget must cover those
    # actual loads. This is not the finite physical queue capacity; the
    # unchanged selected law supplies backpressure/progress separately.
    if p.window.kernel == (1, 1) and schedule.issue_law is not None:
        m_blocks = (p.window.output_spatial[1] + p.row_tile - 1) // p.row_tile
        columns = len(schedule.source.column_intervals)
        budget = p.row_tile + 1 + m_blocks * columns * int(p.bias)
        issue = replace(schedule.issue_law,
            load_issue_budget=max(schedule.issue_law.load_issue_budget, budget))
        schedule = plan_convolution_schedule(schedule.source, schedule.choice,
            retention_law=schedule.retention_law, issue_law=issue,
            epoch_law=schedule.epoch_law)
    return plan_array_retention(schedule, rhs, input_rows, multirow=multirow)
