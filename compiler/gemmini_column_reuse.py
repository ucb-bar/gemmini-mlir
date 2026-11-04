"""Opt-in source-qualified input retention for counted singleton columns.

The caller supplies the original target's synchronous storage-effect law. This
adapter grants no ordering, alias or ownership premise from shapes alone.
"""
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionSchedule, ConvolutionScheduleChoice, plan_convolution_schedule,
)
from merlin.compile.scheduling.window_transfer import ConvolutionInputRetentionLaw


def plan_column_reuse(schedule, *, law, live_tiles):
    """Keep one input K tile across bounded independent original N reductions.

    Consume the shared retention pass before counted realization. Every compute,
    weight transfer and final readout still completes synchronously. ``law``
    must prove that these effects preserve private LHS rows and that the source
    buffers are immutable, mapped and disjoint. Missing premises are refusals.
    """
    if type(schedule) is not ConvolutionSchedule:
        raise ValueError('column reuse requires an original convolution schedule')
    schedule.verify_current()
    p = schedule.source.plan
    if (type(law) is not ConvolutionInputRetentionLaw
            or law.source_identity != schedule.source.identity
            or not all((law.input_preserved, law.independent_outputs,
                        law.immutable_mapped_disjoint_buffers))):
        raise ValueError('column reuse lacks original source/storage lifetime proof')
    law.__post_init__()
    if (type(live_tiles) is not int or not 1 < live_tiles <= law.max_live_tiles
            or live_tiles * p.row_tile + p.accumulator_row > p.accumulator_rows):
        raise ValueError('column reuse exceeds checked live accumulator capacity')
    if (schedule.choice.load_issue or schedule.choice.compute_pairs != 1
            or schedule.issue_law is not None or schedule.epoch_law is not None
            or schedule.source.column_intervals != tuple((c, 1) for c in range(p.output_channels))):
        raise ValueError('column reuse needs the original synchronous singleton source stream')
    return plan_convolution_schedule(schedule.source,
        ConvolutionScheduleChoice(live_tiles, False, 1), retention_law=law)
