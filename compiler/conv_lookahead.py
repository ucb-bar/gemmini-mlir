"""Selected last-use RHS phase transport for singleton rolling consumers.

The source schedule and phase owner establish addresses and exact K coverage.
This adapter supplies no timing claim and changes no local allocation. It only
admits the shared transport pass where its actual consumer uses fresh weights
for a single M tile, so the terminal encoder needs no retained-array opcode.
"""
from dataclasses import dataclass

from merlin.compile.scheduling.convolution_exploration import ConvolutionRhsPhases
from merlin.compile.scheduling.rolling_lookahead import RollingRhsFrontierAdmission


@dataclass(frozen=True)
class ConvolutionRhsLookahead:
    groups: tuple
    law: RollingRhsFrontierAdmission

    def record(self):
        return dict(schema='authored.convolution_rhs_last_use.v1',
            phase_groups=[dict(row_blocks=g.row_blocks,output_rows=g.output_rows)
                          for g in self.groups],
            mechanism='last_use_rhs_phase_transport',
            local_allocation='unchanged original RHS phase and rolling input ring',
            frontier='last original batch/output row; after each complete original K/N reader traversal',
            completion='all original preparation/group/phase drains retained',
            consumer='one M tile, every compute consumes original fresh resident weights',
            qualification='selected dependency legality only; complete-call timing remains separate')


def plan_rhs_lookahead(schedule, rhs, input_rows, ordering):
    """Return a checked plan or a precise pre-emission applicability refusal."""
    p=schedule.source.plan
    if type(rhs) is not ConvolutionRhsPhases or len(rhs.phases)<2:
        return None,'complete RHS already fits; no next original phase'
    if input_rows is None:
        return None,'original input has no checked resident rolling rows'
    if not ordering.get('admitted') or schedule.epoch_law is None:
        return None,'selected native local dependency/epoch admission unavailable'
    if len(schedule.column_blocks)!=1:
        return None,'frontier consumer requires one complete live output-column block'
    if (p.window.output_spatial[1]>p.row_tile or p.column_tile!=p.primitive
            or p.channel_tile!=p.primitive):
        return None,'singleton frontier requires one original primitive column/channel M tile per row'
    # The same selected PRELOAD(weight)->MVIN2 WAR and MVIN2->PRELOAD RAW
    # completion edges used by the original epoch cover overwritten dead RHS
    # rows. ConfigMvin shares the native load lane; finite native queues retain
    # backpressure, and every original real full drain remains present.
    required={('PRELOAD.weight_spad','MVIN2.spad_addr','scratchpad'),
              ('MVIN2.spad_addr','PRELOAD.weight_spad','scratchpad')}
    if not required.issubset({tuple(x) for x in ordering.get('required_completion_edges',())}):
        return None,'selected native weight read/write completion ranges unavailable'
    from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
    groups=plan_rolling_convolution_output_group(schedule,rhs,input_rows,
        max_row_blocks=1,max_output_rows=1,allow_singleton=True)
    if (not isinstance(groups,tuple) or len(groups)!=len(rhs.phases)
            or any(g is None or g.row_blocks!=1 or g.output_rows!=1 for g in groups)):
        return None,'shared planner refused exact singleton phase ownership'
    law=RollingRhsFrontierAdmission(schedule.epoch_law.hardware_identity,
        schedule.epoch_law.numerical_policy_identity,
        'native-ordering:'+ordering['ordering_sha256'],
        scratchpad_read_write_dependencies=True,ordered_configuration=True,
        finite_queue_progress=True,immutable_weights=True,
        full_consumer_transport_completion=True)
    law.require()
    return ConvolutionRhsLookahead(groups,law),None
