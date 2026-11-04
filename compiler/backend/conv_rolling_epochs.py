"""Source-bound complete-K readout epochs for the original NHWC row ring.

The common consumer proves disjoint original N/ACC intervals and closes each
block with a real drain. The common terminal encoder is retained. Configuration
snapshots are unnecessary: every CONFIG_ST in this invocation is identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from . import isa


def _configuration(attributes, dtype, pitch):
    scales = attributes.get('acc_scale_by_column')
    if scales is not None and not scales:
        raise ValueError('rolling epochs require a nonempty source scale vector')
    scale = scales[0] if scales else attributes.get('acc_scale', 1.0)
    relu = bool(attributes.get('relu', False) or 'relu' in attributes.get('epilogue', ()))
    if dtype not in ('i8', 'i32') or dtype == 'i32' and (scales is not None or float(scale) != 1.0 or relu):
        raise ValueError('rolling epochs require the original supported readout')
    stride = pitch * (4 if dtype == 'i32' else 1)
    command = tuple(isa.config_st(stride, scale=scale, activation=int(relu)))
    if scales is not None and any(tuple(isa.config_st(stride, scale=value,
            activation=int(relu))) != command for value in scales):
        raise ValueError('rolling epochs require one unchanged complete store configuration')
    return command


@dataclass(frozen=True)
class RollingReadoutEpoch:
    source_identity: str
    law: object
    configuration: tuple
    admission: dict

    def store(self, ctx, dst, fields, pitch, attributes):
        from .dense_readout import store_tile
        dtype = attributes.get('output_dtype', ctx.spec(dst)['dtype'])
        if _configuration(attributes, dtype, pitch) != self.configuration:
            raise ValueError('rolling readout configuration changed after admission')
        def completed_by_block(phase):
            if phase not in ('before_config', 'after_store'):
                raise ValueError('unknown terminal completion boundary')
            # Actual ACC RAW orders the producer; an identical CONFIG_ST cannot
            # change any pending store. The checked consumer drains at block end.
        store_tile(ctx, dst, fields['accumulator_row'], fields['rows'], fields['columns'],
            pitch, fields['output_row'], fields['output_column'], attributes,
            completion=completed_by_block)


def prepare(schedule, rows, owner, attributes, dtype, pitch):
    """Pure admission. Unknown facts/configuration retain the incumbent."""
    from merlin.compile.scheduling.window_transfer import ConvolutionReadoutEpochLaw
    from merlin.compile.scheduling.convolution_exploration import _phase_blocks
    from .column_batches import _admission
    try:
        configuration = _configuration(attributes, dtype, pitch)
        if pitch != schedule.source.plan.output_channels:
            return None
        # Rechecks the selected fact digest, store/config lane, in-order store
        # completion, finite store capacity and ready/valid backpressure.
        admission = _admission()
        epoch = schedule.epoch_law
        if epoch is None or getattr(owner, 'input_rows', rows) != rows:
            return None
        phases = getattr(owner, 'phases', (owner,))
        blocks = tuple(block for phase in phases
            for block in _phase_blocks(schedule, phase.column_intervals))
        if not blocks or max(map(len, blocks)) < 2:
            return None
        law = ConvolutionReadoutEpochLaw(epoch.hardware_identity,
            epoch.numerical_policy_identity, schedule.source.identity,
            'original-row-ring-uniform-config-private-n-readout-v1',
            max_issued_readouts=max(map(len, blocks)),
            pending_readouts=admission['finite_store_entries'],
            local_read_after_write=epoch.accumulator_result_visibility,
            accumulator_result_visibility=epoch.accumulator_result_visibility,
            ordered_configuration_capture=True, finite_queue_progress=True,
            complete_device_drain=True, exclusive_private_state=True)
        for block in blocks:
            law.require(schedule.source.plan, schedule.source.plan.row_tile, block)
        owner.verify_current()
        return RollingReadoutEpoch(schedule.source.identity, law, configuration,
            dict(admission, configuration=list(configuration),
                store_configuration_snapshot='not assumed; all complete CONFIG_ST words identical',
                completion='actual producer-to-store RAW; real block-final drain before any reuse'))
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
