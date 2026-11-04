"""Selected uniform disjoint-ACC readout/compute admission.

This does not grant causal visibility from an unfinished producer. The checked
library pairs two output groups, moves only completed first-group stores, and
proves their ACC cells disjoint from every next-group compute/load. Original
operand completion epochs and a common real full drain remain intact.
"""
from dataclasses import replace
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.dense_panel_pipeline import DenseReadoutComputeAdmission
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission
from .column_batches import _admission
from . import isa


def enabled():
    return knob('DENSE_DISJOINT_READOUT', default='off', values=('off', 'on'),
                root=Path(__file__).resolve().parents[1]) == 'on'


def options(problem, block, storage, adapter, original, *, bias):
    """Bind selected queue facts; actual footprint proof stays with library."""
    if (bias or problem.tiles[2] // block.n < 2 or storage.accumulator is None
            or 2 * block.m * block.n * problem.tile_m > storage.accumulator.capacity):
        return original, {}, dict(applied=False, reason='requires unbiased two owned ACC groups')
    evidence = _admission()
    # _admission pins the public load/store queue map, finite ready/valid entries
    # and in-order store completion. Stable uniform CONFIG_ST is initialized at
    # the first readout after original staging completion. It never changes
    # while any store is pending. There is no varying-state snapshot premise.
    count = block.m * block.n
    overlap = replace(original.overlap, max_issued_readouts=count,
        readout_configuration='ordered_snapshot_before_store_issue')
    reduction = replace(original.reduction, max_issued_readouts=count,
        readout_configuration='ordered_snapshot_before_store_issue')
    combined = DenseSegmentedAdmission(overlap, reduction, original.qualification_identity)
    identity = 'selected-disjoint-acc-uniform-readout:' + evidence['ordering_sha256']
    law = DenseReadoutComputeAdmission(adapter.configuration_identity,
        adapter.numerical_policy_identity, identity,
        'uniform-config-until-real-full-drain:' + evidence['base_facts_sha256'],
        count, overlap.pending_readouts)
    law.validate_for(adapter, overlap)
    return combined, dict(batch_readouts=True, readout_mode='disjoint',
        readout_admission=law, readout_group_path=('outer_m', 'outer_n')), dict(
            applied=True, admission=evidence, qualification_identity=identity,
            accumulator_rows_per_group=block.m * block.n * problem.tile_m,
            accumulator_capacity_rows=isa.ACC_ROWS,
            numerical_state='one exact uniform CONFIG_ST held until every store drains',
            causal_accumulator_visibility='not assumed; producer already completed',
            ownership='library actual source/footprint proof and two disjoint translated ACC windows')
