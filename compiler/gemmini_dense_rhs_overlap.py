"""Explicit experimental composition: checked RHS panels, then ready readout.

No selection/default/emission policy is changed. Both transformations retain
original source axes and private storage; consumers use the common dense terminal.
"""
from dataclasses import asdict

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_panel_pipeline import DenseReadyReadoutAdmission
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_ready_readout
from merlin.transforms.dense_planning import DenseResources, dense_blocks
from gemmini_dense_readout import ReadoutUnavailable
from gemmini_dense_rhs_panels import plan_rhs_panels


def plan_rhs_ready_readout(incumbent, adapter, problem, geometry, facts_path, *,
                           bias=False, initial=False, outer_order=('m', 'n')):
    """Return (checked result, receipt); refuse before emission if unsupported.

    First consume the original RHS invariant-hoist/full-K plan, then the shared
    exact final-producer/store-frontier transform with a separate causal law.
    No new load/compute overlap, capacity or array-weight persistence is granted.
    """
    from gemmini_dense_pipeline import ordering_premises
    retained, receipt = plan_rhs_panels(incumbent, adapter, problem, geometry,
        facts_path, bias=bias, initial=initial, outer_order=outer_order)
    edge, operand, operand_rows, accumulator, accumulator_rows = geometry
    split = operand_rows // 2
    storage = DenseLoopStorage(
        DenseMemoryView('lhs', operand.name, operand.row_bytes, 0, split),
        DenseMemoryView('rhs', operand.name, operand.row_bytes, split, operand_rows-split),
        DenseMemoryView('acc', accumulator.name, accumulator.row_bytes, 0, accumulator_rows))
    bm, bk, bn = receipt['block']
    resources = DenseResources(operand_rows*operand.row_bytes,
        accumulator_rows*accumulator.row_bytes, max_m_blocks=bm,
        max_k_blocks=bk, max_n_blocks=bn, partial_output_tiles=True,
        partial_output_blocks=True)
    block = next((b for b in dense_blocks(problem, resources,
        k_limit=lambda m,n: storage.k_limit(problem, resources, m, n))
        if (b.m,b.k,b.n) == (bm,bk,bn)), None)
    if block is None:
        raise ReadoutUnavailable('RHS ready readout cannot recover original checked storage/block')
    flags, pending, causal = ordering_premises(facts_path)
    required = ('ordered_accumulator_writes', 'accumulator_result_visibility',
        'ordered_configuration_capture', 'finite_queue_progress',
        'complete_device_drain', 'exclusive_private_state')
    if not all(flags[key] for key in required):
        raise ReadoutUnavailable('RHS ready readout lacks separate causal completion premises')
    identity = 'selected-command-ordering:' + causal['ordering_sha256']
    law = DenseReadyReadoutAdmission(adapter.configuration_identity,
        adapter.numerical_policy_identity,
        identity+':descriptor-acc-write-to-store-completion',
        identity+':ordered-store-configuration', bm*bn, pending['store'])
    try:
        result = schedule_dense_ready_readout(retained, problem, block, resources,
            storage, adapter, law, static_budget=256, timeout_ms=60000)
    except ValueError as error:
        raise ReadoutUnavailable('checked RHS retention/ready composition declined: '+str(error)) from error
    result.revalidate()
    if result.kernel.digest() == retained.kernel.digest():
        raise ReadoutUnavailable('RHS ready composition did not change actual checked commands')
    receipt = dict(receipt, mode='rhs_panels_ready',
        library='plan_rhs_panels -> schedule_dense_ready_readout',
        retained_schedule_sha256=retained.kernel.digest(),
        transformed_schedule_sha256=result.kernel.digest(),
        ready_readout_admission=asdict(law), ready_source_and_epochs=result.realization,
        causal_edge=causal['basis']['accumulator_result_visibility'],
        qualification_identity=None, timing='UNKNOWN; no cost ranking inferred from load/drain counts')
    return result, receipt
