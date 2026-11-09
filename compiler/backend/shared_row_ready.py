"""Exact private-row refinement of a checked causal-ready dense consumer.

This is not a new scheduling law. A combined transfer becomes consecutive
disjoint subrectangles at the same issue position. The union is the original
rectangle; no producer, overwrite, configuration change or drain crosses it.
Finite store backpressure admits the expanded command count, rather than
pretending the combined consumer's issued-store bound counts physical stores.
"""
from dataclasses import asdict
from merlin.sched.ir.kernel import instances
from merlin.sched.ir.expr import evaluate
from merlin.compiler.schedule.dense_panel_pipeline import DRAIN
from merlin.compiler.schedule.dense_ready_readout import DenseReadyReadoutResult


def check_ready_partition(schedule, kernel, specs, starts, actual):
    schedule.revalidate()
    owner=schedule
    # Only the existing final-band wrapper may surround the typed ready owner.
    from .dense_rhs_transfers import DenseRhsTransferResult
    from merlin.compiler.schedule.panel_tile_loads import PanelTileLoadResult
    if type(owner) is DenseRhsTransferResult:
        owner=owner.checked
    if type(owner) is PanelTileLoadResult:
        owner=owner.original
    if type(owner) is not DenseReadyReadoutResult:
        raise ValueError('row-ready partition needs the actual typed causal consumer')
    owner.revalidate()
    binding=schedule.realization.get('ready_readout_target_binding',{})
    if (binding.get('ordering_sha256')!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
            or not binding.get('actual_accumulator_waits')):
        raise ValueError('row-ready partition lacks selected ACC completion edges')
    index=0; max_stores=0; epoch_stores=0; drains=0
    for call,env in instances(schedule.kernel):
        a={key:evaluate(value,env) for key,value in call.args}
        if call.instr not in ('gemmini_mvin_lhs','gemmini_mvout'):
            if actual[index]!=(call.instr,a):
                raise ValueError('row-ready partition moved a nonpartitioned effect')
            index+=1
            if call.instr==DRAIN:
                max_stores=max(max_stores,epoch_stores);epoch_stores=0;drains+=1
            continue
        lhs=call.instr=='gemmini_mvin_lhs'
        field='addr' if lhs else 'acc'
        cursor=a[field]
        for member,(m,_,_) in enumerate(specs):
            instruction,b=actual[index];index+=1
            expected=dict(a,rows=m,dram_row=0)
            expected[field]=cursor
            if (instruction!=f'row_{"lhs" if lhs else "store"}_{member}'
                    or b!=expected or cursor!=a[field]+starts[member]):
                raise ValueError('row-ready partition changed source or local rectangle')
            cursor+=m
            epoch_stores+=int(not lhs)
        if cursor!=a[field]+a['rows']:
            raise ValueError('row-ready subrectangles do not cover the original effect')
    if index!=len(actual) or epoch_stores:
        raise ValueError('row-ready partition changed final completion boundary')
    # All original producer-to-store and store-to-overwrite edges survive:
    # each member footprint is a subset and their disjoint union is exact.
    # No new load/execute overlap or numerical-state snapshot is introduced.
    schedule.revalidate()
    return dict(typed_consumer=type(owner).__name__,
        ready_kernel_sha256=owner.kernel.digest(),partitioned_kernel_sha256=kernel.digest(),
        original_ready_admission=asdict(owner.admission),
        physical_max_issued_stores_per_epoch=max_stores,
        original_drains_preserved=drains,
        target_binding=binding,
        queue_rule='ordered store completion with finite ready/valid backpressure; expanded stores are not simultaneous queue occupancy',
        refinement='consecutive disjoint subrectangles exactly cover each checked DMA/store before any later effect; all producer and reuse edges preserved')
