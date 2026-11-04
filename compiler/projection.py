"""Capacity-guarded raw integer projection construction and completion composition.

No test identity selects this route. Retain the existing M/N block, the original
LHS and RHS private views and the existing RHS two-window issue order. Reuse is
profitable structurally only when full-K LHS fits and its original reload count
exceeds the new count. Numerical/performance qualification is independent.
"""
from dataclasses import asdict, replace
from pathlib import Path
import json

from merlin.compile.scheduling.compact_dense import _digest
from merlin.compile.scheduling.dense_panel_pipeline import verify_dense_panel_pipeline
from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts
from merlin.compile.scheduling.dense_segmented_pipeline import build_dense_segmented_pipeline, _assignment
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan, read_dense_schedule_plan


def construct_projection(original, problem, block, resources, storage, adapter):
    mt, kt, _ = problem.tiles
    if problem.bias or original.output_readout is not None or original.bias_source is not None:
        original.realization['projection_refusal'] = 'requires raw bias-free i32 product'
        return None
    if kt <= 1:
        original.realization['projection_refusal'] = 'no internal K stream to overlap'
        return None
    needed = block.m * kt * problem.tile_m
    unique = mt * kt
    if needed > storage.lhs.capacity:
        original.realization['projection_refusal'] = 'full-K LHS panel exceeds original private view'
        return None
    if 2 * block.n * problem.tile_k > storage.rhs.capacity:
        original.realization['projection_refusal'] = 'two RHS windows exceed original private view'
        return None
    if original.realization['source']['lhs_loads'] <= unique:
        original.realization['projection_refusal'] = 'no redundant LHS transfers to remove'
        return None
    _assignment(problem, block, resources, storage)
    plan = make_dense_schedule_plan(problem,resources,storage,adapter,
        source_identity=original.source_identity,dtypes=('i8','i8','i32'),static_budget=128,
        block_tiles=(block.m,1,block.n),pipeline_kind='retained_lhs_segmented_k',
        pipeline_prefetch=True)
    # Parse and consume the actual scheduling dialect before constructing the
    # emitted stream. The selected decision drives block, storage, prefetch,
    # issue order.
    decision = read_dense_schedule_plan(plan,problem,resources,storage,adapter,
        source_identity=original.source_identity,dtypes=('i8','i8','i32'),static_budget=128)
    actual = build_dense_segmented_pipeline(problem,decision.block,resources,decision.storage,adapter,
        name='gemmini_projection_retained_lhs',dtypes=('i8','i8','i32'),
        prefetch=decision.pipeline_prefetch,issue_order=decision.pipeline_issue_order)

    # A scoped issued bound for ONE complete output block, never a pending-queue
    # capacity. Its kt steps issue <= kt*bm*bn pairs and kt*bn RHS loads.
    # The separate retained-LHS staging epoch issues <= bm*kt loads. Finite
    # ready/valid queue progress comes from the shared selected ordering laws.
    # The completion consumer cannot enlarge these bounds.
    admission = replace(original.admission,
        max_issued_compute_pairs=kt * block.m * block.n,
        max_issued_loads=max(block.m * kt, kt * block.n))
    report = verify_dense_panel_pipeline(actual,problem,block,resources,storage,adapter,admission,
        static_budget=128,timeout_ms=15000)
    report['schedule_ir'] = dict(text=decision.plan_text,sha256=decision.plan_sha256,
        source_identity=original.source_identity,pipeline_kind=decision.pipeline_kind,
        prefetch=decision.pipeline_prefetch,
        loop_transforms=[asdict(t) for t in decision.loop_transforms],
        decisions_consumed=['block_tiles','row_bases','panel_pipeline'],
        construction='build_dense_segmented_pipeline from parsed scheduling decision')
    report['projection_construction'] = dict(lhs_private_rows=storage.lhs.capacity,
        lhs_used_rows=needed,original_lhs_loads=original.realization['source']['lhs_loads'],
        retained_lhs_loads=report['source']['lhs_loads'],block=asdict(block),
        issued_scope=dict(output_block_k_tiles=kt,compute_pairs=kt*block.m*block.n,
            rhs_loads=kt*block.n,retained_staging_loads=block.m*kt),
        pending_capacity='unchanged selected physical queues; ready/valid backpressure')
    checked = replace(original,kernel=actual,realization=report,admission=admission,
        checked_kernel_sha256=actual.digest(),checked_admission_sha256=_digest(asdict(admission)))
    checked.revalidate()
    # Select the same readout construction in both conservative and epoch arms.
    checked = batch_dense_panel_readouts(checked,problem,block,resources,storage,adapter,admission,
        static_budget=128,timeout_ms=15000)
    from merlin.compile.tuning_knobs import knob
    if not knob('projection_completion', default=True, values=(False,True),root=Path(__file__).parent):
        return checked
    from completion_adapter import apply_completion
    premises = json.loads((Path(__file__).parent/'completion_premises.json').read_text())
    # Program obligations: these private views have been verified by _assignment
    # and the full source/epoch checker. Each call owns them until its final
    # drain; lower_matmul emits target.fence for every completion callback.
    premises['exclusive_private_state'] = True
    premises['complete_device_drain'] = True
    return apply_completion(checked,dict(problem=problem,block=block,resources=resources,
        storage=storage,adapter=adapter,admission=admission,premises=premises))
