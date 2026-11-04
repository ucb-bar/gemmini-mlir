"""Optional full-K output-panel lookahead through the checked shared scheduler.

This uses the existing ordered-commit/disjoint-future-load conjunction and real
full drains. It creates neither per-copy completion tokens nor new target
overlap permissions. The ordinary selected dense plan remains the fallback.
"""
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView, DenseBiasSource
from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission, DenseRetainedReductionAdmission
from merlin.compile.scheduling.dense_segmented_pipeline import (
    DenseSegmentedAdmission, OUTPUT_GROUP_PREFETCH, dense_segmented_blocks,
    schedule_dense_segmented_pipeline,
)
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.transforms.dense_planning import DenseProblem, DenseResources

from . import isa


@lru_cache(maxsize=64)
def output_prefetch_schedule(m, k, n, bias=False, column_scales=False, geometry='wide', selection='full_k_groups'):
    """Construct an actual fully checked alternative, or an explicit refusal."""
    from .dense import _adapter
    from .dense_pipeline import PANEL_GEOMETRIES, SEGMENTED_VERIFICATION_TIMEOUT_MS, _record_geometry
    if column_scales:
        return None, 'output-panel prefetch preserves the ordinary uniform readout boundary'
    if m > isa.DIM:
        return None, 'output-panel prefetch experiment covers one primitive M tile'
    dim = isa.DIM
    problem = DenseProblem(m, k, n, dim, dim, dim, 1, 1, 4, bias=bias)
    storage = DenseLoopStorage(
        DenseMemoryView('lhs', 'spad', dim, 0, isa.SPAD_ROWS//2),
        DenseMemoryView('rhs', 'spad', dim, isa.SPAD_ROWS//2, isa.SPAD_ROWS//2),
        DenseMemoryView('acc', 'acc', 4*dim, 0, isa.ACC_ROWS))
    max_m, max_n = PANEL_GEOMETRIES[geometry]
    resources = DenseResources(isa.SPAD_ROWS*dim, isa.ACC_ROWS*4*dim,
        max_m_blocks=max_m, max_n_blocks=max_n, max_k_blocks=problem.tiles[1],
        partial_output_tiles=True, partial_output_blocks=True)
    if selection == 'column_groups' and 2*problem.tiles[1]*dim > storage.rhs.capacity:
        return None, 'column-frontier experiment retains the original full-K two-RHS-panel region'
    if selection == 'column_groups' and problem.tiles[1] < 2:
        return None, 'column frontier requires more than one primitive reduction tile'
    if selection == 'column_groups':
        resources = replace(resources, max_k_blocks=min(16, problem.tiles[1]-1))
    choices = tuple(block for block in dense_segmented_blocks(problem, resources, storage)
        if (selection == 'column_groups' or block.k == problem.tiles[1]) and block.n < problem.tiles[2])
    if not choices:
        return None, 'two disjoint RHS windows do not fit assigned half storage with multiple output groups'
    block = min(choices, key=lambda b: (b.loads, -b.m*b.n, b.m, b.n))
    adapter = _adapter()
    ordering = json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if ordering['completion_order']['execute']['order'] != 'in_order':
        return None, 'selected execute completion order is unknown'
    issued_loads = max(1, block.m*problem.tiles[1], block.k*block.n)
    reduction = DenseRetainedReductionAdmission(adapter.configuration_identity,
        adapter.numerical_policy_identity, ordering['identity'],
        max(1, block.m*block.k*block.n), issued_loads, 2, 8)
    overlap = DensePanelOverlapAdmission(adapter.configuration_identity,
        adapter.numerical_policy_identity, ordering['identity']+':disjoint-owned-operand-windows',
        reduction.max_issued_compute_pairs, reduction.max_issued_loads, 2, 8)
    admission = DenseSegmentedAdmission(overlap, reduction,
        ordering['identity']+':pure-inorder-execute-with-disjoint-future-rhs-loads')
    source = sha256(json.dumps([m,k,n,bias,'i8-i8-i32'],sort_keys=True).encode()).hexdigest()
    bias_source = DenseBiasSource(n, 'i32') if bias else None
    plan = make_dense_schedule_plan(problem, resources, storage, adapter,
        source_identity=source, dtypes=('i8','i8','i32'), static_budget=128,
        block_tiles=(block.m,block.k,block.n),
        pipeline_kind='retained_lhs_segmented_k', pipeline_prefetch=True,
        pipeline_issue_order=(OUTPUT_GROUP_PREFETCH if selection == 'full_k_groups' else 'prefetch_rhs_between_columns'), bias_source=bias_source)
    try:
        result = schedule_dense_segmented_pipeline(problem, resources, storage, adapter,
            admission, name='dense_output_prefetch', dtypes=('i8','i8','i32'),
            schedule_plan=plan, source_identity=source, external_buffers_disjoint=True,
            static_budget=128, timeout_ms=SEGMENTED_VERIFICATION_TIMEOUT_MS,
            bias_source=bias_source)
    except DenseVerificationUnavailable as error:
        return None, 'output-panel prefetch proof unavailable: '+str(error)
    _record_geometry(result, geometry, block, problem, resources, storage)
    result.realization['output_prefetch'] = dict(
        selection=selection, full_k_tiles=problem.tiles[1], segment_k_tiles=block.k, rhs_slots=2,
        rhs_slot_rows=block.k*block.n*dim, retained_lhs_rows=block.m*problem.tiles[1]*dim,
        numerical_boundary='original complete i8*i8 reduction into i32',
        completion='full epoch drains before RHS and accumulator reuse',
        qualification=admission.qualification_identity)
    result.revalidate()
    return result, None


def select_output_prefetch(schedule, attributes=None, *, m, k, n, bias=False,
        column_scales=False, geometry=None):
    """Call after dense selection and before store/completion epoch transforms."""
    from .dense_pipeline import PANEL_GEOMETRIES
    attributes = attributes or {}
    root = Path(__file__).resolve().parents[1]
    selection = attributes.get('dense_output_prefetch', os.environ.get('DENSE_OUTPUT_PREFETCH',
        knob('DENSE_OUTPUT_PREFETCH', default='off', values=('off','full_k_groups','column_groups'), root=root)))
    if selection not in ('off','full_k_groups','column_groups'):
        raise ValueError('unsupported dense output prefetch selection')
    if selection == 'off':
        return schedule, None
    schedule.revalidate()
    original_block = schedule.realization.get('panel_geometry', {}).get('selected_block_tiles')
    original_multiple_groups = (original_block is not None
        and original_block[2] < (n+isa.DIM-1)//isa.DIM)
    if not original_multiple_groups:
        return schedule, dict(schema='authored_dense_output_prefetch.v1',
            selection=selection, applied=False,
            refusal='original checked incumbent does not establish multiple N output groups',
            source_kernel_sha256=schedule.kernel.digest(), candidate_kernel_sha256=None,
            timing='UNKNOWN; incumbent preserved')
    if selection == 'column_groups':
        original_bounds = schedule.realization.get('segmented_epoch_bounds') or {}
        original_partition = schedule.realization.get('storage_partition') or {}
        original_issue = (schedule.realization.get('frontier_issue') or {}).get('issue_order')
        expected_lhs_rows = original_block[0]*((k+isa.DIM-1)//isa.DIM)*isa.DIM
        if not (original_partition.get('selection') == 'half'
                and original_issue == 'prefetch_rhs_between_k_tiles'
                and original_bounds.get('rhs_slots') == 2
                and original_bounds.get('retained_lhs_rows') == expected_lhs_rows):
            return schedule, dict(schema='authored_dense_output_prefetch.v1',
                selection=selection, applied=False,
                refusal='column frontier preserves the original half-storage full-LHS two-RHS epoch topology',
                source_kernel_sha256=schedule.kernel.digest(), candidate_kernel_sha256=None,
                timing='UNKNOWN; incumbent preserved')
    geometry = geometry or os.environ.get('DENSE_PANEL_GEOMETRY',
        knob('DENSE_PANEL_GEOMETRY', default='balanced', values=tuple(PANEL_GEOMETRIES), root=root))
    candidate, refusal = output_prefetch_schedule(m,k,n,bias,column_scales,geometry,selection)
    if candidate is not None and selection == 'column_groups':
        candidate_block = candidate.realization.get('panel_geometry', {}).get('selected_block_tiles')
        if tuple(candidate_block or ()) != tuple(original_block):
            candidate, refusal = None, 'column-frontier alternative must preserve original M/K/N blocking'
    report = dict(schema='authored_dense_output_prefetch.v1', selection=selection,
        applied=candidate is not None, refusal=refusal,
        source_kernel_sha256=schedule.kernel.digest(),
        candidate_kernel_sha256=None if candidate is None else candidate.kernel.digest(),
        target_execution='original ordered accumulator commits, disjoint future loads and full drain',
        timing='UNKNOWN; requires matched complete-call comparison')
    if candidate is None:
        return schedule, report
    if candidate.kernel.args != schedule.kernel.args:
        raise ValueError('output-panel prefetch changed original source ABI')
    # Detached diagnostic metadata does not replace candidate.revalidate().
    candidate.realization['output_prefetch_selection'] = report
    return candidate, report
