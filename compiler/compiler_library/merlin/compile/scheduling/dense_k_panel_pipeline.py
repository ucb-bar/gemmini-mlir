"""Source-checked bounded K panels in two independently owned operand slots.

The shared primitive panel builder and terminal emitter remain unchanged.
Multiple reductions in an epoch consume an explicit ordered-accumulator plus
future-disjoint-load conjunction, rather than extending the primitive law.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from .compact_dense import DenseLoopStorage, _digest
from .dense_panel_pipeline import (DensePanelPipelineResult,
    _build_dense_panel_pipeline_from_windows, _verify_dense_epochs, _calls)
from .dense_segmented_pipeline import DenseSegmentedAdmission
from .library_sources import admission_point, shared_library_sources
from merlin.transforms.dense_planning import dense_blocks


def k_panel_windows(problem, block, resources, storage):
    if (storage.accumulator is None or block.k<1
            or block not in dense_blocks(problem,resources)
            or not storage.admits(problem,block,resources)):
        raise ValueError('K panel differs from original source/resource block derivation')
    rows=(block.m*block.k*problem.tile_m,block.k*block.n*problem.tile_k)
    parents=(storage.lhs,storage.rhs)
    if any(2*count>view.capacity for count,view in zip(rows,parents)):
        raise ValueError('two complete K panels exceed assigned operand windows')
    if 2*sum(count*view.unit_bytes for count,view in zip(rows,parents))>resources.operand_bytes:
        raise ValueError('simultaneously live K panels exceed original operand bytes')
    slots=tuple(DenseLoopStorage(*[replace(view,base=view.base+slot*count,capacity=count)
        for view,count in zip(parents,rows)],storage.accumulator) for slot in (0,1))
    assigned=[v for slot in slots for v in (slot.lhs,slot.rhs)]+[storage.accumulator]
    for i,a in enumerate(assigned):
        if a.base%a.alignment:raise ValueError('K-panel window violates original row alignment')
        for b in assigned[i+1:]:
            if (a.backing==b.backing and a.base*a.unit_bytes<(b.base+b.capacity)*b.unit_bytes
                    and b.base*b.unit_bytes<(a.base+a.capacity)*a.unit_bytes):
                raise ValueError('simultaneously live K-panel windows alias physical bytes')
    return slots


def build_dense_k_panel_pipeline(problem,block,resources,storage,adapter,*,name,dtypes,
        prefetch=True,outer_order=("m","n"),output_readout=None,bias_source=None,
        issue_order="compute_then_prefetch",window_offsets=None):
    if block.k>1 and problem.k%(block.k*problem.tile_k):
        raise ValueError('K-panel overlap requires complete original K panels; partial K uses the primitive route')
    if any(dtype not in ('i8','i16','i32','i64') for dtype in dtypes):
        raise ValueError('K-panel family requires explicit signed integer dtypes')
    if issue_order!="compute_then_prefetch" or window_offsets is not None:
        raise ValueError('K panels require ordered compute and derived private windows')
    if type(prefetch) is not bool:
        raise ValueError('K panels require explicit Boolean prefetch')
    slots=k_panel_windows(problem,block,resources,storage)
    return _build_dense_panel_pipeline_from_windows(problem,block,resources,storage,
        adapter,slots,name=name,dtypes=dtypes,prefetch=prefetch,outer_order=outer_order,
        output_readout=output_readout,bias_source=bias_source,issue_order=issue_order)


@dataclass(frozen=True)
class DenseKPanelPipelineResult(DensePanelPipelineResult):
    checked_k_panel_owner_sha256: str = ''

    def revalidate(self):
        super().revalidate()
        if (type(self.admission) is not DenseSegmentedAdmission
                or sha256(Path(__file__).read_bytes()).hexdigest()!=self.checked_k_panel_owner_sha256):
            raise ValueError('K-panel owner or explicit combined source/execute law changed')
        self.admission.__post_init__()


@admission_point('merlin.compile.scheduling.dense_k_panel_pipeline.schedule_dense_k_panel_pipeline',
    fallback_route='caller-selected checked original schedule; no unavailable proposal emitted')
def schedule_dense_k_panel_pipeline(problem,resources,storage,adapter,admission,*,
        name,dtypes,schedule_plan,source_identity,external_buffers_disjoint,
        static_budget=128,timeout_ms=5000,bias_source=None):
    from .schedule_ir import read_dense_schedule_plan,check_dense_schedule_commands
    if external_buffers_disjoint is not True:
        raise ValueError('K-panel original external buffers must be independently disjoint')
    if type(admission) is not DenseSegmentedAdmission:
        raise ValueError('K-panel reductions need explicit combined execution admission')
    admission.__post_init__()
    if (admission.overlap.hardware_identity!=adapter.configuration_identity
            or admission.overlap.numerical_policy_identity!=adapter.numerical_policy_identity):
        raise ValueError('K-panel execution law differs from selected original adapter')
    decision=read_dense_schedule_plan(schedule_plan,problem,resources,storage,adapter,
        source_identity=source_identity,dtypes=dtypes,static_budget=static_budget,
        output_readout=None,bias_source=bias_source)
    if (decision.pipeline_kind!='k_panel_two_window' or decision.pipeline_prefetch is None
            or decision.pipeline_issue_order!='compute_then_prefetch'
            or decision.pipeline_window_offsets is not None or decision.loop_transforms):
        raise ValueError('K-panel consumer requires its exact bounded two-window decision')
    slots=k_panel_windows(problem,decision.block,resources,decision.storage)
    kernel=_build_dense_panel_pipeline_from_windows(problem,decision.block,resources,
        decision.storage,adapter,slots,name=name,dtypes=dtypes,
        prefetch=decision.pipeline_prefetch,outer_order=decision.outer_order,bias_source=bias_source)
    check_dense_schedule_commands(decision,sum(1 for _ in _calls(kernel.body)),static_budget)
    bound=replace(admission.overlap,
        max_issued_compute_pairs=min(admission.overlap.max_issued_compute_pairs,
            admission.reduction.max_issued_compute_pairs),
        max_issued_loads=min(admission.overlap.max_issued_loads,admission.reduction.max_issued_loads),
        max_issued_readouts=min(admission.overlap.max_issued_readouts,admission.reduction.max_issued_readouts))
    report=_verify_dense_epochs(kernel,problem,resources,decision.storage,adapter,bound,
        static_budget=static_budget,timeout_ms=timeout_ms,bias_source=bias_source,
        ordered_accumulator_commits=True)
    report.update(schema='merlin.dense_k_panel_pipeline.v1',admission=asdict(admission),
        ordered_accumulator_commits=admission.qualification_identity,
        k_panel_windows=[[(v.base,v.capacity) for v in (slot.lhs,slot.rhs)] for slot in slots],
        schedule_ir=dict(sha256=decision.plan_sha256,text=decision.plan_text,
            source_identity=source_identity,pipeline_kind=decision.pipeline_kind,
            pipeline_issue_order=decision.pipeline_issue_order,
            decisions_consumed=('block_tiles','outer_order','row_bases','panel_pipeline')))
    result=DenseKPanelPipelineResult(kernel,report,source_identity,admission,
        kernel.digest(),_digest(asdict(admission)),tuple(sorted(shared_library_sources(version=14).items())),
        bias_source=bias_source,checked_boundary_sha256=_digest((None,
            None if bias_source is None else asdict(bias_source),None)),
        checked_k_panel_owner_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    result.revalidate()
    return result
