"""Retain full-K LHS panels with the incumbent's single-K completion law.

Shared segmented construction is specialized to one K tile per epoch. Actual
source and hazard verification therefore needs only disjoint accumulator writes,
not the unavailable ordered-overlapping-commit conjunction of longer segments.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView, _digest
from merlin.compile.scheduling.dense_panel_pipeline import verify_dense_panel_pipeline
from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts
from merlin.compile.scheduling.dense_segmented_pipeline import build_dense_segmented_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan, dense_schedule_text
from merlin.transforms.pipeline_policy import ShapePredicate, ResourcePredicate, PolicyFacts
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks

@dataclass(frozen=True)
class RetainedLhsResult:
    original: object
    checked: object
    kernel: object
    realization: dict
    source_sha256: str
    def revalidate(self):
        self.original.revalidate()
        self.checked.revalidate()
        if (self.kernel != self.checked.kernel or
            sha256(Path(__file__).read_bytes()).hexdigest() != self.source_sha256):
            raise ValueError('retained LHS helper or checked emitted stream changed')
    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        self.revalidate()
        receipt = self.checked.emit_index_program(control, lower_effect, complete_epoch=complete_epoch)
        self.revalidate()
        return receipt

def plan_segmented(m,k,n,adapter,geometry,incumbent,block,r):
    edge, operand_rows, acc_rows = geometry
    p = DenseProblem(m,k,n,edge,edge,edge,1,1,4)
    lhs_rows = block.m*p.tiles[1]*edge
    split = (operand_rows // (2 * edge)) * edge
    rhs_rows = operand_rows-split
    original_loads = incumbent.realization['source']['lhs_loads']
    unique_loads = p.tiles[0]*p.tiles[1]
    # Hardware-block matrix class, frozen independently of cycle observations.
    # Policy predicates select an alternative; full verification below proves legality.
    facts_sha = sha256((Path(__file__).parent/'rtl_facts/facts.json').read_bytes()).hexdigest()
    facts = PolicyFacts(AnalysisStamp(incumbent.kernel.digest(),
        PipelineContext(resources=facts_sha), 'dense_geometry'),
        shapes=((m,k),(k,n)), resources=(
            ('incumbent_m_tiles',block.m), ('lhs_view_rows',split),
            ('rhs_view_rows',rhs_rows), ('lhs_repeated_loads',max(0,original_loads-unique_loads))))
    guards = (ResourcePredicate('incumbent_m_tiles',1),
        ResourcePredicate('lhs_view_rows',lhs_rows),
        ResourcePredicate('rhs_view_rows',2*block.n*edge),
        ResourcePredicate('lhs_repeated_loads',1), ShapePredicate(0,1,at_least=edge+1))
    if not all(g.match(facts) is True for g in guards):
        return incumbent
    s = DenseLoopStorage(
        DenseMemoryView('lhs','scratchpad',edge,0,split),
        DenseMemoryView('rhs','scratchpad',edge,split,rhs_rows),
        DenseMemoryView('acc','accumulator',edge*4,0,acc_rows))
    decision = make_dense_schedule_plan(p,r,s,adapter,source_identity=incumbent.source_identity,
        dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),
        pipeline_kind='retained_lhs_segmented_k',pipeline_prefetch=True)
    actual = build_dense_segmented_pipeline(p,block,r,s,adapter,
        name='gemmini_retained_lhs_single_k',dtypes=('i8','i8','i32'))
    # Issued counts are compiler bounds. Physical pending queues and snapshot
    # premises remain exactly the incumbent's selected queue-derived admission.
    admission = replace(incumbent.admission,
        max_issued_compute_pairs=block.m*block.n,
        max_issued_loads=max(lhs_rows//edge,block.n))
    report = verify_dense_panel_pipeline(actual,p,block,r,s,adapter,admission,
        static_budget=128,timeout_ms=30000)
    report['schedule_ir'] = dict(text=dense_schedule_text(decision),
        source_identity=incumbent.source_identity,
        pipeline_kind='retained_lhs_segmented_k',prefetch=True,
        construction='shared build_dense_segmented_pipeline with K=1; strict disjoint-output full verifier')
    checked = replace(incumbent,kernel=actual,realization=report,admission=admission,
        checked_kernel_sha256=actual.digest(),checked_admission_sha256=_digest(asdict(admission)))
    checked.revalidate()
    batched = batch_dense_panel_readouts(checked,p,block,r,s,adapter,admission,
        static_budget=128,timeout_ms=30000)
    batched.realization['retained_lhs_single_k'] = {
        'facts_sha256':sha256((Path(__file__).parent/'rtl_facts/facts.json').read_bytes()).hexdigest(),
        'before_kernel_sha256':incumbent.kernel.digest(),
        'actual_kernel_sha256':batched.kernel.digest(),
        'original_lhs_loads':original_loads,'retained_lhs_loads':unique_loads,
        'incumbent_drains':incumbent.realization['dynamic_epoch_drains'],
        'retained_drains':batched.realization['dynamic_epoch_drains'],
        'block':(block.m,block.k,block.n),'lhs_rows':lhs_rows,'rhs_rows':rhs_rows,'rhs_base':split,
        'completion_law':'one K tile per compute epoch; all accumulator commits complete before next K',
        'longer_segment_status':'UNKNOWN: selected overlapping accumulator commit conjunction absent',
        'selection':'one-or-more-M-tile capacity-admitted block class; incumbent M/N block retained; actual redundant LHS count and full-LHS fits original LHS view; unchanged RHS base/windows; repeated original LHS loads only',
        'source_and_hazards':'actual transformed operands/completions checked by verify_dense_panel_pipeline'}
    return RetainedLhsResult(incumbent,batched,batched.kernel,batched.realization,
        sha256(Path(__file__).read_bytes()).hexdigest())
