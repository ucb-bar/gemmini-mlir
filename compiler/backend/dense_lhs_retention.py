"""Optional resident-LHS composition of existing checked dense mechanisms.

The shared segmented builder stages the complete original LHS once, then
streams two RHS windows. Keep its real startup completion outside the epoch
pass. Subsequent RHS reuse uses the unchanged selected RAW/WAR/ACC law and
exact finite consumption proof. No new readiness or capacity fact is granted.
"""
from dataclasses import asdict,dataclass,replace
from hashlib import sha256
from pathlib import Path
from time import monotonic
from merlin.sched.ir import Call,Loop,check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.compact_dense import DenseBiasSource
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN,_calls,_serial_projection,DenseRetainedReductionAdmission
from merlin.compile.scheduling.dense_panel_readout import _propose
from merlin.compile.scheduling.dense_panel_epochs import _plan,_record,_drains
from merlin.compile.scheduling.dense_segmented_pipeline import build_dense_segmented_pipeline,DenseSegmentedAdmission
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan,read_dense_schedule_plan,check_dense_schedule_commands
from .dense_finite_verification import FiniteRetainedResult,verify_retained_commands
from .dense_panel_epochs_finite import _SourceEpochBudget,verify_readout_refinement,prove_consumption,TIMEOUT_SECONDS


def _owners():
    from merlin.compile.scheduling import dense_segmented_pipeline,dense_panel_pipeline,dense_panel_epochs,dense_panel_readout,schedule_ir
    from . import dense_finite_verification,dense_panel_epochs_finite,dense_issue_order
    return tuple((str(Path(m.__file__).resolve()),sha256(Path(m.__file__).read_bytes()).hexdigest())
        for m in (dense_segmented_pipeline,dense_panel_pipeline,dense_panel_epochs,dense_panel_readout,schedule_ir,
                  dense_finite_verification,dense_panel_epochs_finite,dense_issue_order))


def _source_kernel(p,block,r,s,a,bias,issue_order="current"):
    base=build_dense_segmented_pipeline(p,block,r,s,a,name='dense_lhs_retention',dtypes=('i8','i8','i32'),
        prefetch=True,outer_order=('m','n'),issue_order='compute_then_prefetch',bias_source=bias)
    if issue_order=='current':
        _resident_startup(base,a,block_m=block.m)
        return base
    from .dense_issue_order import reorder_resident_frontier
    return reorder_resident_frontier(base,a,issue_order)[0]


def _resident_startup(kernel,a,*,block_m=1):
    """Identify the actual drained immutable LHS preparation, never a receipt."""
    m=kernel.args[0].shape[0]
    if (len(kernel.body)!=1 or type(kernel.body[0]) is not Loop
            or not 0<m<=64 or type(block_m) is not int or not 1<=block_m<=4
            or ((m+15)//16)%block_m or kernel.body[0].extent!=((m+15)//16)//block_m):
        raise ValueError('resident LHS requires one exact original counted M scope up to four tiles')
    scope=kernel.body[0]
    if not scope.body:raise ValueError('resident LHS source lacks startup')
    first=scope.body[0]
    if (type(first) is not Loop or first.extent!=1 or not first.body
            or first.body[-1]!=Call(DRAIN,())
            or any(c==Call(DRAIN,()) for c in _calls(first.body[:-1]))):
        raise ValueError('resident LHS requires a separate complete startup drain')
    calls=tuple(_calls(first.body[:-1]))
    if not calls or any(type(a.project(c)) is not Load or a.project(c).role!='lhs' for c in calls):
        raise ValueError('resident LHS startup contains a non-LHS effect')
    for c in _calls(scope.body[1:]):
        if c==Call(DRAIN,()):continue
        op=a.project(c)
        if type(op) is Load and op.role=='lhs':
            raise ValueError('resident LHS is rewritten after its startup completion')
    # Each dynamic M iteration reuses physical operand/ACC addresses.
    # Its final original full-device completion must survive before the next
    # iteration overwrites LHS/RHS or returns ownership to the caller.
    if tuple(_calls(scope.body))[-1]!=Call(DRAIN,()):
        raise ValueError('resident M scope lacks its final ownership completion')
    return first


def _epoch_proposal(batched,a,law,admission,completion_epochs,*,block_m=1):
    startup=_resident_startup(batched,a,block_m=block_m)
    if not completion_epochs:return batched,[]
    # A resident load need not be consumed by the immediately following K step:
    # it is already complete and is never rewritten. Preserve that completion
    # verbatim and apply the existing proof solely to the ordinary RHS stream.
    scope=batched.body[0]
    tail,streams=_plan(replace(batched,body=(replace(scope,body=scope.body[1:]),)),a,law,admission)
    if len(tail.body)!=1 or replace(tail.body[0],body=())!=replace(scope,body=()):
        raise ValueError('resident LHS epoch proposal changed original M scope')
    actual=replace(batched,body=(replace(scope,body=(startup,*tail.body[0].body)),))
    if _resident_startup(actual,a,block_m=block_m)!=startup:
        raise ValueError('resident M scope changed its startup or terminal ownership')
    return actual,streams


def _identity(original,batched,actual,p,block,r,s,a,admission,law,combined,source,completion_epochs,issue_order):
    return sha256(repr((original.digest(),batched.digest(),actual.digest(),asdict(p),asdict(block),asdict(r),
        asdict(s),repr(a),asdict(admission),asdict(law),asdict(combined),source,completion_epochs,issue_order)).encode()).hexdigest()


@dataclass(frozen=True)
class ResidentLhsEpochResult:
    kernel:object
    realization:dict
    original:object
    batched:object
    problem:object
    block:object
    resources:object
    storage:object
    adapter:object
    admission:object
    law:object
    combined:object
    source:str
    completion_epochs:bool
    issue_order:str
    checked_identity:str
    owner_sha256:str
    library_owners:tuple

    def revalidate(self):
        self.original.revalidate();self.law.validate_for(self.adapter,self.admission);self.combined.__post_init__()
        identity=_identity(self.original.kernel,self.batched,self.kernel,self.problem,self.block,self.resources,self.storage,
            self.adapter,self.admission,self.law,self.combined,self.source,self.completion_epochs,self.issue_order)
        if (sha256(Path(__file__).read_bytes()).hexdigest()!=self.owner_sha256 or _owners()!=self.library_owners
                or identity!=self.checked_identity or self.kernel.digest()!=self.realization['kernel_sha256']):
            raise ValueError('resident LHS source, actual commands, geometry, law or owner changed')
        bias=DenseBiasSource(self.problem.n,'i32') if self.problem.bias else None
        original=_source_kernel(self.problem,self.block,self.resources,self.storage,self.adapter,bias,self.issue_order)
        if original!=self.original.kernel:raise ValueError('resident LHS differs from checked shared source builder')
        batched,_=_propose(original,self.adapter,self.admission.max_issued_readouts)
        actual,_=_epoch_proposal(batched,self.adapter,self.law,self.admission,self.completion_epochs,block_m=self.block.m)
        if batched!=self.batched or actual!=self.kernel:
            raise ValueError('resident LHS differs from checked shared completion transformations')

    def emit_llvm(self,builder,lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate();result=emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call);self.revalidate();return result

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands
        self.revalidate();return materialize_scf_commands(self.kernel)


def retained_lhs_epoch_schedule(p,block,r,s,a,admission,law,source,*,completion_epochs=True,issue_order="current"):
    """Compose resident source -> exact source proof -> batching -> RHS epochs.

    Capacity/source semantics are legality only. The caller exposes this as an
    off-default alternative; there is no case identity or empirical cutoff.
    """
    deadline=monotonic()+TIMEOUT_SECONDS
    admission.__post_init__();law.validate_for(a,admission)
    if (not 1<=p.tiles[0]<=4 or not 1<=block.m<=4 or block.k!=1 or p.tile_n!=16 or law.unknown()
            or type(completion_epochs) is not bool):
        raise ValueError('resident LHS epochs require up to four original M tiles, one-K slices, uniform readout and complete selected law')
    # The actual shared builder independently checks retained LHS plus both RHS
    # slots, every parent window, alignment, ACC placement and complete budget.
    if block.m*p.tiles[1]*p.tile_m>s.lhs.capacity:
        raise ValueError('complete original LHS does not fit its assigned scratchpad half')
    bias=DenseBiasSource(p.n,'i32') if p.bias else None
    reduction=DenseRetainedReductionAdmission(admission.hardware_identity,admission.numerical_policy_identity,
        law.qualification_identity+':resident-read-only-lhs',admission.max_issued_compute_pairs,admission.max_issued_loads,
        admission.pending_compute_pairs,admission.pending_loads,max_issued_readouts=admission.max_issued_readouts,
        pending_readouts=admission.pending_readouts,readout_configuration=admission.readout_configuration)
    combined=DenseSegmentedAdmission(admission,reduction,law.qualification_identity+':ordered-accumulator-disjoint-rhs')
    plan=make_dense_schedule_plan(p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=256,
        block_tiles=(block.m,block.k,block.n),pipeline_kind='retained_lhs_segmented_k',pipeline_prefetch=True,bias_source=bias)
    decision=read_dense_schedule_plan(plan,p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=256,bias_source=bias)
    if (decision.block!=block or decision.storage!=s or decision.pipeline_kind!='retained_lhs_segmented_k'
            or not decision.pipeline_prefetch or decision.loop_transforms or decision.outer_order!=('m','n')
            or decision.pipeline_issue_order!='compute_then_prefetch'):
        raise ValueError('resident LHS source plan changed original geometry, placement or issue order')
    original_kernel=_source_kernel(p,block,r,s,a,bias,issue_order)
    _resident_startup(original_kernel,a,block_m=block.m)
    source_report=verify_retained_commands(original_kernel,p,r,s,a,_SourceEpochBudget(admission),static_budget=256,allow_disjoint_mixed=True)
    check_dense_schedule_commands(decision,source_report['static_commands'],256)
    source_report['schedule_ir']=dict(sha256=decision.plan_sha256,text=decision.plan_text,source_identity=source,
        pipeline_kind=decision.pipeline_kind,prefetch=True,pipeline_issue_order=decision.pipeline_issue_order,
        decisions_consumed=['block_tiles','outer_order','row_bases','panel_pipeline','pipeline_window_offsets','pipeline_issue_order'])
    from . import dense_finite_verification
    original=FiniteRetainedResult(original_kernel,source_report,original_kernel.digest(),sha256(Path(dense_finite_verification.__file__).read_bytes()).hexdigest())
    batched,groups=_propose(original_kernel,a,admission.max_issued_readouts)
    batching=verify_readout_refinement(original_kernel,batched,a,admission,source_report,deadline=deadline)
    if batching['groups']!=groups:raise ValueError('resident LHS readout group accounting changed')
    actual,streams=_epoch_proposal(batched,a,law,admission,completion_epochs,block_m=block.m)
    if check_structure(actual) or _serial_projection(actual)!=_serial_projection(batched):
        raise ValueError('resident LHS epoch proposal changed source effects or issue order')
    if len(tuple(_calls(actual.body)))>256:raise ValueError('resident LHS static command bound exceeded')
    check_integer_domain(actual,a.index_bits)
    consumption=prove_consumption(streams,a,s,deadline=deadline)
    before,after=_drains(batched.body),_drains(actual.body)
    from .dense_issue_order import reorder_resident_frontier
    base=_source_kernel(p,block,r,s,a,bias,'current')
    if issue_order=='current':
        checked_frontier=base
        issue_report=dict(issue_order='current',source_kernel_sha256=base.digest(),actual_kernel_sha256=base.digest(),pass_order=[])
    else:checked_frontier,issue_report=reorder_resident_frontier(base,a,issue_order)
    if checked_frontier!=original_kernel:raise ValueError('resident frontier replay differs from checked source')
    report=dict(issue_order=issue_report,schema='authored_dense_resident_lhs_epochs.v1',kernel_sha256=actual.digest(),source=source,
        schedule_ir=source_report['schedule_ir'],input_schedule=source_report,readout_batching=dict(groups=groups,finite_verification=batching),
        admission=asdict(admission),epoch_law=asdict(law),segmented_admission=asdict(combined),unknown_premises=[],
        dynamic_epoch_drains=after,completion_epochs=completion_epochs,
        panel_epochs=dict(input_dynamic_drains=before,dynamic_drains=after,removed_dynamic_drains=before-after,
            streams=[_record(stream) for stream in streams],consumption_proofs=consumption),
        resident_lhs=dict(rows=block.m*p.tiles[1]*p.tile_m,assigned_rows=s.lhs.capacity,prepared_once=actual.body[0].extent==1,prepared_once_per_m_scope=True,
            startup_completion='original full device drain preserved verbatim',later_lhs_writes=0,
            lhs_loads=source_report['lhs_loads'],rhs_loads=source_report['rhs_loads'],
            m_scope=dict(variable=actual.body[0].var,extent=actual.body[0].extent,block_m=block.m,
                original_m_tiles=p.tiles[0],
                startup='original complete LHS preparation per dynamic M group',
                ownership='original final full-device completion per dynamic M group',
                scope_header_preserved=replace(actual.body[0],body=())==replace(original_kernel.body[0],body=()))),
        qualification='exact bounded source/effect/RHS-consumption proof; completed immutable LHS startup',
        target_execution='not_established')
    result=ResidentLhsEpochResult(actual,report,original,batched,p,block,r,s,a,admission,law,combined,source,completion_epochs,issue_order,
        _identity(original_kernel,batched,actual,p,block,r,s,a,admission,law,combined,source,completion_epochs,issue_order),
        sha256(Path(__file__).read_bytes()).hexdigest(),_owners())
    result.revalidate();return result
