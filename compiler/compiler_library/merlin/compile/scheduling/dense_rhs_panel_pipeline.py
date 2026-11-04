"""Source-bound full-K RHS lookahead with immutable load configuration.

This service consumes the shared output-group lookahead constructor; it never
segments K. Its separate law admits fixed load channels instead of claiming
configuration snapshots for arbitrarily varying concurrent DMA commands.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from .compact_dense import _digest
from .dense_panel_pipeline import _verify_dense_epochs
from .dense_segmented_pipeline import build_dense_segmented_pipeline, OUTPUT_GROUP_PREFETCH, _assignment
from .library_sources import shared_library_sources


@dataclass(frozen=True)
class DenseRhsPanelAdmission:
    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    source_identity: str
    max_issued_compute_pairs: int
    max_issued_loads: int
    pending_compute_pairs: int
    pending_loads: int
    binding_sources: tuple
    execution: str = 'ordered_accumulator_commits_disjoint_rhs_loads_full_drain'
    load_configuration: str = 'channels_0_1_immutable_for_complete_invocation'
    descriptor: str = 'preload_immediately_paired_with_compute'
    max_issued_readouts: int = 1
    pending_readouts: int = 1
    readout_configuration: str = 'separate_drained_readout'

    def __post_init__(self):
        if (any(type(x) is not str or not x for x in (self.hardware_identity,
                self.numerical_policy_identity,self.qualification_identity,self.source_identity))
            or any(type(x) is not int or x<=0 for x in (self.max_issued_compute_pairs,
                self.max_issued_loads,self.pending_compute_pairs,self.pending_loads))
            or self.execution!='ordered_accumulator_commits_disjoint_rhs_loads_full_drain'
            or self.load_configuration!='channels_0_1_immutable_for_complete_invocation'
            or self.descriptor!='preload_immediately_paired_with_compute'
            or self.max_issued_readouts!=1 or self.pending_readouts!=1
            or self.readout_configuration!='separate_drained_readout'
            or type(self.binding_sources) is not tuple or not self.binding_sources):
            raise ValueError('RHS lookahead needs its explicit immutable-configuration combined law')
        for path,digest in self.binding_sources:
            if sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise ValueError('RHS lookahead selected source/configuration binding changed')


@dataclass(frozen=True)
class DenseRhsPanelResult:
    kernel: object
    realization: dict
    source_identity: str
    admission: DenseRhsPanelAdmission
    checked_kernel_sha256: str
    checked_admission_sha256: str
    checked_owner_sha256: str
    checked_library_sources: tuple

    def revalidate(self):
        self.admission.__post_init__()
        if (self.kernel.digest()!=self.checked_kernel_sha256
            or _digest(asdict(self.admission))!=self.checked_admission_sha256
            or sha256(Path(__file__).read_bytes()).hexdigest()!=self.checked_owner_sha256
            or tuple(sorted(shared_library_sources(version=15).items()))!=self.checked_library_sources):
            raise ValueError('RHS lookahead checked program, law or library changed')

    def materialize_scf(self):
        from .scf_commands import materialize_scf_commands
        self.revalidate()
        result=materialize_scf_commands(self.kernel)
        self.revalidate()
        return result

    def emit_llvm(self,builder,lower_call):
        from .llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate()
        result=emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call)
        self.revalidate()
        return result


def schedule_dense_rhs_panels(problem,block,resources,storage,adapter,admission,*,
        name,source_identity,external_buffers_disjoint,static_budget=128,timeout_ms=30000):
    """Check actual alternating full-K N panels against the original product.

    Scope: one primitive M tile, uniform raw accumulator readout, no bias/seed.
    The caller must keep channels 0/1 fixed through the whole invocation; its
    actual emitter/configuration source is pinned by the selected admission.
    All readouts and ownership changes retain real full completion boundaries.
    """
    if (type(admission) is not DenseRhsPanelAdmission or admission.source_identity!=source_identity
        or admission.hardware_identity!=adapter.configuration_identity
        or admission.numerical_policy_identity!=adapter.numerical_policy_identity):
        raise ValueError('RHS panel law differs from original source or selected adapter')
    admission.__post_init__()
    if (external_buffers_disjoint is not True or problem.bias or problem.tiles[0]!=1
        or block.m!=1 or block.k!=problem.tiles[1] or problem.tiles[2]<=block.n
        or problem.tile_n!=problem.tile_k or problem.lhs_bytes!=1
        or problem.rhs_bytes!=1 or problem.accumulator_bytes!=4):
        raise ValueError('RHS lookahead needs one M tile, full K, multiple uniform N panels and no seed')
    _assignment(problem,block,resources,storage)
    kernel=build_dense_segmented_pipeline(problem,block,resources,storage,adapter,
        name=name,dtypes=('i8','i8','i32'),prefetch=True,issue_order=OUTPUT_GROUP_PREFETCH)
    # This is the shared complete-program checker, not a checking projection
    # supplied by candidate code. It proves every actual epoch's disjoint load
    # writes/compute reads and physical bounds, then the entire original product.
    report=_verify_dense_epochs(kernel,problem,resources,storage,adapter,admission,
        static_budget=static_budget,timeout_ms=timeout_ms,
        ordered_accumulator_commits=True,source_phase_fission=True)
    report.update(schema='merlin.dense_rhs_panel_pipeline.v1',
        admission=asdict(admission),retained_operand='lhs',full_k_tiles=block.k,
        block_tiles=[block.m,block.k,block.n],rhs_slots=2,k_segmentation=False,
        configuration='channels 0/1 fixed; stores separately drained',
        frontier_issue=OUTPUT_GROUP_PREFETCH,
        constructor='dense_segmented_pipeline.build_dense_segmented_pipeline:'+OUTPUT_GROUP_PREFETCH)
    result=DenseRhsPanelResult(kernel,report,source_identity,admission,kernel.digest(),
        _digest(asdict(admission)),sha256(Path(__file__).read_bytes()).hexdigest(),
        tuple(sorted(shared_library_sources(version=15).items())))
    result.revalidate()
    return result
