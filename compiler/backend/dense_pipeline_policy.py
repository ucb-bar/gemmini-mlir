"""Offline-evidenced retained/segmented selection over actual checked plans.

The capacity-starved RHS class is a performance hypothesis, not a hardware
grant. Planning calls the existing shared schedule entry for both alternatives;
selection consumes one of those current-source artifacts through PipelinePolicy.
No profiling, simulation, empirical cutoff or default promotion happens here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from merlin.transforms.pipeline_policy import PipelineVariant, PolicyFacts, RegisteredPass, ResourcePredicate
from merlin.transforms.pipeline_state import PassSpec, PipelineSession
from merlin.transforms.shape_policy import AssessedShapePolicy, ShapeScheduleClass, assess_shape_policy

from . import isa

LEVEL = 'authored_dense_pipeline_source'
OUTPUT = 'checked_dense_pipeline_schedule'
RESOURCE_NAMES = frozenset({'m_tiles', 'k_tiles', 'n_tiles', 'retained_legal',
    'segmented_legal', 'retained_n_blocks', 'segmented_n_blocks',
    'rhs_capacity_starved', 'rhs_panel_expanded', 'dim', 'spad_rows', 'acc_rows'})
PLAN_RESOURCE_SUFFIXES = ('m_blocks', 'k_blocks', 'n_blocks', 'm_groups',
    'k_groups', 'n_groups', 'output_groups', 'reduction_groups',
    'lhs_loads', 'rhs_loads', 'completion_epochs')
RESOURCE_NAMES = RESOURCE_NAMES | frozenset(
    prefix+'_'+suffix for prefix in ('retained', 'segmented')
    for suffix in PLAN_RESOURCE_SUFFIXES) | frozenset({'mk_tiles',
        'segmented_m_panel_expanded', 'segmented_operand_loads_not_increased',
        'incumbent_whole_rhs_reuse'})


def _candidate_resources(name, result, tiles):
    """Observe an admitted schedule; these counts do not establish legality.

    Transfer and completion counts walk the actual counted schedule, without
    unrolling it. Group extents describe its admitted panel decomposition.
    They are diagnostics, never a cycle estimate or a performance score.
    """
    resources = {name+'_'+key: 0 for key in PLAN_RESOURCE_SUFFIXES}
    if result is None:
        return resources
    from .dense import _adapter
    from .dense_pipeline import _retained_load_counts
    from merlin.sched.ir import Loop
    block = result.realization['panel_geometry']['selected_block_tiles']
    groups = tuple((extent+width-1)//width for extent, width in zip(tiles, block))
    lhs, rhs = _retained_load_counts(result.kernel, _adapter())
    def completions(body, multiplicity=1):
        total = 0
        for statement in body:
            if type(statement) is Loop:
                total += completions(statement.body, multiplicity*statement.extent)
            elif statement.instr == 'merlin_dense_panel_drain':
                total += multiplicity
        return total
    values = dict(zip(('m_blocks','k_blocks','n_blocks'), block))
    values.update(zip(('m_groups','k_groups','n_groups'), groups))
    values.update(output_groups=groups[0]*groups[2],
        reduction_groups=groups[0]*groups[1]*groups[2],
        lhs_loads=lhs, rhs_loads=rhs,
        completion_epochs=completions(result.kernel.body))
    resources.update((name+'_'+key, value) for key, value in values.items())
    return resources


@dataclass(frozen=True)
class DensePipelineSource:
    m: int
    k: int
    n: int
    bias: bool = False
    column_scales: bool = False
    geometry: str = 'wide'
    storage_split: str = 'half'
    frontier: str = 'off'
    segment_k_limit: int = 16
    segmented_geometry: str | None = None
    segment_partition: str = 'legacy'

    def __post_init__(self):
        from .dense_pipeline import PANEL_GEOMETRIES
        if (any(type(v) is not int or v <= 0 for v in (self.m, self.k, self.n))
                or any(type(v) is not bool for v in (self.bias, self.column_scales))
                or self.geometry not in PANEL_GEOMETRIES
                or self.segmented_geometry is not None and self.segmented_geometry not in PANEL_GEOMETRIES
                or self.storage_split not in ('half', 'adaptive')
                or self.frontier not in ('off', 'k_tiles', 'output_groups')
                or type(self.segment_k_limit) is not int or self.segment_k_limit not in (8,16,32,64)
                or self.segment_partition not in ('legacy','balanced_short')):
            raise ValueError('dense pipeline policy requires exact positive retained source and resource options')


def pipeline_policy_session(source, *, context=None):
    if type(source) is not DensePipelineSource:
        raise ValueError('dense pipeline policy needs its immutable source IR')
    def verify(value):
        if type(value) is not DensePipelineSource:
            raise ValueError('dense pipeline source changed type')
        value.__post_init__()
    verify(source)
    return PipelineSession(source, context=context, ir_level=LEVEL,
        fingerprint=lambda value: sha256(json.dumps(asdict(value), sort_keys=True).encode()).hexdigest(),
        verify_ir=verify)


@dataclass(frozen=True)
class DensePipelineCandidates:
    source: DensePipelineSource
    retained: object
    retained_refusal: str | None
    segmented: object
    segmented_refusal: str | None
    kernel_identities: tuple[tuple[str, str | None], ...]

    def revalidate(self):
        self.source.__post_init__()
        identities = dict(self.kernel_identities)
        for name in ('retained', 'segmented'):
            result, refusal = getattr(self, name), getattr(self, name+'_refusal')
            if result is None:
                if not refusal or identities.get(name) is not None:
                    raise ValueError('missing dense pipeline candidate must preserve its original refusal')
                continue
            result.revalidate()
            if refusal is not None or result.kernel.digest() != identities.get(name):
                raise ValueError('dense pipeline candidate or its original source checks changed')
            s = self.source
            expected = [('lhs', (s.m, s.k), 'i8', 'read'),
                        ('rhs', (s.k, s.n), 'i8', 'read')]
            if s.bias:
                expected.append(('bias', (s.n,), 'i32', 'read'))
            expected.append(('output', (s.m, s.n), 'i32', 'write'))
            if [(a.name, a.shape, a.dtype, a.access) for a in result.kernel.args] != expected:
                raise ValueError('dense pipeline artifact does not implement this original source ABI')
            expected_geometry=(s.segmented_geometry or s.geometry) if name=='segmented' else s.geometry
            if result.realization['panel_geometry']['selection'] != expected_geometry:
                raise ValueError('dense pipeline candidate changed the selected panel resource bounds')
            if name == 'segmented':
                from .dense_segment_options import segment_k_cap
                cap=segment_k_cap((s.k+isa.DIM-1)//isa.DIM,
                    s.segment_k_limit,s.segment_partition,s.frontier)
                expected_issue = {'off':'compute_then_prefetch',
                    'k_tiles':'prefetch_rhs_between_k_tiles',
                    'output_groups':'prefetch_rhs_between_output_groups'}[self.source.frontier]
                if (result.realization.get('segment_k_limit',16) != self.source.segment_k_limit
                        or result.realization.get('segment_partition','legacy') != self.source.segment_partition
                        or result.realization.get('effective_segment_k_cap',cap) != cap
                        or result.realization['panel_geometry']['selected_block_tiles'][1] > cap
                        or result.realization.get('requested_frontier', 'off') != self.source.frontier
                        or result.realization['schedule_ir']['pipeline_issue_order'] != expected_issue):
                    raise ValueError('dense pipeline candidate changed the frozen frontier issue policy')


def pipeline_policy_candidates(source, *, retained_planner=None):
    """Plan both current sources without changing queues, storage or emission.

    Pass dense_pipeline._retained_schedule explicitly when integrating into its
    dispatcher. The callback uses that existing positional ABI, including the
    segmented Boolean, and must not re-enter a public policy dispatcher. An
    explicit frontier additionally passes that frozen option by keyword.
    """
    if type(source) is not DensePipelineSource:
        raise ValueError('dense pipeline candidates need the typed source')
    source.__post_init__()
    if retained_planner is None:
        from .dense_pipeline import _retained_schedule
        retained_planner = _retained_schedule
    if not callable(retained_planner):
        raise ValueError('dense pipeline policy needs the actual non-dispatching planner')
    def plan(segmented):
        options = {} if source.frontier == 'off' else {'frontier': source.frontier if segmented else 'off'}
        if segmented and source.segment_k_limit!=16:
            options['segment_k_limit']=source.segment_k_limit
        if segmented and source.segment_partition!='legacy':
            options['segment_partition']=source.segment_partition
        return retained_planner(source.m, source.k, source.n, source.bias,
            source.column_scales, segmented,
            (source.segmented_geometry or source.geometry) if segmented else source.geometry,
            source.storage_split, **options)
    retained_result = plan(False)
    try:
        segmented_result = plan(True)
    except (ValueError, TimeoutError) as error:
        # An optional candidate whose complete source/epoch check refuses or
        # remains unknown cannot replace the already checked incumbent.
        segmented_result = (None, 'segmented admission refused: '+str(error))
    results = [retained_result, segmented_result]
    (retained, rr), (segmented, sr) = results
    identities = tuple((name, None if result is None else result.kernel.digest())
        for name, result in (('retained', retained), ('segmented', segmented)))
    candidates = DensePipelineCandidates(source, retained, rr, segmented, sr, identities)
    candidates.revalidate()
    return candidates


def pipeline_policy_facts(session, candidates):
    """Facts come from the actual chosen blocks and their complete admission."""
    if type(candidates) is not DensePipelineCandidates or candidates.source != session.module:
        raise ValueError('dense pipeline facts require the actual current-source candidates')
    candidates.revalidate()
    s = candidates.source
    mt, kt = (s.m+isa.DIM-1)//isa.DIM, (s.k+isa.DIM-1)//isa.DIM
    ntile = 1 if s.column_scales else isa.DIM
    nt = (s.n+ntile-1)//ntile
    rb = None if candidates.retained is None else candidates.retained.realization['panel_geometry']['selected_block_tiles']
    sb = None if candidates.segmented is None else candidates.segmented.realization['panel_geometry']['selected_block_tiles']
    if rb is not None and rb[1] != kt:
        raise ValueError('retained policy candidate must actually retain full K')
    if sb is not None:
        # These exact windows and epochs were admitted by the shared segmented
        # planner; do not infer their legality just from N block counts.
        bounds = candidates.segmented.realization['segmented_epoch_bounds']
        if ((sb[1] != kt if s.frontier == 'output_groups' else sb[1] >= kt) or bounds['rhs_slots'] != 2
                or bounds['retained_lhs_rows'] != sb[0]*kt*isa.DIM
                or bounds['rhs_slot_rows'] != sb[1]*sb[2]*isa.DIM):
            raise ValueError('segmented policy needs the actual full-LHS/two-RHS realization')
    starved = rb is not None and sb is not None and rb[2] == 1 and nt > 1 and sb[2] > 1
    resources = {'m_tiles': mt, 'k_tiles': kt, 'n_tiles': nt,
        'retained_legal': int(rb is not None), 'segmented_legal': int(sb is not None),
        'retained_n_blocks': 0 if rb is None else rb[2],
        'segmented_n_blocks': 0 if sb is None else sb[2],
        'rhs_capacity_starved': int(starved),
        'rhs_panel_expanded': int(rb is not None and sb is not None and sb[2] > rb[2]),
        'dim': isa.DIM,
        'spad_rows': isa.SPAD_ROWS, 'acc_rows': isa.ACC_ROWS}
    resources['mk_tiles'] = mt*kt
    for name in ('retained', 'segmented'):
        resources.update(_candidate_resources(name, getattr(candidates, name), (mt, kt, nt)))
    both = rb is not None and sb is not None
    resources['segmented_m_panel_expanded'] = int(both and sb[0] > rb[0])
    resources['segmented_operand_loads_not_increased'] = int(both and all(
        resources['segmented_'+role+'_loads'] <= resources['retained_'+role+'_loads']
        for role in ('lhs', 'rhs')))
    resources['incumbent_whole_rhs_reuse'] = int(rb is not None and
        candidates.retained.realization.get('retained_operand') == 'weight' and rb[2] == nt)
    return PolicyFacts(session.stamp(), shapes=((s.m, s.k), (s.k, s.n)),
        element_types=('i8', 'i8'), layouts=('row_major', 'row_major'),
        operations=frozenset({'dense'}), resources=tuple(sorted(resources.items())))


def pipeline_crossover_facts(session, candidates, incumbent):
    """Compare to the actual final fallback, including capacity retention.

    This observational entry does not choose or construct a new policy. Its
    caller supplies the exact checked retained fallback after any incumbent
    operand selection. Reusing retained-LHS facts would hide whole-RHS reuse.
    The standard candidate validation checks its original source ABI, resource
    geometry and immutable schedule identity before exposing any resources.
    """
    from dataclasses import replace
    if type(incumbent) is not tuple or len(incumbent) != 2:
        raise ValueError('crossover facts require the actual checked fallback and refusal')
    result, refusal = incumbent
    identities = dict(candidates.kernel_identities)
    identities['retained'] = None if result is None else result.kernel.digest()
    actual = replace(candidates, retained=result, retained_refusal=refusal,
        kernel_identities=tuple(identities.items()))
    return pipeline_policy_facts(session, actual)


def pipeline_policy_alternatives():
    return (PipelineVariant('segmented', ('dense_pipeline_segmented',),
                (ResourcePredicate('segmented_legal', 1),)),
            PipelineVariant('retained', ('dense_pipeline_retained',)))


def assess_dense_pipeline_policy(feedback, *, design, predicates=None):
    """Require original paired feedback and caller-frozen train/heldout design.

    The default Boolean class also needs both exact 1 and adjacent outside 0
    facts in that frozen panel. Missing evidence retains the original route.
    """
    if predicates is None:
        predicates = (ResourcePredicate('rhs_capacity_starved', 1),)
    if (type(predicates) is not tuple or not 1 <= len(predicates) <= 16
            or any(type(p) is not ResourcePredicate or p.resource not in RESOURCE_NAMES for p in predicates)):
        raise ValueError('freeze existing typed dense pipeline resource predicates before assessment')
    group = ShapeScheduleClass('rhs_panel_capacity', predicates, 'retained', design)
    return assess_shape_policy((group,), pipeline_policy_alternatives(), feedback,
        fallback='retained', max_regression_ratio=1.0)


def pipeline_policy_registry(candidates):
    if type(candidates) is not DensePipelineCandidates:
        raise ValueError('dense pipeline registry needs actual current-source candidates')
    def entry(name):
        def action(session):
            if session.module != candidates.source:
                raise ValueError('dense pipeline registry cannot reuse a training-source artifact')
            candidates.revalidate()
            return {OUTPUT: (getattr(candidates, name), getattr(candidates, name+'_refusal'))}
        key = 'dense_pipeline_'+name
        return key, RegisteredPass(PassSpec(key, 0, produces=(OUTPUT,),
            mutates_ir=False, ir_level=LEVEL), action)
    return dict(entry(name) for name in ('retained', 'segmented'))


def execute_dense_pipeline_policy(m, k, n, bias=False, column_scales=False, *,
        geometry='wide', storage_split='half', context=None, policy=None,
        domain=None, retained_planner=None, frontier='off'):
    """Return actual ``(checked_schedule_or_None, refusal, receipt)``.

    The caller consumes schedule.emit_llvm with the existing terminal encoder.
    No compile-time execution evidence is produced here. Compact, streaming and
    unsupported full-LHS handling remain with the existing caller's dispatch.
    """
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, storage_split, frontier)
    session = pipeline_policy_session(source, context=context)
    candidates = pipeline_policy_candidates(source, retained_planner=retained_planner)
    facts = pipeline_policy_facts(session, candidates)
    if policy is None:
        policy = assess_dense_pipeline_policy((), design=None)
    if (type(policy) is not AssessedShapePolicy or policy.fallback != 'retained'
            or policy.alternatives != pipeline_policy_alternatives()
            or any(c.incumbent != 'retained' or any(type(p) is not ResourcePredicate
                or p.resource not in RESOURCE_NAMES for p in c.predicates) for c in policy.classes)):
        raise ValueError('dense pipeline policy must preserve the assessed actual paths and incumbent')
    selected = policy.execute(session, pipeline_policy_registry(candidates), facts,
        output=OUTPUT, domain=domain)
    schedule, refusal = selected.artifact
    receipt = dict(schema='authored_dense_pipeline_policy.v1', policy=selected.receipt,
        assessment=policy.report, candidate_kernels=dict(candidates.kernel_identities),
        candidate_refusals={'retained': candidates.retained_refusal,
                            'segmented': candidates.segmented_refusal},
        refusal=refusal, realization=None if schedule is None else dict(schedule.realization))
    return schedule, refusal, receipt


def execute_adaptive_dense_frontier(m, k, n, bias=False, column_scales=False, *,
        geometry='wide', storage_split='half', context=None, retained_planner=None,
        retained_operand='lhs',segment_k_limit=16,segmented_geometry=None,segment_partition=None,
        crossover_policy=None):
    """Apply the frozen native-evidenced capacity class through PipelinePolicy.

    This authored resource rule is separate from AssessedShapePolicy and makes
    no claim that the legacy ELF/input/reference evidence adapter admitted it.
    Both alternatives are actual current-source checked plans. The caller owns
    the incumbent's retained-to-streaming capacity fallback.
    """
    if retained_operand not in ('lhs','capacity'):
        raise ValueError('adaptive retained operand must be lhs or capacity')
    from merlin.transforms.pipeline_policy import PipelinePolicy, execute_policy
    from .dense_segment_options import resolve_segment_partition
    from .dense_crossover_policy import resolve_crossover_policy
    crossover_policy=resolve_crossover_policy(crossover_policy)
    segment_partition=resolve_segment_partition(segment_partition)
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry,
        storage_split, 'k_tiles',segment_k_limit,segmented_geometry,segment_partition)
    if context is None:
        # Resource predicates are unknown without an explicit selected resource
        # context. Bind the same concrete capacities/options consumed below;
        # actual source/storage/epoch checks still establish each candidate.
        from merlin.transforms.pipeline_state import PipelineContext
        resource_identity = sha256(json.dumps(dict(dim=isa.DIM,
            spad_rows=isa.SPAD_ROWS, acc_rows=isa.ACC_ROWS,
            geometry=geometry, storage_split=storage_split,segment_k_limit=segment_k_limit,
            segmented_geometry=segmented_geometry,segment_partition=segment_partition),sort_keys=True).encode()).hexdigest()
        context = PipelineContext(resources=resource_identity)
    session = pipeline_policy_session(source, context=context)
    candidates = pipeline_policy_candidates(source, retained_planner=retained_planner)
    facts = pipeline_policy_facts(session, candidates)
    policy = PipelinePolicy((
        PipelineVariant('segmented', ('dense_pipeline_segmented',),
            (ResourcePredicate('segmented_legal', 1),
             ResourcePredicate('rhs_panel_expanded', 1))),
        PipelineVariant('retained', ('dense_pipeline_retained',))), 'retained')
    selected = execute_policy(session, policy, pipeline_policy_registry(candidates),
        facts, output=OUTPUT)
    schedule, refusal = selected.artifact
    receipt = dict(schema='authored_adaptive_dense_frontier.v1',
        evidence_basis='native paired complete provisional cycle model; independent exact numerical checks',
        legacy_shape_assessor_admission=False, policy=selected.receipt,
        candidate_kernels=dict(candidates.kernel_identities),
        candidate_resources=dict(facts.resources),
        candidate_refusals={'retained': candidates.retained_refusal,
                            'segmented': candidates.segmented_refusal})
    if selected.receipt['selected_variant']=='retained' and retained_operand=='capacity':
        from .dense_retention_policy import execute_capacity_retention
        schedule,refusal=execute_capacity_retention(m,k,n,bias,column_scales,
            geometry=geometry,storage_split=storage_split,incumbent=(schedule,refusal))
        receipt['retained_operand_policy']='capacity'
        receipt['selected_retained_operand']=None if schedule is None else schedule.realization['retained_operand']
    if selected.receipt['selected_variant']=='retained' and crossover_policy=='equal_loads':
        from .dense_crossover_policy import execute_equal_loads_crossover
        schedule,refusal,crossover=execute_equal_loads_crossover(
            session,candidates,(schedule,refusal))
        receipt['crossover']=crossover
        receipt['expansion_policy']=receipt['policy']
        receipt['policy']=crossover['policy']
        receipt['final_selected_variant']=crossover['policy']['selected_variant']
        receipt['final_kernel']=None if schedule is None else schedule.kernel.digest()
    return schedule, refusal, receipt
