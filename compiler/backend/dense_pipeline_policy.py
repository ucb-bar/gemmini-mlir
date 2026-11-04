"""Offline-evidenced retained/segmented selection over actual checked plans.

The capacity-starved RHS class is a performance hypothesis, not a hardware
grant. Planning calls the existing shared schedule entry for both alternatives;
selection consumes one of those current-source artifacts through PipelinePolicy.
No profiling, simulation, per-capsule dispatch or cutoff tuning happens here.
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
    'rhs_capacity_starved', 'dim', 'spad_rows', 'acc_rows'})


@dataclass(frozen=True)
class DensePipelineSource:
    m: int
    k: int
    n: int
    bias: bool = False
    column_scales: bool = False
    geometry: str = 'wide'
    storage_split: str = 'half'

    def __post_init__(self):
        from .dense_pipeline import PANEL_GEOMETRIES
        if (any(type(v) is not int or v <= 0 for v in (self.m, self.k, self.n))
                or any(type(v) is not bool for v in (self.bias, self.column_scales))
                or self.geometry not in PANEL_GEOMETRIES
                or self.storage_split not in ('half', 'adaptive')):
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
            if result.realization['panel_geometry']['selection'] != self.source.geometry:
                raise ValueError('dense pipeline candidate changed the selected panel resource bounds')


def pipeline_policy_candidates(source, *, retained_planner=None, capacity_class_only=False):
    """Plan both current sources without changing queues, storage or emission.

    Pass dense_pipeline._retained_schedule explicitly when integrating into its
    dispatcher. The callback uses that existing positional ABI, including the
    segmented Boolean, and must not re-enter a public policy dispatcher.
    """
    if type(source) is not DensePipelineSource:
        raise ValueError('dense pipeline candidates need the typed source')
    source.__post_init__()
    if retained_planner is None:
        from .dense_pipeline import _retained_schedule
        retained_planner = _retained_schedule
    if not callable(retained_planner):
        raise ValueError('dense pipeline policy needs the actual non-dispatching planner')
    retained, rr = retained_planner(source.m, source.k, source.n, source.bias,
        source.column_scales, False, source.geometry, source.storage_split)
    # The native policy's sole class needs a legal retained single-N panel.
    # Do not build an unused segmented proof when that necessary fact is false.
    # General policy exploration still constructs both alternatives by default.
    ntile = 1 if source.column_scales else isa.DIM
    eligible = (retained is not None
        and retained.realization['panel_geometry']['selected_block_tiles'][2] == 1
        and (source.n + ntile - 1) // ntile > 1)
    if capacity_class_only and not eligible:
        segmented, sr = None, 'native capacity class has no single-N retained panel'
    else:
        segmented, sr = retained_planner(source.m, source.k, source.n, source.bias,
            source.column_scales, True, source.geometry, source.storage_split)
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
        if (sb[1] >= kt or bounds['rhs_slots'] != 2
                or bounds['retained_lhs_rows'] != sb[0]*kt*isa.DIM
                or bounds['rhs_slot_rows'] != sb[1]*sb[2]*isa.DIM):
            raise ValueError('segmented policy needs the actual full-LHS/two-RHS realization')
    starved = rb is not None and sb is not None and rb[2] == 1 and nt > 1 and sb[2] > 1
    resources = {'m_tiles': mt, 'k_tiles': kt, 'n_tiles': nt,
        'retained_legal': int(rb is not None), 'segmented_legal': int(sb is not None),
        'retained_n_blocks': 0 if rb is None else rb[2],
        'segmented_n_blocks': 0 if sb is None else sb[2],
        'rhs_capacity_starved': int(starved), 'dim': isa.DIM,
        'spad_rows': isa.SPAD_ROWS, 'acc_rows': isa.ACC_ROWS}
    return PolicyFacts(session.stamp(), shapes=((s.m, s.k), (s.k, s.n)),
        element_types=('i8', 'i8'), layouts=('row_major', 'row_major'),
        operations=frozenset({'dense'}), resources=tuple(sorted(resources.items())))


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
        domain=None, retained_planner=None):
    """Return actual ``(checked_schedule_or_None, refusal, receipt)``.

    The caller consumes schedule.emit_llvm with the existing terminal encoder.
    No compile-time execution evidence is produced here. Compact, streaming and
    unsupported full-LHS handling remain with the existing caller's dispatch.
    """
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, storage_split)
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


def execute_native_dense_pipeline_policy(m, k, n, bias=False, column_scales=False, *,
        geometry='wide', storage_split='half', assessment=None, retained_planner=None,
        output_policy='off', output_assessment=None):
    """Consume native paired-cycle support through the existing policy engine.

    No workload identifiers enter the runtime facts or policy. Native feedback
    supports an empirical capacity class; source correctness, physical windows,
    continuation and completion are checked afresh by the ordinary planners.
    Missing native support preserves the retained candidate.
    """
    from pathlib import Path
    from merlin.transforms.pipeline_policy import PipelinePolicy, execute_policy
    from merlin.transforms.pipeline_state import PipelineContext
    if assessment is None:
        path = Path(__file__).with_name('data') / 'dense_pipeline_native_policy.json'
        assessment = json.loads(path.read_text()) if path.is_file() else None
    admitted = False
    if assessment is not None:
        if (type(assessment) is not dict or assessment.get('schema') != 'dense_native_pipeline_assessment.v1'
                or assessment.get('guard') != dict(resource='rhs_capacity_starved', at_least=1)
                or assessment.get('incumbent') != 'retained' or assessment.get('alternative') != 'segmented'):
            raise ValueError('native dense assessment changed its declared alternatives or resource guard')
        admitted = assessment.get('admitted') is True and not assessment.get('refusal')
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, storage_split)
    context = PipelineContext(hardware='gemmini-fire-aligned-dim16-ws',
        numerical_policy='i8-i8-i32-modular',
        resources=sha256(json.dumps([isa.DIM, isa.SPAD_ROWS, isa.ACC_ROWS]).encode()).hexdigest())
    session = pipeline_policy_session(source, context=context)
    candidates = pipeline_policy_candidates(source, retained_planner=retained_planner,
        capacity_class_only=True)
    facts = pipeline_policy_facts(session, candidates)
    fallback = PipelineVariant('retained', ('dense_pipeline_retained',))
    paths = [PipelineVariant('segmented', ('dense_pipeline_segmented',),
        (ResourcePredicate('rhs_capacity_starved', 1), ResourcePredicate('segmented_legal', 1)))] if admitted else []
    registry = pipeline_policy_registry(candidates)
    facts, output_path, output_receipt = _native_output_path(session, candidates, facts,
        output_policy=output_policy, assessment=output_assessment)
    if output_path is not None:
        paths.append(output_path[0])
        registry.update(output_path[1])
    paths.append(fallback)
    selected = execute_policy(session, PipelinePolicy(tuple(paths), 'retained'),
        registry, facts, output=OUTPUT)
    schedule, refusal = selected.artifact
    # Per-invocation host wall times are diagnostic, not emitted program facts.
    # Keep deterministic source/candidate/selection custody in the artifact.
    policy_receipt = {key: value for key, value in selected.receipt.items()
        if key != 'host_pass_profile'}
    receipt = dict(schema='authored_dense_native_pipeline_policy.v1', policy=policy_receipt,
        native_assessment_admitted=admitted,
        native_assessment_sha256=None if assessment is None else sha256(
            json.dumps(assessment, sort_keys=True).encode()).hexdigest(),
        evidence=None if assessment is None else assessment.get('evidence'),
        facts=dict(facts.resources), candidate_kernels=dict(candidates.kernel_identities),
        output_policy=output_receipt,
        physical_qualification_inferred=False, runtime_profiling=False)
    return schedule, refusal, receipt


OUTPUT_GUARD = {'m_at_most_dim': 1, 'n_tiles_ge_four_k_tiles': 1, 'n_tiles_min': 8}


def _native_output_path(session, candidates, facts, *, output_policy, assessment):
    """Add one checked output-group path to the existing regional policy."""
    from dataclasses import replace
    from pathlib import Path
    if output_policy not in ('off', 'wide_prefetch'):
        raise ValueError('unknown native dense output policy')
    receipt = dict(requested=output_policy, admitted=False, candidate_kernel=None,
                   refusal=None, physical_qualification_inferred=False)
    if output_policy == 'off':
        return facts, None, receipt
    if assessment is None:
        path = Path(__file__).with_name('data') / 'dense_output_native_policy.json'
        assessment = json.loads(path.read_text()) if path.is_file() else None
    if assessment is not None and (type(assessment) is not dict
            or assessment.get('schema') != 'dense_native_output_assessment.v1'
            or assessment.get('frozen_guard') != OUTPUT_GUARD
            or assessment.get('candidate') != 'output_prefetch'
            or assessment.get('incumbent') != 'regional'):
        raise ValueError('native output assessment changed its frozen alternatives or guard')
    admitted = (assessment is not None and assessment.get('admitted') is True
                and not assessment.get('refusal'))
    receipt.update(admitted=admitted, assessment_sha256=None if assessment is None
        else sha256(json.dumps(assessment,sort_keys=True).encode()).hexdigest(),
        evidence=None if assessment is None else assessment.get('evidence'))
    if not admitted:
        receipt['refusal'] = 'native matched output panel not admitted'
        return facts, None, receipt
    s = candidates.source
    resources = dict(facts.resources)
    mt_small = s.m <= isa.DIM
    aspect = resources['n_tiles'] >= 4*resources['k_tiles']
    coarse = mt_small and aspect and resources['n_tiles'] >= 8
    # Full-K refusal and the current segmented capacity class retain priority.
    eligible = (coarse and not s.column_scales and s.geometry == 'wide'
        and s.storage_split == 'half' and candidates.retained is not None
        and not resources['rhs_capacity_starved'])
    result, refusal = None, 'outside frozen output class or current full-K resource domain'
    if eligible:
        from .dense_output_pipeline import output_group_schedule
        result, refusal = output_group_schedule(s.m,s.k,s.n,s.bias,s.column_scales,
                                               geometry=s.geometry)
    def revalidate():
        if result is None:
            raise ValueError('unavailable output-group candidate cannot execute')
        result.revalidate()
        expected=[('lhs',(s.m,s.k),'i8','read'),('rhs',(s.k,s.n),'i8','read')]
        if s.bias: expected.append(('bias',(s.n,),'i32','read'))
        expected.append(('output',(s.m,s.n),'i32','write'))
        if [(a.name,a.shape,a.dtype,a.access) for a in result.kernel.args] != expected:
            raise ValueError('output-group artifact changed the complete original source ABI')
        realization=result.realization['output_group_prefetch']
        if (realization['original_k_tiles'] != resources['k_tiles']
                or realization['rhs_slots'] != 2
                or realization['actual_kernel_sha256'] != result.kernel.digest()):
            raise ValueError('output-group artifact changed its checked full-K realization')
    if result is not None: revalidate()
    resources.update(m_at_most_dim=int(mt_small),
        n_tiles_ge_four_k_tiles=int(aspect),output_group_legal=int(result is not None))
    facts=replace(facts,resources=tuple(sorted(resources.items())))
    receipt.update(candidate_kernel=None if result is None else result.kernel.digest(),
                   refusal=refusal, frozen_guard=OUTPUT_GUARD)
    def action(current):
        if current.module != s:
            raise ValueError('output registry cannot substitute a training-source artifact')
        revalidate()
        return {OUTPUT:(result,None)}
    key='dense_pipeline_output_prefetch'
    path=PipelineVariant('output_prefetch',(key,),(
        ResourcePredicate('m_at_most_dim',1),ResourcePredicate('n_tiles_ge_four_k_tiles',1),
        ResourcePredicate('n_tiles',8),ResourcePredicate('output_group_legal',1)))
    registry={key:RegisteredPass(PassSpec(key,0,produces=(OUTPUT,),
        mutates_ir=False,ir_level=LEVEL),action)}
    return facts,(path,registry),receipt


def assess_native_output_panel(panel, *, training_baseline, training_candidate,
        heldout_baseline, heldout_candidate, workspace='.'):
    """Offline same-domain native assessment of the prospectively frozen class.

    Native ELF/capsule/model identities retain their original meaning. Neither
    request bindings nor local/executed ELF equality are inferred from them.
    """
    from pathlib import Path
    from dataclasses import replace
    from .dense_pipeline_evidence import load_native_observation, _pin
    from .frontend import parse_source
    from merlin.transforms.pass_exploration import ObservedMetrics, rank_observed_metrics
    if (type(panel) is not dict or panel.get('schema')!='authored.dense_output_frozen_panel.v1'
            or type(panel.get('frozen_at_unix_ns')) is not int
            or panel.get('frozen_guard')!=OUTPUT_GUARD
            or panel.get('candidate')!='output_prefetch' or panel.get('incumbent')!='regional'):
        raise ValueError('native output assessment requires the original frozen panel')
    rosters={key:tuple(panel[key]) for key in ('training_inside','training_outside',
        'held_out_inside','held_out_outside')}
    all_cases=sum(rosters.values(),())
    if not all(rosters.values()) or len(set(all_cases))!=len(all_cases):
        raise ValueError('native output train/heldout classes must be nonempty and disjoint')
    root=Path(workspace).resolve();rows=[];reasons=[];domains=set()
    def collect(case,partition,inside_expected,baseline,candidate):
        source=(root/'public_sources'/(case+'.mlir')).read_bytes()
        cb=parse_source(source.decode());m,k=cb['tensors']['A0']['shape'];kk,n=cb['tensors']['W']['shape']
        if k!=kk:raise ValueError('native output source changed contraction axes')
        kt=(k+isa.DIM-1)//isa.DIM;nt=(n+isa.DIM-1)//isa.DIM
        inside=m<=isa.DIM and nt>=4*kt and nt>=8
        pair=tuple(replace(load_native_observation(path,case=case,alternative=label,
            workspace=root),alternative=name) for path,label,name in
            ((baseline,'retained','regional'),(candidate,'segmented','output_prefetch')))
        a,b=pair
        keys=('facts_sha256','model_sha256','scope_sha256','comparison_domain_sha256','capsule_sha256')
        valid=(all(_pin(getattr(a,key)) and getattr(a,key)==getattr(b,key) for key in keys)
            and all(r.numerical_status=='pass' and r.failure_category is None
                and r.complete_modeled_window and _pin(r.elf_sha256) for r in pair))
        if not valid:reasons.append(case+': incomplete, nonpassing or different-domain native pair')
        if inside!=inside_expected:reasons.append(case+': source is on wrong side of frozen guard')
        if inside and valid and b.cycles>=a.cycles:
            reasons.append(case+': frozen class regresses complete modeled cycles')
        domains.add((a.facts_sha256,a.model_sha256,a.scope_sha256))
        ranking=None if not valid else rank_observed_metrics(tuple((r.alternative,
            ObservedMetrics('modeled_roi',r.comparison_domain_sha256,(('cycles',r.cycles),)))
            for r in pair),objectives=('cycles',))
        rows.append(dict(case=case,partition=partition,shape=[m,k,n],inside_frozen_class=inside,
            original_source_sha256=sha256(source).hexdigest(),observations=[asdict(r) for r in pair],
            ranking=ranking,policy_selection='output_prefetch' if inside and valid else 'regional'))
    for key in ('training_inside','training_outside'):
        for case in rosters[key]:collect(case,'training',key.endswith('_inside'),training_baseline,training_candidate)
    training_supported=not reasons
    # The explicitly requested fresh panel may independently reject a timing
    # hypothesis even when old training numerics are UNKNOWN. It cannot repair
    # missing training evidence or admit the policy in that circumstance.
    response_path=Path(heldout_candidate)
    if not response_path.is_absolute(): response_path=root/response_path
    response=json.loads(response_path.read_bytes())
    requested=response.get('requested_at_unix_ns')
    if type(requested) is not int or requested <= panel['frozen_at_unix_ns']:
        reasons.append('held-out candidate request was not after the frozen panel')
    for key in ('held_out_inside','held_out_outside'):
        for case in rosters[key]:collect(case,'held_out',key.endswith('_inside'),heldout_baseline,heldout_candidate)
    if len(domains)!=1:reasons.append('native model, facts or elapsed-call scope differ')
    return dict(schema='dense_native_output_assessment.v1',admitted=not reasons,refusal=reasons,
        candidate='output_prefetch',incumbent='regional',frozen_guard=OUTPUT_GUARD,
        frozen_panel_sha256=sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest(),
        training_supported=training_supported,held_out_inspected=True,
        held_out_is_diagnostic_when_training_unsupported=not training_supported,
        observations=rows,common_domain=dict(zip(('facts_sha256','model_sha256','scope_sha256'),next(iter(domains))))
            if len(domains)==1 else None,
        evidence='modeled: native Spike timing model; provisional complete same-call paired windows',
        limitations=['numeric cutoff neighbors are host integer checks, not native cycle evidence',
            'unmeasured source modifiers extrapolate the class; current source proofs remain mandatory',
            'outside-class alternatives stay incumbent regardless of isolated timing'],
        physical_qualification=False,exact_current_executable_equivalence_claimed=False)
