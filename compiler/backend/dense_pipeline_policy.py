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

from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
from merlin.transforms.pipeline_policy import PipelineVariant, PolicyFacts, RegisteredPass, ResourcePredicate
from merlin.transforms.pipeline_state import PassSpec, PipelineSession
from merlin.transforms.shape_policy import AssessedShapePolicy, ShapeScheduleClass, assess_shape_policy

from . import isa

LEVEL = 'authored_dense_pipeline_source'
OUTPUT = 'checked_dense_pipeline_schedule'
RESOURCE_NAMES = frozenset({'m_tiles', 'k_tiles', 'n_tiles', 'retained_legal',
    'segmented_legal', 'retained_n_blocks', 'segmented_n_blocks',
    'rhs_capacity_starved', 'segmented_n_gain', 'segmented_input_not_greater',
    'single_m_tile',
    'dim', 'spad_rows', 'acc_rows'})


@dataclass(frozen=True)
class DensePipelineSource:
    m: int
    k: int
    n: int
    bias: bool = False
    column_scales: bool = False
    geometry: str = 'wide'
    storage_split: str = 'half'
    segment_depth: str = 'fixed16'
    issue_order: str = 'compute_then_prefetch'

    def __post_init__(self):
        from .dense_pipeline import PANEL_GEOMETRIES
        if (any(type(v) is not int or v <= 0 for v in (self.m, self.k, self.n))
                or any(type(v) is not bool for v in (self.bias, self.column_scales))
                or self.geometry not in PANEL_GEOMETRIES
                or self.storage_split not in ('half', 'adaptive')
                or self.segment_depth not in ('fixed16', 'capacity')
                or self.issue_order not in ('compute_then_prefetch','prefetch_rhs_between_k_tiles')):
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
            if name=='segmented' and (result.realization.get('segment_depth_policy','fixed16') != self.source.segment_depth
                    or result.realization.get('segmented_issue_order','compute_then_prefetch') != self.source.issue_order):
                raise ValueError('dense pipeline candidate changed the original segment or frontier order')
            if result.realization['panel_geometry']['selection'] != self.source.geometry:
                raise ValueError('dense pipeline candidate changed the selected panel resource bounds')


def pipeline_policy_candidates(source, *, retained_planner=None):
    """Plan both current sources without changing queues, storage or emission.

    Pass dense_pipeline._retained_schedule explicitly when integrating into its
    dispatcher. Legacy alternatives retain the original positional ABI;
    nondefault depth/order append those two original source options. The
    callback must not re-enter a public policy dispatcher.
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
        arguments=(source.m, source.k, source.n, source.bias, source.column_scales,
                   segmented, source.geometry, source.storage_split)
        # Preserve the original planner callback ABI for legacy alternatives.
        if source.segment_depth=='fixed16' and source.issue_order=='compute_then_prefetch':
            return retained_planner(*arguments)
        return retained_planner(*arguments, source.segment_depth, source.issue_order)
    retained, rr = plan(False)
    try:
        segmented, sr = plan(True)
    except DenseVerificationUnavailable as error:
        reason = str(error)
        timed_out = (reason.startswith('compact dense verification budget exhausted: ')
            or (reason.startswith('compact dense proof unavailable: ')
                and reason.endswith(': timeout')))
        # A completed incumbent proof remains valid when the optional
        # segmented proof runs out of time. This is an unavailable alternative,
        # never permission to emit an unchecked kernel. Semantic counterexamples
        # and all other construction failures still stop compilation.
        if not timed_out:
            raise
        segmented, sr = None, 'optional segmented proof unavailable: ' + reason
    identities = tuple((name, None if result is None else result.kernel.digest())
        for name, result in (('retained', retained), ('segmented', segmented)))
    candidates = DensePipelineCandidates(source, retained, rr, segmented, sr, identities)
    candidates.revalidate()
    return candidates


def pipeline_policy_facts(session, candidates, *, effective_retained=None):
    """Facts come from the actual chosen blocks and their complete admission."""
    if type(candidates) is not DensePipelineCandidates or candidates.source != session.module:
        raise ValueError('dense pipeline facts require the actual current-source candidates')
    candidates.revalidate()
    if effective_retained is not None:
        effective_retained.revalidate()
        if candidates.retained is None or effective_retained.kernel.args != candidates.retained.kernel.args:
            raise ValueError('effective retained schedule changed original source arguments')
        for key in ('configuration_identity','numerical_policy_identity'):
            if effective_retained.realization['source'].get(key)!=candidates.retained.realization['source'].get(key):
                raise ValueError('effective retained schedule changed target or numerical custody')
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
    def input_bytes(schedule):
        if schedule is None:
            return None
        source=schedule.realization['source']
        values=[source.get(name) for name in ('requested_lhs_bytes',
            'requested_rhs_bytes','requested_bias_load_bytes')]
        total=source.get('requested_load_bytes')
        return total if (type(total) is int and all(type(v) is int and v>=0 for v in values)
                         and total==sum(values)) else None
    retained_bytes=input_bytes(effective_retained if effective_retained is not None else candidates.retained)
    segmented_bytes=input_bytes(candidates.segmented)
    resources = {'m_tiles': mt, 'k_tiles': kt, 'n_tiles': nt,
        'single_m_tile': int(mt == 1),
        'retained_legal': int(rb is not None), 'segmented_legal': int(sb is not None),
        'retained_n_blocks': 0 if rb is None else rb[2],
        'segmented_n_blocks': 0 if sb is None else sb[2],
        'rhs_capacity_starved': int(starved), 'dim': isa.DIM,
        'segmented_n_gain': 0 if rb is None or sb is None else max(0,sb[2]-rb[2]),
        'segmented_input_not_greater': int(retained_bytes is not None and segmented_bytes is not None
                                         and segmented_bytes<=retained_bytes),
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
        geometry='wide', storage_split='half', segment_depth='fixed16',
        issue_order='compute_then_prefetch', context=None, policy=None,
        domain=None, retained_planner=None):
    """Return actual ``(checked_schedule_or_None, refusal, receipt)``.

    The caller consumes schedule.emit_llvm with the existing terminal encoder.
    No compile-time execution evidence is produced here. Compact, streaming and
    unsupported full-LHS handling remain with the existing caller's dispatch.
    """
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, storage_split,
        segment_depth, issue_order)
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
        source=asdict(source),resources=dict(facts.resources),
        assessment=policy.report, candidate_kernels=dict(candidates.kernel_identities),
        candidate_refusals={'retained': candidates.retained_refusal,
                            'segmented': candidates.segmented_refusal},
        refusal=refusal, realization=None if schedule is None else dict(schedule.realization))
    return schedule, refusal, receipt


def capacity_guarded_schedule(m, k, n, bias=False, column_scales=False, *,
        geometry='wide', storage_split='half', context=None, segment_depth=None,
        issue_order=None, guard_mode=None):
    """Use checked RHS-starvation facts for the selected segmented alternative.

    Both artifacts implement this original source and use the same target
    adapter. The resource guard compares the actual admitted panel widths;
    missing full-LHS capacity or no additional N reuse keeps the retained
    incumbent. This static policy consumes no runtime profiling or training
    kernel. Its paired native evidence is recorded outside candidate execution.
    """
    from merlin.transforms.pipeline_policy import PipelinePolicy, execute_policy
    from merlin.compile.tuning_knobs import knob
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    if guard_mode is None:
        guard_mode=knob('DENSE_CAPACITY_GUARD',default='rhs_starved',
            values=('rhs_starved','rhs_panel_gain'),root=root)
    if guard_mode not in ('rhs_starved','rhs_panel_gain'):
        raise ValueError('unsupported dense capacity guard')
    if segment_depth is None:
        segment_depth=knob('DENSE_SEGMENT_DEPTH',default='fixed16',
            values=('fixed16','capacity'),root=root)
    if issue_order is None:
        issue_order=knob('DENSE_SEGMENT_ISSUE_ORDER',default='compute_then_prefetch',
            values=('compute_then_prefetch','prefetch_rhs_between_k_tiles'),root=root)
    source=DensePipelineSource(m,k,n,bias,column_scales,geometry,storage_split,
        segment_depth,issue_order)
    if context is None:
        from merlin.transforms.pipeline_state import PipelineContext
        from .dense import _adapter
        adapter=_adapter()
        # Custody for the exact resource model already checked by both
        # candidates. This identity grants no target execution law.
        selected_resources={'dim':isa.DIM,'spad_rows':isa.SPAD_ROWS,
                            'acc_rows':isa.ACC_ROWS,'storage_split':storage_split,
                            'geometry':geometry}
        context=PipelineContext(hardware=adapter.configuration_identity,
            numerical_policy=adapter.numerical_policy_identity,
            resources=sha256(json.dumps(selected_resources,sort_keys=True).encode()).hexdigest())
    session=pipeline_policy_session(source,context=context)
    candidates=pipeline_policy_candidates(source)
    m_reuse=knob('DENSE_M_PANEL_REUSE',default='disabled',
        values=('disabled','rhs','traffic_guarded'),root=root)
    effective_retained=candidates.retained
    if guard_mode=='rhs_panel_gain' and effective_retained is not None and m_reuse=='traffic_guarded':
        from .dense_m_panels import traffic_guarded_m_panel_schedule
        effective_retained=traffic_guarded_m_panel_schedule(source.m,source.k,source.n,
            source.bias,source.column_scales,effective_retained)
    facts=pipeline_policy_facts(session,candidates,effective_retained=effective_retained)
    starved=PipelineVariant('segmented_starved',('dense_pipeline_segmented',),(
        ResourcePredicate('rhs_capacity_starved',1),ResourcePredicate('segmented_legal',1)))
    if guard_mode=='rhs_starved':
        variants=(PipelineVariant('segmented',starved.passes,starved.predicates),)
    else:
        # Complete paired evidence for the widened class covers one M tile.
        # Additional M tiles keep the original starvation rule, independently
        # of this narrower performance class and its traffic comparison.
        variants=(PipelineVariant('segmented',('dense_pipeline_segmented',),(
            ResourcePredicate('single_m_tile',1),ResourcePredicate('segmented_n_gain',1),
            ResourcePredicate('segmented_input_not_greater',1),ResourcePredicate('segmented_legal',1))),starved)
    guarded=PipelinePolicy(variants+(
        PipelineVariant('retained',('dense_pipeline_retained',)),), 'retained')
    selected=execute_policy(session,guarded,pipeline_policy_registry(candidates),
        facts,output=OUTPUT)
    schedule,refusal=selected.artifact
    receipt=dict(schema='authored_dense_capacity_guard.v1',policy=selected.receipt,guard_mode=guard_mode,
        source=asdict(source),
        resources=dict(facts.resources),candidate_kernels=dict(candidates.kernel_identities),
        candidate_refusals={'retained':candidates.retained_refusal,
                            'segmented':candidates.segmented_refusal},
        effective_retained_kernel=None if effective_retained is None else effective_retained.checked_kernel_sha256,
        selected_family='retained' if selected.receipt['selected_variant']=='retained' else 'segmented',
        refusal=refusal)
    if schedule is not None and selected.receipt['selected_variant']=='retained' and m_reuse=='traffic_guarded':
        from .dense_m_panels import traffic_guarded_m_panel_schedule
        schedule=traffic_guarded_m_panel_schedule(source.m,source.k,source.n,
            source.bias,source.column_scales,schedule)
    if schedule is not None:
        # Detached diagnostic receipt only; the immutable checked kernel,
        # source and completion admissions still own actual emission.
        schedule.realization['capacity_guard']=json.loads(json.dumps(receipt))
    return schedule,refusal
