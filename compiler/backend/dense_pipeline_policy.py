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

from merlin.transforms.pipeline_policy import PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass, ResourcePredicate, execute_policy
from merlin.transforms.pipeline_state import PassSpec, PipelineSession, PipelineContext
from merlin.transforms.shape_policy import AssessedShapePolicy, ShapeScheduleClass, assess_shape_policy

from . import isa

LEVEL = 'authored_dense_pipeline_source'
OUTPUT = 'checked_dense_pipeline_schedule'
RESOURCE_NAMES = frozenset({'m_tiles', 'k_tiles', 'n_tiles', 'retained_legal',
    'segmented_legal', 'retained_n_blocks', 'segmented_n_blocks',
    'rhs_capacity_starved', 'skinny_rhs_panel_widened', 'skinny_segmented_capacity_improved', 'dim', 'spad_rows', 'acc_rows'})


@dataclass(frozen=True)
class DensePipelineSource:
    m: int
    k: int
    n: int
    bias: bool = False
    column_scales: bool = False
    geometry: str = 'wide'
    storage_split: str = 'half'
    segment_k_tiles: int = 16

    def __post_init__(self):
        from .dense_pipeline import PANEL_GEOMETRIES, SEGMENT_K_LIMITS
        if (any(type(v) is not int or v <= 0 for v in (self.m, self.k, self.n))
                or any(type(v) is not bool for v in (self.bias, self.column_scales))
                or self.geometry not in PANEL_GEOMETRIES
                or self.storage_split not in ('half', 'adaptive')
                or type(self.segment_k_tiles) is not int or self.segment_k_tiles not in SEGMENT_K_LIMITS):
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
            if name == 'segmented' and result.realization['segmented_k_limit'] != self.source.segment_k_tiles:
                raise ValueError('dense segmented candidate changed its selected K resource bound')
            if result.realization['panel_geometry']['selection'] != self.source.geometry:
                raise ValueError('dense pipeline candidate changed the selected panel resource bounds')


def pipeline_policy_candidates(source, *, retained_planner=None, include_segmented=True,
                               require_wider_rhs=False):
    """Plan both current sources without changing queues, storage or emission.

    Pass dense_pipeline._retained_schedule explicitly when integrating into its
    dispatcher. The callback uses that existing positional ABI, including the
    segmented Boolean and selected K-tile limit, and must not re-enter a public policy dispatcher.
    """
    if type(source) is not DensePipelineSource:
        raise ValueError('dense pipeline candidates need the typed source')
    source.__post_init__()
    if retained_planner is None:
        from .dense_pipeline import _retained_schedule
        retained_planner = _retained_schedule
    if not callable(retained_planner):
        raise ValueError('dense pipeline policy needs the actual non-dispatching planner')
    if type(include_segmented) is not bool or type(require_wider_rhs) is not bool:
        raise ValueError('candidate family inclusion must be explicit')
    retained, rr = retained_planner(source.m, source.k, source.n, source.bias,
        source.column_scales, False, source.geometry, source.storage_split, source.segment_k_tiles)
    exclusion = None
    if not include_segmented:
        exclusion = 'segmented family excluded outside the opt-in skinny-row class'
    elif require_wider_rhs and retained is not None:
        from .dense_pipeline import PANEL_GEOMETRIES
        ntile = 1 if source.column_scales else isa.DIM
        ntiles = (source.n + ntile - 1) // ntile
        maximum_width = min(ntiles, PANEL_GEOMETRIES[source.geometry][1])
        retained_width = retained.realization['panel_geometry']['selected_block_tiles'][2]
        if retained_width >= maximum_width:
            exclusion = 'retained RHS panel already spans source N or the selected geometry bound'
    if exclusion is None:
        segmented, sr = retained_planner(source.m, source.k, source.n, source.bias,
            source.column_scales, True, source.geometry, source.storage_split, source.segment_k_tiles)
    else:
        segmented, sr = None, exclusion
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
        'rhs_capacity_starved': int(starved),
        'skinny_rhs_panel_widened': int(mt == 1 and rb is not None and sb is not None
            and sb[2] > rb[2] and not s.column_scales), 'dim': isa.DIM,
        'skinny_segmented_capacity_improved': int(mt == 1 and sb is not None
            and not s.column_scales and (rb is not None and sb[2] > rb[2]
                or rb is None and s.storage_split == 'adaptive')),
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
        geometry='wide', storage_split='half', segment_k_tiles=16, context=None, policy=None,
        domain=None, retained_planner=None):
    """Return actual ``(checked_schedule_or_None, refusal, receipt)``.

    The caller consumes schedule.emit_llvm with the existing terminal encoder.
    No compile-time execution evidence is produced here. Compact, streaming and
    unsupported full-LHS handling remain with the existing caller's dispatch.
    """
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, storage_split, segment_k_tiles)
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


def execute_capacity_pipeline(m, k, n, bias=False, column_scales=False, *,
        geometry='wide', storage_split='half', segment_k_tiles=16, retained_planner=None):
    """Explicit capacity experiment over current checked schedule alternatives.

    This opt-in route is separate from empirical default promotion. It asks the
    existing registered pipeline policy to use segmented K only for a single
    output-row tile whose actual segmented RHS panel is wider than its retained
    panel, or when adaptive placement admits full LHS while full-K retained
    placement is unavailable. Widths come from complete candidate admission, including the resident
    full-LHS and two disjoint RHS slots. All other shapes retain their incumbent.
    The artifact returned by the selected registered pass is the emitted one.
    No timing lookup, startup simulation or capsule identity is consulted.
    """
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, storage_split, segment_k_tiles)
    # The resource identity describes the selected constants used by both
    # candidate planners. Actual legality still comes from their own checks.
    resource_identity = sha256(json.dumps(dict(dim=isa.DIM, spad_rows=isa.SPAD_ROWS,
        acc_rows=isa.ACC_ROWS), sort_keys=True).encode()).hexdigest()
    session = pipeline_policy_session(source,
        context=PipelineContext(resources=resource_identity))
    candidates = pipeline_policy_candidates(source, retained_planner=retained_planner,
        include_segmented=(m <= isa.DIM and k > isa.DIM and not column_scales),
        require_wider_rhs=True)
    facts = pipeline_policy_facts(session, candidates)
    variants = (PipelineVariant('segmented', ('dense_pipeline_segmented',),
        (ResourcePredicate('segmented_legal', 1),
         ResourcePredicate('skinny_segmented_capacity_improved', 1))),
        PipelineVariant('retained', ('dense_pipeline_retained',)))
    selected = execute_policy(session, PipelinePolicy(variants, 'retained'),
        pipeline_policy_registry(candidates), facts, output=OUTPUT)
    schedule, refusal = selected.artifact
    receipt = dict(schema='authored_dense_capacity_pipeline.v1',
        selection_basis='opt-in actual checked RHS panel width; performance unqualified',
        policy=selected.receipt, candidate_kernels=dict(candidates.kernel_identities),
        candidate_refusals={'retained': candidates.retained_refusal,
                            'segmented': candidates.segmented_refusal},
        refusal=refusal, realization=None if schedule is None else dict(schedule.realization))
    return schedule, refusal, receipt
