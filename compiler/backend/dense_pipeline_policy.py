"""Offline-evidenced retained/segmented selection over actual checked plans.

The capacity-starved RHS class is a performance hypothesis, not a hardware
grant. Planning calls the existing shared schedule entry for both alternatives;
selection consumes one of those current-source artifacts through PipelinePolicy.
No profiling, simulation, empirical cutoff or default promotion happens here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    'rhs_capacity_starved', 'dim', 'spad_rows', 'acc_rows'})

# Exact selected RTL facts grounding isa.DIM/SPAD_ROWS/ACC_ROWS (see isa.py).
# This identity enables resource comparisons only. Actual schedules still need
# their original independent source, capacity and epoch admission.
SELECTED_RESOURCE_FACTS = 'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'


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
    return PipelineSession(source, context=context or PipelineContext(resources=SELECTED_RESOURCE_FACTS), ir_level=LEVEL,
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
    not_constructed: frozenset[str] = frozenset()
    retained_capacity_block: tuple[int, int, int] | None = None

    def revalidate(self):
        self.source.__post_init__()
        identities = dict(self.kernel_identities)
        if self.retained_capacity_block is not None:
            from .dense_pipeline import half_retained_capacity
            s = self.source
            _, _, _, block = half_retained_capacity(s.m,s.k,s.n,s.bias,s.column_scales,s.geometry)
            if block is None or self.retained_capacity_block != (block.m,block.k,block.n):
                raise ValueError('regional capacity classification changed its original full-K candidate')
        for name in ('retained', 'segmented'):
            result, refusal = getattr(self, name), getattr(self, name+'_refusal')
            if result is None:
                if name in self.not_constructed:
                    if refusal is not None or identities.get(name) is not None:
                        raise ValueError('unconstructed alternative cannot claim refusal or kernel evidence')
                    continue
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


def pipeline_policy_candidates(source, *, retained_planner=None):
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
    results = [retained_planner(source.m, source.k, source.n, source.bias,
        source.column_scales, segmented, source.geometry, source.storage_split)
        for segmented in (False, True)]
    (retained, rr), (segmented, sr) = results
    identities = tuple((name, None if result is None else result.kernel.digest())
        for name, result in (('retained', retained), ('segmented', segmented)))
    candidates = DensePipelineCandidates(source, retained, rr, segmented, sr, identities)
    candidates.revalidate()
    return candidates


def capacity_policy_candidates(source, *, retained_planner=None):
    """Construct segmented only inside the frozen original-lhs capacity class.

    An omitted alternative has no admission or refusal evidence. Ordinary
    explicit segmented planning and assessed-policy candidate construction are
    unaffected by this regional dispatch short circuit.
    """
    if retained_planner is None:
        from .dense_pipeline import _retained_schedule
        retained_planner = _retained_schedule
    from .dense_pipeline import half_retained_capacity
    from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
    s = source
    # Capacity is deterministic and independent of finite SMT proof budgets.
    # It classifies the original lhs/half plan but cannot admit its emission.
    _, _, _, block = half_retained_capacity(s.m,s.k,s.n,s.bias,s.column_scales,s.geometry)
    cb = None if block is None else (block.m,block.k,block.n)
    try:
        retained, rr = retained_planner(s.m, s.k, s.n, s.bias,
            s.column_scales, False, s.geometry, 'half')
    except DenseVerificationUnavailable as error:
        retained, rr = None, 'retained candidate proof unavailable: '+str(error)
    rb = None if retained is None else retained.realization['panel_geometry']['selected_block_tiles']
    if rb is not None and tuple(rb) != cb:
        raise ValueError('checked retained candidate differs from original capacity classification')
    in_region = not s.column_scales and cb is not None and cb[2] == 1 and s.n > isa.DIM
    segmented, sr = (retained_planner(s.m, s.k, s.n, s.bias,
        s.column_scales, True, s.geometry, 'half') if in_region else (None, None))
    identities = tuple((name, None if result is None else result.kernel.digest())
        for name, result in (('retained', retained), ('segmented', segmented)))
    result = DensePipelineCandidates(s, retained, rr, segmented, sr, identities,
        frozenset() if in_region else frozenset({'segmented'}), cb)
    result.revalidate()
    return result


def pipeline_policy_facts(session, candidates):
    """Keep pure capacity classification separate from actual kernel admission.

    retained_legal/segmented_legal require checked artifacts. A separately
    revalidated original capacity block may classify the frozen region even
    when its optional kernel proof exhausted its finite budget.
    """
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
    capacity_block = candidates.retained_capacity_block or rb
    starved = (not s.column_scales and capacity_block is not None and sb is not None
        and capacity_block[2] == 1 and nt > 1 and sb[2] > 1)
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


def execute_dense_capacity_policy(m, k, n, bias=False, column_scales=False, *,
        geometry='wide', retained_operand='lhs', storage_split='half',
        issue_order='grouped', retained_planner=None):
    """Execute the explicit half-store capacity alternative on current source.

    The regional rule switches only when the original retained capacity plan fits
    complete K but can hold one N tile, whereas its independently checked two-slot segmented
    alternative holds multiple N tiles. Missing retained capacity is excluded:
    the caller must preserve its original streaming fallback. Single-column
    quantization, short K and output domains of one N tile keep the incumbent.

    Both registered artifacts consume the original selected source ABI and
    full source/epoch checks. This explicit experimental policy uses the
    existing PipelinePolicy executor; it supplies no empirical identity or
    retrospective train/heldout binding to AssessedShapePolicy.
    """
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, 'half')
    session = pipeline_policy_session(source)
    candidates = capacity_policy_candidates(source, retained_planner=retained_planner)
    facts = pipeline_policy_facts(session, candidates)
    policy = PipelinePolicy((
        PipelineVariant('segmented', ('dense_pipeline_segmented',),
            (ResourcePredicate('rhs_capacity_starved', 1),
             ResourcePredicate('segmented_legal', 1))),
        PipelineVariant('retained', ('dense_pipeline_retained',))), 'retained')
    # Capacity classification remains tied to the exact lhs incumbent. Once
    # retained is selected, an explicit traffic alternative may consume its own
    # fully checked source artifact. It cannot expand the segmented region.
    if retained_operand not in ('lhs', 'weight', 'traffic'):
        raise ValueError('unknown retained operand alternative')
    if storage_split not in ('half', 'adaptive') or issue_order not in ('grouped', 'interleaved'):
        raise ValueError('unknown segmented storage or issue alternative')
    emission_candidates = candidates
    if not dict(facts.resources)['rhs_capacity_starved'] and retained_operand != 'lhs':
        if retained_planner is None:
            from .dense_pipeline import _retained_schedule
            retained_planner = _retained_schedule
        alternate, alternate_refusal = retained_planner(m, k, n, bias, column_scales,
            False, geometry, 'half', 'grouped', retained_operand)
        identities = tuple((name, None if result is None else result.kernel.digest())
            for name, result in (('retained', alternate), ('segmented', candidates.segmented)))
        emission_candidates = replace(candidates, retained=alternate,
            retained_refusal=alternate_refusal, kernel_identities=identities)
        emission_candidates.revalidate()
    elif dict(facts.resources)['rhs_capacity_starved'] and (storage_split, issue_order) != ('half', 'grouped'):
        if retained_planner is None:
            from .dense_pipeline import _retained_schedule
            retained_planner = _retained_schedule
        alternate, alternate_refusal = retained_planner(m, k, n, bias, column_scales,
            True, geometry, storage_split, issue_order, 'lhs')
        identities = tuple((name, None if result is None else result.kernel.digest())
            for name, result in (('retained', candidates.retained), ('segmented', alternate)))
        emission_candidates = replace(candidates, segmented=alternate,
            segmented_refusal=alternate_refusal, kernel_identities=identities)
        emission_candidates.revalidate()
    selected = execute_policy(session, policy, pipeline_policy_registry(emission_candidates), facts, output=OUTPUT)
    schedule, refusal = selected.artifact
    receipt = dict(schema='authored_dense_capacity_policy.v1',
        hypothesis='one full-K RHS tile versus several bounded segmented RHS tiles',
        capacity_class_storage_split='half', capacity_class_issue_order='grouped',
        retained_capacity_block=candidates.retained_capacity_block,
        retained_capacity_authority='original shared geometry enumeration only; no source/epoch admission',
        storage_split=storage_split, segmented_issue_order=issue_order,
        policy=selected.receipt, candidate_kernels=dict(candidates.kernel_identities),
        emission_candidate_kernels=dict(emission_candidates.kernel_identities),
        candidate_refusals={'retained': candidates.retained_refusal,
                           'segmented': candidates.segmented_refusal},
        candidate_construction={name: 'not_constructed' if name in candidates.not_constructed else 'constructed'
                                for name in ('retained', 'segmented')},
        selection=selected.receipt['selected_variant'],
        retained_operand=retained_operand,
        emitted_kernel=None if schedule is None else schedule.kernel.digest(),
        retained_operand_selection=None if schedule is None else schedule.realization.get('retained_operand_selection'),
        numerical_and_execution_authority='original selected candidate source and epoch checks',
        performance='explicit regional alternative; complete-call paired evidence remains separate')
    return schedule, refusal, receipt


def capacity_pipeline_schedule(m, k, n, bias=False, column_scales=False, *, geometry=None):
    """Two-value dispatch ABI; preserve the selected policy in the emitter receipt."""
    import os
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    from .dense_pipeline import PANEL_GEOMETRIES
    configured = knob('DENSE_PANEL_GEOMETRY', default='balanced', values=tuple(PANEL_GEOMETRIES),
                      root=Path(__file__).resolve().parents[1])
    geometry = geometry or os.environ.get('DENSE_PANEL_GEOMETRY', configured)
    configured_operand = knob('DENSE_RETAINED_OPERAND', default='lhs',
        values=('lhs', 'weight', 'traffic'), root=Path(__file__).resolve().parents[1])
    operand = os.environ.get('DENSE_RETAINED_OPERAND', configured_operand)
    configured_split = knob('DENSE_STORAGE_SPLIT', default='half',
        values=('half', 'adaptive'), root=Path(__file__).resolve().parents[1])
    storage_split = os.environ.get('DENSE_STORAGE_SPLIT', configured_split)
    configured_issue = knob('DENSE_SEGMENTED_ISSUE_ORDER', default='grouped',
        values=('grouped', 'interleaved'), root=Path(__file__).resolve().parents[1])
    issue_order = os.environ.get('DENSE_SEGMENTED_ISSUE_ORDER', configured_issue)
    schedule, refusal, receipt = execute_dense_capacity_policy(
        m, k, n, bias, column_scales, geometry=geometry, retained_operand=operand,
        storage_split=storage_split, issue_order=issue_order)
    if schedule is not None:
        schedule.realization['dense_pipeline_selection'] = receipt
    return schedule, refusal
