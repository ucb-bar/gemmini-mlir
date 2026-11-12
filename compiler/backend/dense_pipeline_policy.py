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

from merlin.compiler.transforms.pipeline_policy import PipelineVariant, PolicyFacts, RegisteredPass, ResourcePredicate
from merlin.compiler.transforms.pipeline_state import PassSpec, PipelineSession
from merlin.compiler.transforms.shape_policy import AssessedShapePolicy, ShapeScheduleClass, assess_shape_policy

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


def execute_capacity_pipeline(m, k, n, bias=False, column_scales=False, *, ready_readout=False):
    """Select the shared segmented path only for a capacity-starved RHS panel.

    This frozen resource class is exposed as an explicit tuning alternative.
    Both candidates retain their original source/storage/epoch admission; policy
    dispatch returns the actual checked artifact consumed by the terminal encoder.
    No evaluation, source identity lookup or runtime profiling occurs here.
    """
    from pathlib import Path
    import os
    from merlin.compiler.selection.knobs import knob
    from merlin.compiler.transforms.pipeline_state import PipelineContext
    from merlin.compiler.transforms.pipeline_policy import PipelinePolicy, execute_policy
    from .dense_pipeline import retained_schedule, _retained_schedule, PANEL_GEOMETRIES
    root = Path(__file__).resolve().parents[1]
    configured = knob('DENSE_PANEL_GEOMETRY', default='balanced',
        values=tuple(PANEL_GEOMETRIES), root=root)
    geometry = os.environ.get('DENSE_PANEL_GEOMETRY', configured)
    split = knob('DENSE_STORAGE_SPLIT', default='half', values=('half','adaptive'), root=root)
    # Construct the incumbent first. An illegal full-K panel must keep the
    # established streaming fallback; a single-N panel has no expansion gain.
    retained, refusal = retained_schedule(m, k, n, bias, column_scales, geometry=geometry, ready_readout=ready_readout)
    family = knob('DENSE_CAPACITY_FAMILY', default='incumbent',
        values=('incumbent', 'segmented_if_legal'), root=root)
    if family == 'segmented_if_legal':
        from .dense_capacity_alternatives import execute_segmented_alternative
        return execute_segmented_alternative(m, k, n, bias, column_scales,
            retained, refusal, ready_readout=ready_readout, storage_split=split)
    block = None if retained is None else retained.realization['panel_geometry']['selected_block_tiles']
    if column_scales or block is None or block[2] != 1 or n <= isa.DIM:
        return retained, refusal, {'selected':'retained', 'reason':'no RHS panel capacity bottleneck'}
    source = DensePipelineSource(m, k, n, bias, column_scales, geometry, split)
    ordering = Path(__file__).with_name('dense_ordering.json')
    context = PipelineContext(resources=sha256(json.dumps(
        {'dim':isa.DIM, 'spad_rows':isa.SPAD_ROWS, 'acc_rows':isa.ACC_ROWS,
         'geometry':geometry, 'split':split}, sort_keys=True).encode()).hexdigest(),
        execution=sha256(ordering.read_bytes()).hexdigest())
    session = pipeline_policy_session(source, context=context)
    transfers = knob('DENSE_RHS_TRANSFERS', default='primitive', values=('primitive','block64'), root=root)
    operand = knob('DENSE_RETAINED_OPERAND', default='lhs', values=('lhs','rhs_reuse','rhs_amortized'), root=root)
    issue = knob('DENSE_SEGMENTED_ISSUE', default='compute_then_prefetch',
        values=('compute_then_prefetch','prefetch_rhs_between_columns','prefetch_rhs_between_k_tiles','prefetch_rhs_grouped_columns'), root=root)
    placement = knob('DENSE_SEGMENTED_PLACEMENT', default='compact', values=('compact','window_halves'), root=root)
    segmented_transfers = knob('DENSE_SEGMENTED_TRANSFERS', default='primitive', values=('primitive','block64'), root=root)
    from .dense_segmented_readout import segmented_readout_selection
    segmented_readout_mode = segmented_readout_selection()
    input_pipeline = knob('DENSE_INPUT_PIPELINE',default='off',values=('off','two_slot'),root=root)
    candidates = pipeline_policy_candidates(source, retained_planner=lambda *args:
        _retained_schedule(*args, rhs_transfers=transfers, retained_operand=operand, segmented_issue=issue, segmented_placement=placement, ready_readout=ready_readout, segmented_transfers=segmented_transfers,
            segmented_readout_mode=segmented_readout_mode,input_pipeline=input_pipeline))
    facts = pipeline_policy_facts(session, candidates)
    policy = PipelinePolicy((
        PipelineVariant('segmented', ('dense_pipeline_segmented',), (
            ResourcePredicate('rhs_capacity_starved', 1),
            ResourcePredicate('segmented_legal', 1))),
        PipelineVariant('retained', ('dense_pipeline_retained',))), fallback='retained')
    result = execute_policy(session, policy, pipeline_policy_registry(candidates), facts, output=OUTPUT)
    schedule, refusal = result.artifact
    selected = 'segmented' if schedule is candidates.segmented else 'retained'
    if schedule is not None:
        schedule.realization['capacity_pipeline_policy'] = {
            'selected':selected, 'resources':dict(facts.resources),
            'candidate_kernels':dict(candidates.kernel_identities), 'policy':result.receipt}
    return schedule, refusal, {'selected':selected, 'policy':result.receipt}
