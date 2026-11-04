"""Optional offline-evidenced selection of existing retained dense geometries.

No timing runs during compilation. Missing panel support, current domain or
resource context keeps the wide incumbent. Every selected path constructs the
ordinary checked schedule and leaves emission with its existing target encoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks
from merlin.transforms.pipeline_policy import (
    PipelineVariant, PolicyFacts, RegisteredPass, ResourcePredicate,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineSession
from merlin.transforms.shape_policy import AssessedShapePolicy, ShapeScheduleClass, assess_shape_policy

from . import isa

LEVEL = 'authored_dense_geometry_source'
OUTPUT = 'checked_dense_geometry_schedule'
RESOURCE_NAMES = frozenset({'m_tiles', 'n_tiles', 'k_tiles', 'output_tiles',
    'primitive_products', 'm_tiles_ge_n_tiles', 'dim', 'spad_rows', 'acc_rows',
    'wide_legal', 'tall_16_legal'})


@dataclass(frozen=True)
class DenseGeometrySource:
    m: int
    k: int
    n: int
    bias: bool = False
    column_scales: bool = False
    storage_split: str = 'half'

    def __post_init__(self):
        if (any(type(v) is not int or v <= 0 for v in (self.m, self.k, self.n))
                or any(type(v) is not bool for v in (self.bias, self.column_scales))
                or self.storage_split not in ('half', 'adaptive')):
            raise ValueError('dense geometry policy needs positive extents and exact retained source options')


def geometry_policy_session(source, *, context=None):
    """Expose the original immutable source/context stamp for offline feedback."""
    if type(source) is not DenseGeometrySource:
        raise ValueError('dense geometry policy requires a typed original source')
    def verify(value):
        if type(value) is not DenseGeometrySource:
            raise ValueError('dense geometry source changed type')
        value.__post_init__()
    verify(source)
    return PipelineSession(source, context=context, ir_level=LEVEL,
        fingerprint=lambda value: sha256(json.dumps(asdict(value), sort_keys=True).encode()).hexdigest(),
        verify_ir=verify)


def _legal(source, geometry):
    # Use the same selected physical views and original candidate admission as
    # dense_pipeline. These predicates grant no substitute source/epoch proof.
    from .dense_pipeline import PANEL_GEOMETRIES
    dim = isa.DIM
    p = DenseProblem(source.m, source.k, source.n, dim, dim,
        1 if source.column_scales else dim, 1, 1, 4, bias=source.bias)
    max_m, max_n = PANEL_GEOMETRIES[geometry]
    r = DenseResources(isa.SPAD_ROWS*dim, isa.ACC_ROWS*4*dim,
        max_m_blocks=max_m, max_n_blocks=max_n, max_k_blocks=p.tiles[1],
        partial_output_tiles=True, partial_output_blocks=True)
    if source.storage_split == 'adaptive':
        from .dense_storage_partition import select_full_k_partition
        return select_full_k_partition(p, r, max_m_blocks=max_m, max_n_blocks=max_n) is not None
    s = DenseLoopStorage(
        DenseMemoryView('lhs', 'spad', dim, 0, isa.SPAD_ROWS//2),
        DenseMemoryView('rhs', 'spad', dim, isa.SPAD_ROWS//2, isa.SPAD_ROWS//2),
        DenseMemoryView('acc', 'acc', 4*dim, 0, isa.ACC_ROWS))
    return any(b.k == p.tiles[1] and s.admits(p, b, r)
        for b in dense_blocks(p, r, k_limit=lambda m, n: s.k_limit(p, r, m, n)))


def geometry_policy_facts(session):
    source = session.module
    source.__post_init__()
    ntile = 1 if source.column_scales else isa.DIM
    mt, nt = (source.m+isa.DIM-1)//isa.DIM, (source.n+ntile-1)//ntile
    kt = (source.k+isa.DIM-1)//isa.DIM
    resources = {'m_tiles': mt, 'n_tiles': nt, 'k_tiles': kt,
        'output_tiles': mt*nt, 'primitive_products': mt*kt*nt,
        'm_tiles_ge_n_tiles': int(mt >= nt),
        'dim': isa.DIM, 'spad_rows': isa.SPAD_ROWS, 'acc_rows': isa.ACC_ROWS,
        'wide_legal': int(_legal(source, 'wide')),
        'tall_16_legal': int(_legal(source, 'tall_16'))}
    return PolicyFacts(session.stamp(), shapes=((source.m, source.k), (source.k, source.n)),
        element_types=('i8', 'i8'), layouts=('row_major', 'row_major'),
        operations=frozenset({'dense'}), resources=tuple(sorted(resources.items())))


def geometry_policy_alternatives():
    return (PipelineVariant('tall_16', ('dense_geometry_tall_16',),
                (ResourcePredicate('tall_16_legal', 1),)),
            PipelineVariant('wide', ('dense_geometry_wide',)))


def assess_dense_geometry_policy(feedback, *, design, predicates=None, min_m_tiles=None):
    """Assess caller-frozen geometry panel and cutoff using exact observations.

    Callers supply original ShapeCaseFeedback/ValidationPartition objects and
    their independent identities. This function never fills missing evidence.
    Pass an immutable tuple of existing ResourcePredicate objects to freeze a
    class without an aspect-ratio restriction. The legacy explicit min_m_tiles
    form retains its tall/square class. Neither form infers a cutoff or fills
    boundary evidence. Resource units are the actual selected primitive tiles.
    """
    if predicates is None:
        if type(min_m_tiles) is not int or min_m_tiles < 1:
            raise ValueError('freeze resource predicates or a positive M tile cutoff before assessment')
        predicates = (ResourcePredicate('m_tiles_ge_n_tiles', 1),
                      ResourcePredicate('m_tiles', min_m_tiles))
    elif min_m_tiles is not None:
        raise ValueError('freeze either explicit predicates or the legacy M tile class')
    if (type(predicates) is not tuple or not 1 <= len(predicates) <= 16
            or any(type(p) is not ResourcePredicate or p.resource not in RESOURCE_NAMES
                   for p in predicates)):
        raise ValueError('dense geometry classes require existing typed selected-resource predicates')
    group = ShapeScheduleClass('retained_geometry', predicates, 'wide', design)
    return assess_shape_policy((group,), geometry_policy_alternatives(), feedback,
        fallback='wide', max_regression_ratio=1.0)


def geometry_policy_registry(*, retained_planner=None):
    """Bind the real non-dispatching planner, optionally supplied by its owner.

    Callback ABI is the existing _retained_schedule positional ABI. Passing it
    explicitly permits dense_pipeline to invoke this hook without importing or
    recursively calling its public policy dispatcher.
    """
    if retained_planner is None:
        from .dense_pipeline import _retained_schedule
        retained_planner = _retained_schedule
    if not callable(retained_planner):
        raise ValueError('dense geometry policy requires a non-dispatching retained planner')
    def entry(geometry):
        name = 'dense_geometry_'+geometry
        def action(session):
            s = session.module
            result, refusal = retained_planner(s.m, s.k, s.n, s.bias,
                s.column_scales, False, geometry, s.storage_split)
            if result is not None:
                result.revalidate()
                if result.realization.get('panel_geometry', {}).get('selection') != geometry:
                    raise ValueError('registered dense planner returned a different geometry')
            return {OUTPUT: (result, refusal)}
        return name, RegisteredPass(PassSpec(name, 0, produces=(OUTPUT,),
            mutates_ir=False, ir_level=LEVEL), action)
    return dict(entry(g) for g in ('wide', 'tall_16'))


def execute_dense_geometry(m, k, n, bias=False, column_scales=False, *,
        storage_split='half', context=None, policy=None, domain=None, retained_planner=None):
    """Return ``(actual_checked_schedule_or_None, refusal, receipt)``.

    The schedule is the artifact executed by PipelinePolicy, not a catalogue
    entry or a cached training kernel. Its normal emit_llvm method remains the
    terminal lowering path. Full-K capacity refusal remains a normal outcome.
    Segmented/streaming and the compact single-product guard stay with callers.
    """
    source = DenseGeometrySource(m, k, n, bias, column_scales, storage_split)
    session = geometry_policy_session(source, context=context)
    facts = geometry_policy_facts(session)
    if policy is None:
        policy = assess_dense_geometry_policy((), design=None, min_m_tiles=1)
    if (type(policy) is not AssessedShapePolicy or policy.fallback != 'wide'
            or policy.alternatives != geometry_policy_alternatives()
            or any(c.incumbent != 'wide' or any(type(p) is not ResourcePredicate
                or p.resource not in RESOURCE_NAMES for p in c.predicates)
                for c in policy.classes)):
        raise ValueError('dense geometry policy must assess the original registered alternatives and incumbent')
    selected = policy.execute(session, geometry_policy_registry(retained_planner=retained_planner),
        facts, output=OUTPUT, domain=domain)
    schedule, refusal = selected.artifact
    receipt = dict(schema='authored_dense_geometry_policy.v1',
        policy=selected.receipt, assessment=policy.report, refusal=refusal,
        realization=None if schedule is None else dict(schedule.realization))
    return schedule, refusal, receipt


def resolve_admitted_geometry(m, k, n, bias=False, column_scales=False, *,
        storage_split='half', assessment=None, retained_planner=None):
    """Return ``(geometry, actual_retained_schedule, refusal, receipt)``.

    The current checked retained result is reusable by the caller's subsequent
    regional pipeline planner. This choice does not select a pipeline family:
    segmented capacity cases must still pass through the regional policy.
    The only empirical guard is the prospectively frozen M/N tile class.
    """
    from pathlib import Path
    from dataclasses import replace
    from merlin.transforms.pipeline_policy import PipelinePolicy, execute_policy
    from merlin.transforms.pipeline_state import PipelineContext
    if assessment is None:
        path = Path(__file__).with_name('data') / 'dense_geometry_native_policy.json'
        assessment = json.loads(path.read_text()) if path.is_file() else None
    admitted = False
    rule = dict(m_tiles_ge_n_tiles=1, m_tiles_min=16,
                legality='actual full-K retained planner physical admission')
    if assessment is not None:
        if (type(assessment) is not dict
                or assessment.get('schema') != 'dense_native_geometry_assessment.v1'
                or assessment.get('frozen_rule') != rule
                or assessment.get('incumbent') != 'wide'
                or assessment.get('candidate') != 'tall_16'):
            raise ValueError('native geometry assessment changed its frozen alternatives or guard')
        admitted = (assessment.get('admitted') is True and not assessment.get('refusal')
                    and storage_split == assessment.get('storage_split'))
    source = DenseGeometrySource(m, k, n, bias, column_scales, storage_split)
    context = PipelineContext(hardware='gemmini-fire-aligned-dim16-ws',
        numerical_policy='i8-i8-i32-modular', resources=sha256(
            json.dumps([isa.DIM, isa.SPAD_ROWS, isa.ACC_ROWS]).encode()).hexdigest())
    session = geometry_policy_session(source, context=context)
    facts = geometry_policy_facts(session)
    if retained_planner is None:
        from .dense_pipeline import _retained_schedule
        retained_planner = _retained_schedule
    cache = {}
    def current_planner(mm, kk, nn, bb, cc, segmented, geometry, split):
        if ((mm, kk, nn, bb, cc, split) != (m, k, n, bias, column_scales, storage_split)
                or segmented):
            raise ValueError('geometry registry cannot substitute a different original source or family')
        if geometry not in cache:
            cache[geometry] = retained_planner(mm, kk, nn, bb, cc, False, geometry, split)
        return cache[geometry]
    resources = dict(facts.resources)
    coarse = resources['m_tiles_ge_n_tiles'] and resources['m_tiles'] >= 16
    regional_priority = False
    if admitted and coarse:
        wide, _ = current_planner(m,k,n,bias,column_scales,False,'wide',storage_split)
        regional_priority = (wide is not None and resources['n_tiles'] > 1
            and wide.realization['panel_geometry']['selected_block_tiles'][2] == 1)
    # A full-K geometry decision must not narrow the independent regional
    # segmented candidate's RHS cap. Its existing capacity policy keeps priority.
    resources['preserves_regional_panel_capacity'] = int(not regional_priority)
    # Physical fitting is separate from source/epoch admission. The latter must
    # succeed before the optional path becomes eligible in the policy engine.
    checked_tall = None
    if admitted and coarse and not regional_priority and resources['tall_16_legal']:
        checked_tall, _ = current_planner(m,k,n,bias,column_scales,False,'tall_16',storage_split)
    resources['tall_16_legal'] = int(checked_tall is not None)
    facts = replace(facts, resources=tuple(sorted(resources.items())))
    fallback = PipelineVariant('wide', ('dense_geometry_wide',))
    paths = (PipelineVariant('tall_16', ('dense_geometry_tall_16',),
        (ResourcePredicate('m_tiles_ge_n_tiles', 1), ResourcePredicate('m_tiles', 16),
         ResourcePredicate('tall_16_legal', 1),
         ResourcePredicate('preserves_regional_panel_capacity', 1))), fallback) if admitted else (fallback,)
    selected = execute_policy(session, PipelinePolicy(paths, 'wide'),
        geometry_policy_registry(retained_planner=current_planner), facts, output=OUTPUT)
    schedule, refusal = selected.artifact
    receipt = dict(schema='authored_dense_native_geometry_policy.v1',
        policy={key:value for key,value in selected.receipt.items() if key!='host_pass_profile'},
        native_assessment_admitted=admitted,
        native_assessment_sha256=None if assessment is None else sha256(
            json.dumps(assessment,sort_keys=True).encode()).hexdigest(),
        evidence=None if assessment is None else assessment.get('evidence'),
        candidate_kernels={g:None if result is None else result.kernel.digest()
                           for g,(result,_) in cache.items()},
        candidate_refusals={g:reason for g,(_,reason) in cache.items()},
        deferred_to_regional_capacity_policy=regional_priority,
        actual_checked_retained_reusable=True, selects_pipeline_family=False,
        physical_qualification_inferred=False, runtime_profiling=False)
    return selected.receipt['selected_variant'], schedule, refusal, receipt


def assess_native_geometry_panel(panel, *, baseline_response, training_response,
        heldout_response, workspace='.'):
    """Offline admission from original hash-bound native records and sources.

    Training must support the frozen rule before this function reads heldout
    feedback. Native observations are kept under their original identity kinds;
    no input/reference/EvaluationRequest bindings are reconstructed from them.
    """
    from pathlib import Path
    from dataclasses import replace
    from .dense_pipeline_evidence import load_native_observation, _pin
    from .frontend import parse_source
    from merlin.transforms.pass_exploration import ObservedMetrics, rank_observed_metrics
    rule = dict(m_tiles_ge_n_tiles=1,m_tiles_min=16,
                legality='actual full-K retained planner physical admission')
    if (type(panel) is not dict or panel.get('schema')!='authored.geometry_experiment.v1'
            or panel.get('frozen_before_collecting_native_results') is not True
            or panel.get('frozen_rule')!=rule or panel.get('candidate')!='tall_16'
            or panel.get('incumbent')!='wide'):
        raise ValueError('native geometry feedback needs the original frozen rule and roster')
    training,heldout=tuple(panel['training']),tuple(panel['held_out'])
    if not training or not heldout or len(set(training+heldout))!=len(training+heldout):
        raise ValueError('native geometry partitions must be nonempty and disjoint')
    root=Path(workspace).resolve()
    reasons,rows,domains=[],[],set()
    def collect(case,partition,response):
        source_path=root/'public_sources'/(case+'.mlir')
        source_bytes=source_path.read_bytes()
        cb=parse_source(source_bytes.decode())
        m,k=cb['tensors']['A0']['shape'];kk,n=cb['tensors']['W']['shape']
        if k!=kk:
            raise ValueError('native geometry source is not the original dense product')
        inside=(m+isa.DIM-1)//isa.DIM>=max(16,(n+isa.DIM-1)//isa.DIM)
        # The native result contains no alternative label; labels below refer
        # only to the explicit immutable request's registered geometry choice.
        pair=tuple(replace(load_native_observation(path,case=case,alternative=label,
            workspace=root),alternative=geometry) for path,label,geometry in
            ((baseline_response,'retained','wide'),(response,'segmented','tall_16')))
        lhs,rhs=pair
        keys=('facts_sha256','model_sha256','scope_sha256','comparison_domain_sha256','capsule_sha256')
        valid=all(_pin(getattr(lhs,key)) and getattr(lhs,key)==getattr(rhs,key) for key in keys)
        valid=valid and all(r.numerical_status=='pass' and r.failure_category is None
            and r.complete_modeled_window and _pin(r.elf_sha256) for r in pair)
        if not valid:reasons.append(case+': incomplete, nonpassing or different-domain native pair')
        if inside and valid and rhs.cycles>=lhs.cycles:
            reasons.append(case+': frozen class has no complete-cycle gain')
        domains.add((lhs.facts_sha256,lhs.model_sha256,lhs.scope_sha256))
        ranking=None if not valid else rank_observed_metrics(tuple((r.alternative,
            ObservedMetrics('modeled_roi',r.comparison_domain_sha256,(('cycles',r.cycles),)))
            for r in pair),objectives=('cycles',))
        rows.append(dict(case=case,partition=partition,shape=[m,k,n],inside_frozen_class=inside,
            original_source_sha256=sha256(source_bytes).hexdigest(),
            observations=[asdict(r) for r in pair],ranking=ranking,
            policy_selection='tall_16' if inside and valid else 'wide'))
        return inside
    for case in training:
        if not collect(case,'training',training_response):
            reasons.append(case+': training source is outside the frozen class')
    training_supported=not reasons
    if training_supported:
        for case in heldout:collect(case,'held_out',heldout_response)
        if not any(r['partition']=='held_out' and r['inside_frozen_class'] for r in rows):
            reasons.append('missing held-out inside-class native geometry')
        if not any(r['partition']=='held_out' and not r['inside_frozen_class'] for r in rows):
            reasons.append('missing held-out outside-class boundary')
    if len(domains)!=1:reasons.append('native panel model, hardware facts or elapsed-call scope differ')
    return dict(schema='dense_native_geometry_assessment.v1',admitted=not reasons,
        refusal=reasons,candidate='tall_16',incumbent='wide',frozen_rule=rule,
        storage_split='half',training_supported=training_supported,
        held_out_inspected=training_supported,observations=rows,
        frozen_panel_sha256=sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest(),
        common_domain=dict(zip(('facts_sha256','model_sha256','scope_sha256'),next(iter(domains))))
            if len(domains)==1 else None,
        evidence='modeled: native Spike timing model; provisional; complete same-call paired windows',
        limitations=['M tile cutoff neighbors15/16/17 are not natively timed',
            'unmeasured source modifiers and shapes extrapolate the frozen resource class',
            'outside-class alternatives are not adopted, even when faster'],
        physical_qualification=False,exact_current_executable_equivalence_claimed=False)
