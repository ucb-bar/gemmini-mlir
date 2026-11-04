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
