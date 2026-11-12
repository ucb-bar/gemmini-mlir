"""Typed selection over the existing checked full-K storage partitions.

Both artifacts retain their source, private storage, completion proof and
terminal encoder. A larger panel is an opt-in hypothesis, never a timing fact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from merlin.compiler.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy,
)
from merlin.compiler.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession
from merlin.compiler.transforms.shape_policy import ShapeScheduleClass, assess_shape_policy
from . import isa

LEVEL = 'authored_dense_storage_source'
OUTPUT = 'checked_dense_storage_schedule'


def storage_alternatives():
    return (PipelineVariant('adaptive', ('dense_storage_adaptive',), (
                ResourcePredicate('thin_m', 1), ResourcePredicate('rhs_expansion', 1),
                ResourcePredicate('adaptive_legal', 1))),
            PipelineVariant('half', ('dense_storage_half',)))


def assess_dense_storage_policy(feedback, *, design):
    """Assess the frozen class using caller-supplied paired complete timing.

    Missing held-out/boundary evidence leaves the half incumbent. No timing or
    observations are manufactured by this adapter during ordinary compilation.
    """
    group = ShapeScheduleClass('full_k_rhs_expansion', (
        ResourcePredicate('thin_m', 1), ResourcePredicate('rhs_expansion', 1)),
        'half', design)
    return assess_shape_policy((group,), storage_alternatives(), feedback,
        fallback='half', max_regression_ratio=1.0)


@dataclass(frozen=True)
class DenseStorageSource:
    m: int
    k: int
    n: int
    bias: bool = False
    column_scales: bool = False
    geometry: str = 'wide'

    def __post_init__(self):
        from .dense_pipeline import PANEL_GEOMETRIES
        if (any(type(x) is not int or x <= 0 for x in (self.m, self.k, self.n))
                or any(type(x) is not bool for x in (self.bias, self.column_scales))
                or self.geometry not in PANEL_GEOMETRIES):
            raise ValueError('storage policy needs the exact typed dense source')


def storage_session(source):
    if type(source) is not DenseStorageSource:
        raise ValueError('storage policy requires an authored dense source')
    source.__post_init__()
    context = PipelineContext(resources=sha256(json.dumps(dict(
        dim=isa.DIM, spad_rows=isa.SPAD_ROWS, acc_rows=isa.ACC_ROWS,
        geometry=source.geometry, adaptive_max_n=5), sort_keys=True).encode()).hexdigest(),
        execution=sha256(Path(__file__).with_name('dense_ordering.json').read_bytes()).hexdigest())
    return PipelineSession(source, context=context, ir_level=LEVEL,
        fingerprint=lambda s: sha256(json.dumps(asdict(s), sort_keys=True).encode()).hexdigest(),
        verify_ir=lambda s: s.__post_init__())


def storage_candidates(source, planner):
    """Call the real retained planner, with exactly the two storage choices."""
    source.__post_init__()
    candidates = {}
    for split in ('half', 'adaptive'):
        # Five primitive N tiles are one admitted 64-byte band and one
        # primitive tail. This bounded experiment preserves both current ready
        # readout and block DMA, rather than bypassing their exact replay proof.
        options = {'storage_n_cap': 5} if split == 'adaptive' else {}
        result, refusal = planner(source.m, source.k, source.n, source.bias,
            source.column_scales, False, source.geometry, split, **options)
        if result is not None:
            result.revalidate()
            block = result.realization['panel_geometry']['selected_block_tiles']
            if block[1] != (source.k+isa.DIM-1)//isa.DIM:
                raise ValueError('storage policy cannot select partial-K artifacts')
            expected = [('lhs', (source.m, source.k), 'i8', 'read'),
                        ('rhs', (source.k, source.n), 'i8', 'read')]
            if source.bias:
                expected.append(('bias', (source.n,), 'i32', 'read'))
            expected.append(('output', (source.m, source.n), 'i32', 'write'))
            if [(a.name, a.shape, a.dtype, a.access) for a in result.kernel.args] != expected:
                raise ValueError('storage candidate changed original source effects')
            if refusal is not None:
                raise ValueError('checked storage candidate has a refusal')
        elif not refusal:
            raise ValueError('missing storage candidate lost its original refusal')
        candidates[split] = (result, refusal)
    return candidates


def storage_facts(session, candidates):
    s = session.module
    blocks = {}
    for name, (result, _) in candidates.items():
        if result is not None:
            result.revalidate()
        blocks[name] = None if result is None else result.realization['panel_geometry']['selected_block_tiles']
    half, adaptive = blocks['half'], blocks['adaptive']
    # This class describes a real larger RHS panel for one primitive M tile.
    # It grants neither a speedup nor new storage/completion semantics.
    expansion = (half is not None and adaptive is not None
        and adaptive[2] > half[2] and not s.column_scales)
    resources = dict(m_tiles=(s.m+isa.DIM-1)//isa.DIM,
        k_tiles=(s.k+isa.DIM-1)//isa.DIM,
        n_tiles=(s.n+(1 if s.column_scales else isa.DIM)-1)//(1 if s.column_scales else isa.DIM),
        half_legal=int(half is not None), adaptive_legal=int(adaptive is not None),
        half_n_blocks=0 if half is None else half[2],
        adaptive_n_blocks=0 if adaptive is None else adaptive[2],
        rhs_expansion=int(expansion), thin_m=int(s.m <= isa.DIM),
        dim=isa.DIM, spad_rows=isa.SPAD_ROWS, acc_rows=isa.ACC_ROWS)
    return PolicyFacts(session.stamp(), shapes=((s.m, s.k), (s.k, s.n)),
        element_types=('i8', 'i8'), layouts=('row_major', 'row_major'),
        operations=frozenset({'dense'}), resources=tuple(sorted(resources.items())))


def execute_storage_policy(source, planner):
    """Return the actual selected checked artifact and a source-bound receipt."""
    session = storage_session(source)
    candidates = storage_candidates(source, planner)
    facts = storage_facts(session, candidates)
    identities = {name: None if r is None else r.kernel.digest()
                  for name, (r, _) in candidates.items()}
    registry = {}
    for name in ('half', 'adaptive'):
        def action(current, name=name):
            if current.module != source:
                raise ValueError('storage registry cannot reuse a training artifact')
            result, refusal = candidates[name]
            if result is not None:
                result.revalidate()
                if result.kernel.digest() != identities[name]:
                    raise ValueError('registered storage artifact changed')
            return {OUTPUT: (result, refusal)}
        key = 'dense_storage_'+name
        registry[key] = RegisteredPass(PassSpec(key, 0, produces=(OUTPUT,),
            mutates_ir=False, ir_level=LEVEL), action)
    policy = PipelinePolicy(storage_alternatives(), fallback='half')
    selected = execute_policy(session, policy, registry, facts, output=OUTPUT)
    result, refusal = selected.artifact
    if result is not None:
        result.realization['storage_policy'] = dict(schema='dense_storage_policy.v1',
            policy=selected.receipt, resources=dict(facts.resources), candidate_kernels=identities)
    return result, refusal
