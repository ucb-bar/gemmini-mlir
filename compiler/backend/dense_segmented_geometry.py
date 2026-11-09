"""Optional geometry stage over two original checked segmented schedules.

This stage follows capacity-family selection. Each geometry keeps its own typed
source/candidate contract; the returned artifact is emitted by the existing
dense terminal encoder. Capacity is legality, not a performance prediction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from merlin.compiler.selection.knobs import knob
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

LEVEL = 'authored_segmented_geometry_source'
OUTPUT = 'checked_segmented_geometry_schedule'


def segmented_geometry_selection():
    return knob('DENSE_SEGMENTED_GEOMETRY', default='incumbent',
        values=('incumbent', 'wide_16'), root=Path(__file__).resolve().parents[1])


@dataclass(frozen=True)
class SegmentedGeometrySource:
    m: int
    k: int
    n: int
    bias: bool
    column_scales: bool
    storage_split: str

    def revalidate(self):
        from .dense_pipeline_policy import DensePipelineSource
        DensePipelineSource(self.m, self.k, self.n, self.bias,
            self.column_scales, 'wide', self.storage_split)


def _segmented_owner(result):
    from .dense_segmented_readout import DenseSegmentedReadyResult, _owners
    from .dense_rhs_transfers import DenseRhsTransferResult
    from .dense_segmented_frontier import DenseGroupedFrontierResult
    from merlin.compiler.schedule.dense_segmented_pipeline import DenseSegmentedPipelineResult
    if type(result) is DenseSegmentedReadyResult:
        result.revalidate()
        result = result.original
    if type(result) not in (DenseRhsTransferResult, DenseGroupedFrontierResult,
                           DenseSegmentedPipelineResult):
        return None
    # A RHS wrapper can also own a full-K schedule. Preserve that incumbent.
    if type(result) is DenseRhsTransferResult:
        candidate = result.checked.original
        if type(candidate) is DenseGroupedFrontierResult:
            candidate = candidate.original
        if type(candidate) is not DenseSegmentedPipelineResult:
            return None
    return _owners(result)[0]


@dataclass(frozen=True)
class SegmentedGeometryCandidates:
    source: SegmentedGeometrySource
    wide: object
    wide_16: object
    wide_16_refusal: str | None
    identities: tuple

    def revalidate(self):
        from .dense_pipeline_policy import DensePipelineSource, DensePipelineCandidates
        self.source.revalidate()
        s = self.source
        for geometry in ('wide', 'wide_16'):
            result = getattr(self, geometry)
            refusal = None if geometry == 'wide' else self.wide_16_refusal
            source = DensePipelineSource(s.m, s.k, s.n, s.bias,
                s.column_scales, geometry, s.storage_split)
            # Reuse the original ABI and geometry checker independently for
            # each geometry. No wide_16 artifact is labeled as a wide result.
            candidates = DensePipelineCandidates(source, None,
                'full-K is outside this segmented geometry stage', result,
                refusal, (('retained', None), ('segmented', dict(self.identities)[geometry])))
            candidates.revalidate()
            if result is not None and _segmented_owner(result) is None:
                raise ValueError('geometry stage requires an actual checked segmented owner')


def execute_segmented_geometry(m, k, n, bias, column_scales, incumbent, refusal,
                               *, ready_readout=False, selection=None):
    """Return the actual selected artifact, original refusal, and stage receipt."""
    mode = segmented_geometry_selection() if selection is None else selection
    if mode not in ('incumbent', 'wide_16'):
        raise ValueError('invalid segmented geometry selection')
    def decline(reason):
        return incumbent, refusal, {'selection': mode, 'selected': 'incumbent',
            'reason': reason}
    if mode == 'incumbent':
        return decline('incumbent selected')
    if incumbent is None or refusal is not None:
        return decline('original incumbent unavailable')
    incumbent.revalidate()
    if column_scales or _segmented_owner(incumbent) is None:
        return decline('source is not a uniform-column segmented schedule')
    if incumbent.realization['panel_geometry']['selection'] != 'wide':
        return decline('original geometry is not wide')
    split = knob('DENSE_STORAGE_SPLIT', default='half', values=('half', 'adaptive'),
        root=Path(__file__).resolve().parents[1])
    source = SegmentedGeometrySource(m, k, n, bias, column_scales, split)
    source.revalidate()
    from .dense_pipeline import retained_schedule
    from merlin.compiler.schedule.compact_verification import DenseVerificationUnavailable
    try:
        alternate, alternate_refusal = retained_schedule(m, k, n, bias,
            column_scales, True, geometry='wide_16', ready_readout=ready_readout)
    except DenseVerificationUnavailable as exc:
        alternate, alternate_refusal = None, type(exc).__name__ + ': ' + str(exc)
    candidates = SegmentedGeometryCandidates(source, incumbent, alternate,
        alternate_refusal, tuple((g, None if r is None else r.kernel.digest())
        for g, r in (('wide', incumbent), ('wide_16', alternate))))
    candidates.revalidate()
    wb = incumbent.realization['panel_geometry']['selected_block_tiles']
    ab = None if alternate is None else alternate.realization['panel_geometry']['selected_block_tiles']
    resources = dict(wide_16_legal=int(alternate is not None),
        wider_n=int(ab is not None and ab[2] > wb[2]),
        preserves_m=int(ab is not None and ab[0] >= wb[0]),
        preserves_k=int(ab is not None and ab[1] == wb[1]),
        distinct_schedule=int(alternate is not None and alternate.kernel.digest() != incumbent.kernel.digest()))
    # Resource predicates deliberately stay UNKNOWN without a selected context.
    # Bind this context to both original checked kernels and their actual blocks.
    context = PipelineContext(resources=sha256(json.dumps({
        'source': asdict(source), 'kernels': dict(candidates.identities),
        'blocks': [wb, ab], 'resources': resources,
        'storage': [incumbent.realization['panel_geometry'],
            None if alternate is None else alternate.realization['panel_geometry']],
    }, sort_keys=True).encode()).hexdigest())
    session = PipelineSession(source, context=context, ir_level=LEVEL,
        fingerprint=lambda s: sha256(json.dumps(asdict(s), sort_keys=True).encode()).hexdigest(),
        verify_ir=lambda s: s.revalidate())
    facts = PolicyFacts(session.stamp(), shapes=((m, k), (k, n)),
        element_types=('i8', 'i8'), layouts=('row_major', 'row_major'),
        operations=frozenset({'dense'}), resources=tuple(sorted(resources.items())))
    def entry(geometry):
        def action(current):
            if current.module != source:
                raise ValueError('segmented geometry source changed before selection')
            candidates.revalidate()
            return {OUTPUT: (getattr(candidates, geometry),
                None if geometry == 'wide' else candidates.wide_16_refusal)}
        key = 'segmented_geometry_' + geometry
        return key, RegisteredPass(PassSpec(key, 0, produces=(OUTPUT,),
            mutates_ir=False, ir_level=LEVEL), action)
    policy = PipelinePolicy((
        PipelineVariant('wide_16', ('segmented_geometry_wide_16',),
            tuple(ResourcePredicate(name, 1) for name in resources)),
        PipelineVariant('wide', ('segmented_geometry_wide',))), fallback='wide')
    selected = execute_policy(session, policy, dict(entry(g) for g in ('wide', 'wide_16')),
        facts, output=OUTPUT)
    result, selected_refusal = selected.artifact
    receipt = {'selection': mode, 'selected': 'wide_16' if result is alternate else 'incumbent',
        'source': asdict(source), 'resources': resources,
        'candidate_kernels': dict(candidates.identities),
        'candidate_refusals': {'wide': None, 'wide_16': alternate_refusal},
        'candidate_blocks': {'wide': wb, 'wide_16': ab}, 'policy': selected.receipt,
        'module_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}
    result.realization['segmented_geometry_stage'] = receipt
    result.revalidate()
    return result, selected_refusal, receipt
