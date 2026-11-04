"""Full-K RHS reuse across counted M panels, using the shared checked pass."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView, DenseBiasSource
from merlin.compile.scheduling.dense_panel_pipeline import (
    DenseRetainedReductionAdmission, dense_retained_choices, schedule_dense_m_panels,
)
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from . import isa


def traffic_guarded_m_panel_schedule(m,k,n,bias,column_scales,incumbent):
    """Choose RHS reuse only on an admitted strict saving in scheduled bytes.

    This traffic hypothesis is deliberately independent of latency. Both sides
    are checked current-source schedules; a tie preserves the actual incumbent.
    """
    from merlin.transforms.pipeline_policy import PolicyFacts,ResourcePredicate
    from merlin.transforms.pipeline_state import PipelineContext,PipelineSession
    if incumbent is None or 'segmented_epoch_bounds' in incumbent.realization:
        return incumbent
    incumbent.revalidate()
    candidate,refusal=m_panel_schedule(m,k,n,bias,column_scales)
    if candidate is not None:
        candidate.revalidate()
        if candidate.kernel.args != incumbent.kernel.args:
            raise ValueError('RHS M-panel alternative changed original source arguments')
    old=incumbent.realization['source']
    new=None if candidate is None else candidate.realization['source']
    def input_bytes(source):
        if source is None:
            return None
        values=[source.get('requested_lhs_bytes'),source.get('requested_rhs_bytes'),
                source.get('requested_bias_load_bytes')]
        if not all(type(v) is int and v>=0 for v in values):
            return None
        total=source.get('requested_load_bytes')
        return total if type(total) is int and total==sum(values) else None
    old_bytes,new_bytes=input_bytes(old),input_bytes(new)
    admitted=(candidate is not None and old_bytes is not None and new_bytes is not None
        and all(old.get(k)==new.get(k) for k in ('configuration_identity','numerical_policy_identity')))
    resources=dict(rhs_admitted=int(admitted),strict_input_byte_saving=
        max(0,old_bytes-new_bytes) if admitted else 0)
    custody=hashlib.sha256(json.dumps(dict(incumbent=incumbent.checked_kernel_sha256,
        candidate=None if candidate is None else candidate.checked_kernel_sha256,
        resources=resources),sort_keys=True).encode()).hexdigest()
    context=PipelineContext(hardware=old['configuration_identity'],
        numerical_policy=old['numerical_policy_identity'],resources=custody)
    source=(m,k,n,bool(bias),bool(column_scales))
    session=PipelineSession(source,context=context,ir_level='checked_dense_m_panel_source',
        fingerprint=lambda value:hashlib.sha256(json.dumps(value).encode()).hexdigest(),
        verify_ir=lambda value:None)
    facts=PolicyFacts(session.stamp(),resources=tuple(sorted(resources.items())))
    predicates=(ResourcePredicate('rhs_admitted',1),ResourcePredicate('strict_input_byte_saving',1))
    use_rhs=all(p.match(facts) is True for p in predicates)
    selected=candidate if use_rhs else incumbent
    selected.realization['m_panel_traffic_guard']=dict(
        schema='authored_dense_m_panel_traffic_guard.v1',
        selected_family='rhs_m_panels' if use_rhs else 'incumbent',
        incumbent_requested_input_bytes=old_bytes,rhs_requested_input_bytes=new_bytes,
        incumbent_kernel=incumbent.checked_kernel_sha256,
        rhs_kernel=None if candidate is None else candidate.checked_kernel_sha256,
        resources=resources,resource_identity=custody,rhs_refusal=refusal,
        hypothesis='strictly fewer checked scheduled input bytes; not a cycle estimate')
    return selected


@lru_cache(maxsize=256)
def m_panel_schedule(m, k, n, bias=False, column_scales=False):
    """Construct one source-checked schedule; refusal emits no alternative."""
    from .dense import _adapter
    a = _adapter()
    d = isa.DIM
    p = DenseProblem(m, k, n, d, d, 1 if column_scales else d, 1, 1, 4, bias=bias)
    storage = DenseLoopStorage(
        DenseMemoryView('lhs', 'spad', d, 0, isa.SPAD_ROWS//2),
        DenseMemoryView('rhs', 'spad', d, isa.SPAD_ROWS//2, isa.SPAD_ROWS//2),
        DenseMemoryView('acc', 'acc', 4*d, 0, isa.ACC_ROWS))
    resources = DenseResources(isa.SPAD_ROWS*d, isa.ACC_ROWS*4*d,
        max_m_blocks=32, max_n_blocks=8, max_k_blocks=p.tiles[1],
        partial_output_tiles=True, partial_output_blocks=True)
    # The shared catalogue ranks exact full/tail-family operand traffic.
    choices = dense_retained_choices(p, resources, storage, roles=('weight',))
    if not choices or p.tiles[0] <= 1:
        return None, 'full-K RHS M panels exceed original storage or have no M reuse'
    selected = choices[0]
    b = selected.block
    evidence = json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if evidence['completion_order']['execute']['order'] != 'in_order':
        return None, 'selected execute completion order is unknown'
    admission = DenseRetainedReductionAdmission(a.configuration_identity,
        a.numerical_policy_identity, evidence['identity'],
        max(1,b.m*b.k*b.n), max(1,(b.m+b.n)*b.k), 2, 8)
    source = hashlib.sha256(json.dumps(dict(m=m,k=k,n=n,bias=bias,
        dtype=['i8','i8','i32']),sort_keys=True).encode()).hexdigest()
    attempt = schedule_dense_m_panels(p,resources,storage,a,admission,
        n_tiles=b.n,m_tiles=b.m,name='dense_rhs_m_panels',
        dtypes=('i8','i8','i32'),source_identity=source,
        external_buffers_disjoint=True,static_budget=128,timeout_ms=30000,
        bias_source=DenseBiasSource(n,'i32') if bias else None)
    if attempt.result is None:
        return None, 'full-K RHS M panel refused: '+','.join(attempt.feedback.get('extension_blockers',()))
    result=attempt.result
    result.realization['rhs_m_panels']=attempt.feedback
    result.realization['rhs_m_panels']['selected_block_tiles']=[b.m,b.k,b.n]
    return result,None
