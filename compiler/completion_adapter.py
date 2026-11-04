"""Backend binding for completion-only dense schedule composition.

The public entry never constructs data commands or changes the route. Unsupported
owners and failed obligations return the valid original kernel with a refusal
receipt. Production supplies source-counted issued bounds and the shared target
premises; apply_completion never silently increases a caller's admission.
"""
from dataclasses import dataclass, is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from collections.abc import Mapping

from merlin.compile.scheduling.dense_completion import (
    CompletionRefusal, coalesce_dense_completion, propose_completion, issued_requirements, _bounds,
    make_completion_plan,
)
from merlin.compile.scheduling.dense_panel_pipeline import (
    DensePanelPipelineResult, DenseRetainedReductionAdmission,
    verify_dense_panel_pipeline, verify_dense_retained_reduction,
)
from merlin.compile.scheduling.dense_segmented_pipeline import (
    DenseSegmentedPipelineResult, verify_dense_segmented_pipeline,
)


def _owner(original):
    # The inherited retained helper is a composition wrapper, not two-window
    # construction. Its actual checked stream stays the source of this pass.
    from segmented_k import RetainedLhsResult
    if type(original) is RetainedLhsResult:
        return original.checked
    if type(original) in (DensePanelPipelineResult, DenseSegmentedPipelineResult):
        return original
    raise CompletionRefusal('unsupported checked result owner: '+type(original).__name__+
        '; no proved completion/ownership interface for this route','unsupported_owner')


def _without_budgets(admission):
    if hasattr(admission,'overlap') and hasattr(admission,'reduction'):
        return replace(admission,overlap=_without_budgets(admission.overlap),
                       reduction=_without_budgets(admission.reduction))
    return replace(admission,max_issued_compute_pairs=1,max_issued_loads=1,max_issued_readouts=1)


def _validate_original(original,context):
    original.revalidate()
    owner=_owner(original)
    if original.kernel!=owner.kernel:
        raise CompletionRefusal('wrapper does not own the actual checked command stream')
    if _without_budgets(context['admission'])!=_without_budgets(owner.admission):
        raise CompletionRefusal('completion context changes selected physical queues or execution identity')
    p,b,r,s,a=(context[key] for key in ('problem','block','resources','storage','adapter'))
    # The backend encoder uses physical Gemmini row numbers. A caller cannot
    # turn aliases into independent memories by renaming a backing or changing
    # its byte units. Derive this binding from the selected target facts.
    from dense import _geometry
    edge,operand_rows,accumulator_rows=_geometry()
    if ((p.tile_m,p.tile_k,p.tile_n,p.lhs_bytes,p.rhs_bytes,p.accumulator_bytes)
            !=(edge,edge,edge,1,1,4)):
        raise CompletionRefusal('problem differs from the backend physical tile/element mapping','context_mismatch')
    views=((s.lhs,'scratchpad',edge,operand_rows),(s.rhs,'scratchpad',edge,operand_rows),
           (s.accumulator,'accumulator',edge*4,accumulator_rows))
    if any(view is None or (view.backing,view.unit_bytes)!=(backing,width)
           or view.base+view.capacity>rows for view,backing,width,rows in views):
        raise CompletionRefusal('unknown physical ownership or row mapping in assigned storage','ownership_refusal')
    if r.operand_bytes>operand_rows*edge or r.accumulator_bytes>accumulator_rows*edge*4:
        raise CompletionRefusal('declared resources exceed selected target capacities','ownership_refusal')
    if not s.admits(p,b,r):
        raise CompletionRefusal('original live operand/accumulator assignments overlap or exceed resources','ownership_refusal')
    if type(owner) is DenseSegmentedPipelineResult:
        verifier=verify_dense_segmented_pipeline
    elif type(owner.admission) is DenseRetainedReductionAdmission:
        verifier=verify_dense_retained_reduction
    else:
        verifier=verify_dense_panel_pipeline
    # Reprove original source, physical windows and exact epochs against THIS
    # caller context. Context declarations cannot borrow an unrelated receipt.
    from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
    from merlin.compile.scheduling.dense_completion import _check_epochs
    _check_epochs(owner.kernel)
    try:
        return verifier(owner.kernel,p,b,r,s,a,owner.admission,static_budget=128,timeout_ms=15000,
                        output_readout=owner.output_readout,bias_source=owner.bias_source)
    except DenseVerificationUnavailable as exc:
        raise CompletionRefusal('original source proof unavailable: '+str(exc),'proof_unavailable') from exc
    except ValueError as exc:
        raise CompletionRefusal('original route validation refused: '+str(exc),'original_validation_refusal') from exc


@dataclass(frozen=True)
class CompletionFallbackResult:
    original: object
    realization: dict

    @property
    def kernel(self):return self.original.kernel

    def __getattr__(self,name):return getattr(self.original,name)


def _fallback(original,reason,category):
    report=dict(original.realization)
    report['completion_placement']=dict(status='unchanged',category=category,reason=reason,
        source_kernel_sha256=original.kernel.digest(),actual_kernel_sha256=original.kernel.digest(),
        removed_dynamic_drains=0)
    return (replace(original,realization=report) if is_dataclass(original)
            else CompletionFallbackResult(original,report))


def apply_completion(original,context):
    """Return the actual production result, or a precisely explained original.

    context is a dictionary with problem, block, resources, storage, adapter,
    admission and a plain premises dictionary. Missing/false laws and inadequate
    issued budgets never receive an optimized result. Source proof failure is
    distinguished from a malformed original; neither fabricates correctness.
    """
    try:
        if isinstance(context,Mapping):
            context=dict(context)
        else:
            supplied=context
            context={key:getattr(supplied,key) for key in
                     ('problem','block','resources','storage','adapter','admission','premises')}
            for key in ('completion_plan','issued_budget_proof'):
                if hasattr(supplied,key):context[key]=getattr(supplied,key)
        if type(context.get('premises')) is not dict:
            raise CompletionRefusal('completion context requires a plain premises dictionary')
        context['premises']=dict(context['premises'])
        source_report=_validate_original(original,context)
        context={**context,'original_validation':dict(
            owner=type(_owner(original)).__name__,kernel_sha256=original.kernel.digest(),
            schema=source_report.get('schema'),source=source_report.get('source'))}
        if 'completion_plan' not in context:
            context={**context,'completion_plan':make_completion_plan(original,context)}
        return coalesce_dense_completion(original,context)
    except CompletionRefusal as exc:
        return _fallback(original,str(exc),exc.category)
    except (ValueError,KeyError,AttributeError,TypeError) as exc:
        return _fallback(original,type(exc).__name__+': '+str(exc),'context_or_original_validation_refusal')


def selected_premises(original,adapter):
    root=Path(__file__).resolve().parent
    document=json.loads((root/'COMPLETION_PREMISES.json').read_text())
    facts=(root/'rtl_facts/facts.json').read_bytes()
    ordering=(root/'COMPLETION_ORDERING.json').read_bytes()
    if (sha256(facts).hexdigest()!=document['inputs']['facts']['sha256']
            or sha256(ordering).hexdigest()!=document['inputs']['ordering']['sha256']):
        raise CompletionRefusal('shared target ordering inputs changed; reselect premises')
    flags={key:row['value'] is True and row['status']=='fact'
           for key,row in document['premises'].items()}
    # Program proofs: original source/epoch verification checks private storage;
    # production emits each retained drain with the same target Builder.fence.
    # No callback, ownership or completion claim is accepted from schedule text.
    flags['exclusive_private_state']=True
    flags['complete_device_drain']=adapter.synchronous_completion is True
    return flags


def make_completion_context(original,problem,block,resources,storage,adapter):
    """Source-scoped issued proof; no pending queue field is ever increased."""
    owner=_owner(original)
    proposal,groups=propose_completion(original.kernel,adapter,problem)
    required,epochs=issued_requirements(proposal,adapter)
    def widen(admission):
        if hasattr(admission,'overlap') and hasattr(admission,'reduction'):
            return replace(admission,overlap=widen(admission.overlap),reduction=widen(admission.reduction))
        return replace(admission,**{key:max(getattr(admission,key),value) for key,value in required.items()})
    admission=widen(owner.admission)
    context=dict(problem=problem,block=block,resources=resources,storage=storage,adapter=adapter,
        admission=admission,premises=selected_premises(original,adapter),
        issued_budget_proof=dict(source_kernel_sha256=original.kernel.digest(),
            actual_kernel_sha256=proposal.digest(),requirements=required,epochs=epochs,
            original_bounds=_bounds(owner.admission),admitted_bounds=_bounds(admission),
            basis='exact finite loop multiplicities of unchanged source data calls within each retained drain'))
    context['completion_plan']=make_completion_plan(original,context)
    return context


def finish_completion(original,problem,block,resources,storage,adapter):
    try:
        context=make_completion_context(original,problem,block,resources,storage,adapter)
    except CompletionRefusal as exc:
        return _fallback(original,str(exc),exc.category)
    except (ValueError,KeyError,AttributeError,TypeError) as exc:
        return _fallback(original,type(exc).__name__+': '+str(exc),'context_construction_refusal')
    return apply_completion(original,context)
