"""Production binding for completion-only scheduling, shared with safety probes."""
from pathlib import Path
import json
from hashlib import sha256

from merlin.compile.scheduling.dense_k_completion import (
    complete_dense_k_stream, CompletionNotApplicable,
)


def selected_premises():
    """Shared target facts plus invocation ownership/completion program proof.

    lower_matmul emits a fixed configuration before all schedule data, and its
    sole complete_epoch callback is Builder.fence, the original full drain.
    The backend serializes accelerator groups with entry/final full drains;
    these scratchpad/ACC views belong exclusively to this invocation. The
    completion pass validates actual assigned windows and every emitted effect.
    No concurrent user of accelerator state is introduced by this lowering.
    """
    root = Path(__file__).parent
    try:
        record = json.loads((root/'COMPLETION_PREMISES.json').read_text())
        for key in ('facts', 'ordering'):
            pin = record['inputs'][key]
            path = root / pin['path']
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != pin['sha256']:
                return {'target_identity': {'value':None, 'reason':'selected '+key+' identity changed'}}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {'target_identity': {'value':None, 'reason':'selected target premises unavailable: '+str(exc)}}
    premises = {name: dict(value) for name,value in record['premises'].items()}
    premises['complete_device_drain'] = True
    premises['exclusive_private_state'] = True
    return premises


def apply_completion(original, context):
    """Return the actual emission result, preserving valid originals on refusal.

    The caller supplies all proof context; no missing law or budget is filled
    here. RetainedLhsResult is an existing backend wrapper around a checked
    panel result, not a two-window construction. Keep its source owner alive
    and let the generic consumer inspect its actual retained schedule family.
    """
    from segmented_k import RetainedLhsResult
    if type(context) is not dict:
        fields = ('problem','block','resources','storage','adapter','admission','premises')
        context = {key:getattr(context,key) for key in fields if hasattr(context,key)}
    checked = original.checked if type(original) is RetainedLhsResult else original
    try:
        original.revalidate()
        result = complete_dense_k_stream(checked, context)
        if type(original) is RetainedLhsResult:
            # Preserve both source owners: result's checked panel owner and the
            # backend retention construction which produced that actual kernel.
            return CompletionComposition(original, result)
        return result
    except (CompletionNotApplicable, ValueError) as exc:
        # A proof refusal is an unchanged-emission outcome, not numerical PASS
        # or a claimed optimized schedule. A malformed original still fails its
        # own revalidate/normal emission; this handler grants no new validity.
        report = dict(original.realization)
        report['completion'] = {'status':'unchanged', 'reason':str(exc),
                                'outcome':'unchanged_fallback', 'failure_kind':'proof_refusal'}
        return UnchangedCompletionResult(original, report)


class UnchangedCompletionResult:
    """Diagnostic wrapper which preserves the original owner and emission."""
    def __init__(self, original, report):
        self.original, self.kernel, self.realization = original, original.kernel, report
    @property
    def admission(self):
        return getattr(self.original, 'admission', getattr(getattr(self.original, 'checked', None), 'admission', None))
    def revalidate(self):
        self.original.revalidate()
        if self.kernel != self.original.kernel:
            raise ValueError('unchanged fallback kernel changed')
    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        self.revalidate()
        return self.original.emit_index_program(control, lower_effect, complete_epoch=complete_epoch)
    def materialize_scf(self):
        self.revalidate()
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands
        return materialize_scf_commands(self.kernel)


class CompletionComposition:
    def __init__(self, original, checked):
        self.original, self.checked = original, checked
        self.kernel, self.realization = checked.kernel, checked.realization
    @property
    def admission(self):
        return self.checked.admission
    @property
    def source_identity(self):
        return self.checked.source_identity
    @property
    def context(self):
        return self.checked.context
    def revalidate(self):
        self.original.revalidate()
        self.checked.revalidate()
        if self.kernel != self.checked.kernel or self.realization != self.checked.realization:
            raise ValueError('retained/completion composition changed')
    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        self.revalidate()
        receipt = self.checked.emit_index_program(control, lower_effect, complete_epoch=complete_epoch)
        self.revalidate()
        return receipt
    def materialize_scf(self):
        self.revalidate()
        return self.checked.materialize_scf()
