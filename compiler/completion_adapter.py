"""Production source-bound entry for completion placement, with explicit unknowns."""
from pathlib import Path
from hashlib import sha256
import json

from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw, STREAM_FLAGS
from merlin.compile.scheduling.dense_completion import (
    checked_source, effective_admission, schedule_dense_completion, normalize_completion_context,
)


def completion_law(context):
    """Only literal True supplies a premise; missing, None and false remain unknown."""
    a = context['adapter']
    premises = context.get('premises',{})
    if type(premises) is not dict:
        raise ValueError('completion premises must be a plain dictionary')
    return DensePanelEpochLaw(a.configuration_identity,a.numerical_policy_identity,
        context.get('ordering_identity','explicit-caller-ordering-premises'),
        **{name:premises.get(name) is True for name in (*STREAM_FLAGS,'accumulator_result_visibility')})


def apply_completion(original, context):
    """Return the actual checked result consumed by the ordinary target emitter.

    This entry does not enlarge input budgets. The production caller constructs
    its separately checked issued scope before calling this same public entry.
    Invalid ownership/storage/source/completion inputs refuse with ValueError;
    unknown ordering laws and insufficient issued budgets keep boundaries.
    """
    context = normalize_completion_context(original,context)
    law = completion_law(context)
    from merlin.compile.scheduling.dense_completion_ir import (
        make_dense_completion_plan,consume_dense_completion_plan,
    )
    plan = context.get('completion_plan')
    if plan is None:
        plan = make_dense_completion_plan(original,context)
    strategy,text = consume_dense_completion_plan(plan,original,context)
    if strategy == 'drained':
        original.revalidate()
        return original
    result = schedule_dense_completion(original,context,law)
    if result is not original:
        result.realization['completion_schedule_ir'] = {
            'text':text,'decisions_consumed':['completion_epochs.strategy','scope','preserve_readout'],
            'strategy':strategy,'scope':'output_block','preserve_readout':True}
    return result


def production_completion(original,problem,block,resources,storage,adapter,*,enabled=True):
    root = Path(__file__).resolve().parent
    selected = json.loads((root/'completion_premises.json').read_text())
    facts = root/'rtl_facts/facts.json'
    if sha256(facts.read_bytes()).hexdigest() != selected['inputs']['facts']['sha256']:
        raise ValueError('completion target premises select different RTL facts')
    premises = {name:row.get('value') is True for name,row in selected['premises'].items()
                if row.get('status') == 'fact'}
    # Program obligations: the exact same b.fence callback closes both paths;
    # target.py emits side-effecting full fence with a memory clobber. Backend
    # invocations issue serially and privately own their validated operand/ACC
    # allocation. The consumer rechecks the true route allocation and source.
    premises.update(complete_device_drain=True,exclusive_private_state=True)
    source = checked_source(original)
    context = dict(problem=problem,block=block,resources=resources,storage=storage,
        adapter=adapter,admission=effective_admission(source.admission),premises=premises,
        ordering_identity=selected['ordering_identity'])
    law = completion_law(context)
    try:
        from merlin.compile.scheduling.dense_completion_pipeline import run_dense_completion_pipeline
        return run_dense_completion_pipeline(original,context,law,apply_completion,enabled=enabled)
    except ValueError as refusal:
        # This original result was constructed and checked before composition.
        # A new consumer's unsupported proof cannot change its data program.
        original.revalidate()
        original.realization['completion_refusal'] = str(refusal)
        return original
