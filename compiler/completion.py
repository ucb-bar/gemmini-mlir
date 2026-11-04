"""Selected target adapter for the reusable dense completion-placement pass."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Block, Region

from merlin.compile.scheduling.dense_completion import (
    COMPLETION_PASS, CompletionNotApplicable, DenseCompletionLaw,
    DenseCompletionOwnership, coalesce_dense_k_completions,
)
from merlin.compile.scheduling.schedule_ir import dense_schedule_text
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, PropertyPredicate,
    RegisteredPass, execute_policy,
)
from merlin.transforms.pipeline_state import PipelineContext, PipelineSession, PassSpec
from merlin.sched.ir import check_structure
from merlin.xdsl_dialects.schedule import PlanOp, BindOp, ApplyPolicyOp, HandleType

MODES = ("per_k", "inner_stream", "ordered_stream")


def selected_law(root, adapter):
    """Read the supplied target table, retaining its qualification caveats."""
    premise = json.loads((root / "COMPLETION_PREMISES.json").read_text())
    ordering_bytes = (root / "COMMAND_ORDERING_SELECTION.json").read_bytes()
    facts_bytes = (root / "rtl_facts/facts.json").read_bytes()
    ordering = json.loads(ordering_bytes)
    if (premise.get("schema") != "merlin.public_completion_premises.v1"
            or sha256(ordering_bytes).hexdigest() != premise["inputs"]["ordering"]["sha256"]
            or sha256(facts_bytes).hexdigest() != premise["inputs"]["facts"]["sha256"]
            or ordering["ordering_sha256"] != premise["ordering_identity"]):
        raise CompletionNotApplicable("missing or mismatched selected target ordering")
    facts = json.loads(facts_bytes)["facts"]
    classes = facts["timing_structure"]["command_classes"]["classes"]
    queues = {k: classes[k]["command_queue"] for k in ("load", "execute")}
    if any(q["status"] != "derived" for q in queues.values()):
        raise CompletionNotApplicable("unknown physical pending queues")

    def flag(name):
        row = premise["premises"].get(name, {})
        return row.get("value") if row.get("status") == "fact" else None

    law = DenseCompletionLaw(adapter.configuration_identity,
        premise["ordering_identity"], adapter.numerical_policy_identity,
        *(flag(name) for name in ("local_read_after_write", "local_write_after_read",
            "ordered_accumulator_writes", "ordered_configuration_capture", "finite_queue_progress")),
        queues["load"]["entries_total"],
        min(c["entries"] for c in queues["execute"]["chain"]) // 2,
        next(a["rows"] for a in facts["arrays"] if a["name"] == "mesh"),
        tuple(premise["coverage_caveats"]))
    law.require(adapter)
    return law


def make_completion_plan(kernel, ordering_identity, mode):
    if mode not in MODES:
        raise ValueError("unsupported K completion setting")
    bind = BindOp(result_types=[HandleType()], properties={
        "target": StringAttr("dense-completion:" + kernel.digest() + ":" + ordering_identity)})
    choice = ApplyPolicyOp(operands=[[bind.handle]], properties={"policy": StringAttr(mode)})
    return ModuleOp([PlanOp(properties={"sym_name": StringAttr("dense_k_completion")},
        regions=[Region(Block([bind, choice]))])])


def consume_completion_plan(plan, kernel, ordering_identity):
    plan.verify()
    top = tuple(plan.body.block.ops)
    if len(top) != 1 or type(top[0]) is not PlanOp:
        raise ValueError("completion plan needs exactly one schedule.plan")
    ops = tuple(top[0].body.block.ops)
    if (len(ops) != 2 or type(ops[0]) is not BindOp or type(ops[1]) is not ApplyPolicyOp
            or ops[0].target.data != "dense-completion:" + kernel.digest() + ":" + ordering_identity
            or tuple(ops[1].handles) != (ops[0].handle,)
            or ops[1].policy.data not in MODES):
        raise ValueError("completion schedule is stale or contains unconsumed decisions")
    return ops[1].policy.data


def select_completion(original, storage, adapter, *, root, mode,
                      exclusive_storage, full_drain_implementation, serialized_invocation,
                      schedule_plan=None):
    """Consume a bound dialect decision and select a registered checked route.

    Unknown target/program facts and proof-budget refusals select the unchanged
    original.  The returned kernel is the one passed to the ordinary LLVM
    effect emitter; there is no alternate payload implementation.
    """
    original.revalidate()
    reason = None
    law = None
    candidate = None
    ownership = DenseCompletionOwnership(original.kernel.digest(), exclusive_storage,
        full_drain_implementation, serialized_invocation)
    try:
        law = selected_law(root, adapter)
    except (OSError, KeyError, ValueError) as error:
        reason = str(error)
    identity = law.ordering_identity if law is not None else "unknown"
    plan = schedule_plan if schedule_plan is not None else make_completion_plan(original.kernel, identity, mode)
    mode = consume_completion_plan(plan, original.kernel, identity)
    if mode != "per_k" and law is not None:
        try:
            candidate = coalesce_dense_k_completions(original, storage, adapter, law, ownership, mode=mode)
            candidate = replace(candidate, selected_sources=tuple(
                (str(root / name), sha256((root / name).read_bytes()).hexdigest())
                for name in ("COMPLETION_PREMISES.json", "COMMAND_ORDERING_SELECTION.json",
                    "rtl_facts/facts.json", "completion.py", "dense.py", "target.py")))
        except CompletionNotApplicable as error:
            reason = str(error)

    context = PipelineContext(hardware=adapter.configuration_identity,
        numerical_policy=adapter.numerical_policy_identity,
        resources=original.kernel.digest(), execution=identity)
    session = PipelineSession(original.kernel, context=context, ir_level="dense_kernel",
        fingerprint=lambda k: k.digest(),
        verify_ir=lambda k: (_ for _ in ()).throw(ValueError("invalid dense Kernel"))
            if check_structure(k) else None)
    facts = PolicyFacts(session.stamp(),
        capabilities=(("ordered_k_completion", candidate is not None),))
    inputs = PassSpec("completion_inputs", 50, mutates_ir=False,
        produces=COMPLETION_PASS.requires, ir_level="dense_kernel")
    keep = PassSpec("keep_k_completion", 60, requires=COMPLETION_PASS.requires,
        produces=COMPLETION_PASS.produces, mutates_ir=False, ir_level="dense_kernel")
    emit = PassSpec("bind_completion_emission", 70, requires=COMPLETION_PASS.produces,
        produces=("completion_result",), mutates_ir=False, ir_level="dense_kernel")

    def transform(s):
        if candidate is None or s.module != original.kernel:
            raise ValueError("completion selection lost checked source custody")
        candidate.revalidate()
        s.module = candidate.kernel
        return {"checked_completion_placement": candidate}

    registry = {
        inputs.name: RegisteredPass(inputs, lambda s: dict(zip(inputs.produces,
            (original, law, ownership), strict=True))),
        COMPLETION_PASS.name: RegisteredPass(COMPLETION_PASS, transform),
        keep.name: RegisteredPass(keep, lambda s: {"checked_completion_placement": original}),
        emit.name: RegisteredPass(emit, lambda s: {"completion_result": s.analysis("checked_completion_placement")}),
    }
    policy = PipelinePolicy((
        PipelineVariant("ordered_stream", (inputs.name, COMPLETION_PASS.name, emit.name),
            (PropertyPredicate("capability", "ordered_k_completion"),)),
        PipelineVariant("per_k", (inputs.name, keep.name, emit.name)),), "per_k")
    result = execute_policy(session, policy, registry, facts, output="completion_result")
    receipt = dict(result.receipt, requested_mode=mode, refusal=reason,
        schedule_ir=dense_schedule_text(plan), selected_law=None if law is None else asdict(law),
        ownership=asdict(ownership), decisions_consumed=("schedule.bind", "schedule.apply_policy"))
    if result.artifact is not original:
        result.artifact.realization["completion_policy"] = receipt
    return result.artifact, receipt
