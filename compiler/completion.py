"""Selected Gemmini effects for the reusable counted completion consumer.

Hardware facts, program obligations, and the study's protected boundaries are
kept separate. The unchanged panel checker still checks the original products;
completion_transform proves only the newly merged ordering/ownership scopes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN, DensePanelPipelineResult
from merlin.compile.scheduling.llvm_program import emit_index_effect_program
from merlin.sched.ir import Call, check_structure
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, PropertyPredicate,
    RegisteredPass, execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

from completion_transform import Access, Effect, OrderingLaw, CompletionRefused, coalesce_k_completions


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _source_pins():
    root = Path(__file__).resolve().parent
    return tuple((str(p.relative_to(root)), _sha(p)) for p in (
        root / "completion.py", root / "completion_transform.py", root / "dense.py",
        root / "target.py", root / "backend.py", root / "compiler.py",
        root / "logical_contract.yaml", root / "segmented_k.py", root / "ordering/COMPLETION_PREMISES.json",
        root / "ordering/COMMAND_ORDERING_SELECTION.json", root / "rtl_facts/facts.json",
    ))


def selected_law(storage):
    """Bind the public premises to the exact selected facts, preserving caveats.

    Complete-device-drain and exclusive-private-state are program obligations,
    never promoted from the hardware premise table's null fields. Here they are
    discharged conditionally on the same Builder.fence callback and the closed,
    nonescaping dense command consumer checked below. Emission binds the callback
    again; the caller keeps invocation and output-block completion.
    """
    root = Path(__file__).resolve().parent
    try:
        premise_path = root / "ordering/COMPLETION_PREMISES.json"
        ordering_path = root / "ordering/COMMAND_ORDERING_SELECTION.json"
        facts_path = root / "rtl_facts/facts.json"
        premises = json.loads(premise_path.read_text())
        ordering_selection = json.loads(ordering_path.read_text())
        facts = json.loads(facts_path.read_text())["facts"]
        if (premises["schema"] != "merlin.public_completion_premises.v1"
                or premises["inputs"]["facts"]["sha256"] != _sha(facts_path)
                or premises["inputs"]["ordering"]["sha256"] != _sha(ordering_path)
                or premises["ordering_identity"] != ordering_selection["ordering_sha256"]):
            raise CompletionRefused("selected ordering/facts do not match public completion premise pins")
        ordering = ordering_selection["ordering"]
        if ordering["status"] != "derived" or ordering["completion_order"]["execute"]["order"] != "in_order":
            raise CompletionRefused("missing ordered execute completion semantics")
        flags = {}
        for name in ("local_read_after_write", "local_write_after_read", "ordered_accumulator_writes",
                     "ordered_configuration_capture", "finite_queue_progress"):
            row = premises["premises"][name]
            flags[name] = row["value"] if row["status"] == "fact" else None
        # Explicitly preserve tracked=None. These pairs have selected ordering
        # semantics but no tracking certification. No load-queue WAW is assumed.
        dependencies = []
        for load in ("MVIN", "MVIN2"):
            for reader in ("COMPUTE_PRELOADED.a_spad", "PRELOAD.weight_spad"):
                for older, younger in ((load+".spad_addr", reader), (reader, load+".spad_addr")):
                    edge = next((r for r in ordering["cross_queue"] if
                        (r["older"], r["younger"], r["space"]) == (older, younger, "scratchpad")), None)
                    if edge is None or edge["overlap"] != "completion" or edge["tracked"] is False:
                        raise CompletionRefused("a consumed local dependency has unknown/conflicting ordering")
                    dependencies.append(dict(edge))
        classes = facts["timing_structure"]["command_classes"]
        if classes["status"] != "derived":
            raise CompletionRefused("finite dispatch capacities are unknown")
        pending = {}
        for role in ("load", "execute"):
            entries = classes["classes"][role]["entries"]
            if entries["status"] != "derived" or type(entries["value"]) is not int or entries["value"] < 1:
                raise CompletionRefused("selected dispatch entry count is unknown")
            pending[role] = entries["value"]
        # Three assigned, nonescaping local views; check canonical backings.
        # They are the original allocation, not a newly enlarged resource bound.
        views = (storage.lhs, storage.rhs, storage.accumulator)
        memories = {m["name"]: m["bytes"] for m in facts["memories"]}
        for view in views:
            if (view.backing not in memories or view.base < 0 or view.capacity <= 0
                    or (view.base + view.capacity) * view.unit_bytes > memories[view.backing]):
                raise CompletionRefused("original local view exceeds selected memory capacity")
        for i, a in enumerate(views):
            for b in views[i+1:]:
                if (a.backing == b.backing and a.base*a.unit_bytes < (b.base+b.capacity)*b.unit_bytes
                        and b.base*b.unit_bytes < (a.base+a.capacity)*a.unit_bytes):
                    raise CompletionRefused("original privately owned local views overlap")
        identity = sha256(json.dumps(dict(premises=_sha(premise_path),
            ordering=_sha(ordering_path), facts=_sha(facts_path)), sort_keys=True).encode()).hexdigest()
        law = OrderingLaw(identity, **flags, pending_load_commands=pending["load"],
            pending_execute_commands=pending["execute"], complete_device_drain=True,
            exclusive_private_state=True)
        law.require()
        return law, dict(
            premise_table_sha256=_sha(premise_path), ordering_sha256=_sha(ordering_path),
            facts_sha256=_sha(facts_path), selected_dependencies=dependencies,
            uncertain_tracking=[r for r in dependencies if r["tracked"] is None],
            coverage_caveats=premises["coverage_caveats"],
            accumulator_result_visibility=premises["premises"]["accumulator_result_visibility"],
            accumulator_visibility_used=False,
            program_obligations=dict(
                complete_device_drain="same target.Builder.fence as original; final K drain and every readout/block drain retained; callback bound during emission",
                exclusive_private_state="original disjoint assigned views; complete closed command projection; no callback/host observer inside the stream; verbatim source-pinned backend entry/exit drains; sequential invocation owns local state through its closing drains"),
            assigned_views=[asdict(v) for v in views],
            qualification="selected semantics; tracked=None stays uncertain; not RTL certification")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise CompletionRefused("missing or malformed selected target premises: " + str(error)) from error


def dense_effect_projector(adapter, storage):
    def access(view, row, rows, writes):
        return Access(view.backing, row, rows, view.base, view.base + view.capacity, writes)

    def project(call):
        if call == Call(DRAIN, ()):
            return Effect("drain")
        try:
            op = adapter.project(call)
        except (ValueError, KeyError):
            raise CompletionRefused("unknown dense target effect")
        if type(op) is Load:
            view = storage.lhs if op.role == "lhs" else storage.rhs if op.role == "weight" else None
            if view is None:
                raise CompletionRefused("unknown operand storage role")
            return Effect("load", (access(view, op.row, op.rows, True),))
        if type(op) is Preload:
            return Effect("preload", (access(storage.rhs, op.weight_row, op.weight_rows, False),))
        if type(op) is Compute:
            # The adjacent actual preload carries the accumulator destination.
            # This is supplied without a side table by the checked pair walker
            # below, after decoding the actual source Call.
            return Effect("compute", (access(storage.lhs, op.input_row, op.rows, False),))
        if type(op) is Store:
            return Effect("readout", (), (access(storage.accumulator, op.accumulator_row, op.rows, False),))
        raise CompletionRefused("unmapped dense configuration, bias or observer")
    return project, access


def checked_dense_effects(kernel, adapter, storage):
    """Decode every actual field; bind immediate preload/compute pairs in IR."""
    from merlin.sched.ir import Loop
    project, access = dense_effect_projector(adapter, storage)
    by_call = {}

    def visit(body):
        pending = None
        for call in body:
            if type(call) is Loop:
                if pending is not None:
                    raise CompletionRefused("preload/compute pair crosses a counted scope")
                visit(call.body)
                continue
            effect = project(call)
            if effect.kind == "preload":
                if pending is not None:
                    raise CompletionRefused("overwritten preload descriptor")
                pending = adapter.project(call)
            elif effect.kind == "compute":
                if pending is None:
                    raise CompletionRefused("compute lacks its adjacent actual preload")
                effect = Effect(effect.kind, effect.local, (
                    access(storage.accumulator, pending.accumulator_row, pending.rows, True),))
                pending = None
            elif pending is not None:
                raise CompletionRefused("interposed effect splits preload/compute pair")
            # The same immutable Call may recur only with the same effects.
            if call in by_call and by_call[call] != effect:
                raise CompletionRefused("identical call has ambiguous accumulator effects")
            by_call[call] = effect
        if pending is not None:
            raise CompletionRefused("unconsumed preload descriptor")
    visit(kernel.body)
    return by_call.__getitem__


@dataclass(frozen=True)
class CompletionResult:
    original: object
    kernel: object
    realization: dict
    source_identity: str
    source_pins: tuple
    report_sha256: str

    def revalidate(self):
        self.original.revalidate()
        if (self.kernel.digest() != self.realization["kernel_sha256"]
                or self.source_pins != _source_pins()
                or sha256(json.dumps(self.realization, sort_keys=True).encode()).hexdigest() != self.report_sha256):
            raise ValueError("completion source, checked emitted IR, target premises or proof report changed")

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from target import Builder
        self.revalidate()
        if (type(getattr(complete_epoch, "__self__", None)) is not Builder
                or getattr(complete_epoch, "__func__", None) is not Builder.fence
                or getattr(control, "builder", None) is not complete_epoch.__self__):
            raise ValueError("ordered K completion requires the original real Builder.fence callback")
        receipt = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return receipt


def apply_ordered_k_completion(original, problem, storage, adapter, *, mode):
    """Registered counted-IR A/B paths with the original unconditional fallback."""
    if mode not in ("per_k", "ordered_stream", "ordered_initial"):
        raise ValueError("unknown dense K completion policy")
    from segmented_k import RetainedLhsResult
    if type(original) is DensePanelPipelineResult:
        checked = original
    elif type(original) is RetainedLhsResult:
        # Consume its checked ACTUAL retained stream, never its panel precursor.
        # The wrapper's own revalidation still owns retained storage/source
        # provenance. All retained addresses and its segmentation stay fixed.
        original.revalidate()
        checked = original.checked
        if type(checked) is not DensePanelPipelineResult or checked.kernel != original.kernel:
            return original
    else:
        # Cross-invocation resident lifetimes need their own emission consumer.
        return original
    original.revalidate()
    premises = None
    reason = None
    try:
        law, premises = selected_law(storage)
        supported = True
    except CompletionRefused as error:
        law, supported, reason = None, None, str(error)

    def verify(kernel):
        errors = check_structure(kernel)
        if errors:
            raise ValueError("invalid counted completion schedule: " + str(errors))

    context = PipelineContext(hardware=adapter.configuration_identity,
        numerical_policy=adapter.numerical_policy_identity,
        resources=checked.checked_kernel_sha256,
        execution=None if law is None else law.identity)
    session = PipelineSession(original.kernel, ir_level="counted_dense", context=context,
        fingerprint=lambda k: k.digest(), verify_ir=verify)
    facts = PolicyFacts(session.stamp(), shapes=((problem.m,problem.k),(problem.k,problem.n)),
        operations=frozenset(("dense",)), capabilities=(("ordered_k_completion", supported),
        ("completion_enabled", mode != "per_k")))
    chosen = {"result": original}

    def source(s):
        original.revalidate()
        return {"completion_source": original.kernel.digest()}

    def transform(s):
        if s.analysis("completion_source") != original.kernel.digest():
            raise ValueError("completion source analysis is stale")
        try:
            project = checked_dense_effects(s.module, adapter, storage)
            kernel, report = coalesce_k_completions(s.module, project, law, drain=DRAIN,
                ordered_initial=mode == "ordered_initial")
            realization = dict(schema="fresh6.dense_ordered_k_result.v1", kernel_sha256=kernel.digest(),
                source=checked.realization["source"],
                dynamic_epoch_drains=report["actual_dynamic_drains"], completion_transform=report,
                selected_premises=premises, original_schedule=original.realization)
            result = CompletionResult(original, kernel, realization, checked.source_identity,
                _source_pins(), sha256(json.dumps(realization, sort_keys=True).encode()).hexdigest())
            result.revalidate()
            s.module = kernel
            chosen["result"] = result
            return {"completion_checked": dict(status="APPLIED", report=report)}
        except CompletionRefused as error:
            return {"completion_checked": dict(status="UNCHANGED_FALLBACK", reason=str(error))}

    def fallback(s):
        s.analysis("completion_source")
        return {"completion_checked": dict(status="UNCHANGED_FALLBACK", reason=reason or "per_k selected")}

    def result(s):
        s.analysis("completion_checked")
        chosen["result"].revalidate()
        return {"completion_result": chosen["result"]}

    specs = (
        PassSpec("completion_source",0,produces=("completion_source",),mutates_ir=False,ir_level="counted_dense"),
        PassSpec("ordered_k_completion",1,requires=("completion_source",),produces=("completion_checked",),ir_level="counted_dense"),
        PassSpec("per_k_completion",1,requires=("completion_source",),produces=("completion_checked",),mutates_ir=False,ir_level="counted_dense"),
        PassSpec("completion_result",2,requires=("completion_checked",),produces=("completion_result",),mutates_ir=False,ir_level="counted_dense"),
    )
    registry = {spec.name: RegisteredPass(spec, action) for spec, action in zip(specs,
        (source, transform, fallback, result))}
    policy = PipelinePolicy((
        PipelineVariant("ordered_stream", ("completion_source","ordered_k_completion","completion_result"),
            (PropertyPredicate("capability","completion_enabled"),
             PropertyPredicate("capability","ordered_k_completion"))),
        PipelineVariant("per_k", ("completion_source","per_k_completion","completion_result")),
    ), fallback="per_k")
    return execute_policy(session,policy,registry,facts,output="completion_result").artifact
