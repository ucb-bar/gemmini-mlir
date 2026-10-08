"""Exact generated-component execution features for host-selected feedback.

Merlin supplies the ordinary sandboxed compile/execute/readback service. This
adapter owns Gemmini instruction/operand decoding and refuses missing execution
or stage correspondence. No retained workload or historical calibration is read.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from merlin.benchharness import hash_tree
from merlin.common.jsonio import canonical_sha256
from merlin.perf.component_cost import (
    COMPLETE_STAGES,
    ComponentCostRegion,
    ComponentCostScope,
    ComponentFeatureObservation,
)
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS

from .executed_features import census
from .no_fsm_audit import audit_elf
from .operand_features import read_telemetry, summarize

_STAGE_ISSUER = object()


def _sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pin(path, digest):
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_absolute()
        or not path.is_file()
        or path.resolve() != path
        or _sha(path) != digest
    ):
        raise ValueError("component execution artifact or dependency changed")
    return path


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str

    def verify(self):
        return _pin(self.path, self.sha256)


@dataclass(frozen=True)
class ExecutedComponent:
    """One actual ordinary executor result, with explicit execution regime.

    ``regions`` are supplied only when the existing execution owner has verified
    source/ELF/event scope closure. The adapter never invents phases from function
    names or duplicates a cold histogram to manufacture warm evidence.
    """

    regime: str
    compiler_sha256: str
    member_sha256: str
    target_sha256: str
    scope_sha256: str
    domain_sha256: str
    elf: Artifact
    histogram: Artifact
    console: Artifact
    execution_receipt: Artifact
    dependencies: tuple[Artifact, ...]
    inputs: tuple[Artifact, ...]
    regions: tuple[ComponentCostRegion, ...] = ()
    operand_telemetry: Artifact | None = None
    operand_producer_qualification: Artifact | None = None


@dataclass(frozen=True)
class ComponentExecutionService:
    """An explicit trusted normal executor and its machine/runtime dependencies.

    The callback owns ordinary package sandboxing, source/build/invocation joins,
    complete readback, and exact original-golden/effect qualification. This service
    does not establish those properties merely by hashing its files.
    """

    execute: Callable
    implementation: Artifact
    engine: Artifact
    machine_configuration: Artifact
    runtime_dependencies: tuple[Artifact, ...]
    engine_kind: str
    configuration_json: str
    _code: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        callback = self.execute.func if type(self.execute) is partial else self.execute
        object.__setattr__(self, "_code", getattr(callback, "__code__", None))

    def verify(self):
        if self.engine_kind != "spike":
            raise ValueError(
                "component PC features require an explicitly selected Spike functional engine"
            )
        configuration = json.loads(self.configuration_json)
        if not isinstance(configuration, dict) or not configuration:
            raise ValueError(
                "component execution needs explicit machine/ISA/runtime configuration"
            )
        prefix = configuration.get("spike_argv_prefix")
        if (
            not isinstance(prefix, list)
            or not all(type(token) is str for token in prefix)
            or not prefix
            or prefix[0] != str(self.engine.path)
            or "-g" not in prefix
            or not isinstance(configuration.get("isa"), str)
            or not configuration["isa"]
            or "--isa=" + configuration["isa"] not in prefix
            or "--extension=gemmini" not in prefix
        ):
            raise ValueError(
                "component PC features require explicit Spike argv, ISA, extension and histogram mode"
            )
        self.implementation.verify()
        self.engine.verify()
        self.machine_configuration.verify()
        if json.loads(self.machine_configuration.path.read_text()) != configuration:
            raise ValueError(
                "component execution configuration differs from its exact machine/runtime artifact"
            )
        if not self.runtime_dependencies:
            raise ValueError(
                "component execution runtime dependency closure is missing"
            )
        for artifact in self.runtime_dependencies:
            artifact.verify()
        callback = self.execute
        if type(callback) is partial:
            from merlin_experiments.phase2.component_execution import (
                ComponentNormalExecutionBinding,
            )

            binding = callback.keywords.get("binding")
            if (
                callback.func is not _normal_development_execute
                or callback.args
                or set(callback.keywords) != {"binding"}
                or type(binding) is not ComponentNormalExecutionBinding
                or binding.verify() != configuration.get("normal_execution_sha256")
            ):
                raise ValueError(
                    "component executor lacks its closed normal host execution binding"
                )
            pinned_members = {artifact.path for artifact in self.runtime_dependencies}
            if {
                path.resolve() for path in self.implementation.path.parent.rglob("*.py")
            } - pinned_members:
                raise ValueError(
                    "selected component companion Python membership changed"
                )
            callback = callback.func
        source = inspect.getsourcefile(callback)
        if (
            source is None
            or Path(source).resolve() != self.implementation.path
            or self._code is None
            or getattr(callback, "__code__", None) is not self._code
        ):
            raise ValueError(
                "component executor differs from its pinned normal execution owner"
            )
        if configuration.get("executor_qualname") != callback.__qualname__:
            raise ValueError(
                "component machine/runtime artifact does not bind its selected executor"
            )
        return configuration


@dataclass(frozen=True)
class ComponentStageAssessment:
    """Actual selected evaluator outcome, with separately observed stage facets.

    Successful file joins supply an observed subset. Missing semantic lifts never
    become a complete witness; a future evaluator must produce actual typed
    products and evaluated effects before this conditional promotion can succeed.
    """

    inspection_path: Artifact
    stages: tuple
    unresolved: tuple[str, ...]
    output_roster: tuple[str, ...]
    effect_roster: tuple[str, ...]
    _issuer: object = field(repr=False, compare=False)

    def complete_witness(
        self,
        *,
        member,
        candidate_sha256,
        target_descriptor,
        frontend,
        capsule_root,
        evidence_root,
        required_effects,
        **_,
    ):
        from merlin_experiments.phase1.component_witness import (
            ComponentStageWitness,
            verify_component_stage_witness,
        )

        if self._issuer is not _STAGE_ISSUER:
            raise ValueError(
                "component stage assessment was not issued by the selected observed evaluator"
            )
        self.inspection_path.verify()
        selected = inspect_component_execution_records(
            member=member,
            candidate_sha256=candidate_sha256,
            target_descriptor=target_descriptor,
            frontend=frontend,
            capsule_root=capsule_root,
            evidence_root=evidence_root,
            required_effects=required_effects,
            **_,
        )
        self.inspection_path.verify()
        if tuple(selected["unresolved"]) != self.unresolved:
            raise ValueError(
                "component stage assessment differs from independently reopened actual observations"
            )
        if self.unresolved:
            raise NotImplementedError(
                "ordinary component stage/effect witness UNKNOWN: "
                + "; ".join(self.unresolved)
            )
        witness = ComponentStageWitness(
            member["program_sha256"],
            candidate_sha256,
            member.get("sha256", member.get("capsule_fingerprint")),
            _sha(target_descriptor),
            frontend,
            self.stages,
            self.output_roster,
            self.effect_roster,
            "original_independent_golden",
        )
        verify_component_stage_witness(
            witness,
            member=member,
            candidate_sha256=candidate_sha256,
            target_descriptor_sha256=_sha(target_descriptor),
            frontend=frontend,
            capsule_root=capsule_root,
            evidence_root=evidence_root,
            required_effects=required_effects,
        )
        return witness


def evaluate_component_execution_stages(**inputs):
    """Evaluate actual source/product joins independently of completed qualification."""
    from merlin_experiments.phase1.component_witness import ComponentStageFile

    inspection = inspect_component_execution_records(**inputs)
    generated = Path(inputs["result_path"]).parent / "generated"
    from merlin_experiments.phase2.contracts import mapping_file

    capsule = mapping_file(
        Path(inputs["capsule_root"]) / "capsule.yaml", yaml_file=True
    )
    source = (
        Path(inputs["capsule_root"])
        / capsule.get("interface_mlir", "capsule.interface.mlir")
    ).resolve()
    stages = [ComponentStageFile("source", source, _sha(source), ())]
    # A target conversion is retained as observed evidence, not relabeled as
    # preparation/partition/emitted host+device without an actual semantic lift.
    return ComponentStageAssessment(
        Artifact(
            (generated / "component_execution_inspection.json").resolve(),
            _sha(generated / "component_execution_inspection.json"),
        ),
        tuple(stages),
        tuple(inspection["unresolved"]),
        (),
        (),
        _STAGE_ISSUER,
    )


def verify_component_execution_witness(**inputs):
    return evaluate_component_execution_stages(**inputs).complete_witness(**inputs)


def inspect_component_execution_records(
    *,
    member,
    result_path,
    candidate_root,
    compiler_snapshot,
    candidate_sha256,
    capsule_root,
    evidence_root,
    target_descriptor,
    frontend,
    required_effects,
    timeout_s,
):
    """Reopen actual ordinary artifacts without granting missing semantic facets.

    This is a private diagnostic. Successful invocations establish observed file
    joins; they do not prove that partition decisions, alias/lifetime/epoch
    obligations or full readback were evaluated. A complete typed stage witness
    is unavailable until those target services provide their actual authorities.
    """
    from merlin.common import invocation_record
    from merlin_experiments.phase2 import contracts as contract

    deadline = time.monotonic() + min(float(timeout_s), ITERATION_MAX_SECONDS)
    candidate_root, compiler_snapshot = Path(candidate_root), Path(compiler_snapshot)
    capsule_root, evidence_root, result_path = (
        Path(capsule_root),
        Path(evidence_root),
        Path(result_path),
    )
    if frontend not in {"mlir", "pytorch"}:
        raise ValueError("component execution has an unknown source frontend")
    if (
        contract.exact_tree_record(candidate_root)["sha256"] != candidate_sha256
        or contract.exact_tree_record(compiler_snapshot)["sha256"] != candidate_sha256
    ):
        raise ValueError(
            "component execution does not use the admitted exact compiler bytes"
        )
    if (
        result_path.resolve() != result_path
        or not result_path.is_relative_to(evidence_root.resolve())
        or result_path.is_symlink()
    ):
        raise ValueError("component result escapes its private grade evidence owner")
    result = contract.mapping_file(result_path)
    if result.get("capsule") != member["name"]:
        raise ValueError("component result belongs to a different independent member")
    capsule = contract.mapping_file(capsule_root / "capsule.yaml", yaml_file=True)
    source = (
        capsule_root / capsule.get("interface_mlir", "capsule.interface.mlir")
    ).resolve()
    if (
        not source.is_relative_to(capsule_root.resolve())
        or _sha(source) != member["program_sha256"]
    ):
        raise ValueError(
            "component source differs from the independent admitted program"
        )
    generated = result_path.parent / "generated"
    if generated.is_symlink() or not generated.is_dir():
        raise ValueError("ordinary component generated artifacts are absent")
    records, unresolved = [], []
    for path in sorted(generated.rglob("invocation.json")):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "component artifact inspection exceeded its development deadline"
            )
        if path.resolve() != path or path.is_symlink():
            raise ValueError(
                "component invocation observation escapes its evidence owner"
            )
        record = invocation_record.verify(path)
        records.append({"path": str(path), "sha256": _sha(path), "record": record})
    if not records:
        unresolved.append("actual ordinary invocation observations absent")
    source_copy = generated / "input.interface.mlir"
    source_join = (
        source_copy.is_file() and _sha(source_copy) == member["program_sha256"]
    )
    if not source_join:
        unresolved.append("ordinary compile input does not join admitted source")

    def emitted(name, product, *, stdout=False):
        if not product.is_file() or product.is_symlink():
            return False
        pin = {"path": str(product), "sha256": _sha(product)}
        matches = []
        for row in records:
            record = row["record"]
            if record["stage"] not in name:
                continue
            observed = record["stdout"] if stdout else None
            if (
                (observed and observed["sha256"] == pin["sha256"])
                or pin in record["outputs"]
            ) and any(
                item["sha256"] == member["program_sha256"] for item in record["inputs"]
            ):
                matches.append(row)
        return len(matches) == 1

    target_join = emitted(
        ("lower_interface_to_target",), generated / "lowered.target.mlir", stdout=True
    )
    buffer_join = emitted(
        ("emit_command_buffer", "emit_analysis_bundle"),
        generated / "command_buffer.json",
    )
    artifact_join = emitted(
        ("emit_target_artifact", "lower_target_to_llvm", "emit_analysis_bundle"),
        generated / "lowered.llvm.mlir",
        stdout=True,
    )
    if buffer_join:
        from .cmdbuf import validate

        buffer = json.loads((generated / "command_buffer.json").read_text())
        if validate(buffer):
            raise ValueError(
                "observed component command buffer fails the actual ABI structural validator"
            )
    object_products, elf_products, engine_products = [], [], []
    for row in records:
        record = row["record"]
        if record["stage"] == "object":
            object_products.extend(record.get("outputs", ()))
        elif record["stage"] == "elf":
            elf_products.extend(record.get("outputs", ()))
        elif record["stage"] in {
            "execution",
            "engine_execution",
            "component_cold",
            "component_warm",
        }:
            engine_products.append(row)
    object_to_elf = bool(object_products) and any(
        all(pin in row["record"].get("inputs", ()) for pin in object_products)
        for row in records
        if row["record"]["stage"] == "elf"
    )
    elf_to_engine = bool(elf_products) and any(
        all(pin in row["record"].get("inputs", ()) for pin in elf_products)
        for row in engine_products
    )
    if not object_to_elf:
        unresolved.append("actual object/link input-product join unavailable")
    if not elf_to_engine:
        unresolved.append("actual linked ELF/engine invocation join unavailable")
    # Reopening source and emitted products is useful, but the current target
    # protocol supplies no independently evaluated preparation/partition facet,
    # FX/import chain, or complete all-domain effect authority. Keep each absent
    # facet explicit even if numerical grading and a final ELF audit succeeded.
    if frontend == "pytorch":
        unresolved.append("actual FX capture/import correspondence unavailable")
    unresolved.extend(
        (
            "preparation applicability proof unavailable",
            "partition semantic lift unavailable",
            "emitted host/device correspondence unavailable",
            "complete output/effect execution authority unavailable",
        )
    )
    unresolved.extend(
        "effect obligation UNKNOWN: " + name for name in sorted(set(required_effects))
    )
    report = {
        "schema": "gemmini_component_execution_inspection_v1",
        "status": "UNKNOWN",
        "source_sha256": member["program_sha256"],
        "compiler_sha256": candidate_sha256,
        "target_descriptor_sha256": _sha(target_descriptor),
        "result_sha256": _sha(result_path),
        "observed_file_joins": {
            "admitted_source_to_input": source_join,
            "target_stdout_to_file": target_join,
            "command_buffer_product": buffer_join,
            "artifact_stdout_to_file": artifact_join,
            "object_to_link_input": object_to_elf,
            "linked_elf_to_engine_input": elf_to_engine,
        },
        "invocations": records,
        "unresolved": unresolved,
        "scope": "actual ordinary file/invocation inspection only; no complete stage/effect witness",
    }
    path = generated / "component_execution_inspection.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def _decode(
    service,
    record,
    *,
    compiler_sha256,
    member,
    corpus,
    target_sha256,
    scope,
    regime,
    workspace,
):
    if type(record) is not ExecutedComponent or record.regime != regime:
        raise ValueError(
            "component executor must return one typed observation for the requested execution regime"
        )
    if (
        record.compiler_sha256,
        record.member_sha256,
        record.target_sha256,
        record.scope_sha256,
    ) != (compiler_sha256, member.source_sha256, target_sha256, scope.sha256):
        raise ValueError(
            "component execution belongs to different compiler/source/target/timer inputs"
        )
    artifacts = [
        record.elf,
        record.histogram,
        record.console,
        record.execution_receipt,
        *record.dependencies,
        *record.inputs,
    ]
    if not record.dependencies or not record.inputs:
        raise ValueError(
            "component execution lacks complete build/runtime or input identities"
        )
    for artifact in artifacts:
        artifact.verify()
    # This receipt comes from the selected ordinary executor. Hash/field matching
    # is input identity checking, not an independent hardware execution certificate.
    from merlin.common import invocation_record

    receipt = invocation_record.verify(record.execution_receipt.path)
    configuration = service.verify()
    prefix = configuration["spike_argv_prefix"]
    input_pins = {(pin["path"], pin["sha256"]) for pin in receipt["inputs"]}
    output_pins = {
        (pin["path"], pin["sha256"])
        for pin in [receipt["stdout"], receipt["stderr"], *receipt["outputs"]]
    }
    dependency_pins = {(pin["path"], pin["sha256"]) for pin in receipt["dependencies"]}
    if (
        receipt.get("kind") != "subprocess"
        or receipt.get("stage") != "component_" + regime
        or receipt.get("executable")
        != {"path": str(service.engine.path), "sha256": service.engine.sha256}
        or receipt["argv"][: len(prefix)] != prefix
        or receipt["argv"].count(str(record.elf.path)) != 1
        or (str(record.elf.path), record.elf.sha256) not in input_pins
        or any(
            (str(a.path), a.sha256) not in input_pins | dependency_pins
            for a in record.inputs
        )
        or any(
            (str(a.path), a.sha256) not in dependency_pins
            for a in (
                service.machine_configuration,
                *service.runtime_dependencies,
                *record.dependencies,
            )
        )
        or any(
            (str(a.path), a.sha256) not in output_pins
            for a in (record.histogram, record.console)
        )
    ):
        raise ValueError(
            "ordinary component invocation record differs from actual execution inputs/products"
        )
    binding = {
        "elf_sha256": record.elf.sha256,
        "histogram_sha256": record.histogram.sha256,
        "scope_id": "component_" + regime + "_whole_execution",
        "engine": {
            "kind": service.engine_kind,
            "sha256": service.engine.sha256,
            "machine_configuration_sha256": service.machine_configuration.sha256,
        },
    }
    decoded = census(
        record.elf.path,
        record.histogram.path,
        scope_id=binding["scope_id"],
        execution_binding=binding,
    )
    if decoded["status"] != "pass":
        raise ValueError(
            "component execution has unknown or unmapped emitted instruction classes"
        )
    audit = audit_elf(record.elf.path.read_bytes())
    if audit["status"] != "pass":
        raise ValueError(
            "component final ELF fails the primitive-only instruction audit"
        )
    if record.operand_telemetry is not None:
        if record.operand_producer_qualification is None:
            raise ValueError(
                "operand observations require independent copied-engine producer qualification"
            )
        for artifact in (
            record.operand_telemetry,
            record.operand_producer_qualification,
        ):
            artifact.verify()
            artifacts.append(artifact)
        # Qualification authority is owned by the selected executor. Retain its
        # original bytes; the adapter never interprets a claimed boolean as proof.
        decoded["operand_observation"] = summarize(
            read_telemetry(record.operand_telemetry.path)
        )
    decoded["unknown_features"].update(
        {
            "ordered_fp_dependencies": "UNKNOWN: unordered PC histogram",
            "ordered_gpr_dependencies": "UNKNOWN: unordered PC histogram",
            "dependency_distance": "UNKNOWN: no chronological register trace",
            "exact_replay_and_fallback_counts": "UNKNOWN: no semantic execution scope",
            "fences_and_drains": "UNKNOWN: no joined ordered event timeline",
            "live_allocations": "UNKNOWN: no allocation/lifetime observer",
            "cache_footprint": "UNKNOWN: no addressed memory trace",
            "code_footprint": "UNKNOWN: no declared code/cache measurement scope",
        }
    )
    functional_status = legality_status = "UNKNOWN"
    # This adapter has actual engine observations, not the independent original
    # golden and all-domain effect verifier. Neither an invocation record nor
    # a primitive-only ELF audit grants functional or legality PASS.
    regions = record.regions
    if not regions:
        regions = tuple(
            ComponentCostRegion(stage, (stage,), ("unpriced_" + stage,), {})
            for stage in COMPLETE_STAGES
        )
    else:
        raise NotImplementedError(
            "priced component stages require verified cold/warm stage timeline scopes"
        )
    artifact = workspace / (regime + "_executed_features.json")
    artifact.write_text(json.dumps(decoded, sort_keys=True, indent=2) + "\n")
    artifacts.append(Artifact(artifact.resolve(), _sha(artifact)))
    return regions, functional_status, legality_status, artifacts


@dataclass(frozen=True)
class UnobservedComponentRegime:
    regime: str
    reason: str


def _normal_development_execute(
    *,
    binding,
    compiler,
    member,
    corpus,
    target_descriptor,
    scope,
    regime,
    workspace,
    timeout_s,
    configuration,
):
    from merlin.common import invocation_record
    from merlin.runtime.backends import base as backends

    binding.verify()
    if (
        scope != binding.scope
        or Path(target_descriptor).resolve() != binding.target_experiment.path
    ):
        raise ValueError(
            "normal component execution selected another exact target/scope"
        )
    if regime == "warm":
        return UnobservedComponentRegime(
            "warm",
            "ordinary contract executor has no independently qualified warm PC/stage scope",
        )
    if regime != "cold":
        raise ValueError("ordinary component execution has an unknown regime")
    backend = backends.get_backend(binding.target_experiment.target)
    observer = getattr(backend, "observe_component_execution", None)
    if not callable(observer):
        raise TypeError(
            "selected ordinary engine launch has no actual PC observation producer"
        )
    root = binding.output / canonical_sha256(
        [str(workspace), str(compiler), member.source_sha256]
    )
    binding.output.mkdir(parents=True, exist_ok=True)
    inputs = tuple(
        Artifact(path.resolve(), _sha(path))
        for path in sorted(member.source_dir.rglob("*"))
        if path.is_file()
    )
    machine = binding.output / (binding.sha256 + "_machine.json")
    dependencies = (
        *tuple(
            Artifact(Path(path), digest)
            for path, digest in configuration["dependencies"]
        ),
        Artifact(machine, _sha(machine)),
    )
    with observer(
        configuration=configuration,
        dependencies=tuple(a.path for a in dependencies),
        inputs=tuple(a.path for a in inputs),
        regime=regime,
    ):
        raw = binding.execute(
            compiler=compiler,
            member=member,
            corpus=corpus,
            workspace=root,
            timeout_s=timeout_s,
        )
    raw_path = root / "component_normal_execution.json"
    raw_path.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n")
    found = []
    for path in root.rglob("invocation.json"):
        record = invocation_record.verify(path)
        if record.get("kind") == "subprocess" and record["stage"] == "component_cold":
            found.append((path, record))
    if len(found) != 1:
        raise ValueError(
            "normal component execution needs exactly one observed cold Spike invocation; cache/split scope unresolved"
        )
    path, record = found[0]
    elf_path = Path(record["argv"][-1])
    observed_inputs = {pin["path"]: pin["sha256"] for pin in record["inputs"]}
    if observed_inputs.get(str(elf_path)) != _sha(elf_path):
        raise ValueError(
            "normal observed Spike invocation has no exact current ELF input"
        )
    return ExecutedComponent(
        regime,
        str(hash_tree(compiler)["sha256"]),
        member.source_sha256,
        _sha(target_descriptor),
        scope.sha256,
        configuration["domain_sha256"],
        Artifact(elf_path, _sha(elf_path)),
        Artifact(Path(record["stderr"]["path"]), record["stderr"]["sha256"]),
        Artifact(Path(record["stdout"]["path"]), record["stdout"]["sha256"]),
        Artifact(path, _sha(path)),
        dependencies,
        inputs,
    )


def prepare_component_execution_service(
    *,
    baseline,
    qualification,
    view,
    runtime,
    corpus,
    target_experiment,
    contract_root,
    source_root,
    scope,
    output,
):
    """Closed normal service selection from an exact target-owned descriptor.

    Missing independent certificates, native tools, runtime or source stage proof
    refuses. Existing historical calibration and run directories are not read.
    """
    import yaml
    from merlin.runtime.backends import base as backends
    from merlin_experiments.phase2.component_execution import (
        prepare_component_normal_execution,
    )
    from merlin_experiments.phase2.development_feedback import sweep_workers

    document = yaml.safe_load(Path(target_experiment.path).read_text())
    selected = (
        document.get("component_feedback") if isinstance(document, dict) else None
    )
    required = {"schema", "certificate", "rtl_facts", "machine_configuration"}
    if (
        not isinstance(selected, dict)
        or set(selected) != required
        or selected["schema"] != "gemmini.component_feedback.v1"
    ):
        raise ValueError(
            "component_feedback descriptor requires pinned independent certificate, RTL facts and machine configuration"
        )
    pins = {}
    for name in ("certificate", "rtl_facts", "machine_configuration"):
        pin = selected[name]
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
            raise ValueError(
                "component feedback selection requires exact absolute file pins"
            )
        pins[name] = Artifact(Path(pin["path"]), pin["sha256"])
        pins[name].verify()
    if sweep_workers() != 1:
        raise ValueError(
            "component normal execution requires one inner worker until engine observation context propagation is qualified"
        )
    configuration = json.loads(pins["machine_configuration"].path.read_text())
    required_machine = {
        "schema",
        "engine",
        "isa",
        "harts",
        "memory",
        "domain_sha256",
        "runtime_dependencies",
    }
    if (
        not isinstance(configuration, dict)
        or set(configuration) != required_machine
        or configuration["schema"] != "gemmini.component_machine.v1"
        or type(configuration["harts"]) is not int
        or configuration["harts"] < 1
        or not isinstance(configuration["isa"], str)
        or not configuration["isa"]
    ):
        raise ValueError(
            "component machine selection requires exact engine, ISA, harts, memory and runtime closure"
        )
    engine = Artifact(
        Path(configuration["engine"]["path"]), configuration["engine"]["sha256"]
    )
    engine.verify()
    runtime_files = tuple(
        Artifact(Path(pin["path"]), pin["sha256"])
        for pin in configuration["runtime_dependencies"]
    )
    for artifact in runtime_files:
        artifact.verify()
    if not runtime_files:
        raise ValueError("component machine execution runtime closure is empty")
    binding = prepare_component_normal_execution(
        baseline=baseline,
        qualification=qualification,
        view=view,
        runtime=runtime,
        corpus=corpus,
        target_experiment=target_experiment,
        contract_root=contract_root,
        source_root=source_root,
        scope=scope,
        output=output,
        certificate_path=pins["certificate"].path,
        certificate_sha256=pins["certificate"].sha256,
        rtl_facts_path=pins["rtl_facts"].path,
    )
    backend = backends.get_backend(target_experiment.target)
    flags, libdir = backend.spike_extension()
    required_extensions = {
        path.resolve() for path in libdir.glob("libgemmini*.so*") if path.is_file()
    }
    required_extensions.update(
        Path(flag.split("=", 1)[1]).resolve()
        for flag in flags
        if flag.startswith("--extlib=")
    )
    if not required_extensions or required_extensions - {
        artifact.path for artifact in runtime_files
    }:
        raise ValueError(
            "selected ordinary Spike extension/runtime files lack explicit machine dependency pins"
        )
    if Path(backend.spike_path()).resolve() != engine.path:
        raise ValueError(
            "explicit machine engine differs from the selected ordinary target launch owner"
        )
    prefix = [
        str(engine.path),
        "-g",
        *flags,
        "-p" + str(configuration["harts"]),
        "--isa=" + configuration["isa"],
    ]
    memory = configuration["memory"]
    if memory is not None:
        if (
            not isinstance(memory, list)
            or len(memory) != 2
            or any(type(value) is not int or value <= 0 for value in memory)
        ):
            raise ValueError(
                "component selected memory span must be explicit positive base/extent or absent"
            )
        prefix.append("-m" + hex(memory[0]) + ":" + hex(memory[1]))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    machine = output / (binding.sha256 + "_machine.json")
    dependencies = {path: digest for path, digest in binding.source_pins}
    dependencies.update(
        {
            path.resolve(): _sha(path)
            for path in Path(__file__).resolve().parent.rglob("*.py")
        }
    )
    dependencies.update({a.path: a.sha256 for a in (*runtime_files, *pins.values())})
    complete = {
        **configuration,
        "spike_argv_prefix": prefix,
        "executor_qualname": _normal_development_execute.__qualname__,
        "normal_execution_sha256": binding.sha256,
        "dependencies": [
            (str(path), digest)
            for path, digest in sorted(
                dependencies.items(), key=lambda row: str(row[0])
            )
        ],
    }
    machine.write_text(json.dumps(complete, sort_keys=True, indent=2) + "\n")
    machine.chmod(0o400)
    service = ComponentExecutionService(
        partial(_normal_development_execute, binding=binding),
        Artifact(Path(__file__).resolve(), _sha(__file__)),
        engine,
        Artifact(machine, _sha(machine)),
        tuple(
            Artifact(path, digest)
            for path, digest in sorted(
                dependencies.items(), key=lambda row: str(row[0])
            )
        ),
        "spike",
        json.dumps(complete, sort_keys=True),
    )
    # The generated exact machine selection also joins the actual engine record.
    # Keep self hashing out of the file: the recorder adds this exact pin from
    # its explicit observer service rather than rewriting a circular commitment.
    service.verify()
    return service


def component_feature_provider(
    *,
    service,
    baseline,
    candidate,
    member,
    corpus,
    target_descriptor,
    scope,
    workspace,
    timeout_s,
):
    """Execute and decode two exact compiler arms under explicit cold/warm regimes."""
    if (
        type(service) is not ComponentExecutionService
        or type(scope) is not ComponentCostScope
    ):
        raise ValueError(
            "component features require selected normal execution service and exact complete-cost scope"
        )
    service.verify()
    deadline = time.monotonic() + min(float(timeout_s), ITERATION_MAX_SECONDS)
    target_sha256 = _sha(target_descriptor)
    result = []
    for arm, compiler in (("baseline", Path(baseline)), ("candidate", Path(candidate))):
        compiler_sha = str(hash_tree(compiler)["sha256"])
        records, regime_rows = [], []
        for regime in ("cold", "warm"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "component features exhausted the development simulation budget"
                )
            work = Path(workspace) / arm / regime
            work.mkdir(parents=True)
            record = service.execute(
                compiler=compiler,
                member=member,
                corpus=corpus,
                target_descriptor=target_descriptor,
                scope=scope,
                regime=regime,
                workspace=work,
                timeout_s=remaining,
                configuration=service.verify(),
            )
            if type(record) is UnobservedComponentRegime:
                if record.regime != regime or not record.reason:
                    raise ValueError(
                        "unobserved regime requires exact regime and concrete missing authority"
                    )
                refusal = work / "unobserved_regime.json"
                refusal.write_text(
                    json.dumps(
                        {"regime": regime, "status": "UNKNOWN", "reason": record.reason}
                    )
                    + "\n"
                )
                unknown = tuple(
                    ComponentCostRegion(stage, (stage,), ("unpriced_" + stage,), {})
                    for stage in COMPLETE_STAGES
                )
                regime_rows.append(
                    (
                        unknown,
                        "UNKNOWN",
                        "UNKNOWN",
                        [Artifact(refusal.resolve(), _sha(refusal))],
                    )
                )
            else:
                regime_rows.append(
                    _decode(
                        service,
                        record,
                        compiler_sha256=compiler_sha,
                        member=member,
                        corpus=corpus,
                        target_sha256=target_sha256,
                        scope=scope,
                        regime=regime,
                        workspace=work,
                    )
                )
            records.append(record)
        # Cold and warm must be separately observed. A repeated static ELF is
        # valid; reusing the same invocation/histogram/readback identity is not.
        if type(records[0]) is not ExecutedComponent:
            raise ValueError("ordinary component execution has no actual cold witness")
        if type(records[1]) is ExecutedComponent and (
            records[0].execution_receipt.path == records[1].execution_receipt.path
            or records[0].histogram.path == records[1].histogram.path
            or records[0].console.path == records[1].console.path
        ):
            raise ValueError(
                "component cold and warm regimes reuse one execution observation"
            )
        if type(records[1]) is ExecutedComponent and [
            (a.path, a.sha256) for a in records[0].inputs
        ] != [(a.path, a.sha256) for a in records[1].inputs]:
            raise ValueError(
                "component cold/warm executions use different admitted inputs"
            )
        if (
            type(records[1]) is ExecutedComponent
            and records[0].domain_sha256 != records[1].domain_sha256
        ):
            raise ValueError(
                "component cold/warm records must share the declared combined regime domain"
            )
        artifacts = [
            *regime_rows[0][3],
            *regime_rows[1][3],
            service.implementation,
            service.engine,
            service.machine_configuration,
            *service.runtime_dependencies,
        ]
        result.append(
            ComponentFeatureObservation(
                compiler_sha,
                member.source_sha256,
                corpus.capsules_sha256,
                target_sha256,
                scope.sha256,
                records[0].domain_sha256,
                records[0].elf.sha256,
                canonical_sha256(
                    [
                        (str(a.path), a.sha256)
                        for record in records
                        if type(record) is ExecutedComponent
                        for a in record.dependencies
                    ]
                ),
                canonical_sha256([a.sha256 for a in records[0].inputs]),
                tuple(a.sha256 for a in artifacts),
                regime_rows[0][0],
                regime_rows[1][0],
                "PASS" if all(row[1] == "PASS" for row in regime_rows) else "UNKNOWN",
                tuple((str(a.path), a.sha256) for a in artifacts),
                legality_status="UNKNOWN",
            )
        )
        if str(hash_tree(compiler)["sha256"]) != compiler_sha:
            raise ValueError("component execution changed compiler bytes")
    service.verify()
    if result[0].inputs_sha256 != result[1].inputs_sha256:
        raise ValueError(
            "component compiler arms execute different admitted input bytes"
        )
    return tuple(result)


def prepare_component_feature_provider(*, execution_service, require_normal=False):
    if type(execution_service) is not ComponentExecutionService:
        raise ValueError("a selected ordinary component execution service is required")
    execution_service.verify()
    if require_normal and (
        type(execution_service.execute) is not partial
        or execution_service.execute.func is not _normal_development_execute
    ):
        raise ValueError(
            "selected component feedback requires the closed admitted normal execution factory"
        )
    provider = partial(component_feature_provider, service=execution_service)
    # The generic host must include these private exact pins in its factory's
    # explicit dependency mapping. Callback source alone does not bind engine,
    # ISA/runtime configuration or admitted executor selection.
    provider.component_source_pins = {
        artifact.path: artifact.sha256
        for artifact in (
            execution_service.implementation,
            execution_service.engine,
            execution_service.machine_configuration,
            *execution_service.runtime_dependencies,
        )
    }
    return provider
