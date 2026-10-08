"""Actual process observations remain distinct from stage/functional/cycle proof."""

import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin.benchharness import hash_tree
from merlin.common import invocation_record
from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf.component_cost import COMPLETE_STAGES, ComponentCostScope
from test_no_fsm_audit import elf_with_instructions, rocc

from mlir_oot import component_feedback as feedback


def pin(path):
    path = Path(path).resolve()
    return feedback.Artifact(path, hashlib.sha256(path.read_bytes()).hexdigest())


def ordinary_fixture_executor(
    *,
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
    """Observe an inert test producer; it grants no source execution authority."""
    elf = workspace / "component.elf"
    elf.write_bytes(elf_with_instructions([rocc(2), rocc(4)]))
    executable = Path(configuration["spike_argv_prefix"][0])
    inputs = tuple(pin(Path(path)) for path in configuration["inputs"])
    dependencies = tuple(pin(Path(path)) for path in configuration["dependencies"])
    invocation_record.run(
        [*configuration["spike_argv_prefix"], str(elf)],
        directory=workspace,
        stage="component_" + regime,
        inputs=(elf, *(a.path for a in inputs)),
        dependencies=tuple(a.path for a in dependencies),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    receipt = next((workspace / "invocations").glob("*/invocation.json"))
    record = invocation_record.verify(receipt)
    assert record["executable"]["path"] == str(executable)
    return feedback.ExecutedComponent(
        regime,
        hash_tree(compiler)["sha256"],
        member.source_sha256,
        hashlib.sha256(target_descriptor.read_bytes()).hexdigest(),
        scope.sha256,
        sha("explicit combined cold/warm test domain"),
        pin(elf),
        pin(record["stderr"]["path"]),
        pin(record["stdout"]["path"]),
        pin(receipt),
        dependencies,
        inputs,
    )


def setup(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for root, content in (
        (baseline, "original compiler"),
        (candidate, "candidate compiler"),
    ):
        root.mkdir()
        (root / "compiler").write_text(content)
    target = tmp_path / "target.json"
    target.write_text('{"configuration":"explicit fixture only"}')
    engine = tmp_path / "inert-producer"
    engine.write_text(
        "#!" + sys.executable + "\nimport sys\n"
        "print('retained complete console fixture')\n"
        "print('PC Histogram size:2\\n80000000 7\\n80000004 3',file=sys.stderr)\n"
    )
    engine.chmod(0o755)
    runtime = tmp_path / "runtime.bin"
    runtime.write_bytes(b"explicit inert runtime input")
    payload = tmp_path / "input.bin"
    payload.write_bytes(b"admitted equal input bytes")
    machine = tmp_path / "machine.json"
    configuration = {
        "isa": "rv64gc",
        "spike_argv_prefix": [str(engine), "-g", "--isa=rv64gc", "--extension=gemmini"],
        "executor_qualname": ordinary_fixture_executor.__qualname__,
        "inputs": [str(payload)],
        "dependencies": [str(machine), str(runtime)],
    }
    machine.write_text(json.dumps(configuration))
    service = feedback.ComponentExecutionService(
        ordinary_fixture_executor,
        pin(Path(__file__)),
        pin(engine),
        pin(machine),
        (pin(runtime),),
        "spike",
        json.dumps(configuration),
    )
    scope = ComponentCostScope(sha("timer"), sha("accuracy"), sha("input policy"))
    arguments = {
        "baseline": baseline,
        "candidate": candidate,
        "member": SimpleNamespace(source_sha256=sha("member")),
        "corpus": SimpleNamespace(capsules_sha256=sha("corpus")),
        "target_descriptor": target,
        "scope": scope,
        "workspace": tmp_path / "observations",
        "timeout_s": 5,
    }
    return service, arguments


def test_actual_engine_invocation_features_do_not_promote_missing_stages(tmp_path):
    service, arguments = setup(tmp_path)
    provider = feedback.prepare_component_feature_provider(execution_service=service)
    baseline, candidate = provider(**arguments)
    assert baseline.compiler_sha256 != candidate.compiler_sha256
    assert baseline.inputs_sha256 == candidate.inputs_sha256
    for observation in (baseline, candidate):
        assert observation.functional_status == observation.legality_status == "UNKNOWN"
        assert tuple(region.id for region in observation.cold) == COMPLETE_STAGES
        assert tuple(region.id for region in observation.warm) == COMPLETE_STAGES
        assert all(
            region.context == {} for region in observation.cold + observation.warm
        )
    decoded = json.loads(
        next(
            arguments["workspace"].glob("baseline/cold/*executed_features.json")
        ).read_text()
    )
    assert decoded["features"]["primitive_commands"]["LOAD_CMD"] == 7
    assert decoded["features"]["primitive_commands"]["COMPUTE_AND_FLIP_CMD"] == 3
    assert decoded["unknown_features"]["hardware_cycles"] == "UNKNOWN"


@pytest.mark.parametrize("missing", ["spike_argv_prefix", "isa"])
def test_missing_explicit_machine_configuration_refuses(tmp_path, missing):
    service, _ = setup(tmp_path)
    configuration = json.loads(service.configuration_json)
    configuration.pop(missing)
    with pytest.raises(ValueError, match="explicit Spike"):
        replace(service, configuration_json=json.dumps(configuration)).verify()


def test_invocation_artifact_mutation_refuses_before_decoding(tmp_path):
    service, arguments = setup(tmp_path)
    work = arguments["workspace"]
    work.mkdir()
    record = service.execute(
        compiler=arguments["baseline"],
        member=arguments["member"],
        corpus=arguments["corpus"],
        target_descriptor=arguments["target_descriptor"],
        scope=arguments["scope"],
        regime="cold",
        workspace=work,
        timeout_s=3,
        configuration=service.verify(),
    )
    record.histogram.path.write_text("PC Histogram size:1\n80000000 1\n")
    with pytest.raises(ValueError, match="changed"):
        feedback._decode(
            service,
            record,
            compiler_sha256=hash_tree(arguments["baseline"])["sha256"],
            member=arguments["member"],
            corpus=arguments["corpus"],
            target_sha256=pin(arguments["target_descriptor"]).sha256,
            scope=arguments["scope"],
            regime="cold",
            workspace=work,
        )


def test_selected_full_support_resolves_companion_decoder_and_requires_service(
    monkeypatch,
):
    from merlin.runtime.backends.base import get_backend

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root / "merlin-support"))
    backend = get_backend("gemmini")
    artifact, executed, service = backend.component_execution_types()
    assert (
        Path(sys.modules[artifact.__module__].__file__).resolve()
        == root / "mlir_oot/component_feedback.py"
    )
    assert executed.__module__ == service.__module__ == artifact.__module__
    assert (
        root / "mlir_oot/component_feedback.py"
        in backend.component_feedback_dependencies()
    )
    with pytest.raises(ValueError, match="execution service"):
        backend.prepare_component_feature_provider(execution_service=None)


def inspection_arguments(
    tmp_path, *, source_text="fixture source bytes; no semantic witness claimed"
):
    from merlin_experiments.phase2 import contracts

    compiler = tmp_path / "compiler"
    compiler.mkdir()
    (compiler / "compiler").write_text("exact immutable fixture compiler")
    snapshot = tmp_path / "snapshot"
    shutil.copytree(compiler, snapshot)
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    source = capsule / "program.mlir"
    source.write_text(source_text)
    (capsule / "capsule.yaml").write_text("interface_mlir: program.mlir\n")
    grade = tmp_path / "grade"
    generated = grade / "case" / "generated"
    generated.mkdir(parents=True)
    result = generated.parent / "capsule_result.json"
    result.write_text(
        json.dumps({"capsule": "case", "status": "PASS", "all_effects": "PASS"})
    )
    copied = generated / "input.interface.mlir"
    copied.write_bytes(source.read_bytes())
    for stage, name in (
        ("lower_interface_to_target", "lowered.target.mlir"),
        ("emit_target_artifact", "lowered.llvm.mlir"),
    ):
        process = invocation_record.run(
            [sys.executable, "-c", "print('actual observed fixture product')"],
            directory=generated,
            stage=stage,
            inputs=(copied,),
            capture_output=True,
            text=True,
            check=True,
        )
        (generated / name).write_text(process.stdout)
    arguments = {
        "member": {"name": "case", "program_sha256": pin(source).sha256},
        "result_path": result,
        "candidate_root": compiler,
        "compiler_snapshot": snapshot,
        "candidate_sha256": contracts.exact_tree_record(compiler)["sha256"],
        "capsule_root": capsule,
        "evidence_root": grade,
        "target_descriptor": source,
        "frontend": "mlir",
        "required_effects": ("alias", "epoch", "repeated_invocation"),
        "timeout_s": 3,
    }
    return arguments


def test_observed_stage_products_and_positive_metadata_cannot_fill_semantic_gaps(
    tmp_path,
):
    arguments = inspection_arguments(tmp_path)
    inspection = feedback.inspect_component_execution_records(**arguments)
    assert inspection["observed_file_joins"]["admitted_source_to_input"]
    assert inspection["observed_file_joins"]["target_stdout_to_file"]
    assert inspection["observed_file_joins"]["artifact_stdout_to_file"]
    assert inspection["status"] == "UNKNOWN"
    with pytest.raises(NotImplementedError, match="effect obligation UNKNOWN: alias"):
        feedback.verify_component_execution_witness(**arguments)


def test_full_ordinary_run_owner_retains_actual_engine_command(tmp_path, monkeypatch):
    import importlib

    from merlin.runtime.backends import spike_model
    from merlin.runtime.backends.base import get_backend

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root / "merlin-support"))
    backend = get_backend("gemmini")
    implementation = importlib.import_module(backend.__name__ + ".gemmini")
    engine = tmp_path / "engine"
    engine.write_text(
        "#!" + sys.executable + "\nprint('retained ordinary engine console')\n"
    )
    engine.chmod(0o755)
    elf = tmp_path / "ordinary.elf"
    elf.write_bytes(elf_with_instructions([rocc(2)]))
    library = tmp_path / "libgemmini.so"
    library.write_bytes(b"inert explicitly observed library fixture")
    monkeypatch.setattr(implementation, "spike_path", lambda: engine)
    monkeypatch.setattr(
        implementation, "spike_extension", lambda: (("--extension=gemmini",), tmp_path)
    )
    for name in ("declared_harts", "declared_isa", "declared_memory"):
        monkeypatch.setattr(spike_model, name, lambda _: None)
    console = backend.run_elf(elf, simulator="spike", timeout=2)
    record = invocation_record.verify(
        next((tmp_path / "invocations").glob("*/invocation.json"))
    )
    assert console == "retained ordinary engine console\n"
    assert record["kind"] == "subprocess" and record["stage"] == "engine_execution"
    assert record["argv"] == [str(engine), "--extension=gemmini", str(elf)]
    assert record["inputs"] == [{"path": str(elf), "sha256": pin(elf).sha256}]
    assert {"path": str(library), "sha256": pin(library).sha256} in record[
        "dependencies"
    ]


def unobserved_warm_executor(**kwargs):
    if kwargs["regime"] == "warm":
        return feedback.UnobservedComponentRegime(
            "warm", "no qualified warm region in actual normal harness"
        )
    return ordinary_fixture_executor(**kwargs)


def test_missing_warm_observation_is_unknown_and_never_reuses_cold_histogram(tmp_path):
    service, arguments = setup(tmp_path)
    configuration = json.loads(service.configuration_json)
    configuration["executor_qualname"] = unobserved_warm_executor.__qualname__
    service.machine_configuration.path.write_text(json.dumps(configuration))
    service = replace(
        service,
        execute=unobserved_warm_executor,
        machine_configuration=pin(service.machine_configuration.path),
        configuration_json=json.dumps(configuration),
    )
    observations = feedback.prepare_component_feature_provider(
        execution_service=service
    )(**arguments)
    assert observations[0].inputs_sha256 == observations[1].inputs_sha256
    for arm in ("baseline", "candidate"):
        assert (
            len(
                list(
                    (arguments["workspace"] / arm / "cold" / "invocations").glob(
                        "*/invocation.json"
                    )
                )
            )
            == 1
        )
        assert not (arguments["workspace"] / arm / "warm" / "invocations").exists()
        assert (
            json.loads(
                (
                    arguments["workspace"] / arm / "warm" / "unobserved_regime.json"
                ).read_text()
            )["status"]
            == "UNKNOWN"
        )
    assert all(region.context == {} for row in observations for region in row.warm)


def test_closed_selected_factory_refuses_missing_explicit_independent_machine_configuration(
    tmp_path, monkeypatch
):
    from merlin.runtime.backends.base import get_backend

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root / "merlin-support"))
    target = tmp_path / "descriptor.yaml"
    target.write_text("target: gemmini\n")
    backend = get_backend("gemmini")
    with pytest.raises(ValueError, match="pinned independent certificate"):
        backend.prepare_component_execution_service(
            baseline=tmp_path,
            qualification=None,
            view=None,
            runtime=(),
            corpus=None,
            target_experiment=SimpleNamespace(path=target),
            contract_root=tmp_path,
            source_root=tmp_path,
            scope=None,
            output=tmp_path / "private",
        )


def test_observation_context_changes_only_the_actual_selected_spike_launch(
    tmp_path, monkeypatch
):
    import importlib

    from merlin.runtime.backends import spike_model
    from merlin.runtime.backends.base import get_backend

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root / "merlin-support"))
    backend = get_backend("gemmini")
    owner = importlib.import_module(backend.__name__ + ".gemmini")
    engine = tmp_path / "ordinary-engine"
    engine.write_text("#!" + sys.executable + "\nprint('ordinary observed fixture')\n")
    engine.chmod(0o755)
    elf = tmp_path / "fixture.elf"
    elf.write_bytes(elf_with_instructions([rocc(2)]))
    runtime = tmp_path / "extension.bin"
    runtime.write_bytes(b"explicit fixture runtime")
    monkeypatch.setattr(owner, "spike_path", lambda: engine)
    monkeypatch.setattr(
        owner, "spike_extension", lambda: (("--extension=gemmini",), tmp_path)
    )
    for name in ("declared_harts", "declared_isa", "declared_memory"):
        monkeypatch.setattr(spike_model, name, lambda _: None)
    prefix = [str(engine), "-g", "--extension=gemmini", "-p1", "--isa=rv64gc"]
    configuration = {
        "spike_argv_prefix": prefix,
        "harts": 1,
        "isa": "rv64gc",
        "memory": None,
    }
    with backend.observe_component_execution(
        configuration=configuration, dependencies=(runtime,), inputs=(), regime="cold"
    ):
        backend.run_elf(elf, simulator="spike", timeout=2)
    backend.run_elf(elf, simulator="spike", timeout=2)
    records = [
        invocation_record.verify(path)
        for path in (tmp_path / "invocations").glob("*/invocation.json")
    ]
    observed = next(record for record in records if record["stage"] == "component_cold")
    ordinary = next(
        record for record in records if record["stage"] == "engine_execution"
    )
    assert observed["argv"] == [*prefix, str(elf)]
    assert ordinary["argv"] == [str(engine), "--extension=gemmini", str(elf)]
    assert {"path": str(runtime), "sha256": pin(runtime).sha256} in observed[
        "dependencies"
    ]


def test_selected_factory_does_not_adopt_transport_callback_as_execution_authority(
    tmp_path, monkeypatch
):
    from merlin.runtime.backends.base import get_backend

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root / "merlin-support"))
    backend = get_backend("gemmini")
    with pytest.raises(ValueError, match="execution service"):
        backend.prepare_component_feature_provider(execution_service=object())
    service, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="closed admitted normal execution factory"):
        feedback.prepare_component_feature_provider(
            execution_service=service, require_normal=True
        )


def test_prepared_bound_owner_has_live_context_and_no_mutable_partial_service(tmp_path):
    from dataclasses import FrozenInstanceError

    service, _ = setup(tmp_path)
    prepared = feedback.prepare_component_feature_provider(execution_service=service)
    owner = prepared.__self__
    assert owner.component_source_pins[service.engine.path] == service.engine.sha256
    with pytest.raises(FrozenInstanceError):
        owner.service = None
    service.engine.path.write_bytes(b"changed selected runtime")
    with pytest.raises(ValueError, match="changed"):
        _ = owner.component_source_pins


def test_source_applicability_dict_flags_cannot_supply_independent_source_authority(
    tmp_path,
):
    arguments = inspection_arguments(tmp_path)
    with pytest.raises(ValueError, match="source applicability"):
        feedback.inspect_component_execution_records(
            **arguments,
            source_applicability={
                "observable_input_mutation": "N_A",
                "all_runtime_effects": "PASS",
            },
        )


def test_actual_source_applicability_join_retains_physical_runtime_unknowns(tmp_path):
    from merlin_experiments.phase1.component_source_applicability import (
        evaluate_component_source_applicability,
    )

    arguments = inspection_arguments(
        tmp_path,
        source_text=(
            "module { func.func @identity(%arg0: tensor<2xf32>) -> tensor<2xf32> "
            "{ func.return %arg0 : tensor<2xf32> } }"
        ),
    )
    source = arguments["capsule_root"] / "program.mlir"
    authority = evaluate_component_source_applicability(
        source=source,
        source_program_sha256=arguments["member"]["program_sha256"],
        frontend="mlir",
    )
    record = feedback.inspect_component_execution_records(
        **arguments, source_applicability=authority
    )
    source_only = record["source_applicability"]
    assert source_only["facts"]["static_input_domain"]["status"] == "PASS"
    assert source_only["facts"]["observable_input_mutation"]["status"] == "N_A"
    assert set(source_only["runtime_effects"].values()) == {"UNKNOWN"}
    assert "effect obligation UNKNOWN: alias" in record["unresolved"]
    with pytest.raises(NotImplementedError, match="effect obligation UNKNOWN: alias"):
        feedback.verify_component_execution_witness(
            **arguments, source_applicability=authority
        )


def test_source_applicability_cannot_be_reused_for_different_observed_source(tmp_path):
    from merlin_experiments.phase1.component_source_applicability import (
        evaluate_component_source_applicability,
    )

    arguments = inspection_arguments(tmp_path)
    other = tmp_path / "different_source.mlir"
    other.write_bytes((arguments["capsule_root"] / "program.mlir").read_bytes())
    authority = evaluate_component_source_applicability(
        source=other,
        source_program_sha256=arguments["member"]["program_sha256"],
        frontend="mlir",
    )
    with pytest.raises(ValueError, match="exact independent observed source"):
        feedback.inspect_component_execution_records(
            **arguments, source_applicability=authority
        )
