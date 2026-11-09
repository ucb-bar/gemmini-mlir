'''Mixed-program owner extraction and public callback/record preservation.

Substitute host/device products below exercise routing and metadata, not target
arithmetic, native code generation or physical completion qualification.
'''
import inspect
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
from gemmini.backend import host_runtime as public
from gemmini.backend import hostcontract, hostlink

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import lower

SOURCE = '''module {func.func @forward(%x:tensor<2x3xf32>,%y:tensor<2x3xf32>)
 -> (tensor<6xf32>,tensor<6xf32>) {
 %a=tensor.collapse_shape %x [[0,1]]:tensor<2x3xf32> into tensor<6xf32>
 %b=tensor.collapse_shape %y [[0,1]]:tensor<2x3xf32> into tensor<6xf32>
 func.return %a,%b:tensor<6xf32>,tensor<6xf32>
}}'''


def test_public_records_keep_exact_owner_aliases_and_pickle_names(tmp_path):
    for name in ("DeviceRegion", "DeviceRequest", "DeviceArtifact", "WholeProgramResult", "WholeProgramError"):
        cls = getattr(public, name)
        assert cls is getattr(hostcontract, name) and cls.__module__ == public.__name__
    artifact = public.DeviceArtifact(tmp_path / "region.mlir", "kernel", command_buffer={"commands": [{}]})
    assert pickle.loads(pickle.dumps(artifact)) == artifact
    request = public.DeviceRequest("source", "kernel", "a"*64, ("input",), ("output",), (1,), tmp_path)
    assert pickle.loads(pickle.dumps(request)) == request
    assert public.whole_program_command_buffer is hostcontract.whole_program_command_buffer
    assert inspect.getsourcefile(public.DeviceArtifact) == public.__file__  # Public import/pickle surface.
    assert inspect.getsourcefile(hostlink.materialize_combined_llvm_dialect) == hostlink.__file__


@pytest.mark.parametrize("reuse", [False, True])
def test_routing_retains_source_hook_exact_ports_and_shared_host_lowering(tmp_path, monkeypatch, reuse):
    from gemmini.backend.host_frontend_kernel_cache import typed_device_function_identity

    calls, lowered, source_hooks = [], [], []
    original = public._source_function
    def source_function(region, symbol):
        source_hooks.append(symbol)
        return original(region, symbol)
    monkeypatch.setattr(public, "_source_function", source_function)
    def regions(module):
        function = module.body.block.first_op
        return tuple(public.DeviceRegion((op,), tuple(op.operands), tuple(op.results), "test layout")
                     for op in function.body.block.ops if op.name == "tensor.collapse_shape")
    def compile_device(request):
        calls.append(request)
        module = parse_mlir_text(request.source_text)
        assert module.body.block.first_op.sym_name.data == request.symbol
        assert request.input_types == ("tensor<2x3xf32>",) and request.output_types == ("tensor<6xf32>",)
        path = request.workdir / "device.mlir"
        path.write_text("// substitute device product; never executed\n")
        return public.DeviceArtifact(path, request.symbol, command_buffer={"commands": [{"transport": "substitute"}]})
    def shared(text, work, **options):
        lowered.append((text, options))
        work.mkdir()
        path = work / "model.ll"
        path.write_text("; substitute host product; never executed\n")
        return SimpleNamespace(ll_path=path, stats={})
    monkeypatch.setattr(lower, "lower_model", shared)
    result = public.lower_whole_program(SOURCE, tmp_path / "mixed", compile_device, regions=regions,
        host_driver="python", host_optimization="baseline", reuse_device_kernels=reuse)
    assert len(calls) == (1 if reuse else 2) and len(source_hooks) == 2
    assert len(lowered) == 1 and lowered[0][1]["targets"] == ()
    assert result.shim_c.is_file() and result.routed_source.is_file()
    module = parse_mlir_text(result.routed_source.read_text())
    function = module.body.block.first_op
    assert tuple(map(str,function.function_type.inputs)) == ("tensor<2x3xf32>", "tensor<2x3xf32>")
    assert tuple(map(str,function.function_type.outputs)) == ("tensor<6xf32>", "tensor<6xf32>")
    rows = result.stats["adapter_device_calls"]
    assert len(rows) == 2 and rows[0]["source_ordinals"] != rows[1]["source_ordinals"]
    assert result.stats["adapter_device_code_sharing"] == {
        "enabled": reuse, "source_calls": 2, "compiled_kernels": 1 if reuse else 2, "reused_calls": int(reuse)}
    assert "merlin_device_group_0" in result.shim_c.read_text()
    assert typed_device_function_identity(calls[0].source_text)
    result.stats["adapter_schedule_instruction_count"] = 7
    commands = public.whole_program_command_buffer(result)
    assert commands["kernel_abi"]["outputs"] == ["output0", "output1"]
    assert commands["params"]["global_program_plan"]["schedule_instruction_count"] == 7
    assert commands["commands"] == []  # Host semantics stay in the compiled task.


def test_declined_callback_is_fatal_before_shared_host_lowering(tmp_path, monkeypatch):
    monkeypatch.setattr(lower, "lower_model", lambda *a, **k: pytest.fail("host fallback reached"))
    def regions(module):
        operation = module.body.block.first_op.body.block.first_op
        return (public.DeviceRegion((operation,), tuple(operation.operands), tuple(operation.results)),)
    def declined(request):
        path = request.workdir / "device.mlir"
        path.write_text("not executed")
        return public.DeviceArtifact(path, request.symbol, command_buffer={"commands": []})
    with pytest.raises(public.WholeProgramError, match="empty or declined"):
        public.lower_whole_program(SOURCE, tmp_path, declined, regions=regions,
                                   host_driver="python", host_optimization="baseline")
    assert not (tmp_path / "routed.mlir").exists()


def test_public_materialization_passes_original_policy_hook_without_backedge(tmp_path, monkeypatch):
    selected = []
    def policy(value):
        selected.append(value)
        return "baseline"
    monkeypatch.setattr(public, "_selected_host_optimization", policy)
    original = public.WholeProgramResult(Path("host.ll"), None, None, (), Path("routed.mlir"), Path("report.json"))
    def materialize(result, output, *, select_host_optimization, **options):
        assert result is original and output == tmp_path / "combined.mlir"
        assert select_host_optimization is policy
        assert options["cflags"] == ("--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d")
        select_host_optimization(options["host_optimization"])
        return output
    monkeypatch.setattr(hostlink, "materialize_combined_llvm_dialect", materialize)
    assert public.materialize_combined_llvm_dialect(original, tmp_path / "combined.mlir") == tmp_path / "combined.mlir"
    assert selected == [None]

@pytest.mark.parametrize("mode", ["off", "provided", "missing", "dependency", "identity"])
def test_worker_optional_pass_uses_only_selected_import_and_preserves_refusals(tmp_path, monkeypatch, mode):
    import builtins
    import json
    import sys
    from types import ModuleType

    from gemmini.backend import host_runtime_worker as worker

    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    work = tmp_path / "host"
    work.mkdir()
    response = tmp_path / "result.json"
    request = tmp_path / "request.json"
    feature = "scalar_window_reductions"
    request.write_text(json.dumps({"source": str(source), "work": str(work), "result": str(response),
        "options": {"features": [feature] if mode != "off" else []}}))
    monkeypatch.setattr(sys, "argv", ["worker", str(request)])
    name = "merlin.llvmlower.scalar_window_reduction"
    module = ModuleType(name)
    module.FEATURE = "wrong-feature" if mode == "identity" else feature
    prepared, imports, lowered = [], [], []
    selected = SOURCE + "\\n// selected test pass; no arithmetic is executed\\n"
    def prepare(text, *, enabled):
        assert text == SOURCE and enabled is True
        prepared.append(text)
        return selected, {"schema": "test-only", "rewrites": 1}
    module.prepare_scalar_window_reductions = prepare
    monkeypatch.setitem(sys.modules, name, module)
    original_import = builtins.__import__
    def selected_import(import_name, *args, **kwargs):
        if import_name == name:
            imports.append(import_name)
            if mode == "off":
                pytest.fail("unselected optional pass imported")
            if mode == "missing":
                raise ModuleNotFoundError("selected pass absent", name=name)
            if mode == "dependency":
                raise ModuleNotFoundError("selected pass dependency absent", name="selected_pass_dependency")
        return original_import(import_name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", selected_import)
    def compile_host(text, directory, **options):
        lowered.append(text)
        assert options["features"] == frozenset()
        assert options["targets"] == () and options["retain_llvm_dialect"] is False
        path = directory / "model.ll"
        path.write_text("; test-only product, not native execution")
        return SimpleNamespace(ll_path=path, stats={})
    monkeypatch.setattr(worker, "lower_model", compile_host)
    if mode in {"missing", "identity"}:
        with pytest.raises(ValueError, match="reviewed source overlay|different feature identity"):
            worker.main()
    elif mode == "dependency":
        with pytest.raises(ModuleNotFoundError) as failed:
            worker.main()
        assert failed.value.name == "selected_pass_dependency"
    else:
        worker.main()
        result = json.loads(response.read_text())
        assert lowered == [selected if mode == "provided" else SOURCE]
        assert prepared == ([SOURCE] if mode == "provided" else [])
        assert imports == ([name] if mode == "provided" else [])
        assert ("scalar_window_reductions" in result["stats"]) == (mode == "provided")
        if mode == "provided":
            assert (work / "model.scalar-windows.mlir").read_text() == selected
            assert result["stats"]["scalar_window_reductions"] == {"schema": "test-only", "rewrites": 1}
    if mode in {"missing", "dependency", "identity"}:
        assert not lowered and not prepared and not response.exists()
