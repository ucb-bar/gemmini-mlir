"""The target edge supplies source-bound backend hooks and simulator options."""

import json

from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen.core_aten_device import load_execution_provider, selected_facts

PROVIDER = Path(__file__).resolve().parents[1] / "core_aten/execution_provider.py"


def test_runner_settings_follow_the_selected_spike_installation(monkeypatch):
    from merlin.runtime.backends import spike

    provider = load_execution_provider("gemmini", PROVIDER)
    monkeypatch.setattr(spike, "spike_path", lambda: "/selected/env/riscv-tools/bin/spike")
    options = provider.runner_options()
    assert options["extension"] == "gemmini"
    assert options["extlib"] == Path("/selected/env/riscv-tools/lib/libgemmini.so")
    assert options["path_prepend"] == [Path("/selected/env/bin")]


@pytest.mark.parametrize('readout_options', [None, dict(scheme=dict(readout_strategy='f32_product_rne'))])
def test_provider_binds_catalog_and_derives_precision_from_selected_facts(tmp_path, monkeypatch, readout_options):
    provider = load_execution_provider("gemmini", PROVIDER)
    (tmp_path / "model.mlir").write_text("builtin.module {}")

    def builder(*args):
        return None

    backend = SimpleNamespace(
        build_catalog=lambda source: (None, {"covered_contractions": 1}), merlin_builder=lambda llvm_bin: builder, final_elf_audit=lambda elf: None
    )
    original_load = provider.load_module
    monkeypatch.setattr(provider, "load_module", lambda package, name, **kw:
                        backend if name == "mlir_oot.golden_device_catalog" else original_load(package, name, **kw))
    original_core = provider._core_aten_module
    monkeypatch.setattr(provider, "_core_aten_module", lambda package, name: original_core(PROVIDER.parents[2], name))
    fixture = Path(__file__).resolve().parent / "fixtures/core_aten_device/facts.json"
    eligible = [(None, SimpleNamespace(dtypes=("i8", "i8", "i32")))]
    with selected_facts("synthetic_core_aten", fixture) as document:
        route = provider.routing(
            tmp_path, target="synthetic_core_aten", package=tmp_path, facts=document, eligible=eligible, readout_options=readout_options
        )
        assert (route.operand_dtype, route.accum_dtype) == ("i8", "i32")
        assert route.select is None and route.catalog_builder is not builder
        # Even the legacy route without a policy receipt must go through admission.
        manifest, obj = route.catalog_builder(tmp_path / 'model.mlir', tmp_path / 'work')
        assert obj is None
        assert json.loads(manifest.read_text())['status'] == 'declined'
        assert provider.routing(
            tmp_path,
            target="synthetic_core_aten",
            package=tmp_path,
            facts=document,
            eligible=[(None, SimpleNamespace(dtypes=("f32", "f32", "f32")))],
        ) is None
        backend.build_catalog = lambda source: (None, {"covered_contractions": 0})
        assert (
            provider.routing(
                tmp_path, target="synthetic_core_aten", package=tmp_path, facts=document, eligible=eligible, readout_options=readout_options
            )
            is None
        )


def test_other_targets_cannot_load_this_provider():
    path = PROVIDER
    with pytest.raises((ValueError, FileNotFoundError), match="selected target|no capability contract"):
        load_execution_provider("synthetic_core_aten", path)


@pytest.mark.parametrize("m,n,k", [(3, 2, 1), (17, 19, 31), (3, 2, 32), (33, 35, 65)])
def test_actual_catalog_schedule_bounds_independent_ws_primitives(m, n, k):
    provider = load_execution_provider("gemmini", PROVIDER)
    package = PROVIDER.parents[2]
    source = f"""builtin.module {{
      func.func @forward(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
        %c = arith.constant 0 : i32
        %e = tensor.empty() : tensor<{m}x{n}xi32>
        %z = linalg.fill ins(%c : i32) outs(%e : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
        %r = linalg.matmul ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>)
          outs(%z : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
        func.return %r : tensor<{m}x{n}xi32>
      }}
    }}"""
    policy = dict(
        numerical_semantics=dict(
            accumulator_dtype="i32",
            internal_arithmetic=dict(
                signed_operand_bits=8,
                mac_result_bits=20,
                mac_result_overflow="wrap_to_20_bits",
                full_operation_overflow_policy="bounded_exact_requires_each_partial_sum",
            ),
        )
    )
    proof = provider.admit_numeric_schedule(source, package=package, numeric_policy=policy)
    (kernel,) = proof["kernels"]
    assert max(kernel["primitive_k"]) <= kernel["padded_primitive_k"] <= 31
    assert sum(kernel["k_split"]) == k
    assert kernel["max_partial_sum_bound"] <= 524287
    assert kernel["initialized_accumulator_rows"]
    if k > kernel["padded_primitive_k"]:
        assert kernel["initialized_accumulator_rows"] == kernel["accumulated_rows"]
    policy["numerical_semantics"]["internal_arithmetic"].update(mac_result_bits=8, mac_result_overflow="wrap_to_8_bits")
    with pytest.raises(ValueError, match="padded mesh K"):
        provider.admit_numeric_schedule(source, package=package, numeric_policy=policy)
