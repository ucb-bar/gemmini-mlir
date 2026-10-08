"""Pure resource routing/refusal checks; no compiler or simulator execution."""

import importlib
import importlib.util
import json

import pytest
from gemmini_conformance.support import ROOT, backend


def test_default_harness_is_provider_owned_and_never_chipyard(monkeypatch, tmp_path):
    implementation = importlib.import_module(backend().__name__ + ".gemmini")
    monkeypatch.delenv("MERLIN_GEMMINI_HARNESS_DIR", raising=False)
    monkeypatch.setattr(implementation, "chipyard_root", lambda: pytest.fail("implicit Chipyard fallback"))
    assert implementation.rocc_tests_dir() == ROOT / "resources/gemmini-rocc-tests"
    assert (implementation.rocc_tests_dir() / "include/gemmini.h").is_file()
    for relative in (
        "riscv-tests/env/encoding.h",
        "riscv-tests/benchmarks/common/crt.S",
        "riscv-tests/benchmarks/common/syscalls.c",
        "riscv-tests/benchmarks/common/test.ld",
        "rocc-software/src/xcustom.h",
    ):
        assert (implementation.rocc_tests_dir() / relative).is_file()
    monkeypatch.setattr(implementation, "__file__", str(tmp_path / "backend/gemmini.py"))
    assert implementation.rocc_tests_dir() == tmp_path / "resources/gemmini-rocc-tests"
    assert not implementation.rocc_tests_dir().exists()


def test_explicit_harness_override_is_not_replaced(monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_GEMMINI_HARNESS_DIR", str(tmp_path))
    assert backend().rocc_tests_dir() == tmp_path


def _conv():
    module = importlib.import_module(backend().__name__ + ".gemmini_loop_conv")
    module.derive_native_conv_contract.cache_clear()
    return module


def test_native_conv_without_derivable_facts_refuses(monkeypatch, tmp_path):
    """This provider ships no facts pin: they are derived for it.  Where they cannot be derived (no
    cache committed to this provider's contract, and no elaboration to extract from), the contract is
    UNKNOWN by name, never built on borrowed evidence."""
    from merlin.targetgen.rtl import facts

    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))

    def extract(path, target):
        raise RuntimeError("no elaboration on this host")

    monkeypatch.setattr(facts, "_dump_facts_for_kind", extract)
    facts.clear_resolution_cache()
    module = _conv()
    with pytest.raises(module.UnsupportedNativeConv, match="deriving them failed"):
        module.derive_native_conv_contract()


def test_explicit_facts_preserve_stale_hardware_refusal(monkeypatch, tmp_path):
    # Deliberately false evidence proves refusal, never hardware qualification.
    hw = tmp_path / "synthetic.hw"
    hw.write_text("synthetic changed hardware")
    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps(
            {
                "inputs": {"core_hw_sha256": "0" * 64},
                "facts": {"interfaces": [{"name": "funct_decode_table", "hw_source": str(hw)}]},
            }
        )
    )
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(facts))
    monkeypatch.delenv("MERLIN_GEMMINI_HARNESS_DIR", raising=False)
    module = _conv()
    with pytest.raises(module.UnsupportedNativeConv, match="stale core hardware identity"):
        module.derive_native_conv_contract()


def test_calibration_identity_and_inputs_are_provider_owned(monkeypatch):
    path = ROOT / "cost_model/calibrate.py"
    spec = importlib.util.spec_from_file_location("oot_calibration_resource_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("MERLIN_GEMMINI_HARNESS_DIR", raising=False)
    assert module.TARGET == "gemmini"
    assert module.STAGEF == ROOT / "cost_model/validation"
    assert (module.STAGEF / "resident_rhs_ablation.c").is_file()
    assert (module.STAGEF / "dispatch_batching_ablation.c").is_file()
    assert module.paths()["rocc"] == ROOT / "resources/gemmini-rocc-tests"


def test_resource_and_coefficient_bytes_preserved():
    inventory = json.loads((ROOT / "resource_migration.json").read_text())
    changed = {"cost_model/calibrate.py", "backend/gemmini.py", "backend/gemmini_loop_conv.py"}
    for row in inventory["files"]:
        if row["destination"] not in changed:
            assert row["source_sha256"] == row["destination_sha256"], row["destination"]
