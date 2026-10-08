"""Provider-owned configuration only: no compiler or simulator execution."""

import os
import subprocess
import threading
from pathlib import Path

import pytest

from merlin.runtime.backends import base

KEYS = ("MERLIN_GEMMINI_GSIM_EMU", "MERLIN_GEMMINI_VERILATOR", "MERLIN_GEMMINI_GSIM_MAXCYCLES")


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native launch"))


@pytest.mark.parametrize("cap", [None, 27])
def test_pure_environment_policy_copies_without_process_mutation(runtime, cap):
    backend, paths = runtime
    source = {KEYS[0]: "wrong", KEYS[1]: "wrong", KEYS[2]: "17", "KEEP": "value"}
    before = dict(source)
    process = dict(os.environ)
    result = backend.runtime_environment(binaries=paths, gsim_max_cycles=cap, environment=source)
    assert source == before and dict(os.environ) == process
    assert result is not source and result["KEEP"] == "value"
    assert result[KEYS[0]] == str(paths["gsim"]) and result[KEYS[1]] == str(paths["verilator"])
    assert result.get(KEYS[2]) == (None if cap is None else str(cap))


@pytest.mark.parametrize("binaries", [{}, {"gsim": "missing"}, {"gsim": 1, "verilator": 2}])
def test_pure_environment_policy_rejects_invalid_binary_mapping(runtime, binaries):
    backend, _ = runtime
    before = dict(os.environ)
    with pytest.raises((TypeError, ValueError, OSError)):
        backend.runtime_environment(binaries=binaries, gsim_max_cycles=None, environment={})
    assert dict(os.environ) == before


@pytest.mark.parametrize("environment", [None, [], {"VALUE": 1}, {1: "value"}])
def test_pure_environment_policy_rejects_malformed_environment(runtime, environment):
    backend, paths = runtime
    with pytest.raises(ValueError, match="map strings"):
        backend.runtime_environment(binaries=paths, gsim_max_cycles=None, environment=environment)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(Path(__file__).resolve().parents[1]))
    backend = base.get_backend("gemmini")
    paths = {engine: tmp_path / engine for engine in ("gsim", "verilator")}
    for path in paths.values():
        path.write_bytes(b"synthetic binary bytes; never executed")
    return backend, paths


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("cap", [None, 1234])
@pytest.mark.parametrize("fail", [False, True])
def test_binds_both_engines_and_restores_even_on_refusal(runtime, monkeypatch, present, cap, fail):
    backend, paths = runtime
    for key in KEYS:
        if present:
            monkeypatch.setenv(key, "17")
        else:
            monkeypatch.delenv(key, raising=False)
    previous = {key: os.environ.get(key) for key in KEYS}
    try:
        with backend.pinned_runtime(binaries=paths, gsim_max_cycles=cap) as configured:
            assert configured is backend
            assert configured.gsim_path() == paths["gsim"]
            assert configured.verilator_path() == paths["verilator"]
            assert configured.gsim_max_cycles() == str(cap if cap is not None else configured.GSIM_MAX_CYCLES)
            if cap is None:
                assert KEYS[2] not in os.environ
            if fail:
                raise RuntimeError("simulated shared pin refusal")
    except RuntimeError as exc:
        assert fail and str(exc) == "simulated shared pin refusal"
    assert {key: os.environ.get(key) for key in KEYS} == previous


@pytest.mark.parametrize("cap", [True, False, 0, -1, "10", 1.5])
def test_invalid_cap_never_changes_environment(runtime, cap):
    backend, paths = runtime
    before = dict(os.environ)
    with pytest.raises(ValueError):
        with backend.pinned_runtime(binaries=paths, gsim_max_cycles=cap):
            pytest.fail("invalid configuration entered")
    assert dict(os.environ) == before


def test_nested_scopes_restore_outer_configuration(runtime, tmp_path):
    backend, paths = runtime
    alternate = tmp_path / "alternate"
    alternate.write_bytes(b"another inert file")
    before = dict(os.environ)
    with backend.pinned_runtime(binaries=paths, gsim_max_cycles=11):
        with backend.pinned_runtime(binaries={"gsim": alternate, "verilator": alternate}, gsim_max_cycles=None):
            assert backend.gsim_path() == alternate
            assert backend.verilator_path() == alternate
        assert backend.gsim_path() == paths["gsim"]
        assert backend.gsim_max_cycles() == "11"
    assert dict(os.environ) == before


def test_scopes_serialize_across_threads(runtime):
    backend, paths = runtime
    attempted, entered = threading.Event(), threading.Event()
    failures = []

    def second_scope():
        attempted.set()
        try:
            with backend.pinned_runtime(binaries=paths, gsim_max_cycles=22):
                entered.set()
                assert backend.gsim_max_cycles() == "22"
        except BaseException as exc:
            failures.append(exc)

    before = dict(os.environ)
    with backend.pinned_runtime(binaries=paths, gsim_max_cycles=11):
        worker = threading.Thread(target=second_scope)
        worker.start()
        assert attempted.wait(2)
        assert not entered.wait(0.05)
        assert backend.gsim_max_cycles() == "11"
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert entered.is_set() and not failures
    assert dict(os.environ) == before


@pytest.mark.parametrize("invalid", ["missing", "extra", "absent", "directory"])
def test_invalid_binary_configuration_refuses_before_mutation(runtime, tmp_path, invalid):
    backend, paths = runtime
    paths = dict(paths)
    if invalid == "missing":
        paths.pop("verilator")
    elif invalid == "extra":
        paths["other"] = paths["gsim"]
    else:
        paths["gsim"] = tmp_path / "absent" if invalid == "absent" else tmp_path
    before = dict(os.environ)
    with pytest.raises((ValueError, FileNotFoundError)):
        with backend.pinned_runtime(binaries=paths, gsim_max_cycles=None):
            pytest.fail("invalid configuration entered")
    assert dict(os.environ) == before
