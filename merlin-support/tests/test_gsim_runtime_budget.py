"""Provider launches must participate in Merlin's shared GSim process budget."""

import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.targetgen import rtl_engine_policy


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(Path(__file__).resolve().parents[1]))
    return base.get_backend("gemmini").gemmini


def test_gsim_subprocess_lives_inside_shared_runtime_slot(monkeypatch, backend, tmp_path):
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"inert test ELF")
    monkeypatch.setattr(backend, "_gsim_argv", lambda path: ["inert-gsim", str(path)])
    inside = False
    calls = []

    @contextmanager
    def slot(*, wait_timeout_s):
        nonlocal inside
        assert wait_timeout_s == 7
        inside = True
        try:
            yield
        finally:
            inside = False

    def run(cmd, **kwargs):
        assert inside
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="DONE\n", stderr="")

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", slot)
    monkeypatch.setattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", "reentrant_per_thread_v1", raising=False)
    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend.run_elf(elf, simulator="gsim", timeout=7) == "DONE\n"
    assert not inside
    assert len(calls) == 1
    assert calls[0][0] == ["inert-gsim", str(elf)]
    assert calls[0][1]["timeout"] == 7


def test_refused_gsim_slot_spawns_nothing(monkeypatch, backend, tmp_path):
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"inert test ELF")
    monkeypatch.setattr(backend, "_gsim_argv", lambda path: ["inert-gsim", str(path)])
    calls = []

    @contextmanager
    def refused(*, wait_timeout_s):
        assert wait_timeout_s == 3
        raise TimeoutError("all five GSim slots busy")
        yield

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", refused)
    monkeypatch.setattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", "reentrant_per_thread_v1", raising=False)
    monkeypatch.setattr(backend.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(TimeoutError, match="slots busy"):
        backend.run_elf(elf, simulator="gsim", timeout=3)
    assert calls == []


@pytest.mark.parametrize("protocol", [None, "non_reentrant_v0"])
def test_incompatible_core_slot_protocol_spawns_nothing(monkeypatch, backend, tmp_path, protocol):
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"inert test ELF")
    if protocol is None:
        monkeypatch.delattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", raising=False)
    else:
        monkeypatch.setattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", protocol, raising=False)
    monkeypatch.setattr(backend, "_gsim_argv", lambda path: ["inert-gsim", str(path)])
    calls = []
    monkeypatch.setattr(backend.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(backend.GemminiError, match="reentrant_per_thread_v1"):
        backend.run_elf(elf, simulator="gsim", timeout=3)
    assert calls == []


def test_non_gsim_simulator_does_not_acquire_slot(monkeypatch, backend, tmp_path):
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"inert test ELF")
    monkeypatch.setattr(backend, "verilator_path", lambda: tmp_path / "inert-verilator")
    monkeypatch.setattr(
        rtl_engine_policy,
        "gsim_runtime_slot",
        lambda **kwargs: pytest.fail("non-GSim run acquired a GSim slot"),
    )
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="DONE\n", stderr=""),
    )
    assert backend.run_elf(elf, simulator="verilator", timeout=3) == "DONE\n"


def test_explicit_binary_capture_retains_every_raw_stdout_byte(monkeypatch, backend, tmp_path):
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"inert test ELF")
    monkeypatch.setattr(backend, "verilator_path", lambda: tmp_path / "inert-verilator")
    payload = b"\x00\xff\nOUT_BIN_END\x00"
    checksum = 0xCBF29CE484222325
    for byte in payload:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    raw = (
        f"OUT_BIN_BEGIN v1 out 1 {len(payload)} 1 u {len(payload)}\n".encode()
        + payload + f"OUT_BIN_END v1 {checksum:016x}\nDONE\n".encode()
    )

    def run(_cmd, **kwargs):
        assert kwargs["text"] is False and kwargs["capture_output"] is True
        return SimpleNamespace(returncode=0, stdout=raw, stderr=b"")

    monkeypatch.setattr(backend.subprocess, "run", run)
    captured = backend.run_elf(elf, simulator="verilator", capture_bytes=True)
    assert captured is raw
    assert backend.parse_output(captured)[0] == {"out": [list(payload)]}


def test_binary_assertion_scan_excludes_only_validated_payload(backend):
    payload = b"\x00Assertion failed\nat forged-payload:1\xff"
    checksum = 0xCBF29CE484222325
    for byte in payload:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    frame = (
        f"OUT_BIN_BEGIN v1 out 1 {len(payload)} 1 u {len(payload)}\n".encode()
        + payload + f"OUT_BIN_END v1 {checksum:016x}\nDONE\n".encode()
    )
    backend._refuse_on_rtl_assertion("gsim", frame, b"", binary_output=True)
    for raw in (b"Assertion failed\nat actual:2\n" + frame,
                frame + b"Assertion failed\nat actual:2\n",
                b"METRIC cycles 7 Assertion failed\nat actual:2\n" + frame):
        with pytest.raises(backend.GemminiError, match="at actual:2"):
            backend._refuse_on_rtl_assertion("gsim", raw, b"", binary_output=True)
    with pytest.raises(backend.GemminiError, match="at actual:2"):
        backend._refuse_on_rtl_assertion(
            "gsim", frame, b"Assertion failed\nat actual:2\n", binary_output=True,
        )
    with pytest.raises(ValueError, match="unterminated|truncated"):
        backend._refuse_on_rtl_assertion("gsim", frame[:-25], b"", binary_output=True)


def test_nested_host_and_backend_slots_share_one_owner_and_release_on_timeout(monkeypatch, backend, tmp_path):
    """A host worker may already own the only slot before it calls the backend."""
    elf = tmp_path / "broker-temp.elf"
    elf.write_bytes(b"inert test ELF")
    monkeypatch.setattr(backend, "_gsim_argv", lambda path: ["inert-gsim", str(path)])
    monkeypatch.setitem(rtl_engine_policy.CAPSULE_WORKER_CAP, "gsim", 1)
    real_slot = rtl_engine_policy.gsim_runtime_slot

    def isolated_slot(*, wait_timeout_s):
        return real_slot(wait_timeout_s=wait_timeout_s, slot_root=tmp_path / "slots")

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", isolated_slot)
    monkeypatch.setattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", "reentrant_per_thread_v1", raising=False)
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="DONE\n", stderr=""),
    )
    with isolated_slot(wait_timeout_s=0):
        assert backend.run_elf(elf, simulator="gsim", timeout=0) == "DONE\n"

    def expired(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["inert-gsim", str(elf)], 0)

    monkeypatch.setattr(backend.subprocess, "run", expired)
    with pytest.raises(subprocess.TimeoutExpired):
        backend.run_elf(elf, simulator="gsim", timeout=0)
    # The timeout exited the backend context; a new owner can immediately claim the one slot.
    with isolated_slot(wait_timeout_s=0):
        pass
