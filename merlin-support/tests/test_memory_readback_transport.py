"""An explicit oracle export may add only its closed, byte-bound arguments."""

import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.runtime.backends import spike_model
from merlin.targetgen import rtl_engine_policy


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(Path(__file__).resolve().parents[1]))
    return base.get_backend("gemmini").gemmini


@pytest.fixture
def selected(tmp_path):
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"selected ELF bytes")
    out = tmp_path / "output.bin"
    regions = tmp_path / "regions.txt"
    regions.write_bytes(b"0x80002000 16\n")
    request = {
        "schema": "oracle_memory_readback_v1",
        "transport": "gsim_coherent_dump_v1",
        "output_path": str(out),
        "regions": [{"base": 0x80002000, "bytes": 16}],
        "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        "regions_path": str(regions),
    }
    return elf, out, regions, request


def test_closed_transport_declarations(backend):
    assert backend.memory_readback_transport("spike") == "htif_signature_v1"
    assert backend.memory_readback_transport("gsim") == "gsim_coherent_dump_v1"
    assert backend.memory_readback_transport("spike", policy_transport="coherent_packet_v1") == "htif_signature_v1"
    assert (
        backend.memory_readback_transport("gsim", policy_transport="coherent_packet_v1")
        == "gsim_coherent_packet_v1"
    )
    assert backend.memory_readback_transport("gsim", policy_transport="unknown") is None
    assert backend.memory_readback_transport("verilator") is None


def test_packet_request_selects_only_closed_two_region_gsim_command(monkeypatch, backend, selected):
    elf, out, regions, request = selected
    request["transport"] = "gsim_coherent_packet_v1"
    request["regions"] = [{"base": 0x80002000, "bytes": 8}, {"base": 0x80003000, "bytes": 4096}]
    regions.write_text("0x80002000 8\n0x80003000 4096\n", encoding="ascii")
    monkeypatch.setattr(backend, "_gsim_argv", lambda path: ["selected-gsim", str(path)])
    monkeypatch.setattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", "reentrant_per_thread_v1")
    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **_kwargs: nullcontext())

    def run(argv, **_kwargs):
        assert argv == [
            "selected-gsim", str(elf), f"+dump-regions={regions}",
            f"+dump-out={out}", "+dump-mode=coherent-packet",
        ]
        out.write_bytes(b"GSIMPKT1 complete fixture")
        return SimpleNamespace(returncode=0, stdout="DONE\n", stderr="")

    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend.run_elf(elf, simulator="gsim", memory_readback=request) == "DONE\n"
    out.unlink()
    request["regions"] = [{"base": 0x80004000, "bytes": 8}, {"base": 0x80002000, "bytes": 4096}]
    regions.write_text("0x80004000 8\n0x80002000 4096\n", encoding="ascii")
    assert backend.run_elf(elf, simulator="gsim", memory_readback=request) == "DONE\n"
    out.unlink()
    request["regions"] = [
        {"base": 0x90003000, "bytes": 8},
        {"base": 0x80002000, "bytes": 256 * 1024 * 1024},
    ]
    regions.write_text("0x90003000 8\n0x80002000 268435456\n", encoding="ascii")
    assert backend.run_elf(elf, simulator="gsim", memory_readback=request) == "DONE\n"
    out.unlink()
    request["regions"] = [{"base": 0x80004000, "bytes": 8}, {"base": 0x80002000, "bytes": 4096}]
    request["regions"][0]["bytes"] = 16
    regions.write_text("0x80004000 16\n0x80002000 4096\n", encoding="ascii")
    with pytest.raises(ValueError, match="packet"):
        backend.run_elf(elf, simulator="gsim", memory_readback=request)
    request["regions"][0]["bytes"] = 8
    request["regions"][1]["bytes"] = 256 * 1024 * 1024 + 1
    regions.write_text(f"0x80004000 8\n0x80002000 {request['regions'][1]['bytes']}\n", encoding="ascii")
    with pytest.raises(ValueError, match="bounded"):
        backend.run_elf(elf, simulator="gsim", memory_readback=request)
    monkeypatch.setattr(backend.subprocess, "run", lambda *_a, **_k: pytest.fail("native spawned"))
    with pytest.raises(ValueError, match="transport"):
        backend.run_elf(elf, simulator="spike", memory_readback=request)


def test_gsim_request_only_adds_fixed_arguments_inside_existing_slot(monkeypatch, backend, selected):
    elf, out, regions, request = selected
    monkeypatch.setattr(backend, "_gsim_argv", lambda path: ["selected-gsim", str(path), "+max-cycles=20"])
    monkeypatch.setattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", "reentrant_per_thread_v1")
    entered = False

    def slot(*, wait_timeout_s):
        assert wait_timeout_s == 7
        nonlocal entered
        entered = True
        return nullcontext()

    def run(argv, **kwargs):
        assert entered and kwargs["timeout"] == 7
        assert argv == [
            "selected-gsim", str(elf), "+max-cycles=20", f"+dump-regions={regions}",
            f"+dump-out={out}", "+dump-mode=coherent",
        ]
        out.write_bytes(b"completed binary dump")
        return SimpleNamespace(returncode=0, stdout="DONE\n", stderr="")

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", slot)
    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend.run_elf(elf, simulator="gsim", timeout=7, memory_readback=request) == "DONE\n"


def test_gsim_request_preserves_two_or_three_exact_physical_regions(monkeypatch, backend, selected):
    elf, out, regions, request = selected
    monkeypatch.setattr(backend, "_gsim_argv", lambda path: ["selected-gsim", str(path)])
    monkeypatch.setattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", "reentrant_per_thread_v1")
    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **_kwargs: nullcontext())

    def run(argv, **_kwargs):
        assert argv == [
            "selected-gsim", str(elf), f"+dump-regions={regions}",
            f"+dump-out={out}", "+dump-mode=coherent",
        ]
        out.write_bytes(b"completed binary dump")
        return SimpleNamespace(returncode=0, stdout="DONE\n", stderr="")

    monkeypatch.setattr(backend.subprocess, "run", run)
    for sizes in ((16, 64), (16, 64, 8)):
        current = 0x80002000
        request["regions"] = []
        for size in sizes:
            request["regions"].append({"base": current, "bytes": size})
            current += size
        regions.write_text("".join(f"{row['base']:#x} {row['bytes']}\n" for row in request["regions"]), encoding="ascii")
        assert backend.run_elf(elf, simulator="gsim", memory_readback=request) == "DONE\n"
        out.unlink()
    request["regions"].reverse()
    regions.write_text("".join(f"{row['base']:#x} {row['bytes']}\n" for row in request["regions"]), encoding="ascii")
    with pytest.raises(ValueError, match="overlap|unordered"):
        backend.run_elf(elf, simulator="gsim", memory_readback=request)


def test_spike_request_uses_one_byte_signature_without_other_flags(monkeypatch, backend, selected):
    elf, out, _regions, request = selected
    request["transport"] = "htif_signature_v1"
    request["regions_path"] = None
    monkeypatch.setattr(backend, "spike_path", lambda: Path("/selected/spike"))
    monkeypatch.setattr(backend, "spike_extension", lambda: (("--extension=gemmini",), Path("/selected/lib")))
    monkeypatch.setattr(spike_model, "declared_memory", lambda _elf: None)
    monkeypatch.setattr(spike_model, "declared_harts", lambda _elf: None)
    monkeypatch.setattr(spike_model, "declared_isa", lambda _elf: None)

    def run(argv, **_kwargs):
        assert argv == [
            "/selected/spike", "--extension=gemmini", f"+signature={out}",
            "+signature-granularity=1", str(elf),
        ]
        out.write_bytes(b"00\n" * 16)
        return SimpleNamespace(returncode=0, stdout="DONE\n", stderr="")

    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend.run_elf(elf, simulator="spike", memory_readback=request) == "DONE\n"


def test_spike_declared_memory_bounds_are_not_ignored(monkeypatch, backend, selected):
    elf, _out, _regions, request = selected
    request["transport"] = "htif_signature_v1"
    request["regions_path"] = None
    monkeypatch.setattr(spike_model, "declared_memory", lambda _elf: (0x80000000, 0x1000))
    monkeypatch.setattr(backend.subprocess, "run", lambda *_a, **_k: pytest.fail("native spawned"))
    with pytest.raises(ValueError, match="declared simulator memory"):
        backend.run_elf(elf, simulator="spike", memory_readback=request)


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda r: r.update(schema="unknown"), "transport"),
        (lambda r: r.update(transport="htif_signature_v1"), "transport"),
        (lambda r: r.update(unexpected="+evil"), "field"),
        (lambda r: r.update(elf_sha256="0" * 64), "ELF bytes"),
        (lambda r: r["regions"].append({"base": 0x80002008, "bytes": 16}), "overlap"),
        (lambda r: r["regions"][0].update(bytes=True), "bounded"),
        (lambda r: r["regions"][0].update(base=-1), "bounded"),
    ],
)
def test_invalid_requests_spawn_nothing(monkeypatch, backend, selected, change, reason):
    elf, _out, _regions, request = selected
    change(request)
    monkeypatch.setattr(backend.subprocess, "run", lambda *_a, **_k: pytest.fail("native spawned"))
    with pytest.raises(ValueError, match=reason):
        backend.run_elf(elf, simulator="gsim", memory_readback=request)


def test_manifest_and_output_paths_are_fail_closed(monkeypatch, backend, selected):
    elf, out, regions, request = selected
    monkeypatch.setattr(backend.subprocess, "run", lambda *_a, **_k: pytest.fail("native spawned"))
    regions.write_text("0x80002000 15\n", encoding="ascii")
    with pytest.raises(ValueError, match="manifest"):
        backend.run_elf(elf, simulator="gsim", memory_readback=request)
    regions.write_text("0x80002000 16\n", encoding="ascii")
    out.write_bytes(b"stale")
    with pytest.raises(ValueError, match="not fresh"):
        backend.run_elf(elf, simulator="gsim", memory_readback=request)


@pytest.mark.parametrize("mutation", ["elf", "manifest", "request", "missing_output"])
def test_postrun_identity_or_missing_output_refuses(monkeypatch, backend, selected, mutation):
    elf, out, regions, request = selected
    monkeypatch.setattr(backend, "_gsim_argv", lambda path: ["selected-gsim", str(path)])
    monkeypatch.setattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", "reentrant_per_thread_v1")
    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **_kwargs: nullcontext())

    def run(_argv, **_kwargs):
        if mutation != "missing_output":
            out.write_bytes(b"completed binary dump")
        if mutation == "elf":
            elf.write_bytes(b"changed ELF bytes")
        elif mutation == "manifest":
            regions.write_text("0x80002000 8\n", encoding="ascii")
        elif mutation == "request":
            request["regions"][0]["bytes"] = 8
        return SimpleNamespace(returncode=0, stdout="DONE\n", stderr="")

    monkeypatch.setattr(backend.subprocess, "run", run)
    with pytest.raises(ValueError, match="changed|output"):
        backend.run_elf(elf, simulator="gsim", memory_readback=request)
