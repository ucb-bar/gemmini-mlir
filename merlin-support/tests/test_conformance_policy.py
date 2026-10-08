"""OOT policy regression fixtures; no agent, simulator, board or network execution."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from gemmini_conformance import kernel_slot, workloads

ROOT = Path(__file__).resolve().parents[1]


def test_all_historical_rung_documents_are_identical():
    frozen = json.loads((ROOT / "tests/conformance_workloads.json").read_text())
    assert set(frozen["rung_document_sha256"]) == set(workloads.RUNGS) | set(workloads.QUANT_RUNGS)
    for name, digest in frozen["rung_document_sha256"].items():
        document = json.dumps(workloads.build(name), sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(document).hexdigest() == digest


def test_authoring_retains_visible_repair_then_once_only_heldout(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_slot, "build_prompt", lambda *_: "synthetic prompt")
    monkeypatch.setattr(
        kernel_slot,
        "_run_agent",
        lambda *_a, **_k: {
            "text": "```python\ndef generate_driver(cb): return 'synthetic C'\n```",
            "usage": {"tokens": 3},
        },
    )
    calls = []

    def certify(rung, *_):
        calls.append(rung)
        return {"correct": len(calls) != 1}

    monkeypatch.setattr(kernel_slot, "_certify", certify)
    result = kernel_slot.generate_kernel(workdir=tmp_path, rounds=3)
    assert result["success"] and result["round"] == 1
    assert calls == ["C0", "C0", "C1", "C4", "C4e"]
    assert result["usage"] == {"tokens": 3}


def test_cheat_scan_preserved():
    assert not kernel_slot._scan_cheat("def generate_driver(cb): return ''")
    assert "reference_outputs" in kernel_slot._scan_cheat("reference_outputs(cb)")


def test_underivable_selected_facts_refuse_without_native_fallback(monkeypatch, tmp_path):
    """Facts this provider does not pin are derived for it; a derivation that cannot run refuses by
    name, and nothing falls back to another provider's evidence."""
    from merlin.targetgen.rtl import facts
    from gemmini_conformance.support import backend

    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))

    def extract(path, target):
        raise RuntimeError("no elaboration on this host")

    monkeypatch.setattr(facts, "_dump_facts_for_kind", extract)
    facts.clear_resolution_cache()
    with pytest.raises(FileNotFoundError, match="deriving them failed"):
        backend().gemmini_codegen_mlir.emit_kernel_mlir(workloads.build("C0"))
    with pytest.raises(FileNotFoundError, match="deriving them failed"):
        kernel_slot.isa_reference()


@pytest.mark.parametrize("selected", [True, False])
def test_real_cli_help_and_wrong_provider_refusal(tmp_path, selected):
    env = dict(os.environ, PYTHONPATH=str(ROOT), MERLIN_TARGET_PATH=str(ROOT) if selected else "")
    # Refuse accidental reintroduction of native target evaluation imports, even when
    # they remain available in a migration checkout.
    script = """
import importlib.abc,runpy,sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.startswith('merlin.targetgen.eval.gemmini') or name=='merlin.targetgen.agent.gemmini_kernel_slot':
            raise AssertionError('native conformance fallback: '+name)
sys.meta_path.insert(0,NoNative())
sys.argv=['gemmini_conformance','--help']
runpy.run_module('gemmini_conformance',run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    if selected:
        assert result.returncode == 0, result.stderr
        assert "--summary-only" in result.stdout
    else:
        assert result.returncode != 0
        assert "requires this support provider" in result.stderr
