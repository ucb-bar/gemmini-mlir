"""Pure regressions for selected Gemmini structural checks; no RTL extraction."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def checks(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "backend" / "rtl_checks.py"
    spec = importlib.util.spec_from_file_location("gemmini_structural_checks_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_default_facts", lambda target: {"mesh": [16, 16]})
    return module


def trace(classes):
    return {"source": "synthetic", "instructions": [
        {"index": index, "class": name, "funct": None, "decoded": {}}
        for index, name in enumerate(classes)
    ]}


GOOD = ["FENCE", "CONFIG_EX", "CONFIG_ST", "PRELOAD", "COMPUTE_PRELOADED", "MVOUT", "FENCE"]


def test_good_protocol_preserves_report_types_and_passes(checks):
    from merlin.targetgen.rtl_checks import Check, CheckReport

    report = checks.screen(trace(GOOD), target="explicit-synthetic-facts")
    assert isinstance(report, CheckReport)
    assert all(isinstance(check, Check) for check in report.checks)
    assert report.verdict == "ok"
    for identifier in ("T0.preload_before_compute", "T0.config_before_use", "T0.fence_bracket"):
        assert next(check for check in report.checks if check.id == identifier).status == "pass"


@pytest.mark.parametrize(("removed", "identifier"), [
    ("PRELOAD", "T0.preload_before_compute"),
    ("CONFIG_EX", "T0.config_before_use"),
    ("CONFIG_ST", "T0.config_before_use"),
    ("FENCE", "T0.fence_bracket"),
])
def test_protocol_negatives_remain_errors(checks, removed, identifier):
    report = checks.screen(trace([name for name in GOOD if name != removed]), target="explicit-synthetic-facts")
    finding = next(check for check in report.checks if check.id == identifier)
    assert finding.status == "fail"
    assert finding.severity == "error"
    assert report.verdict == "reject"


def test_store_tile_geometry_is_preserved_in_compiler_and_renderer(checks):
    capsule = {"inputs": [
        {"role": "input", "shape": [25, 16]},
        {"role": "weight", "shape": [16, 17]},
    ], "operation": {"op": "matmul"}}
    facts = {"facts": {"arrays": [{"name": "mesh", "rows": 16, "cols": 16}]}}
    assert checks.project_facts(facts) == {"mesh": [16, 16]}
    assert checks.expected_mvout_count(capsule, {"mesh": [16, 16]})[0] == 4
    compiled = checks.compile_trace_checks(facts, capsule)
    assert "MVOUT_COUNT 4{{$}}" in compiled
    assert "COMPUTE_PRESENT yes" in compiled
    rendered = checks.render_trace(trace(GOOD), facts)
    assert "MVOUT_COUNT 1\n" in rendered
    assert "COMPUTE_PRESENT yes\n" in rendered
