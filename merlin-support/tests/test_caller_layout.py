"""Public caller layouts are exactly the bytes and readback the selected renderer uses."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.commandbuffer import materialize_inputs
from merlin.targetgen.contract.build_service import load_build_package

SUPPORT = Path(__file__).resolve().parents[1]


def _command():
    return {
        "target": "gemmini",
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "A", "access": "read"}, {"tensor": "Y", "access": "write"}],
            "outputs": ["Y"],
        },
        "tensors": {
            "A": {"shape": [2, 3], "dtype": "i8", "role": "input"},
            "Y": {"shape": [2, 3], "dtype": "i8", "role": "output"},
        },
        "commands": [],
        "params": {},
    }


def test_legacy_layout_projection_matches_the_same_rendered_caller():
    build = load_build_package(SUPPORT / "build_support/__init__.py")
    cb = _command()

    def ceil(value):
        return build.format.ceil_dim(value, 4)

    layout = build.describe_whole_program_layout(cb, legacy_helpers=(ceil, build.format.buffer_extent))
    source = build.render_whole_program(
        cb,
        inputs={"A": [[1, 2, 3], [4, 5, 6]]},
        legacy_helpers=(ceil, build.format.pad_rowmajor, build.format.buffer_extent, materialize_inputs),
    )
    assert layout["policy"] == {"mode": "legacy_aligned_row_major_v1", "row_alignment_elements": 4}
    assert layout["tensors"][0]["logical_strides_elements"] == [4, 1]
    assert layout["tensors"][0]["physical_extents"] == [4, 4]
    assert "T_A[16]" in source and "{1,2,3,0,4,5,6,0" in source
    assert "T_Y[i * 4 + j]" in source


def test_explicit_grouped_layout_uses_the_same_checked_encoding_as_renderer():
    build = load_build_package(SUPPORT / "build_support/__init__.py")
    cb = _command()
    cb["params"]["storage_encodings"] = {
        "A": GroupedAxesStorage((2, 3), "i8", ((0,), (1,)), (2, 3), (5, 1), 10).to_dict(),
        "Y": GroupedAxesStorage((2, 3), "i8", ((0,), (1,)), (2, 3), (4, 1), 8).to_dict(),
    }
    layout = build.describe_whole_program_layout(cb)
    source = build.render_whole_program(cb, inputs={"A": [[1, 2, 3], [4, 5, 6]]})
    assert layout["policy"] == {"mode": "declared_grouped_axes_storage_v1"}
    assert [row["logical_strides_elements"] for row in layout["tensors"]] == [[5, 1], [4, 1]]
    assert "T_A[10]" in source and "{1,2,3,0,0,4,5,6,0,0}" in source
    assert "T_Y[8]" in source and "T_Y[0 + (((i * 3 + j) / 3) % 2) * 4" in source


def test_explicit_compact_rows_do_not_inherit_legacy_padding():
    build = load_build_package(SUPPORT / "build_support/__init__.py")
    cb = _command()
    cb["params"]["storage_encodings"] = {
        name: GroupedAxesStorage((2, 3), "i8", ((0,), (1,)), (2, 3), (3, 1), 6).to_dict() for name in ("A", "Y")
    }
    layout = build.describe_whole_program_layout(cb)
    source = build.render_whole_program(cb, inputs={"A": [[1, 2, 3], [4, 5, 6]]})
    assert [row["logical_strides_elements"] for row in layout["tensors"]] == [[3, 1], [3, 1]]
    assert all(row["storage_elements"] == 6 for row in layout["tensors"])
    assert "T_A[6]" in source and "{1,2,3,4,5,6}" in source
    assert "T_Y[6]" in source


def test_selected_provider_refuses_wrong_facts_and_reports_only_layout():
    from backend import gemmini

    cb = _command()
    facts = {"inputs": {"target": "gemmini"}, "facts": {"arrays": [{"name": "mesh", "cols": 16}]}}
    layout = gemmini.describe_caller_layout(cb, target="gemmini", facts=facts)
    assert layout["policy"]["row_alignment_elements"] == 16
    assert layout["tensors"][0]["logical_strides_elements"] == [16, 1]
    assert not any("value" in row or "weight" in row for row in layout["tensors"])
    altered = copy.deepcopy(facts)
    altered["facts"]["arrays"][0]["cols"] = 8
    with pytest.raises(Exception, match="geometry"):
        gemmini.describe_caller_layout(cb, target="gemmini", facts=altered)


def test_host_receipt_uses_actual_selected_renderer_and_exact_public_bytes(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback.caller_layout import inspect_caller_layout

    monkeypatch.setenv("MERLIN_TARGET_PATH", str(SUPPORT))
    submission = tmp_path / "submission"
    submission.mkdir()
    command = submission / "command_buffer.json"
    command.write_text(json.dumps(_command()))
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"inputs": {"target": "gemmini"}, "facts": {"arrays": [{"name": "mesh", "cols": 16}]}}))
    receipt = inspect_caller_layout(
        submission=submission, command_buffer_member=command.name, target="gemmini", facts_path=facts
    )
    assert receipt["status"] == "layout_only"
    assert receipt["policy"] == {"mode": "legacy_aligned_row_major_v1", "row_alignment_elements": 16}
    assert receipt["tensors"][0]["logical_strides_elements"] == [16, 1]
    assert len(receipt["provider_sha256"]) == len(receipt["core_layout_sha256"]) == 64
    assert "all_pass" not in receipt
    explicit = _command()
    explicit["params"]["storage_encodings"] = {
        name: GroupedAxesStorage((2, 3), "i8", ((0,), (1,)), (2, 3), (3, 1), 6).to_dict() for name in ("A", "Y")
    }
    command.write_text(json.dumps(explicit))
    compact = inspect_caller_layout(
        submission=submission, command_buffer_member=command.name, target="gemmini", facts_path=facts
    )
    assert compact["policy"] == {"mode": "declared_grouped_axes_storage_v1"}
    assert compact["tensors"][0]["logical_strides_elements"] == [3, 1]
    assert compact["command_buffer_sha256"] != receipt["command_buffer_sha256"]
