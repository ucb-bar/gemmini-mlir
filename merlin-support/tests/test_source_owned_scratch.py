"""The pure caller allocates source-owned scratch, never invents its initial values."""

import copy
from pathlib import Path

import pytest

from merlin.runtime.commandbuffer import materialize_inputs
from merlin.targetgen.contract.build_service import load_build_package

SUPPORT = Path(__file__).resolve().parents[1]


def _render(command, *, scratch):
    build = load_build_package(SUPPORT / "build_support/__init__.py")
    return build.render_whole_program(
        command,
        inputs={"arg0": [[3, 4], [5, 6]]},
        source_owned_mutables=scratch,
        legacy_helpers=(
            lambda value: build.format.ceil_dim(value, 4),
            build.format.pad_rowmajor,
            build.format.buffer_extent,
            materialize_inputs,
        ),
    )


@pytest.fixture
def command():
    return {
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": "arg0", "access": "read"},
                {"tensor": "tmp9", "access": "readwrite"},
                {"tensor": "tmp_host", "access": "readwrite"},
                {"tensor": "out", "access": "write"},
            ],
            "outputs": ["out"],
        },
        "tensors": {
            "arg0": {"shape": [2, 2], "dtype": "i8", "role": "input"},
            "tmp9": {"shape": [2, 2], "dtype": "i8", "role": "intermediate"},
            "tmp_host": {"shape": [2, 2], "dtype": "i8", "role": "intermediate"},
            "out": {"shape": [2, 2], "dtype": "i8", "role": "output"},
        },
        "commands": [{"opcode": "MATMUL", "operands": {"dst": "tmp9", "lhs": "arg0", "rhs": "arg0"}}],
    }


def test_command_and_host_produced_scratch_are_allocated_without_host_values(command):
    source = _render(command, scratch=("tmp9", "tmp_host"))
    declarations = [line for line in source.splitlines() if line.startswith("static ")]
    assert len([line for line in declarations if "T_arg0[" in line and " = {" in line]) == 1
    for name in ("tmp9", "tmp_host"):
        assert len([line for line in declarations if f"T_{name}[" in line and " = {" not in line]) == 1
    assert "gemmini_kernel((void*)T_arg0, (void*)T_tmp9, (void*)T_tmp_host, (void*)T_out)" in source


@pytest.mark.parametrize("scratch", [(), ("tmp9",), ("tmp9", "tmp9"), ("arg0", "tmp9", "tmp_host")])
def test_incomplete_duplicate_or_entry_scratch_roster_refuses(command, scratch):
    with pytest.raises(RuntimeError, match="scratch"):
        _render(command, scratch=scratch)


def test_readwrite_output_carry_refuses_even_with_scratch_roster(command):
    changed = copy.deepcopy(command)
    changed["kernel_abi"]["args"][-1]["access"] = "readwrite"
    with pytest.raises(RuntimeError, match="scratch"):
        _render(changed, scratch=("tmp9", "tmp_host"))


def test_captured_leaf_roster_cannot_omit_a_real_entry(command):
    build = load_build_package(SUPPORT / "build_support/__init__.py")
    with pytest.raises(build.CodegenError, match="scratch"):
        build.render_whole_program(command, inputs={}, source_owned_mutables=("tmp9", "tmp_host"))
