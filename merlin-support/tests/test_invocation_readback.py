"""A trusted output transport changes the caller, never the selected capsule."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.commandbuffer import materialize_inputs
from merlin.runtime.out_packet import ExpectedOutput, out_bin_packet_capacity
from merlin.targetgen.contract.build_service import load_build_package
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, FULL_VALUES_BIN, ReadbackPolicy

SUPPORT = Path(__file__).resolve().parents[1]


def _command():
    return {
        "target": "gemmini",
        "commands": [],
        "params": {"console_value_cap": 1},
        "tensors": {
            "arg": {"shape": [2, 3], "dtype": "i8", "role": "input"},
            "out": {"shape": [2, 3], "dtype": "i8", "role": "output"},
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "arg", "access": "read"}, {"tensor": "out", "access": "write"}],
            "outputs": ["out"],
        },
    }


def _render(command, *, policy=None):
    build = load_build_package(SUPPORT / "build_support/__init__.py")
    return build.render_whole_program(
        command,
        inputs={"arg": [[1, -2, 3], [4, 5, -6]]},
        legacy_helpers=(
            lambda value: build.format.ceil_dim(value, 4),
            build.format.pad_rowmajor,
            build.format.buffer_extent,
            materialize_inputs,
        ),
        readback_policy=policy,
    )


def test_full_values_override_only_cap_transport_and_do_not_mutate_capsule():
    command = _command()
    saved = copy.deepcopy(command)
    legacy = _render(command)
    explicit_legacy = _render(command, policy=None)
    full = _render(command, policy=ReadbackPolicy(FULL_VALUES_B64))
    assert legacy == explicit_legacy
    assert "OUTSUM out" in legacy and "OUT_B64_BEGIN" not in legacy
    assert '"out_b64.h"' in full and "OUT_B64_BEGIN v1 out" in full
    assert "OUTSUM out" not in full and "merlin_out_b64_finish" in full
    assert "merlin_out_b64_words_i32" not in full
    assert "merlin_out_b64_word(&merlin_out" in full
    assert command == saved


def test_binary_full_values_use_length_writer_without_changing_b64_or_capsule():
    command = _command()
    saved = copy.deepcopy(command)
    b64 = _render(command, policy=ReadbackPolicy(FULL_VALUES_B64))
    binary = _render(command, policy=ReadbackPolicy(FULL_VALUES_BIN))
    assert '"out_bin.h"' in binary and '"out_b64.h"' in binary
    assert "OUT_BIN_BEGIN v1 out" in binary and "OUT_BIN_END v1" in binary
    assert "merlin_out_bin_word(&merlin_out" in binary
    assert "merlin_out_bin_finish(&merlin_out)" in binary
    assert "printbuf);" in binary and "OUTSUM" not in binary
    assert "OUT_B64_BEGIN" not in binary
    assert b64 == _render(command, policy=ReadbackPolicy(FULL_VALUES_B64))
    assert command == saved


def test_coherent_dump_uses_same_caller_storage_and_call_with_one_signature_region():
    command = _command()
    saved = copy.deepcopy(command)
    legacy = _render(command)
    coherent = _render(command, policy=ReadbackPolicy("coherent_dump_v1"))
    for unchanged in (
        "T_arg[",
        "T_out[",
        "gemmini_kernel(",
        "gemmini_fence();",
        'printf("METRIC cycles',
        'printf("METRIC cycle_window_gemmini_region 1',
        'printf("DONE\\n");',
    ):
        assert unchanged in legacy and unchanged in coherent
    assert '"out_bin.h"' not in coherent and '"out_b64.h"' not in coherent
    assert "OUT " not in coherent and "OUTSUM" not in coherent
    assert "OUT_B64" not in coherent and "OUT_BIN" not in coherent
    assert '.set begin_signature, T_out\\n' in coherent
    assert '.set end_signature, T_out+16\\n' in coherent  # selected 4x4 padded i8 allocation
    assert command == saved


@pytest.mark.parametrize("dtype", ["i1", "f16", "bf16", "f64"])
def test_coherent_dump_refuses_unreviewed_physical_output_types(dtype):
    command = _command()
    command["tensors"]["out"]["dtype"] = dtype
    with pytest.raises(RuntimeError, match="coherent dump"):
        _render(command, policy=ReadbackPolicy("coherent_dump_v1"))


def test_coherent_dump_proposes_one_window_for_two_and_three_declared_outputs():
    command = _command()
    command["tensors"]["other"] = {"shape": [2, 3], "dtype": "i32", "role": "output"}
    command["kernel_abi"]["args"].append({"tensor": "other", "access": "write"})
    with pytest.raises(RuntimeError, match="coherent dump"):
        _render(command, policy=ReadbackPolicy("coherent_dump_v1"))
    command["kernel_abi"]["outputs"].append("other")
    two = _render(command, policy=ReadbackPolicy("coherent_dump_v1"))
    assert "T_out[" in two and "T_other[" in two
    assert '.set begin_signature, T_other\\n' in two
    assert '.set end_signature, T_out+16\\n' in two
    assert "OUT_B64" not in two and "OUT_BIN" not in two
    command["kernel_abi"]["outputs"].reverse()
    assert _render(command, policy=ReadbackPolicy("coherent_dump_v1")) == two
    command["kernel_abi"]["outputs"].reverse()
    command["tensors"]["third"] = {"shape": [1, 4], "dtype": "f32", "role": "output"}
    command["kernel_abi"]["args"].append({"tensor": "third", "access": "write"})
    command["kernel_abi"]["outputs"].append("third")
    three = _render(command, policy=ReadbackPolicy("coherent_dump_v1"))
    assert "T_third[" in three
    assert '.set begin_signature, T_third\\n' in three
    assert '.set end_signature, T_out+16\\n' in three
    command["kernel_abi"]["outputs"][:] = ["other", "third", "out"]
    assert _render(command, policy=ReadbackPolicy("coherent_dump_v1")) == three
    command["kernel_abi"]["outputs"].append("other")
    with pytest.raises(RuntimeError, match="coherent dump"):
        _render(command, policy=ReadbackPolicy("coherent_dump_v1"))


def test_coherent_packet_preserves_output_storage_and_full_raw_spike_signature():
    command = _command()
    command["tensors"]["other"] = {"shape": [2, 3], "dtype": "i32", "role": "output"}
    command["kernel_abi"]["args"].append({"tensor": "other", "access": "write"})
    command["kernel_abi"]["outputs"].append("other")
    saved = copy.deepcopy(command)
    physical = _render(command, policy=ReadbackPolicy("coherent_dump_v1"))
    packet = _render(command, policy=ReadbackPolicy("coherent_packet_v1"))
    for unchanged in (
        "T_arg[", "T_out[", "T_other[", "gemmini_kernel(", "gemmini_fence();",
        '.set begin_signature, T_other\\n', '.set end_signature, T_out+16\\n',
        'printf("METRIC cycles', 'printf("METRIC cycle_window_gemmini_region 1',
        'printf("DONE\\n");',
    ):
        assert unchanged in physical and unchanged in packet
    assert 'unsigned char merlin_readback_packet[' in packet
    expected_capacity = out_bin_packet_capacity((
        ExpectedOutput("out", 2, 3, "i8", True, 1, (2, 3)),
        ExpectedOutput("other", 2, 3, "i32", True, 4, (2, 3)),
    ))
    assert f"merlin_readback_packet[{expected_capacity}]" in packet
    assert 'volatile uint64_t merlin_readback_packet_used' in packet
    assert '"out_bin_memory.h"' in packet and '"out_bin.h"' in packet
    assert packet.count("OUT_BIN_BEGIN v1 ") == 2
    assert packet.count("OUT_BIN_END v1 ") == 2
    assert packet.count("merlin_out_bin_word(&merlin_out") == 2
    assert packet.count("merlin_out_bin_finish(&merlin_out)") == 2
    assert 'merlin_out_bin_memory_cstr(&merlin_packet_sink, "DONE\\n")' in packet
    assert packet.index("merlin_out_bin_memory_finish(&merlin_packet_sink)") < packet.index('printf("DONE\\n")')
    assert "printbuf" not in packet and "OUTSUM" not in packet
    assert command == saved


def test_invalid_invocation_policy_refuses_without_reinterpreting_capsule():
    command = _command()
    with pytest.raises(ValueError, match="policy"):
        _render(command, policy={"schema": "merlin_readback_policy_v1", "transport": FULL_VALUES_B64})
    assert command["params"] == {"console_value_cap": 1}


def test_typed_bulk_readback_requires_exact_contiguous_legacy_layout():
    build = load_build_package(SUPPORT / "build_support/__init__.py")
    command = _command()
    for spec in command["tensors"].values():
        spec["shape"] = [2, 4]
        spec["dtype"] = "i32"
    helpers = (
        lambda value: build.format.ceil_dim(value, 4),
        build.format.pad_rowmajor,
        build.format.buffer_extent,
        materialize_inputs,
    )
    source = build.render_whole_program(
        command, inputs={"arg": [[1, 2, 3, 4], [5, 6, 7, 8]]},
        legacy_helpers=helpers, readback_policy=ReadbackPolicy(FULL_VALUES_B64),
    )
    assert "merlin_out_b64_words_i32(&merlin_out, T_out, 8ULL)" in source
    packet = build.render_whole_program(
        command, inputs={"arg": [[1, 2, 3, 4], [5, 6, 7, 8]]},
        legacy_helpers=helpers, readback_policy=ReadbackPolicy("coherent_packet_v1"),
    )
    assert "merlin_out_bin_words_i32(&merlin_out, T_out, 8ULL)" in packet
    command["tensors"]["arg"]["shape"] = [2, 3]
    command["tensors"]["out"]["shape"] = [2, 3]
    padded = build.render_whole_program(
        command, inputs={"arg": [[1, 2, 3], [4, 5, 6]]},
        legacy_helpers=helpers, readback_policy=ReadbackPolicy(FULL_VALUES_B64),
    )
    assert "merlin_out_b64_words_i32" not in padded
    assert "merlin_out_b64_word(&merlin_out" in padded
    packet_padded = build.render_whole_program(
        command, inputs={"arg": [[1, 2, 3], [4, 5, 6]]},
        legacy_helpers=helpers, readback_policy=ReadbackPolicy("coherent_packet_v1"),
    )
    assert "merlin_out_bin_words_i32" not in packet_padded
    assert "merlin_out_bin_word(&merlin_out" in packet_padded


def test_typed_bulk_readback_requires_exact_contiguous_explicit_encoding():
    build = load_build_package(SUPPORT / "build_support/__init__.py")
    command = _command()
    for spec in command["tensors"].values():
        spec["dtype"] = "i32"
    compact = GroupedAxesStorage((2, 3), "i32", ((0,), (1,)), (2, 3), (3, 1), 6).to_dict()
    command["params"]["storage_encodings"] = {"arg": compact, "out": compact}
    source = build.render_whole_program(
        command, inputs={"arg": [[1, 2, 3], [4, 5, 6]]},
        readback_policy=ReadbackPolicy(FULL_VALUES_B64),
    )
    assert "merlin_out_b64_words_i32(&merlin_out, T_out + 0, 6ULL)" in source
    packet = build.render_whole_program(
        command, inputs={"arg": [[1, 2, 3], [4, 5, 6]]},
        readback_policy=ReadbackPolicy("coherent_packet_v1"),
    )
    assert "merlin_out_bin_words_i32(&merlin_out, T_out + 0, 6ULL)" in packet
    command["params"]["storage_encodings"]["out"] = GroupedAxesStorage(
        (2, 3), "i32", ((0,), (1,)), (2, 3), (4, 1), 8,
    ).to_dict()
    padded = build.render_whole_program(
        command, inputs={"arg": [[1, 2, 3], [4, 5, 6]]},
        readback_policy=ReadbackPolicy(FULL_VALUES_B64),
    )
    assert "merlin_out_b64_words_i32" not in padded
    assert "merlin_out_b64_word(&merlin_out" in padded
    packet_padded = build.render_whole_program(
        command, inputs={"arg": [[1, 2, 3], [4, 5, 6]]},
        readback_policy=ReadbackPolicy("coherent_packet_v1"),
    )
    assert "merlin_out_bin_words_i32" not in packet_padded
    assert "merlin_out_bin_word(&merlin_out" in packet_padded
