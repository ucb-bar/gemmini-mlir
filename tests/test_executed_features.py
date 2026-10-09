"""Reject misleading scope and incomplete execution evidence before calibration."""

import hashlib
from unittest.mock import patch

import pytest
from test_no_fsm_audit import elf_with_instructions, rocc

from mlir_oot.executed_features import census, parse_pc_histogram

BASE = 0x80000000


def fixture(tmp_path, words, rows):
    elf = tmp_path / "fixture.elf"
    hist = tmp_path / "fixture.histogram"
    elf.write_bytes(elf_with_instructions(words))
    hist.write_text(
        "PC Histogram size:"
        + str(len(rows))
        + "\n"
        + "".join(f"{pc:x} {count}\n" for pc, count in rows)
    )
    return elf, hist


def test_weighted_primitives_are_executed_counts_and_zero_is_known(tmp_path):
    elf, hist = fixture(
        tmp_path,
        [rocc(2), rocc(4), rocc(6), rocc(3)],
        [(BASE, 7), (BASE + 4, 11), (BASE + 8, 3)],
    )
    result = census(elf, hist)
    assert result["features"]["executed_instructions"] == 21
    commands = result["features"]["primitive_commands"]
    assert commands["LOAD_CMD"] == 7
    assert commands["COMPUTE_AND_FLIP_CMD"] == 11
    assert commands["PRELOAD_CMD"] == 3
    assert commands["STORE_CMD"] == 0
    assert result["features"]["array_work"] is None
    assert result["features"]["requested_dma_bytes"] is None
    assert (
        result["feature_status"]["/features/primitive_commands/LOAD_CMD"]["unit"]
        == "command"
    )
    assert result["scope"]["excludes_harness"].startswith("UNKNOWN")


@pytest.mark.parametrize(
    "text",
    [
        "80000000 3\n",
        "PC Histogram size:2\n80000000 3\n",
        "PC Histogram size:2\n80000000 3\n80000000 2\n",
        "PC Histogram size:1\n80000001 3\n",
        "PC Histogram size:1\n80000000 -3\n",
        "PC Histogram size:1\n80000000 3 extra\n",
        "PC Histogram size:0\n",
        "PC Histogram size:1\n10000000000000000 3\n",
        "PC Histogram size:1\n80000000 3\nPC Histogram size:1\n80000000 3\n",
    ],
)
def test_incomplete_or_invalid_histogram_refused(text):
    with pytest.raises(ValueError):
        parse_pc_histogram(text)


def test_compressed_instructions_and_outside_elf_are_not_custom_counts(tmp_path):
    # Two 16-bit C.NOPs and a 32-bit primitive. Boot-ROM PC is explicitly unknown.
    elf, hist = fixture(
        tmp_path,
        [0x00010001, rocc(2)],
        [(BASE, 2), (BASE + 2, 5), (BASE + 4, 7), (0x1000, 13)],
    )
    result = census(elf, hist)
    assert result["features"]["executed_instructions"] == 27
    assert result["features"]["primitive_commands"]["LOAD_CMD"] == 7
    assert result["scope"]["elf_mapped_instructions"] == 14
    assert result["scope"]["outside_elf_instructions"] == 13


def test_histogram_cannot_start_in_middle_of_32bit_instruction(tmp_path):
    elf, hist = fixture(tmp_path, [rocc(2)], [(BASE + 2, 1)])
    with pytest.raises(ValueError, match="inside an instruction"):
        census(elf, hist)


def test_fsm_in_final_elf_refused_even_when_histogram_does_not_execute_it(tmp_path):
    elf, hist = fixture(tmp_path, [rocc(2), rocc(8)], [(BASE, 7)])
    with pytest.raises(ValueError, match="no-FSM"):
        census(elf, hist)


def test_execution_binding_must_match_bytes_and_scope(tmp_path):
    elf, hist = fixture(tmp_path, [rocc(2)], [(BASE, 7)])
    binding = {
        "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        "histogram_sha256": hashlib.sha256(hist.read_bytes()).hexdigest(),
        "scope_id": "kernel",
        "engine": {"name": "fixture"},
    }
    good = census(elf, hist, scope_id="kernel", execution_binding=binding)
    assert good["engine"] == {"name": "fixture"}
    for key in ("elf_sha256", "histogram_sha256", "scope_id"):
        with pytest.raises(ValueError, match="execution binding"):
            census(
                elf,
                hist,
                scope_id="kernel",
                execution_binding=dict(binding, **{key: "wrong"}),
            )


def test_explicit_function_scope_does_not_follow_a_call_or_divide_invocations(tmp_path):
    elf, hist = fixture(tmp_path, [rocc(2), rocc(4)], [(BASE, 5), (BASE + 4, 11)])
    table = (
        "1: 0000000080000000 4 FUNC GLOBAL DEFAULT 1 first\n"
        "2: 0000000080000004 4 FUNC GLOBAL DEFAULT 1 second\n"
    )
    with patch("mlir_oot.executed_features.subprocess.run") as run:
        run.return_value.stdout = table
        result = census(elf, hist, symbols=["first"], scope_id="first_function")
        assert result["features"]["executed_instructions"] == 5
        assert result["features"]["primitive_commands"]["COMPUTE_AND_FLIP_CMD"] == 0
        assert result["scope"]["alias_or_shared_invocation_attribution"].startswith(
            "UNKNOWN"
        )
        with pytest.raises(ValueError, match="define every"):
            census(elf, hist, symbols=["missing"], scope_id="missing")
        with pytest.raises(ValueError, match="unique"):
            census(elf, hist, symbols=["first", "first"])
        run.return_value.stdout = table + table.splitlines()[0] + "\n"
        with pytest.raises(ValueError, match="ambiguous"):
            census(elf, hist, symbols=["first"])


def test_function_scope_must_be_executable_and_observed(tmp_path):
    elf, hist = fixture(tmp_path, [rocc(2)], [(BASE, 7)])
    with patch("mlir_oot.executed_features.subprocess.run") as run:
        run.return_value.stdout = "1: 0000000080000010 4 FUNC GLOBAL DEFAULT 1 bad\n"
        with pytest.raises(ValueError, match="outside executable"):
            census(elf, hist, symbols=["bad"])


def test_symbol_extent_cannot_hide_middle_of_instruction(tmp_path):
    elf, hist = fixture(tmp_path, [rocc(2)], [(BASE, 7)])
    with patch("mlir_oot.executed_features.subprocess.run") as run:
        run.return_value.stdout = "1: 0000000080000000 2 FUNC GLOBAL DEFAULT 1 bad\n"
        with pytest.raises(ValueError, match="split an instruction"):
            census(elf, hist, symbols=["bad"])


def test_readelf_hexadecimal_function_size_supported(tmp_path):
    elf, hist = fixture(tmp_path, [rocc(2)], [(BASE, 7)])
    with patch("mlir_oot.executed_features.subprocess.run") as run:
        run.return_value.stdout = (
            "1: 0000000080000000 0x4 FUNC GLOBAL DEFAULT 1 first\n"
        )
        assert (
            census(elf, hist, symbols=["first"])["features"]["executed_instructions"]
            == 7
        )
