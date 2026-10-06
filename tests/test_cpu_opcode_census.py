"""Weighted instruction roles include RV64 compressed and FP semantics."""

import pytest
from test_executed_features import fixture
from test_no_fsm_audit import elf_with_instructions, rocc

from mlir_oot.cpu_opcode_census import classify, opcode_index
from mlir_oot.executed_features import census


@pytest.mark.parametrize(
    "word,width,expected",
    [
        (0x02000053, 4, "fp_add_sub"),
        (0x10000053, 4, "fp_mul"),
        (0x1A000053, 4, "fp_div"),
        (0x58000053, 4, "fp_sqrt"),
        (0xC0000053, 4, "fp_convert"),
        (0xD0000053, 4, "fp_convert"),
        (0x42000053, 4, "fp_convert"),
        (0x43, 4, "fp_fma"),
        (0x4F, 4, "fp_fma"),
        (0x027000B3, 4, "integer_mul"),
        (0x027040B3, 4, "integer_div_rem"),
        (0xB0002573, 4, "csr"),
        (0x73, 4, "system"),
        (0x6F, 4, "branch_direct"),
        (0x8067, 4, "branch_indirect"),
        (0x63, 4, "branch_conditional"),
        (0x3003, 4, "load_integer"),
        (0x3007, 4, "load_fp"),
        (0x3023, 4, "store_integer"),
        (0x3027, 4, "store_fp"),
        (0x2000, 2, "load_fp"),
        (0x6000, 2, "load_integer"),
        (0x2081, 2, "integer"),
        (0x8082, 2, "branch_indirect"),
        (0x9082, 2, "branch_indirect"),
        (0x9002, 2, "system"),
        (0xA001, 2, "branch_direct"),
        (0xC001, 2, "branch_conditional"),
        (0xE002, 2, "store_integer"),
        (rocc(2), 4, "custom"),
        (0, 6, "unknown_encoding"),
        (0, 8, "unknown_encoding"),
    ],
)
def test_opcode_roles_match_local_spike_encodings(word, width, expected):
    assert classify(word, width) == expected


def test_weighted_scope_categories_conserve_retired_instructions(tmp_path):
    # Two C.NOPs, FADD.D, CSR, and an executed Gemmini load.
    elf, hist = fixture(
        tmp_path,
        [0x00010001, 0x02000053, 0xB0002573, rocc(2)],
        [
            (0x80000000, 2),
            (0x80000002, 3),
            (0x80000004, 7),
            (0x80000008, 11),
            (0x8000000C, 13),
        ],
    )
    data = census(elf, hist)
    classes = data["features"]["cpu_opcode_classes"]
    assert sum(classes.values()) == data["features"]["executed_instructions"] == 36
    assert classes["integer"] == 5 and classes["fp_add_sub"] == 7
    assert classes["csr"] == 11 and classes["custom"] == 13
    assert classes["fp_div"] == 0
    assert data["features"]["unique_executed_pcs"] == 5
    assert data["features"]["touched_instruction_bytes"] == 16
    assert (
        data["feature_status"]["/features/cpu_opcode_classes/fp_add_sub"]["unit"]
        == "instruction"
    )


def test_opcode_index_keeps_compressed_and_custom_instruction_boundaries():
    index = opcode_index(elf_with_instructions([0x00010001, rocc(2)]))
    assert index == {0x80000000: "integer", 0x80000002: "integer", 0x80000004: "custom"}
