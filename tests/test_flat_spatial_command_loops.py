"""Close compact command loops against full static primitive sequences."""

import pytest
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i64
from xdsl.utils.exceptions import VerifyException

from mlir_oot.codegen.builder import iconst
from mlir_oot.execute_wave_estimator import commands_from_function
from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_flat_conv import GoldenFlatConv
from mlir_oot.ir.gemmini_dialect import ComputeOp, PreloadOp
from mlir_oot.tables import rtl_facts as F


def build(shape, *, compact, **options):
    return GoldenFlatConv(shape, loop_spatial=compact, **options).build()


@pytest.mark.parametrize(
    "shape,options",
    [
        (ConvShape(5, 7, 20, 19, bn=2, explicit_halo=True), {}),
        (
            ConvShape(17, 19, 65, 73, bn=4),
            {
                "virtual_padding": True,
                "band_rows": 4,
                "wide_a": True,
                "separate_b_bank": True,
            },
        ),
        (
            ConvShape(14, 14, 32, 19, bn=2, stride=2),
            {"virtual_padding": True, "wide_a": True},
        ),
        (ConvShape(3, 1, 16, 17, bn=2, explicit_halo=True), {}),
        (
            ConvShape(7, 7, 65, 73, bn=4, output_dtype="i8", scale=0.03125, relu=True),
            {
                "virtual_padding": True,
                "wide_a": True,
                "separate_b_bank": True,
                "pingpong_b": True,
            },
        ),
    ],
)
def test_all_primitive_addresses_extents_initialization_and_order_match(shape, options):
    original, compact = [
        build(shape, compact=choice, **options) for choice in (False, True)
    ]
    # Includes CFG execution to completion, first weight-flip, stationary reuse,
    # each exact accumulator address/initialization and separate bounded tails.
    first, second = [
        list(commands_from_function(module.body.block.first_op, pointer_index_bits=64))
        for module in (original, compact)
    ]
    assert first == second
    lower(compact).verify()


def dynamic_module():
    module = build(ConvShape(5, 7, 32, 19, bn=2, explicit_halo=True), compact=True)
    op = next(op for op in module.walk() if isinstance(op, PreloadOp) and op.operands_)
    return module, op


@pytest.mark.parametrize(
    "name,value",
    [
        ("c_max", -1),
        ("c_max", 1024),
        ("c_reserved_rows", 1025),
        ("c_reserved_rows", 16),
        ("c_accumulate", 2),
        ("c", 0),
    ],
)
def test_unencodable_or_overlapping_declared_c_ranges_refuse(name, value):
    module, op = dynamic_module()
    op.attributes[name] = IntegerAttr(value, i64)
    with pytest.raises(VerifyException):
        module.verify()


def test_operand_outside_declared_range_refuses_before_any_lowering_mutation():
    module, op = dynamic_module()
    # Metadata alone cannot certify the actual loop's computed row.
    op.attributes["c_max"] = IntegerAttr(0, i64)
    module.verify()
    before = str(module)
    with pytest.raises(ValueError, match="executed dynamic C"):
        lower(module)
    assert str(module) == before


def test_dynamic_a_must_also_obey_declared_rows_on_same_execution():
    module, _ = dynamic_module()
    op = next(op for op in module.walk() if isinstance(op, ComputeOp) and op.operands_)
    op.attributes["a_max"] = IntegerAttr(0, i64)
    module.verify()
    with pytest.raises(ValueError, match="executed dynamic A"):
        lower(module)


def test_dynamic_declaration_outside_function_has_no_execution_proof():
    _, op = dynamic_module()
    row = iconst(0)
    detached = op.clone()
    detached.operands = [row.results[0]]
    module = ModuleOp([row, detached])
    module.verify()
    before = str(module)
    with pytest.raises(ValueError, match="no complete reachable function trace"):
        lower(module)
    assert str(module) == before


@pytest.mark.parametrize("option", [1, "yes", None])
def test_option_requires_explicit_boolean(option):
    with pytest.raises(ValueError, match="boolean"):
        build(ConvShape(5, 7, 32, 19, explicit_halo=True), compact=option)


def test_long_loop_has_one_resident_weight_load_and_bounded_destinations():
    shape = ConvShape(14, 14, 64, 73, bn=4)
    module = build(
        shape, compact=True, virtual_padding=True, wide_a=True, separate_b_bank=True
    )

    dynamic = [op for op in module.walk() if isinstance(op, PreloadOp) and op.operands_]
    assert dynamic and all(op.a("bd") == 0xFFFFFFFF for op in dynamic)
    assert all(
        op.a("c_max") + op.a("c_rows") <= op.a("c_reserved_rows") <= F.ACC_ROWS
        for op in dynamic
    )
    assert list(
        commands_from_function(module.body.block.first_op, pointer_index_bits=64)
    ) == list(
        commands_from_function(
            build(
                shape,
                compact=False,
                virtual_padding=True,
                wide_a=True,
                separate_b_bank=True,
            ).body.block.first_op,
            pointer_index_bits=64,
        )
    )


@pytest.mark.parametrize("selection", [None, 1, "yes", True])
def test_normal_source_policy_refuses_unknown_or_inert_selection_before_io(selection):
    from pathlib import Path

    from mlir_oot.captured_requant_bundle import build as build_capture

    with pytest.raises(ValueError, match="boolean selection and flat spatial"):
        build_capture(
            Path("absent"),
            Path("absent"),
            Path("absent"),
            spatial_command_loops=selection,
        )


def test_normal_source_policy_emits_the_exact_same_primitive_stream():
    from mlir_oot.conv_schedule import select_kernel

    shape = ConvShape(5, 9, 33, 73, bn=4)
    original, compact = [
        select_kernel(
            shape, flat_spatial=True, virtual_padding=True, spatial_command_loops=choice
        )[0]
        for choice in (False, True)
    ]
    assert not original.loop_spatial and compact.loop_spatial
    assert original.conv == compact.conv
    assert list(
        commands_from_function(
            original.build().body.block.first_op, pointer_index_bits=64
        )
    ) == list(
        commands_from_function(
            compact.build().body.block.first_op, pointer_index_bits=64
        )
    )
