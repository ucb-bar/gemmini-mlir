"""Prove retained rows preserve complete emitted operand order and bounds."""

import pytest
from test_resident_conv_command_loops import executed_commands

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_resident_stripe_conv import GoldenResidentStripeConv
from mlir_oot.ir.gemmini_dialect import ComputeOp


@pytest.mark.parametrize(
    "shape,stripe_rows",
    [
        (ConvShape(3, 1, 16, 17, bn=2), 2),
        (ConvShape(20, 23, 32, 19, bn=2), 7),
        (ConvShape(5, 31, 48, 33, bn=3, output_dtype="i8", scale=0.002, relu=True), 3),
        (ConvShape(5, 23, 32, 19, bn=2), 1),
    ],
)
def test_same_complete_command_stream_with_x_n_and_stripe_tails(shape, stripe_rows):
    control = GoldenResidentStripeConv(shape, stripe_rows=stripe_rows).build()
    candidate = GoldenResidentStripeConv(
        shape, stripe_rows=stripe_rows, compact_inner_commands=True
    ).build()
    assert executed_commands(control) == executed_commands(candidate)
    before = sum(isinstance(op, ComputeOp) for op in control.walk())
    after = sum(isinstance(op, ComputeOp) for op in candidate.walk())
    assert after <= before
    if stripe_rows > 2:
        assert after < before


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_refuse_untyped_command_option(value):
    with pytest.raises(ValueError, match="boolean"):
        GoldenResidentStripeConv(
            ConvShape(5, 23, 32, 19, bn=2), compact_inner_commands=value
        )


@pytest.mark.parametrize(
    "shape",
    [ConvShape(56, 56, 128, 128, bn=4), ConvShape(5, 23, 31, 19, bn=2)],
)
def test_resource_and_semantic_refusals_preserved(shape):
    with pytest.raises(ValueError):
        GoldenResidentStripeConv(shape, compact_inner_commands=True)
