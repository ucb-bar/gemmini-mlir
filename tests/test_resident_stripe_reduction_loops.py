"""Compare full tap/K/N/row/X command order with retained K loops."""

import pytest
from test_resident_conv_command_loops import executed_commands

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_resident_stripe_conv import GoldenResidentStripeConv
from mlir_oot.ir.gemmini_dialect import PreloadOp


@pytest.mark.parametrize("compact_rows", [False, True])
@pytest.mark.parametrize(
    "shape,stripe_rows",
    [
        (ConvShape(3, 1, 16, 17, bn=2), 2),
        (ConvShape(5, 23, 32, 19, bn=2), 3),
        (ConvShape(5, 31, 48, 33, bn=3, output_dtype="i8", scale=0.002, relu=True), 3),
        (ConvShape(7, 19, 80, 65, bn=4, output_dtype="i8", scale=0.001), 2),
    ],
)
def test_complete_source_command_order_with_k_x_n_and_row_tails(
    shape, stripe_rows, compact_rows
):
    control = GoldenResidentStripeConv(shape, stripe_rows=stripe_rows).build()
    candidate = GoldenResidentStripeConv(
        shape,
        stripe_rows=stripe_rows,
        compact_inner_commands=compact_rows,
        compact_reduction_commands=True,
    ).build()
    assert executed_commands(control) == executed_commands(candidate)
    dynamic_b = [
        op
        for op in candidate.walk()
        if isinstance(op, PreloadOp) and "bd" in op.dynamic_rows()
    ]
    assert bool(dynamic_b) == (shape.cin > 16)
    for op in dynamic_b:
        assert (
            op.a("bd_min")
            >= GoldenResidentStripeConv(shape, stripe_rows=stripe_rows).bbase
        )
        assert op.a("bd_alignment") == 16


@pytest.mark.parametrize("option", [1, None, "yes"])
def test_refuse_untyped_reduction_retention(option):
    with pytest.raises(ValueError, match="boolean"):
        GoldenResidentStripeConv(
            ConvShape(5, 23, 32, 19, bn=2), compact_reduction_commands=option
        )


@pytest.mark.parametrize(
    "shape",
    [
        ConvShape(56, 56, 128, 128, bn=4),
        ConvShape(5, 23, 31, 19, bn=2),
        ConvShape(5, 23, 32, 19, bn=2, stride=2),
    ],
)
def test_existing_resource_and_source_layout_refusals_preserved(shape):
    with pytest.raises(ValueError):
        GoldenResidentStripeConv(shape, compact_reduction_commands=True)
