"""Compare complete executed primitive operands, including DMA and K order."""

from collections import Counter

import pytest
from merlin.llvmlower.static_llvm_cfg import (
    StaticInt,
    StaticPointer,
    trace_static_function,
)
from xdsl.dialects import llvm

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_device_lower import _encoded, lower
from mlir_oot.golden_resident_conv import GoldenResidentConv
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa


def executed_commands(module):
    module.verify()
    lower(module.clone()).verify()
    fn = module.body.block.first_op
    args = [StaticPointer(i, StaticInt(0, 64)) for i in range(3)]
    commands = []
    for step in trace_static_function(
        fn, args, observe=lambda op: isinstance(op, G._GemminiOp), pointer_index_bits=64
    ):
        op = step.operation
        pointers = tuple(v for v in step.inputs if isinstance(v, StaticPointer))
        if isinstance(op, G.ComputeOp) and step.inputs:
            value = step.inputs[0]
            assert isinstance(value, StaticInt) and value.value <= op.a("a_max")
            command = isa.compute(
                a_addr=value.value,
                bd_addr=op.a("bd", isa.GARBAGE_ADDR),
                a_cols=op.a("a_cols"),
                a_rows=op.a("a_rows"),
                bd_cols=op.a("bd_cols", isa.DIM),
                bd_rows=op.a("bd_rows", isa.DIM),
                accumulate=bool(op.a("accumulate", False)),
            )
        elif isinstance(op, G.PreloadOp) and step.inputs:
            value = step.inputs[0]
            assert isinstance(value, StaticInt) and value.value <= op.a("c_max")
            command = isa.preload(
                bd_addr=op.a("bd"),
                c_addr=isa.acc_addr(value.value, accumulate=bool(op.a("c_accumulate"))),
                bd_cols=op.a("bd_cols"),
                bd_rows=op.a("bd_rows"),
                c_cols=op.a("c_cols"),
                c_rows=op.a("c_rows"),
            )
        else:
            command = _encoded(op)
        commands.append((op.name, command, pointers))
    return commands


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize(
    "shape,rows",
    [
        (ConvShape(3, 1, 16, 17, bn=2, output_dtype="i32"), 3),
        (ConvShape(5, 7, 32, 33, bn=2, output_dtype="i32"), 2),
        (ConvShape(7, 7, 48, 67, bn=4, output_dtype="i8", scale=0.01, relu=True), 2),
        (ConvShape(9, 5, 80, 19, bn=2, output_dtype="i8", scale=0.001), 2),
    ],
)
def test_retention_preserves_every_command_pointer_and_short_group(
    shape, rows, prefetch
):
    control = GoldenResidentConv(shape, rows_per_tile=rows, prefetch_b=prefetch).build()
    candidate = GoldenResidentConv(
        shape, rows_per_tile=rows, prefetch_b=prefetch, compact_commands=True
    ).build()
    assert executed_commands(control) == executed_commands(candidate)
    # The N, K and spatial loops are ordinary LLVM branches. Retention is a
    # compiler hint; the complete actual target program still undergoes audit.
    if shape.oh // rows > 1:
        assert any(isinstance(op, llvm.CondBrOp) for op in candidate.walk())
    assert "gemmini.resident_conv_compact_commands" in candidate.attributes


@pytest.mark.parametrize("residue", [False, True])
def test_retained_spatial_mapping_preserves_strided_source_planes(residue):
    shape = ConvShape(9, 9, 32, 19, stride=2, bn=2, output_dtype="i32")
    kwargs = {"rows_per_tile": 2, "source_stride": True, "row_residue": residue}
    a = GoldenResidentConv(shape, **kwargs).build()
    b = GoldenResidentConv(shape, **kwargs, compact_commands=True).build()
    assert executed_commands(a) == executed_commands(b)


def test_large_regular_geometry_changes_static_size_not_issued_work():
    shape = ConvShape(
        14,
        14,
        256,
        256,
        bn=4,
        output_dtype="i8",
        scale=0.0012096002465113997,
        relu=True,
    )
    control = GoldenResidentConv(shape).build()
    compact = GoldenResidentConv(shape, compact_commands=True).build()
    left, right = executed_commands(control), executed_commands(compact)
    assert left == right
    counts = Counter(row[0] for row in right)
    assert counts["gemmini.compute"] == 32256
    assert counts["gemmini.preload"] == 32256
    assert sum(isinstance(op, G.ComputeOp) for op in compact.walk()) < 500


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_retention_refuses_untyped_options(value):
    with pytest.raises(ValueError, match="boolean"):
        GoldenResidentConv(ConvShape(3, 3, 16, 16), compact_commands=value)


@pytest.mark.parametrize(
    "shape",
    [ConvShape(14, 14, 256, 256, bn=8), ConvShape(14, 14, 1024, 256, bn=4)],
)
def test_retention_preserves_accumulator_and_input_resource_refusals(shape):
    with pytest.raises(ValueError):
        GoldenResidentConv(shape, compact_commands=True)
