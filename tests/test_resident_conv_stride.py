"""Decode the actual command stream against an independent scalar convolution."""

from collections import defaultdict

import numpy as np
import pytest
from merlin.llvmlower.static_llvm_cfg import (
    StaticInt,
    StaticPointer,
    trace_static_function,
)
from xdsl.dialects.builtin import IntegerAttr, i64
from xdsl.utils.exceptions import VerifyException

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_device_lower import _encoded, lower
from mlir_oot.golden_resident_conv import GoldenResidentConv
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F


@pytest.mark.parametrize(
    "shape,group,base,residue",
    [
        (ConvShape(5, 21, 32, 19, stride=2, bn=2), 1, 336, False),
        (ConvShape(9, 5, 48, 67, stride=2, bn=4), 2, 8192, False),
        (ConvShape(6, 13, 16, 17, bn=2), 1, 128, False),
        (ConvShape(17, 13, 32, 271, stride=2, bn=7), 1, 576, False),
        (ConvShape(9, 5, 48, 67, stride=2, bn=4), 3, 288, True),
        (ConvShape(17, 13, 32, 271, stride=2, bn=12), 2, 640, True),
    ],
)
def test_actual_strided_commands_match_scalar_source_and_increasing_k(
    shape, group, base, residue
):
    generator = GoldenResidentConv(
        shape,
        rows_per_tile=group,
        weight_base=base,
        source_stride=True,
        row_residue=residue,
    )
    module = generator.build()
    module.verify()
    lower(module.clone()).verify()
    a = ((np.arange(shape.h * shape.w * shape.cin) * 37) % 256 - 128).astype(np.int64)
    b = ((np.arange(9 * shape.cin * shape.cout) * 53) % 256 - 128).astype(np.int64)
    memory = [a, b]
    scratch, acc, order = {}, {}, defaultdict(list)
    loads = {}
    active_b = None
    active_k = None
    destination = None
    stride = 1
    output = np.empty(shape.oh * shape.ow * shape.cout, dtype=np.int64)
    written = set()
    args = [StaticPointer(i, StaticInt(0, 64)) for i in range(3)]
    for step in trace_static_function(
        module.body.block.first_op,
        args,
        observe=lambda op: isinstance(op, G._GemminiOp),
        pointer_index_bits=64,
    ):
        op = step.operation
        if isinstance(op, G.ConfigExOp):
            stride = op.a("a_stride", 1)
            assert (_encoded(op)[1] >> 16) & 65535 == shape.stride == stride
        elif isinstance(op, G.ConfigLdOp):
            loads[op.a("load_id")] = (op.a("stride"), op.a("block_stride", F.DIM))
        elif isinstance(op, G.MvinOp):
            ptr = step.inputs[0]
            load_id = op.a("load_id")
            row_stride, block_stride = loads[load_id]
            for block in range((op.a("cols") + F.DIM - 1) // F.DIM):
                width = min(F.DIM, op.a("cols") - block * F.DIM)
                for row in range(op.a("rows")):
                    address = op.a("local") + block * block_stride + row
                    values = np.zeros(F.DIM, dtype=np.int64)
                    if ptr.base is not None:
                        offset = ptr.offset.value + row * row_stride + block * F.DIM
                        assert ptr.base == load_id and offset + width <= len(
                            memory[ptr.base]
                        )
                        values[:width] = memory[ptr.base][offset : offset + width]
                    scratch[address] = (
                        values,
                        None if ptr.base is None else ptr.offset.value // shape.cout,
                    )
        elif isinstance(op, G.PreloadOp):
            if op.a("bd") != isa.GARBAGE_ADDR:
                active_b = np.stack(
                    [scratch[op.a("bd") + k][0] for k in range(op.a("bd_rows"))]
                )
                active_k = scratch[op.a("bd")][1]
            destination = op
        elif isinstance(op, G.ComputeOp):
            start = op.a("a") if not step.inputs else step.inputs[0].value
            for lane in range(op.a("a_rows")):
                left = scratch[start + lane * stride][0][: op.a("a_cols")]
                address = (destination.a("c") & 0x3FFF) + lane
                if not destination.a("c") & isa.ACC_ACCUMULATE_BIT:
                    acc[address] = np.zeros(F.DIM, dtype=np.int64)
                    order[address] = []
                acc[address] += left @ active_b[: op.a("a_cols")]
                order[address].append(active_k)
        elif isinstance(op, G.MvoutOp):
            ptr = step.inputs[0]
            assert ptr.base == 2
            for block in range((op.a("cols") + F.DIM - 1) // F.DIM):
                width = min(F.DIM, op.a("cols") - block * F.DIM)
                for row in range(op.a("rows")):
                    address = (op.a("local") & 0x3FFF) + block * F.DIM + row
                    assert order[address] == list(range(0, 9 * shape.cin, F.DIM))
                    index = ptr.offset.value // 4 + row * shape.cout + block * F.DIM
                    assert not written.intersection(range(index, index + width))
                    written.update(range(index, index + width))
                    output[index : index + width] = acc[address][:width]
    assert written == set(range(output.size))
    expected = np.zeros_like(output)
    for y in range(shape.oh):
        for x in range(shape.ow):
            for kh in range(3):
                for kw in range(3):
                    iy, ix = y * shape.stride + kh - 1, x * shape.stride + kw - 1
                    if 0 <= iy < shape.h and 0 <= ix < shape.w:
                        left = a[
                            (iy * shape.w + ix) * shape.cin : (iy * shape.w + ix + 1)
                            * shape.cin
                        ]
                        weights = b.reshape(9, shape.cin, shape.cout)[kh * 3 + kw]
                        index = (y * shape.ow + x) * shape.cout
                        expected[index : index + shape.cout] += left @ weights
    assert np.array_equal(output, expected)
    # The allocated row map is a bijection, including added zero padding.
    height = generator.plane // generator.input_pitch
    assert {generator.input_row(y) for y in range(height)} == set(range(height))


@pytest.mark.parametrize("stride", [0, -1, 65536])
def test_mesh_row_stride_refuses_invalid_isa_values(stride):
    op = G.ConfigExOp(
        operands=[[]],
        result_types=[[]],
        attributes={
            "dataflow": IntegerAttr(1, i64),
            "a_stride": IntegerAttr(stride, i64),
        },
    )
    with pytest.raises(VerifyException, match="positive16bits"):
        op.verify()


def test_remaining_weight_slot_requires_disjoint_complete_extents():
    shape = ConvShape(28, 28, 256, 256, stride=2, bn=4)
    generator = GoldenResidentConv(shape, weight_base=14400, source_stride=True)
    assert generator.plane * (shape.cin // F.DIM) == 14400
    assert generator.bbase + shape.bn * F.DIM <= F.SPAD_ROWS
    assert len(generator.row_tiles) * shape.bn * F.DIM <= F.ACC_ROWS
    with pytest.raises(ValueError, match="overlaps weight"):
        GoldenResidentConv(shape, weight_base=14384, source_stride=True)
    with pytest.raises(ValueError, match="resources"):
        GoldenResidentConv(shape, weight_base=F.SPAD_ROWS, source_stride=True)
    with pytest.raises(ValueError, match="bank lookahead"):
        GoldenResidentConv(
            shape, weight_base=14400, prefetch_b=True, source_stride=True
        )
    with pytest.raises(ValueError, match="aligned Cin"):
        GoldenResidentConv(ConvShape(5, 21, 17, 19, stride=2), source_stride=True)


def test_source_stride_selector_consumes_shape_resources_and_ranking_only(tmp_path):
    from mlir_oot.captured_requant_bundle import build
    from mlir_oot.conv_schedule import select_kernel

    shape = ConvShape(28, 28, 256, 256, stride=2, bn=4)
    control, kind = select_kernel(shape, flat_spatial=True, virtual_padding=True)
    assert kind == "spatial_flat_wide_a_separate_b"
    candidate, selected = select_kernel(
        shape, flat_spatial=True, virtual_padding=True, source_stride_resident=True
    )
    assert selected == "resident_source_stride_planes"
    decision = candidate.source_stride_decision
    assert decision["applied"] and not decision["timing_claim"]
    assert decision["control"]["score"] == 636224
    assert decision["candidate"]["score"] == 578048
    assert candidate.bbase == 14400 and candidate.rows_per_tile == 1
    assert control.conv == candidate.conv
    # Large accumulator footprints and wide output rows have explicit fallbacks.
    for bad in [
        ConvShape(14, 14, 512, 512, stride=2, bn=16),
        ConvShape(56, 56, 128, 128, stride=2, bn=4),
    ]:
        fallback, kind = select_kernel(
            bad, flat_spatial=True, virtual_padding=True, source_stride_resident=True
        )
        assert not fallback.source_stride_decision["applied"]
        assert fallback.source_stride_decision["refusal"]
        assert kind != "resident_source_stride_planes"
    with pytest.raises(ValueError, match="proved virtual padding"):
        build(
            tmp_path / "missing",
            tmp_path / "tools",
            tmp_path / "output",
            source_stride_resident=True,
        )
    assert not (tmp_path / "output").exists()


def test_resource_retile_uses_complete_extents_and_wide_transfer_counts():
    from dataclasses import replace

    from mlir_oot.conv_schedule import source_stride_resource_layout

    source = ConvShape(
        14, 14, 512, 512, stride=2, bn=16, output_dtype="i8", wide_b=True
    )
    candidate, decision = source_stride_resource_layout(source)
    assert candidate.conv == replace(source, bn=8)
    assert decision["max_resource_bn"] == 9
    assert decision["selected_bn"] == 8
    assert decision["input_rows"] == decision["weight_base"] == 8192
    assert decision["accumulator_rows"] == 896
    assert decision["weight_rows"] == 128
    assert decision["weight_and_output_transfer_commands"] == 2360
    assert decision["panel_retile"] and not decision["timing_claim"]
    from mlir_oot.conv_schedule import select_kernel

    retained, kind = select_kernel(
        source, flat_spatial=True, virtual_padding=True, source_stride_resident=True
    )
    assert retained.conv == replace(source, wide_b=True)
    assert kind == "spatial_flat_wide_a_separate_b"
    assert not retained.source_stride_decision["applied"]
    assert retained.source_stride_decision["resources"]["selected_bn"] == 8
    # Nine panels fit memory, but create more partial wide transfer commands.
    source = ConvShape(17, 13, 32, 271, stride=2, bn=16)
    tail, detail = source_stride_resource_layout(source)
    assert tail.conv == replace(source, bn=7)
    assert detail["max_resource_bn"] == 7
    assert detail["accumulator_rows"] == 1008
    assert tail.bbase == 576
    with pytest.raises(ValueError, match="no complete"):
        source_stride_resource_layout(ConvShape(28, 28, 512, 512, stride=2))


def test_residue_layout_groups_output_rows_without_source_retile(tmp_path):
    from mlir_oot.captured_requant_bundle import build
    from mlir_oot.conv_schedule import select_kernel, source_stride_resource_layout

    source = ConvShape(
        14, 14, 512, 512, stride=2, bn=16, output_dtype="i8", wide_b=True
    )
    candidate, decision = source_stride_resource_layout(source, row_residue=True)
    assert candidate.conv == source
    assert candidate.row_tiles == ((0, 2, 15), (2, 2, 15), (4, 2, 15), (6, 1, 7))
    assert decision["input_rows"] == decision["weight_base"] == 8192
    assert decision["accumulator_rows"] == 1024
    assert candidate.output_pitch == 8 and candidate.residue_rows == 8
    selected, kind = select_kernel(
        source,
        flat_spatial=True,
        virtual_padding=True,
        source_stride_resident=True,
        source_stride_row_residue=True,
    )
    assert kind == "resident_source_stride_planes"
    assert selected.conv == source and selected.row_residue
    assert selected.source_stride_decision["applied"]
    assert not selected.source_stride_decision["timing_claim"]
    # The previous 28-wide arm retains its byte-preserving source row order.
    retained, _ = select_kernel(
        ConvShape(28, 28, 256, 256, stride=2),
        flat_spatial=True,
        virtual_padding=True,
        source_stride_resident=True,
        source_stride_row_residue=True,
    )
    assert not retained.row_residue
    with pytest.raises(ValueError, match="requires source stride"):
        build(
            tmp_path / "missing",
            tmp_path / "tools",
            tmp_path / "output",
            source_stride_row_residue=True,
        )
    for y, count, span in candidate.row_tiles:
        for row in range(count):
            for kh in range(3):
                for x in range(source.ow):
                    assert candidate.source_offset(y, kh, 0) + (row * 8 + x) * 2 == (
                        candidate.source_offset(y + row, kh, x * 2)
                    )
    with pytest.raises(ValueError, match="row residue"):
        GoldenResidentConv(source, row_residue=True)
    with pytest.raises(ValueError, match="row residue"):
        GoldenResidentConv(
            ConvShape(5, 5, 16, 16), source_stride=True, row_residue=True
        )
