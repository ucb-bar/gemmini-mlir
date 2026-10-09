"""Exact bounded stem command streams, including DMA pointers and pool state."""

import pytest
from merlin.llvmlower.static_llvm_cfg import (
    StaticInt,
    StaticPointer,
    trace_static_function,
)
from xdsl.dialects.builtin import IntegerAttr, i64

from mlir_oot.golden_device_lower import _encoded, lower
from mlir_oot.golden_stem_pool import GoldenStemPool, StemPoolShape
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa


def command_stream(module):
    function = module.body.block.first_op
    args = [StaticPointer(i, StaticInt(0, 64)) for i in range(3)]
    result = []
    for step in trace_static_function(
        function,
        args,
        observe=lambda op: isinstance(op, G._GemminiOp),
        pointer_index_bits=64,
    ):
        op = step.operation
        operands = step.inputs
        if isinstance(op, G.PreloadOp) and operands:
            encoded = isa.preload(
                bd_addr=op.a("bd"),
                c_addr=isa.acc_addr(
                    operands[0].value, accumulate=bool(op.a("c_accumulate"))
                ),
                **{
                    key: op.a(key) for key in ("bd_cols", "bd_rows", "c_cols", "c_rows")
                },
            )
            operands = ()
        elif isinstance(op, G.ComputeOp) and operands:
            encoded = isa.compute(
                a_addr=operands[0].value,
                bd_addr=isa.GARBAGE_ADDR,
                a_cols=op.a("a_cols"),
                a_rows=op.a("a_rows"),
                accumulate=bool(op.a("accumulate")),
            )
            operands = ()
        else:
            encoded = _encoded(op)
        result.append((op.name, encoded, operands))
    return result


@pytest.mark.parametrize(
    "shape",
    [
        StemPoolShape(5, 35, 19, 0.03125),
        StemPoolShape(17, 35, 19, 0.03125),
        StemPoolShape(3, 1, 17, 0.03125),
        StemPoolShape(11, 64, 63, 0.03125),
        StemPoolShape(224, 224, 64, 0.0020730062387883663),
    ],
)
def test_exact_order_extents_addresses_dma_pointers_and_pool_configuration(shape):
    ordinary, compact = [
        GoldenStemPool(shape, loop_spatial=x).build() for x in (False, True)
    ]
    assert command_stream(ordinary) == command_stream(compact)
    lower(compact).verify()


def test_declared_bounds_do_not_substitute_for_executed_address_proof():
    module = GoldenStemPool(StemPoolShape(11, 64, 19), loop_spatial=True).build()
    op = next(
        op for op in module.walk() if isinstance(op, G.PreloadOp) and op.operands_
    )
    op.attributes["c_max"] = IntegerAttr(0, i64)
    before = str(module)
    with pytest.raises(ValueError, match="executed dynamic C"):
        lower(module)
    assert str(module) == before


@pytest.mark.parametrize("value", [1, "yes", None])
def test_explicit_boolean_required(value):
    with pytest.raises(ValueError, match="boolean"):
        GoldenStemPool(StemPoolShape(5, 35, 19), loop_spatial=value)


def test_resource_refusal_applies_before_compact_schedule():
    with pytest.raises(ValueError, match="capacity"):
        GoldenStemPool(StemPoolShape(224, 224, 64, pooled_rows=8), loop_spatial=True)


def test_default_option_is_byte_preserving():
    shape = StemPoolShape(17, 35, 19, 0.03125)
    assert str(GoldenStemPool(shape).build()) == str(
        GoldenStemPool(shape, loop_spatial=False).build()
    )


def test_source_bundle_boolean_refuses_before_creating_artifacts(tmp_path):
    from mlir_oot.stem_pool_bundle import build

    destination = tmp_path / "not-created"
    with pytest.raises(ValueError, match="boolean"):
        build(
            tmp_path / "missing-source",
            tmp_path / "missing-toolchain",
            destination,
            loop_spatial=1,
        )
    assert not destination.exists()
