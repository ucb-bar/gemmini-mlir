"""Check actual B/C operand packing and refuse unproved local row ranges."""

import pytest
from merlin.llvmlower.static_llvm_cfg import trace_static_function
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin, IntegerAttr, ModuleOp, i64
from xdsl.parser import Parser
from xdsl.utils.exceptions import VerifyException

from mlir_oot.codegen.builder import FnBuilder
from mlir_oot.golden_device_lower import lower
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa


def fixture(*, dynamic_c=False, offset=0, unknown=False):
    builder = FnBuilder([i64] if unknown else [])

    def command(index):
        b = (
            builder.entry.args[0]
            if unknown
            else builder.add_i(
                builder.mul_i(index, builder.const(16)), builder.const(8192 + offset)
            )
        )
        values = {
            "bd_min": 8192,
            "bd_max": 8240 + offset,
            "bd_reserved_rows": 8256 + offset,
            "bd_alignment": 16,
            "bd_cols": 11,
            "bd_rows": 7,
            "c_cols": 9,
            "c_rows": 5,
        }
        operands = [b]
        if dynamic_c:
            operands.append(builder.mul_i(index, builder.const(8)))
            values.update(c_max=24, c_reserved_rows=29, c_accumulate=1)
        else:
            values["c"] = isa.acc_addr(3)
        builder.add(
            G.PreloadOp(
                operands=[operands],
                result_types=[[]],
                attributes={
                    key: IntegerAttr(value, i64) for key, value in values.items()
                },
            )
        )

    builder.for_loop(0, 4, 1, command)
    return ModuleOp(
        [
            llvm.FuncOp(
                "preload",
                llvm.LLVMFunctionType([i64] if unknown else []),
                linkage=llvm.LinkageAttr("external"),
                body=builder.finish(),
            )
        ]
    )


@pytest.mark.parametrize("dynamic_c", [False, True])
def test_actual_lowered_integer_operands_match_isa_and_roundtrip(dynamic_c):
    module = fixture(dynamic_c=dynamic_c)
    context = Context()
    for dialect in (Builtin, llvm.LLVM, G.GEMMINI):
        context.load_dialect(dialect)
    parsed = Parser(context, str(module)).parse_module()
    lowered = lower(parsed)
    rows = list(
        trace_static_function(
            lowered.body.block.first_op,
            [],
            observe=lambda op: isinstance(op, llvm.InlineAsmOp),
            pointer_index_bits=64,
        )
    )
    assert len(rows) == 4
    for index, step in enumerate(rows):
        expected = isa.preload(
            bd_addr=8192 + index * 16,
            c_addr=isa.acc_addr(index * 8, accumulate=True)
            if dynamic_c
            else isa.acc_addr(3),
            bd_cols=11,
            bd_rows=7,
            c_cols=9,
            c_rows=5,
        )
        assert tuple(value.value for value in step.inputs) == expected[1:]


@pytest.mark.parametrize("dynamic_c", [False, True])
def test_wave_provider_closes_the_same_dynamic_rows(dynamic_c):
    from mlir_oot.execute_wave_estimator import commands_from_function

    module = fixture(dynamic_c=dynamic_c)
    commands = list(
        commands_from_function(module.body.block.first_op, pointer_index_bits=64)
    )
    assert len(commands) == 4
    for index, command in enumerate(commands):
        assert command.kind == "preload"
        assert command.fields["bd"] == 8192 + index * 16
        assert command.fields["c"] == (
            isa.acc_addr(index * 8, accumulate=True) if dynamic_c else isa.acc_addr(3)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("bd_min", -1),
        ("bd_max", 9000),
        ("bd_reserved_rows", 16385),
        ("bd_alignment", 3),
        ("bd", 8192),
        ("unknown_parameter", 1),
    ],
)
def test_bad_declarations_refuse(field, value):
    module = fixture()
    op = next(op for op in module.walk() if isinstance(op, G.PreloadOp))
    op.attributes[field] = IntegerAttr(value, i64)
    with pytest.raises(VerifyException):
        module.verify()


def test_missing_alignment_witness_refuses():
    module = fixture()
    op = next(op for op in module.walk() if isinstance(op, G.PreloadOp))
    del op.attributes["bd_alignment"]
    with pytest.raises(VerifyException):
        module.verify()


@pytest.mark.parametrize("unknown,offset", [(False, 1), (True, 0)])
def test_actual_unknown_or_misaligned_ssa_refuses_before_mutation(unknown, offset):
    module = fixture(unknown=unknown, offset=offset)
    before = str(module)
    with pytest.raises(ValueError):
        lower(module)
    assert str(module) == before


def test_swapped_b_c_operand_roles_refuse_actual_range():
    module = fixture(dynamic_c=True)
    op = next(op for op in module.walk() if isinstance(op, G.PreloadOp))
    op.operands = tuple(reversed(op.operands))
    with pytest.raises(ValueError):
        lower(module)


def test_garbage_is_static_case_not_dynamic_range():
    module = fixture()
    op = next(op for op in module.walk() if isinstance(op, G.PreloadOp))
    for key in ("bd_min", "bd_max", "bd_reserved_rows", "bd_alignment"):
        del op.attributes[key]
    op.attributes["bd"] = IntegerAttr(isa.GARBAGE_ADDR, i64)
    op.operands = ()
    lower(module)


@pytest.mark.parametrize("field,value", [("bd_min", 8193), ("bd_max", 8208)])
def test_actual_ssa_outside_declared_interval_refuses_before_mutation(field, value):
    module = fixture()
    op = next(op for op in module.walk() if isinstance(op, G.PreloadOp))
    op.attributes[field] = IntegerAttr(value, i64)
    before = str(module)
    with pytest.raises(ValueError, match="executed dynamic B"):
        lower(module)
    assert str(module) == before


def test_generator_multiple_operand_order_is_b_then_c():
    from mlir_oot.golden_gemm import GoldenGemm, Shape

    generator = GoldenGemm(Shape(16, 16, 16))
    b, c = generator.fb.const(8192), generator.fb.const(0)
    generator._rocc(
        "preload",
        {
            "bd_min": 8192,
            "bd_max": 8192,
            "bd_reserved_rows": 8208,
            "bd_alignment": 16,
            "bd_cols": 16,
            "bd_rows": 16,
            "c_cols": 16,
            "c_rows": 16,
            "c_max": 0,
            "c_reserved_rows": 16,
        },
        operands=(b, c),
    )
    command = generator.fb.blk.last_op
    assert command.dynamic_rows() == ("bd", "c")
    assert tuple(command.operands) == (b, c)
    with pytest.raises(ValueError, match="mutually exclusive"):
        generator._rocc("preload", {}, b, operands=(b, c))
