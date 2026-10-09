"""Decode nondefault semantic fields from the actual lowered command operands."""

import pytest
from merlin.llvmlower import toolchain
from merlin.llvmlower.static_llvm_cfg import trace_static_function
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Float32Type, Float64Type, FloatAttr, IntegerAttr, ModuleOp, StringAttr, i64
from xdsl.utils.exceptions import VerifyException

from mlir_oot.codegen.builder import FnBuilder
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_resident_conv import GoldenResidentConv
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa


def command(cls, values):
    return cls(operands=[[]], result_types=[[]], attributes={
        name: FloatAttr(value, Float32Type()) if type(value) is float else IntegerAttr(value, i64)
        for name, value in values.items()
    })


def test_actual_lowered_values_preserve_every_declared_execute_field(tmp_path):
    builder = FnBuilder([])
    builder.add(command(G.FlushOp, {"skip": 1}))
    builder.add(command(G.ConfigExOp, {
        "dataflow": 0, "act": 1, "sys_shift": 0x87654321, "acc_scale": .5,
        "a_stride": 7, "c_stride": 11, "a_transpose": 1, "b_transpose": 1,
        "set_only_strides": 1,
    }))
    module = ModuleOp([llvm.FuncOp("config", llvm.LLVMFunctionType([]),
                                  linkage=llvm.LinkageAttr("external"), body=builder.finish())])
    lowered = lower(module.clone())
    commands = list(trace_static_function(lowered.body.block.first_op, [],
        observe=lambda op: isinstance(op, llvm.InlineAsmOp), pointer_index_bits=64))
    assert len(commands) == 2
    flush, execute = [tuple(value.value for value in step.inputs) for step in commands]
    assert flush == (1, 0)
    rs1, rs2 = execute
    assert rs1 & 3 == 0
    assert (rs1 >> 2) & 1 == 0
    assert (rs1 >> 3) & 3 == 1
    assert (rs1 >> 7) & 7 == 7
    assert (rs1 >> 16) & 65535 == 7
    assert rs1 >> 32 == 0x3f000000
    assert rs2 >> 48 == 11
    assert rs2 & 0xffffffff == 0x87654321
    llvm_bin = toolchain.clang().parent
    if (llvm_bin / "mlir-translate").is_file():
        receipt = compile_module(module, llvm_bin, tmp_path)
        assert receipt["object_nofsm_status"] == "pass"


@pytest.mark.parametrize("field,value", [
    ("a_stride", 0), ("c_stride", 65536), ("c_stride", -1),
    ("act", 2), ("dataflow", 2), ("sys_shift", -1), ("sys_shift", 2**32),
    ("a_transpose", 2), ("b_transpose", -1), ("set_only_strides", 2),
    ("unknown_semantic_parameter", 1), ("acc_scale", 1e300),
])
def test_unrepresentable_or_unknown_semantics_refuse(field, value):
    if field == "acc_scale":
        op = command(G.ConfigExOp, {"dataflow": 1})
        op.attributes[field] = FloatAttr(value, Float64Type())
    else:
        op = command(G.ConfigExOp, {"dataflow": 1, field: value})
    with pytest.raises(VerifyException):
        op.verify()


def test_untyped_flags_and_unconsumed_operands_refuse():
    op = command(G.ConfigExOp, {"dataflow": 1})
    op.attributes["a_transpose"] = StringAttr("false")
    with pytest.raises(VerifyException, match="one-bit"):
        op.verify()
    for cls in (G.ConfigExOp, G.FlushOp):
        builder = FnBuilder([i64])
        op = cls(operands=[[builder.entry.args[0]]], result_types=[[]],
                 attributes={"dataflow": IntegerAttr(1, i64)} if cls is G.ConfigExOp else {})
        with pytest.raises(VerifyException, match="no operands"):
            op.verify()


@pytest.mark.parametrize("source_stride", [False, True])
def test_default_and_previously_qualified_stride_objects_unchanged(tmp_path, monkeypatch, source_stride):
    import mlir_oot.golden_device_lower as lowering
    llvm_bin = toolchain.clang().parent
    if not (llvm_bin / "mlir-translate").is_file():
        pytest.skip("actual LLVM compiler unavailable")
    module = (GoldenResidentConv(ConvShape(5, 21, 32, 19, stride=2, bn=2),
                                weight_base=336, source_stride=True).build()
              if source_stride else GoldenGemm(Shape(17, 19, 65, output_dtype="i32")).build())
    original = lowering._encoded
    def previous(op):
        if isinstance(op, G.ConfigExOp):
            return isa.config_ex(dataflow=op.a("dataflow"), a_stride=op.a("a_stride", 1))
        if isinstance(op, G.FlushOp):
            return isa.flush()
        return original(op)
    monkeypatch.setattr(lowering, "_encoded", previous)
    control = compile_module(module.clone(), llvm_bin, tmp_path / "control")
    monkeypatch.setattr(lowering, "_encoded", original)
    candidate = compile_module(module.clone(), llvm_bin, tmp_path / "candidate")
    assert candidate["object_sha256"] == control["object_sha256"]
