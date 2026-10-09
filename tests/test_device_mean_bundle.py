"""Independent typed source, private writer, and target resource contracts."""

import json
from pathlib import Path

import pytest
from merlin.llvmlower.fresh_tensor_writer import (
    FreshTensorWriterContract,
    rewrite_fresh_tensor_writers,
)
from xdsl.dialects.builtin import StringAttr

from mlir_oot.device_mean_bundle import (
    IntegerSumPlan,
    adapter_source,
    merlin_callbacks,
    rewrite,
)
from mlir_oot.direct_conv_binding import serialize
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_gemm import GoldenGemm
from mlir_oot.golden_resadd_proof import op_name
from mlir_oot.guarded_mean_bundle import build_and_apply, inspect, packed_nhwc_route
from mlir_oot.no_fsm_audit import audit_elf


def source():
    text = Path(__file__).with_name("guarded_mean_fixture.mlir").read_text()
    text = text.replace("tensor<1x2x1x2", "tensor<3x17x1x2").replace(
        "tensor<1x2x", "tensor<3x17x"
    )
    text = text.replace("%a:tensor<3x17x1x2xi8>", "%physical:tensor<3x1x2x17xi8>")
    return text.replace(
        " %dq =",
        """ %init_transpose = tensor.empty() : tensor<3x17x1x2xi8>
 %a = linalg.transpose ins(%physical:tensor<3x1x2x17xi8>) outs(%init_transpose:tensor<3x17x1x2xi8>) permutation=[0,3,1,2]
 %dq =""",
    )


def test_source_binding_has_explicit_private_i32_scratch_and_i8_result():
    module = parse_module(source())
    quantizer = next(
        op for op in module.walk() if op_name(op) == "quant_ext.quantize_per_tensor"
    )
    route = packed_nhwc_route(inspect(quantizer), require_word_lanes=False)
    assert route["output_shape"] == [3, 17] and route["input_matrix_shape"] == [6, 17]
    rewrite(module, quantizer, route, "mean", "a" * 64)
    module.verify()
    parse_module(serialize(module, [])).verify()
    call = next(op for op in module.walk() if op.name == "func.call")
    assert list(call.operands[1].type.get_shape()) == [3, 17]
    assert str(call.operands[1].type.element_type) == "i32"
    assert str(call.operands[2].type.element_type) == "i8"
    assert all(
        value.owner.name == "tensor.empty" and sum(1 for _ in value.uses) == 1
        for value in call.operands[1:]
    )
    report = rewrite_fresh_tensor_writers(
        module,
        [FreshTensorWriterContract("mean", 2, (1, 2), "mean__borrowed_write", 64)],
    )
    assert len(report) == 1 and report[0]["result_argument"] == 2


@pytest.mark.parametrize(
    "batches,channels,count", [(1, 17, 7), (3, 24, 49), (2, 65, 128)]
)
def test_target_producer_handles_reduction_and_channel_tails(
    tmp_path, batches, channels, count
):
    plan = IntegerSumPlan(batches, channels, count)
    facts = plan.proof()
    assert not facts["overflow"] and facts["initial_value"] == 0
    emitted = GoldenGemm(plan.shape()).build()
    emitted.body.block.first_op.properties["sym_name"] = StringAttr("sum_kernel")
    compiled = compile_module(
        emitted,
        Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin"),
        tmp_path,
    )
    assert compiled["object_nofsm_status"] == "pass"
    assert audit_elf((tmp_path / "kernel.o").read_bytes())["status"] == "pass"


@pytest.mark.parametrize(
    "plan",
    [
        IntegerSumPlan(True, 3, 2),
        IntegerSumPlan(1, 0, 2),
        IntegerSumPlan(1, 3, 129),
        IntegerSumPlan(1 << 62, 17, 2),
    ],
)
def test_unproved_geometry_refuses(plan):
    with pytest.raises(ValueError):
        plan.validate()


def test_unproved_physical_shape_or_changed_certificate_refuses():
    module = parse_module(source())
    quantizer = next(
        op for op in module.walk() if op_name(op) == "quant_ext.quantize_per_tensor"
    )
    route = packed_nhwc_route(inspect(quantizer), require_word_lanes=False)
    with pytest.raises(ValueError, match="extent"):
        adapter_source(route | {"input_matrix_shape": [5, 17]}, "mean")
    with pytest.raises(ValueError, match="certificate"):
        adapter_source(route | {"proof": route["proof"] | {"sum_max": 0}}, "mean")


def test_explicit_device_option_refuses_missing_layout_without_io(tmp_path):
    with pytest.raises(ValueError, match="physical NHWC"):
        build_and_apply(
            tmp_path / "absent", tmp_path, tmp_path / "out", device_integer_sum=True
        )
    with pytest.raises(ValueError, match="boolean"):
        build_and_apply(
            tmp_path / "absent", tmp_path, tmp_path / "out", device_integer_sum=1
        )
    assert list(tmp_path.iterdir()) == []


def test_callback_refuses_consistent_ir_with_wrong_scratch_element_type(tmp_path):
    module = parse_module(source())
    quantizer = next(
        op for op in module.walk() if op_name(op) == "quant_ext.quantize_per_tensor"
    )
    route = packed_nhwc_route(inspect(quantizer), require_word_lanes=False)
    rewrite(module, quantizer, route, "mean", "a" * 64)
    saved = {key: value for key, value in route.items() if key != "input"}
    saved.update(symbol="mean", producer_contract=IntegerSumPlan(3, 17, 2).proof())
    (tmp_path / "mean.json").write_text(
        json.dumps(
            {
                "schema": "device_integer_sum_mean_bundle_v1",
                "source_sha256": "a" * 64,
                "routes": [saved],
            }
        )
    )
    original, wrong = tmp_path / "original.mlir", tmp_path / "wrong.mlir"
    original.write_text(serialize(module, []))
    wrong.write_text(
        original.read_text().replace("tensor<3x17xi32>", "tensor<3x17xi64>")
    )
    parse_module(wrong.read_text()).verify()
    prepare, _ = merlin_callbacks(tmp_path, tmp_path, (lambda path, work: path, None))
    assert prepare(original, tmp_path) == original
    with pytest.raises(ValueError, match="type/extent ABI"):
        prepare(wrong, tmp_path)
