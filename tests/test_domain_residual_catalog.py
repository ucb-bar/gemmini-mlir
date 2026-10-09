import hashlib
import json
from pathlib import Path

import pytest
from merlin.llvmlower.quantized_affine_pair import derive
from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    StringAttr,
    TensorType,
    UnitAttr,
    i8,
)
from xdsl.ir import Block, Region

from mlir_oot.captured_residual_bundle import binding_attributes, canonical
from mlir_oot.direct_conv_binding import serialize
from mlir_oot.domain_residual_catalog import inspect_choices
from mlir_oot.frontend.parse import parse_module

SOURCE = {
    "lhs_scale": 0.020491356030106544,
    "rhs_scale": 0.017152275890111923,
    "output_scale": 0.030738085508346558,
    "relu": True,
}
PROPOSAL = {
    "source": SOURCE,
    "predictor": {"p": 135, "q": 113, "scale": 0.004938124679028988},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed_fixture(tmp_path, *, previous_result=True):
    """Unit source/metadata fixture; dummy objects establish no hardware result."""
    ty = TensorType(i8, [16, 64])
    block = Block(arg_types=[ty, ty])
    root = func.FuncOp("unrelated_input", ([ty, ty], [ty]), Region(block))
    from xdsl.dialects.builtin import ModuleOp

    module = ModuleOp([root])
    declarations = []
    routes = []
    sources = [
        {"lhs_scale": 1.0, "rhs_scale": 1.0, "output_scale": 1.0, "relu": True},
        SOURCE,
    ]
    coefficients = [
        {"p": 1, "q": 1, "scale": 1.0},
        {"p": 583, "q": 488, "scale": 0.0011434712214395404},
    ]
    rhs = block.args[1]
    for index, (source, coeff) in enumerate(zip(sources, coefficients, strict=True)):
        proof = derive(**source, **coeff)
        assert proof["mismatched_pairs"] == 0
        symbol = f"arbitrary_{index}"
        route = {
            "symbol": symbol,
            "kernel": symbol + "_kernel",
            "region": "audit_" + str(index),
            "m": 16,
            "n": 64,
            "proof": {
                "source": source,
                "coefficients": coeff,
                "primitive": {"readout": coeff["scale"]},
                "pairs": 65536,
                "exact": True,
            },
            "numeric_policy": {
                "kind": "exact_source",
                "max_output_lsb": 0,
                "domain": "all65536",
            },
        }
        route["proof_sha256"] = hashlib.sha256(
            canonical(route["proof"]).encode()
        ).hexdigest()
        directory = tmp_path / symbol
        directory.mkdir()
        for name in ("kernel.gemmini.mlir", "kernel.o", "adapter.c", "adapter.o"):
            (directory / name).write_text("structural fixture only " + name)
        route["compilation"] = {
            "target_ir_sha256": digest(directory / "kernel.gemmini.mlir"),
            "object_sha256": digest(directory / "kernel.o"),
        }
        route["adapter_compilation"] = {
            "source_sha256": digest(directory / "adapter.c"),
            "object_sha256": digest(directory / "adapter.o"),
        }
        empty = tensor.EmptyOp([], ty)
        selected_rhs = rhs if previous_result else block.args[1]
        call = func.CallOp(symbol, [block.args[0], selected_rhs, empty.tensor], [ty])
        block.add_ops([empty, call])
        rhs = call.results[0]
        access = ArrayAttr(
            [
                DictionaryAttr({"bufferization.access": StringAttr(v)})
                for v in ("read", "read", "write")
            ]
        )
        declaration = func.FuncOp(
            symbol, ([ty] * 3, [ty]), Region(), visibility="private", arg_attrs=access
        )
        declaration.attributes.update(
            binding_attributes(route, "source_identity", "receipt_identity")
        )
        declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
        module.body.block.add_op(declaration)
        declarations.append(declaration)
        routes.append(route)
    block.add_op(func.ReturnOp(rhs))
    module.verify()
    (tmp_path / "rewritten.mlir").write_text(serialize(module, declarations))
    for name in ("residual.o", "native_oracle.c"):
        (tmp_path / name).write_text("structural fixture " + name)
    record = {
        "implementation": "wide_integer",
        "numeric_policy": {"max_output_lsb": 0},
        "routes": routes,
        "source_sha256": "source_identity",
        "capture_receipt_sha256": "receipt_identity",
        "rewritten_sha256": digest(tmp_path / "rewritten.mlir"),
        "object_sha256": digest(tmp_path / "residual.o"),
        "native_oracle_sha256": digest(tmp_path / "native_oracle.c"),
    }
    (tmp_path / "residual.json").write_text(json.dumps(record))
    return record


def test_actual_upstream_exact_relu_result_selects_domain_coefficient(tmp_path):
    sealed_fixture(tmp_path)
    _, _, _, choices = inspect_choices(tmp_path, [PROPOSAL])
    selected = choices[1]
    assert selected["operand_intervals"] == [[-128, 127], [0, 127]]
    assert selected["choice"]["certificate"]["admitted_pairs"] == 32768
    assert selected["choice"]["predictor"] == PROPOSAL["predictor"]
    assert not selected["choice"]["certificate"]["exact_for_complete_type_domain"]


def test_unknown_operand_does_not_inherit_range_from_callee_or_proposal(tmp_path):
    sealed_fixture(tmp_path, previous_result=False)
    _, _, _, choices = inspect_choices(tmp_path, [PROPOSAL])
    assert choices[1]["operand_intervals"] == [[-128, 127], [-128, 127]]
    assert choices[1]["choice"] is None


def test_stale_producer_implementation_refuses(tmp_path):
    sealed_fixture(tmp_path)
    (tmp_path / "arbitrary_0/kernel.o").write_text("changed")
    with pytest.raises(ValueError, match="implementation changed"):
        inspect_choices(tmp_path, [PROPOSAL])


def test_changed_numeric_attribute_refuses_before_choice(tmp_path):
    record = sealed_fixture(tmp_path)
    module = parse_module((tmp_path / "rewritten.mlir").read_text())
    declaration = next(
        op
        for op in module.body.block.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == "arbitrary_0"
    )
    declaration.attributes["gemmini.residual_proof_sha256"] = StringAttr(
        "not_source_proof"
    )
    (tmp_path / "rewritten.mlir").write_text(serialize(module, []))
    record["rewritten_sha256"] = digest(tmp_path / "rewritten.mlir")
    (tmp_path / "residual.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="declaration differs"):
        inspect_choices(tmp_path, [PROPOSAL])


def test_multiple_actual_calls_refuse_until_every_call_domain_supported(tmp_path):
    record = sealed_fixture(tmp_path)
    module = parse_module((tmp_path / "rewritten.mlir").read_text())
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    call.parent.insert_op_before(
        func.CallOp(
            call.callee, list(call.arguments), [result.type for result in call.results]
        ),
        call,
    )
    (tmp_path / "rewritten.mlir").write_text(serialize(module, []))
    record["rewritten_sha256"] = digest(tmp_path / "rewritten.mlir")
    (tmp_path / "residual.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="one actual source call"):
        inspect_choices(tmp_path, [PROPOSAL])


def test_numeric_certificate_tampering_refuses(tmp_path):
    record = sealed_fixture(tmp_path)
    record["routes"][0]["proof"]["source"]["relu"] = False
    (tmp_path / "residual.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="certificate changed"):
        inspect_choices(tmp_path, [PROPOSAL])
