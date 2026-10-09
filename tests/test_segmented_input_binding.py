"""Actual selected source, target object, descriptor and native oracle closure."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from merlin.llvmlower.fresh_tensor_writer import (
    FreshTensorWriterContract,
    rewrite_fresh_tensor_writers,
)
from merlin.xdsl_dialects._common import text
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

from mlir_oot.captured_requant_bundle import dense_adapter, scalar_oracle
from mlir_oot.captured_residual_bundle import compile_adapter
from mlir_oot.descriptor_writer import shim
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.segmented_input_binding import (
    adapter_source,
    derive,
    linker_path,
    merlin_callbacks,
    sha,
)

LLVM = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")
SOURCE = """module {
func.func @forward(%a:tensor<1x3x5x2xi8>,%b:tensor<2x3xi8>)
 ->(tensor<1x3x5x2xi8>,tensor<4x3xi8>) attributes {llvm.emit_c_interface} {
 %empty=tensor.empty():tensor<1x3x5x2xi8>
 %owner=func.call @produce(%a,%empty):(tensor<1x3x5x2xi8>,tensor<1x3x5x2xi8>)->tensor<1x3x5x2xi8>
 %slice="tensor.extract_slice"(%owner)<{static_offsets=array<i64: 0,0,1,0>,
 static_sizes=array<i64: 1,2,2,2>,static_strides=array<i64: 1,2,2,1>,
 operandSegmentSizes=array<i32: 1,0,0,0>}>:(tensor<1x3x5x2xi8>)->tensor<1x2x2x2xi8>
 %flat="tensor.collapse_shape"(%slice)<{reassociation=[[0 : i64,1 : i64,2 : i64,3 : i64]]}>
 :(tensor<1x2x2x2xi8>)->tensor<8xi8>
 %view="tensor.expand_shape"(%flat)<{reassociation=[[0 : i64,1 : i64]],
 static_output_shape=array<i64: 4,2>}>:(tensor<8xi8>)->tensor<4x2xi8>
 %out=tensor.empty():tensor<4x3xi8>
 %r=func.call @consume(%view,%b,%out):(tensor<4x2xi8>,tensor<2x3xi8>,tensor<4x3xi8>)->tensor<4x3xi8>
 return %owner,%r:tensor<1x3x5x2xi8>,tensor<4x3xi8>
}
func.func private @produce(tensor<1x3x5x2xi8>,tensor<1x3x5x2xi8>)->tensor<1x3x5x2xi8>
 attributes {llvm.emit_c_interface}
func.func private @consume(tensor<4x2xi8>,tensor<2x3xi8>,tensor<4x3xi8>)->tensor<4x3xi8>
 attributes {llvm.emit_c_interface,merlin.numeric_contract="exact"}
}"""


def fixture():
    module = parse_module(SOURCE)
    for declaration in module.body.block.ops:
        if declaration.name != "func.func" or declaration.body.blocks:
            continue
        accesses = (
            ("read", "write")
            if declaration.sym_name.data == "produce"
            else ("read", "read", "write")
        )
        declaration.properties["arg_attrs"] = ArrayAttr(
            [DictionaryAttr({"bufferization.access": StringAttr(v)}) for v in accesses]
        )
    shape = Shape(
        4, 3, 2, bm=1, bn=1, cache_a=True, reuse_b=True, wide_b=True, scale=0.25
    )
    emitted = GoldenGemm(shape).build()
    emitted.body.block.first_op.properties["sym_name"] = StringAttr("consume_kernel")
    route = {
        "symbol": "consume",
        "kernel": "consume_kernel",
        "direct_conv": False,
        "schedule": asdict(shape),
        "schedule_kind": "fixture_resident_a",
        "numeric_contract": {"max_output_lsb_error": 0},
        "proof": {
            "fixture_contract": "ordered modular i32 contraction and binary32 store scale"
        },
        "compilation": {
            "target_ir_sha256": hashlib.sha256(
                (str(emitted) + "\n").encode()
            ).hexdigest()
        },
    }
    writers = (
        FreshTensorWriterContract("produce", 1, (1,), "produce__borrowed_write", 64),
        FreshTensorWriterContract("consume", 2, (2,), "consume__borrowed_write", 64),
    )
    return module, shape, route, writers


def test_absent_option_returns_identical_callbacks_without_io(tmp_path):
    base = (lambda *args: None, lambda *args: None)
    callbacks = merlin_callbacks(LLVM, (), (), base)
    assert callbacks[:2] == base and callbacks[2]() == ()
    assert list(tmp_path.iterdir()) == []


def test_derivation_uses_geometry_and_pinned_control_not_symbol_names():
    module, _, route, _ = fixture()
    bindings, refused = derive(module, [route])
    assert len(bindings) == 1 and not refused
    binding = bindings[0]
    assert binding.owner_shape == (1, 3, 5, 2)
    assert binding.contract.address.offset(3, 1) == 27
    assert binding.original["proof"] is route["proof"]
    route["compilation"]["target_ir_sha256"] = "0" * 64
    bindings, refused = derive(module, [route])
    assert not bindings and "compiled target IR" in refused[0]["reason"]


def test_descriptor_emission_refuses_extent_changes_and_invalid_c_names():
    module, shape, route, _ = fixture()
    binding = derive(module, [route])[0][0]
    with pytest.raises(ValueError, match="extent"):
        adapter_source(shape, "consumer", "kernel", binding.contract.address, (31,))
    with pytest.raises(ValueError, match="identifier"):
        adapter_source(
            shape, "bad.name", "kernel", binding.contract.address, binding.owner_shape
        )


def test_selected_normal_catalog_emits_and_executes_borrowed_input(tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    module, shape, route, writers = fixture()
    original = tmp_path / "original.mlir"
    original.write_text(text(module, generic=True))

    def base_prepare(source, work):
        return source

    def base_build(source, work):
        work = Path(work)
        control = GoldenGemm(shape).build()
        control.body.block.first_op.properties["sym_name"] = StringAttr(route["kernel"])
        compilation = compile_module(control, LLVM, work / "control")
        adapter = dense_adapter(shape, route["symbol"], route["kernel"])
        c = work / "control" / "adapter.c"
        c.write_text(adapter)
        compile_adapter(c, c.with_suffix(".o"), LLVM)
        obj = work / "control.o"
        subprocess.run(
            [
                str(linker_path(LLVM)),
                "-r",
                str(c.with_suffix(".o")),
                str(work / "control" / "kernel.o"),
                "-o",
                str(obj),
            ],
            check=True,
        )
        oracle = work / "control.c"
        oracle.write_text(
            "#include <stdint.h>\n"
            + adapter
            + scalar_oracle(shape, route["kernel"], False)
        )
        manifest = {
            "source_sha256": sha(source),
            "compilation": compilation,
            "fused_requantizations": [route],
            "native_oracle_sources": [str(oracle)],
        }
        path = work / "catalog.json"
        path.write_text(json.dumps(manifest))
        return path, obj

    prepare, build, selected_writers = merlin_callbacks(
        LLVM, [route], writers, (base_prepare, base_build), enabled=True
    )
    selected = prepare(original, tmp_path / "prepared")
    path, obj = build(selected, tmp_path / "catalog")
    catalog = json.loads(path.read_text())
    assert catalog["source_sha256"] == sha(selected)
    assert catalog["compilation"]["object_sha256"] == sha(obj)
    assert catalog["fused_requantizations"][0]["symbol"] == "consume__segmented_a"
    assert catalog["segmented_input_acceptance"]["schedule_selection_controls_emission"]
    assert selected_writers()[1].symbol == "consume__segmented_a"
    parsed = parse_module(selected.read_text())
    report = rewrite_fresh_tensor_writers(parsed, selected_writers())
    llvm = lower_to_llvm_ir(
        text(parsed, generic=True), workdir=tmp_path / "lower", vectorize=False
    )
    source = tmp_path / "model.ll"
    source.write_text(llvm)
    native = tmp_path / "native.c"
    native.write_text(
        "\n".join(Path(p).read_text() for p in catalog["native_oracle_sources"])
        + """
void _mlir_ciface_produce(memref4*r,memref4*a,memref4*c) {
 for(int i=0;i<30;i++)((int8_t*)c->aligned)[c->offset+i]=((int8_t*)a->aligned)[a->offset+i];*r=*c;
}
"""
    )
    bridge = tmp_path / "bridge.c"
    bridge.write_text(shim(report))
    host_obj, library = tmp_path / "model.o", tmp_path / "model.so"
    subprocess.run(
        [str(LLVM / "clang"), "-O2", "-fPIC", "-c", str(source), "-o", str(host_obj)],
        check=True,
    )
    subprocess.run(
        [
            "cc",
            "-O2",
            "-fPIC",
            "-shared",
            str(host_obj),
            str(native),
            str(bridge),
            str(mlir_runtime_c()),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
    )
    invoke = HostModel.load(str(library))
    a = (np.arange(30) - 11).astype(np.int8).reshape(1, 3, 5, 2)
    b = np.array([[-1, 2, 1], [3, 0, -2]], dtype=np.int8)
    owner, output = np.zeros_like(a), np.zeros((4, 3), dtype=np.int8)
    expected = a[:, ::2, 1::2, :].reshape(4, 2).astype(np.int32) @ b.astype(np.int32)
    expected = np.rint(expected.astype(np.float32) * np.float32(shape.scale)).astype(
        np.int8
    )
    for _ in range(3):
        invoke([(value.ctypes.data, value.shape) for value in (a, b, owner, output)])
        np.testing.assert_array_equal(owner, a)
        np.testing.assert_array_equal(output, expected)
    bad = tmp_path / "bad.c"
    bad.write_text(
        native.read_text()
        + """
int main(void) {
 int8_t x[30]={0},b[6]={0},c[12]={0};memref2 r={0},bdesc={b,b,0,{2,3},{3,1}},cdesc={c,c,0,{4,3},{3,1}};
 consume__segmented_a_input_descriptor a={x,x,0,{1,3,5,3},{30,10,2,1}};
 _mlir_ciface_consume__segmented_a(&r,&a,&bdesc,&cdesc);return 0;
}
"""
    )
    executable = tmp_path / "bad"
    subprocess.run(["cc", "-O2", str(bad), "-lm", "-o", str(executable)], check=True)
    assert subprocess.run([str(executable)], check=False).returncode < 0
