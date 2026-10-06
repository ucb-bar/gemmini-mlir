"""Actual typed writer ownership and complete native ranked-ABI closure."""

import copy
import ctypes
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from merlin.llvmlower.quantized_affine_joint import derive_joint
from merlin.llvmlower.quantized_affine_pair import predictor_table, source_table
from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

from mlir_oot.captured_residual_bundle import binding_attributes, canonical
from mlir_oot.descriptor_writer import DescriptorWriterContract, shim, transform
from mlir_oot.frontend.parse import parse_module
from mlir_oot.joint_residual_catalog import adapter, oracle, rewrite

SOURCE = {
    "lhs_scale": 0.011258588172495365,
    "rhs_scale": 0.00940733402967453,
    "output_scale": 0.011643771082162857,
    "relu": True,
}


@lru_cache
def proof():
    return derive_joint(
        **SOURCE,
        predictors=[
            {"p": 298, "q": 249, "scale": scale, "relu": True}
            for scale in (0.0032446938566863537, 0.0032447930425405502)
        ],
    )


def fixture(m=48, name="test_residual"):
    old = {
        "symbol": name,
        "m": m,
        "n": 64,
        "shape": [m, 64],
        "proof": {
            "source": SOURCE,
            "primitive": {"readout": 0.00037060913746245205},
            "coefficients": {"p": 2609, "q": 2180, "scale": 0.00037060913746245205},
            "exact": True,
        },
        "numeric_policy": {"kind": "exact_source", "max_output_lsb": 0},
    }
    original = {
        "routes": [old],
        "source_sha256": "a" * 64,
        "capture_receipt_sha256": "b" * 64,
    }
    joint = {
        "original_symbol": name,
        "symbol": name + "__joint",
        "kernel": name + "__joint_kernel",
        "m": m,
        "n": 64,
        "original_route": old,
        "proof": proof(),
        "proof_sha256": hashlib.sha256(canonical(proof()).encode()).hexdigest(),
        "result_argument": 2,
        "fully_written_arguments": [2, 3],
    }
    module = parse_module(f"""module {{
func.func @forward(%a:tensor<{m}x64xi8>,%b:tensor<{m}x64xi8>) -> (tensor<{m}x64xi8>,tensor<{m}x64xi8>,tensor<{m}x64xi8>) attributes {{llvm.emit_c_interface}} {{
 %out=tensor.empty():tensor<{m}x64xi8>
 %r=func.call @{name}(%a,%b,%out):(tensor<{m}x64xi8>,tensor<{m}x64xi8>,tensor<{m}x64xi8>)->tensor<{m}x64xi8>
 return %a,%b,%r:tensor<{m}x64xi8>,tensor<{m}x64xi8>,tensor<{m}x64xi8>
}}
func.func private @{name}(tensor<{m}x64xi8>,tensor<{m}x64xi8>,tensor<{m}x64xi8>)->tensor<{m}x64xi8> attributes {{llvm.emit_c_interface}}
}}""")
    decl = module.body.block.last_op
    decl.properties["arg_attrs"] = ArrayAttr(
        [
            DictionaryAttr({"bufferization.access": StringAttr(a)})
            for a in ("read", "read", "write")
        ]
    )
    decl.attributes.update(
        binding_attributes(
            old, original["source_sha256"], original["capture_receipt_sha256"]
        )
    )
    return module, original, {"routes": [joint]}


def test_typed_private_second_output_and_source_binding():
    module, original, record = fixture()
    rewrite(module, original, record)
    call = next(o for o in module.walk() if isinstance(o, func.CallOp))
    assert len(call.arguments) == 4 and isinstance(
        call.arguments[3].owner, tensor.EmptyOp
    )
    assert sum(1 for _ in call.arguments[3].uses) == 1
    decl = module.body.block.last_op
    assert decl.sym_name.data == record["routes"][0]["symbol"]
    assert [a.data["bufferization.access"].data for a in decl.arg_attrs] == [
        "read",
        "read",
        "write",
        "write",
    ]
    assert decl.attributes["gemmini.source_sha256"].data == original["source_sha256"]
    assert set(decl.attributes["gemmini.qparams"].data) == set(SOURCE)
    assert "gemmini.wide_integer_coefficients" not in decl.attributes
    assert (
        decl.attributes["gemmini.joint_residual_workspace_bytes"].value.data == 48 * 64
    )
    routes = transform(
        module, [DescriptorWriterContract(record["routes"][0]["symbol"], 2, (2, 3))]
    )
    assert routes[0]["fresh_writers"] == [2, 3]
    assert sum(o.name == "memref.alloc" for o in module.walk()) == 2
    module.verify()


@pytest.mark.parametrize(
    "fault", ["source", "effect", "type", "live", "collision", "proof", "writer"]
)
def test_invalid_selection_refuses_before_any_edit(fault):
    module, original, record = fixture()
    decl = module.body.block.last_op
    if fault == "source":
        decl.attributes["gemmini.source_sha256"] = StringAttr("changed")
    elif fault == "effect":
        decl.properties["arg_attrs"] = ArrayAttr(
            [
                DictionaryAttr({"bufferization.access": StringAttr(a)})
                for a in ("write", "read", "write")
            ]
        )
    elif fault == "type":
        record["routes"][0]["m"] = 32
    elif fault == "live":
        call = next(o for o in module.walk() if isinstance(o, func.CallOp))
        call.arguments[2].replace_all_uses_with(call.arguments[0])
    elif fault == "collision":
        clone = decl.clone()
        clone.properties["sym_name"] = StringAttr(record["routes"][0]["symbol"])
        module.body.block.add_op(clone)
    elif fault == "proof":
        record = copy.deepcopy(record)
        record["routes"][0]["proof"]["predictors"][0]["scale"] *= 2
    elif fault == "writer":
        record["routes"][0]["fully_written_arguments"] = [2]
    before = str(module)
    with pytest.raises(ValueError):
        rewrite(module, original, record)
    assert str(module) == before


def test_route_name_does_not_select_numeric_strategy():
    one, original, record = fixture(name="unrelated_operation")
    two, second, other = fixture(name="arbitrary_external")
    rewrite(one, original, record)
    rewrite(two, second, other)
    assert record["routes"][0]["proof"] == other["routes"][0]["proof"]


@pytest.mark.parametrize("guard", [False, True])
def test_native_ranked_abi_all65536_pairs_and_dirty_guards(tmp_path, guard):
    _, _, record = fixture(m=1024)
    route = record["routes"][0]
    source = tmp_path / "joint.c"
    source.write_text(adapter(route, first_output_guard=guard) + oracle(route))
    lib = tmp_path / "joint.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            str(source),
            "-lm",
            "-o",
            str(lib),
        ],
        check=True,
    )

    class Memref(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * 2),
            ("strides", ctypes.c_int64 * 2),
        ]

    a = np.repeat(np.arange(-128, 128, dtype=np.int8), 256)
    b = np.tile(np.arange(-128, 128, dtype=np.int8), 256)
    first = np.full(65536 + 17, 0x55, dtype=np.int8)
    second = np.full(65536 + 17, 0x66, dtype=np.int8)
    # Nonzero odd offsets exercise the scalar decoder fallback and descriptor offsets.
    arrays = (a, b, first, second)
    offsets = (0, 0, 3, 5)
    descriptors = [
        Memref(v.ctypes.data, v.ctypes.data, off, (1024, 64), (64, 1))
        for v, off in zip(arrays, offsets, strict=True)
    ]
    result = Memref()
    call = getattr(ctypes.CDLL(str(lib)), "_mlir_ciface_" + route["symbol"])
    call(*map(ctypes.byref, (result, *descriptors)))
    expected = source_table(**SOURCE).ravel()
    np.testing.assert_array_equal(first[3 : 3 + 65536], expected)
    np.testing.assert_array_equal(
        second[5 : 5 + 65536], predictor_table(**proof()["predictors"][1]).ravel()
    )
    assert np.all(first[:3] == 0x55) and np.all(first[3 + 65536 :] == 0x55)
    assert np.all(second[:5] == 0x66) and np.all(second[5 + 65536 :] == 0x66)
    np.testing.assert_array_equal(
        a, np.repeat(np.arange(-128, 128, dtype=np.int8), 256)
    )
    np.testing.assert_array_equal(b, np.tile(np.arange(-128, 128, dtype=np.int8), 256))
    assert result.aligned == first.ctypes.data and result.offset == 3


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_actual_upstream_private_lifetime_and_live_inputs(tmp_path, optimization):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang
    from merlin.xdsl_dialects._common import text

    module, original, record = fixture()
    rewrite(module, original, record)
    route = record["routes"][0]
    writers = transform(module, [DescriptorWriterContract(route["symbol"], 2, (2, 3))])
    source = tmp_path / "model.ll"
    source.write_text(
        lower_to_llvm_ir(
            text(module, generic=True), workdir=tmp_path / "lower", vectorize=False
        )
    )
    adapter_c, bridge = tmp_path / "adapter.c", tmp_path / "bridge.c"
    adapter_c.write_text(adapter(route, first_output_guard=True) + oracle(route))
    bridge.write_text(shim(writers))
    obj, lib = tmp_path / "model.o", tmp_path / f"model{optimization}.so"
    subprocess.run(
        [str(clang()), optimization, "-fPIC", "-c", str(source), "-o", str(obj)],
        check=True,
    )
    subprocess.run(
        [
            "cc",
            optimization,
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            str(obj),
            str(adapter_c),
            str(bridge),
            str(mlir_runtime_c()),
            "-lm",
            "-o",
            str(lib),
        ],
        check=True,
    )
    codes = (np.arange(3072, dtype=np.int64) * 40503 + 29) % 65536
    a = (codes // 256 - 128).astype(np.int8).reshape(48, 64)
    b = (codes % 256 - 128).astype(np.int8).reshape(48, 64)
    inputs = [a.copy(), b.copy()]
    results = [np.full((48, 64), 77, dtype=np.int8) for _ in range(3)]
    expected = source_table(**SOURCE)[
        a.astype(np.int16) + 128, b.astype(np.int16) + 128
    ]
    invoke = HostModel.load(str(lib))
    for _ in range(3):
        invoke([(v.ctypes.data, v.shape) for v in (*inputs, *results)])
        np.testing.assert_array_equal(inputs[0], a)
        np.testing.assert_array_equal(inputs[1], b)
        np.testing.assert_array_equal(results[0], a)
        np.testing.assert_array_equal(results[1], b)
        np.testing.assert_array_equal(results[2], expected)


def test_actual_target_catalog_compilation_and_binding_checks(tmp_path):
    from mlir_oot.captured_residual_bundle import sha
    from mlir_oot.direct_conv_binding import serialize
    from mlir_oot.joint_residual_catalog import build
    from mlir_oot.residual_mixed_catalog import merlin_callbacks

    llvm_bin = Path(
        "/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin"
    )
    if not (llvm_bin / "clang").is_file():
        pytest.skip("owned LLVM toolchain unavailable")
    module, original, _ = fixture()
    original_bundle = tmp_path / "original"
    original_bundle.mkdir()
    dummy = original_bundle / "native_oracle.c"
    dummy.write_text("void original_unused(void) {}\n")
    subprocess.run(
        [
            str(llvm_bin / "clang"),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-c",
            str(dummy),
            "-o",
            str(original_bundle / "residual.o"),
        ],
        check=True,
    )
    original.update(
        numeric_policy={"max_output_lsb": 0},
        object_sha256=sha(original_bundle / "residual.o"),
        native_oracle_sha256=sha(dummy),
    )
    (original_bundle / "residual.json").write_text(
        json.dumps(original, indent=2) + "\n"
    )
    certificate = tmp_path / "proof.json"
    certificate.write_text(json.dumps(proof(), indent=2) + "\n")
    selected = tmp_path / "selected"
    result = build(
        original_bundle, [certificate], llvm_bin, selected, first_output_guard=True
    )
    assert result["nofsm_audit"]["status"] == "pass"
    assert result["routes"][0]["original_route"] == original["routes"][0]
    assert result["routes"][0]["compilation"]["object_nofsm_status"] == "pass"
    source = tmp_path / "source.mlir"
    source.write_text(serialize(module, []))
    base_calls = []

    def prepare_base(source, work):
        base_calls.append("prepare")
        return source

    def build_base(source, work):
        base_calls.append("build")
        path = Path(work) / "catalog.json"
        path.write_text(
            json.dumps(
                {
                    "compilation": {"schema": "unit_test_base"},
                    "total_device_contractions": 0,
                    "native_oracle_sources": [],
                }
            )
        )
        return path, original_bundle / "residual.o"

    default_prepare, _ = merlin_callbacks(
        llvm_bin, original_bundle, (prepare_base, build_base)
    )
    default_bytes = source.read_bytes()
    assert default_prepare(source, tmp_path) == source
    assert source.read_bytes() == default_bytes
    prepare, compile_catalog = merlin_callbacks(
        llvm_bin, original_bundle, (prepare_base, build_base), joint_bundle=selected
    )
    work = tmp_path / "work"
    work.mkdir()
    prepared = prepare(source, work)
    parsed = parse_module(prepared.read_text())
    parsed.verify()
    assert (
        len(next(o for o in parsed.walk() if isinstance(o, func.CallOp)).arguments) == 4
    )
    # Use a distinct empty base object; the ordinary provider's original object
    # is deliberately linked only once, just as in a real composed catalog.
    empty = tmp_path / "empty.c"
    empty.write_text("void remaining_catalog(void) {}\n")
    subprocess.run(
        [
            str(llvm_bin / "clang"),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-c",
            str(empty),
            "-o",
            str(work / "base.o"),
        ],
        check=True,
    )

    def real_base(source, directory):
        path, _obj = build_base(source, directory)
        return path, work / "base.o"

    prepare, compile_catalog = merlin_callbacks(
        llvm_bin, original_bundle, (prepare_base, real_base), joint_bundle=selected
    )
    catalog, obj = compile_catalog(prepared, work)
    manifest = json.loads(catalog.read_text())
    assert obj.is_file() and manifest["compilation"]["object_nofsm_status"] == "pass"
    assert manifest["residual_additions"][0]["fully_written_arguments"] == [2, 3]
    assert manifest["total_device_contractions"] == 1
    assert len(manifest["native_oracle_sources"]) == 2
    certificate.write_text(certificate.read_text() + " ")
    calls_before = list(base_calls)
    with pytest.raises(ValueError, match="binding changed"):
        prepare(source, work)
    assert base_calls == calls_before
