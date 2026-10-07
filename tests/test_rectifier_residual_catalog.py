"""Normal typed source binding and complete portable ranked oracle."""

import copy
import ctypes
import hashlib
import subprocess

import numpy as np
import pytest
from merlin.llvmlower.quantized_affine_pair import derive, source_table
from merlin.llvmlower.quantized_affine_rectifier import derive as synthesize
from test_joint_residual_catalog import SOURCE, fixture
from xdsl.dialects import func

from mlir_oot.captured_residual_bundle import canonical
from mlir_oot.descriptor_writer import DescriptorWriterContract, transform
from mlir_oot.golden_rectified_resadd import Capabilities
from mlir_oot.joint_residual_catalog import adapter, oracle, rewrite
from mlir_oot.rectifier_residual_catalog import kernel


def selected(m=48, name="source_affine"):
    module, original, record = fixture(m=m, name=name)
    route = record["routes"][0]
    proof = derive(**SOURCE, p=298, q=249, scale=0.0032446938566863537)
    certificate = synthesize(proof, max_pairs=1, indicator_family="axis_offsets")
    route.update(
        symbol=name + "__rectifier",
        kernel=name + "__rectifier_kernel",
        proof=certificate,
        proof_sha256=hashlib.sha256(canonical(certificate).encode()).hexdigest(),
        fully_written_arguments=[2],
        rectifier_pipeline=True,
        capabilities=vars(Capabilities(*([True] * 6))),
        coalesce_internal_spad=False,
        ordering_source=None,
    )
    return module, original, record


@pytest.mark.parametrize("name", ["renamed_source", "arbitrary_other_source"])
def test_same_source_relation_uses_existing_private_single_writer_route(name):
    module, original, record = selected(name=name)
    route = record["routes"][0]
    rewrite(module, original, record)
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    assert len(call.arguments) == 3
    decl = module.body.block.last_op
    assert (
        decl.attributes["gemmini.rectifier_proof_sha256"].data == route["proof_sha256"]
    )
    assert "gemmini.output_guard_proof_sha256" not in decl.attributes
    assert "gemmini.wide_integer_coefficients" not in decl.attributes
    writers = transform(module, [DescriptorWriterContract(route["symbol"], 2, (2,))])
    assert writers[0]["fresh_writers"] == [2]
    assert sum(op.name == "memref.alloc" for op in module.walk()) == 1
    module.verify()


def test_batch_schedule_preserves_typed_writer_and_numeric_binding():
    module, original, record = selected(m=80, name="arbitrary_affine_source")
    route = record["routes"][0]
    route.update(
        panel_batch=4,
        coalesce_internal_spad=True,
        ordering_source="/scratch2/agustin/wt/chipyard-stock/generators/gemmini/src/main/scala/gemmini",
    )
    implementation = kernel(route)
    assert "gemmini.rectifier_panel_batch" in implementation.attributes
    rewrite(module, original, record)
    writers = transform(module, [DescriptorWriterContract(route["symbol"], 2, (2,))])
    assert writers[0]["fresh_writers"] == [2]
    assert sum(op.name == "memref.alloc" for op in module.walk()) == 1
    module.verify()


@pytest.mark.parametrize("factor", [2, True, 4.0])
def test_unknown_catalog_batch_refuses_before_source_mutation(factor):
    module, original, record = selected()
    record["routes"][0]["panel_batch"] = factor
    before = str(module)
    with pytest.raises(ValueError, match="explicit panel batch"):
        rewrite(module, original, record)
    assert str(module) == before


def test_legacy_catalog_without_batch_has_identical_default_bytes():
    _, _, record = selected()
    route = record["routes"][0]
    original = str(kernel(route))
    route["panel_batch"] = 1
    assert str(kernel(route)) == original


@pytest.mark.parametrize(
    "mutation", ["source", "certificate", "alias", "capability", "tail"]
)
def test_changed_source_or_unsupported_provider_refuses_before_rewrite(mutation):
    module, original, record = selected()
    old = str(module)
    route = record["routes"][0]
    if mutation == "source":
        route["original_route"] = copy.deepcopy(route["original_route"])
        route["original_route"]["proof"]["source"]["lhs_scale"] = 1
    elif mutation == "certificate":
        route["proof"]["relation"][0]["correction"] = 2
    elif mutation == "capability":
        route["capabilities"]["spad_execute_write"] = False
    elif mutation == "tail":
        route["m"] = 17
    else:
        call = next(op for op in module.walk() if isinstance(op, func.CallOp))
        call.operands = (call.arguments[0], call.arguments[1], call.arguments[0])
        old = str(module)
    with pytest.raises(ValueError):
        rewrite(module, original, record)
    assert str(module) == old


def test_ranked_oracle_all65536_pairs_misaligned_offsets_and_guards(tmp_path):
    _, _, record = selected(m=1024)
    route = record["routes"][0]
    source = tmp_path / "oracle.c"
    source.write_text(adapter(route, first_output_guard=False) + oracle(route))
    so = tmp_path / "oracle.so"
    subprocess.run(
        [
            "/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang",
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            str(source),
            "-lm",
            "-o",
            str(so),
        ],
        check=True,
        capture_output=True,
    )

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * 2),
            ("strides", ctypes.c_int64 * 2),
        ]

    count = 65536
    a = np.repeat(np.arange(-128, 128, dtype=np.int8), 256)
    b = np.tile(np.arange(-128, 128, dtype=np.int8), 256)
    arrays = []
    descriptors = []
    for offset, data in [(3, a), (5, b), (7, np.full(count, 101, dtype=np.int8))]:
        raw = np.full(count + offset + 64, 0xDB, dtype=np.uint8)
        raw[offset : offset + count] = data.view(np.uint8)
        arrays.append(raw)
        descriptors.append(
            Descriptor(
                raw.ctypes.data,
                raw.ctypes.data,
                offset,
                (ctypes.c_int64 * 2)(1024, 64),
                (ctypes.c_int64 * 2)(64, 1),
            )
        )
    before = [raw.copy() for raw in arrays]
    function = getattr(ctypes.CDLL(str(so)), "_mlir_ciface_" + route["symbol"])
    function.argtypes = [ctypes.POINTER(Descriptor)] * 4
    returned = Descriptor()
    function(ctypes.byref(returned), *[ctypes.byref(x) for x in descriptors])
    assert np.array_equal(arrays[0], before[0]) and np.array_equal(arrays[1], before[1])
    assert np.array_equal(
        arrays[2][7 : 7 + count].view(np.int8), source_table(**SOURCE).ravel()
    )
    assert np.array_equal(arrays[2][:7], before[2][:7]) and np.array_equal(
        arrays[2][7 + count :], before[2][7 + count :]
    )
    assert bytes(returned) == bytes(descriptors[2])
