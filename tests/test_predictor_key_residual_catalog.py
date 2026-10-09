"""Normal source/effect binding for an exact key correction with private scratch."""

import copy
import ctypes
import subprocess

import numpy as np
import pytest
from merlin.llvmlower.quantized_affine_pair import derive, source_table
from merlin.llvmlower.quantized_affine_rectifier import derive as synthesize
from test_joint_residual_catalog import SOURCE
from test_rectifier_residual_catalog import selected
from xdsl.dialects import func

from mlir_oot.descriptor_writer import DescriptorWriterContract, transform
from mlir_oot.joint_residual_catalog import rewrite
from mlir_oot.rectifier_residual_catalog import adapter, kernel, oracle

ORDERING = (
    "/scratch2/agustin/wt/chipyard-stock/generators/gemmini/src/main/scala/gemmini"
)


def key_selected(m=80, name="renamed_quantized_add"):
    module, original, record = selected(m=m, name=name)
    route = record["routes"][0]
    proof = derive(**SOURCE, p=298, q=249, scale=0.0032446938566863537)
    route.update(
        proof=synthesize(proof, max_pairs=1, indicator_family="predictor_key"),
        identity_i32_acc_dma_accumulate=True,
        coalesce_internal_spad=True,
        ordering_source=ORDERING,
        panel_batch=4,
    )
    return module, original, record


@pytest.mark.parametrize("name", ["renamed_quantized_add", "independent_binding"])
def test_typed_route_retains_public_abi_and_private_scratch_contract(name):
    module, original, record = key_selected(name=name)
    route = record["routes"][0]
    rewrite(module, original, record)
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    assert len(call.arguments) == 3
    declaration = module.body.block.last_op
    assert (
        declaration.attributes["gemmini.rectifier_adapter_scratch_bytes"].value.data
        == 4096
    )
    assert (
        declaration.attributes["gemmini.rectifier_adapter_scratch_alignment"].value.data
        == 64
    )
    writers = transform(module, [DescriptorWriterContract(route["symbol"], 2, (2,))])
    assert writers[0]["fresh_writers"] == [2]
    assert sum(op.name == "memref.alloc" for op in module.walk()) == 1
    module.verify()


@pytest.mark.parametrize(
    "mutation",
    [
        "absent_rmw",
        "untyped_rmw",
        "absent_ordering",
        "uncoalesced",
        "certificate",
        "tail",
    ],
)
def test_unsupported_key_route_refuses_before_source_mutation(mutation):
    module, original, record = key_selected()
    route = record["routes"][0]
    if mutation == "absent_rmw":
        route.pop("identity_i32_acc_dma_accumulate")
    elif mutation == "untyped_rmw":
        route["identity_i32_acc_dma_accumulate"] = 1
    elif mutation == "absent_ordering":
        route["ordering_source"] = None
    elif mutation == "uncoalesced":
        route["coalesce_internal_spad"] = False
    elif mutation == "certificate":
        route["proof"] = copy.deepcopy(route["proof"])
        route["proof"]["relation"][0]["key"]["seed"] += 1
    else:
        route["m"] = 17
    before = str(module)
    with pytest.raises(ValueError):
        rewrite(module, original, record)
    assert str(module) == before


def test_existing_axis_route_keeps_four_pointer_primitive_without_scratch():
    for batch in (1, 4):
        _, _, record = selected(m=80)
        route = record["routes"][0]
        if batch == 4:
            route.update(
                panel_batch=4, coalesce_internal_spad=True, ordering_source=ORDERING
            )
        assert len(kernel(route).body.block.first_op.function_type.inputs) == 4
        assert "scratch" not in adapter(route)
        assert "scratch" not in oracle(route)


def test_all_distinct_source_pairs_ranked_native_and_private_stack(tmp_path):
    _, _, record = key_selected(m=1024)
    route = record["routes"][0]
    source = tmp_path / "adapter.c"
    source.write_text(adapter(route) + oracle(route))
    clang = "/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang"
    shared = tmp_path / "adapter.so"
    subprocess.run(
        [
            clang,
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            str(source),
            "-lm",
            "-o",
            str(shared),
        ],
        check=True,
        capture_output=True,
    )
    llvm = tmp_path / "adapter.ll"
    # The native oracle can inline away scratch. Qualify the actual target ABI
    # separately with an external device primitive declaration.
    source.write_text(adapter(route))
    subprocess.run(
        [
            clang,
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-O2",
            "-S",
            "-emit-llvm",
            "-ffreestanding",
            str(source),
            "-o",
            str(llvm),
        ],
        check=True,
        capture_output=True,
    )
    assert "alloca [4096 x i8], align 64" in llvm.read_text()

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
    arrays, descriptors = [], []
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
    function = getattr(ctypes.CDLL(str(shared)), "_mlir_ciface_" + route["symbol"])
    function.argtypes = [ctypes.POINTER(Descriptor)] * 4
    returned = Descriptor()
    function(ctypes.byref(returned), *[ctypes.byref(item) for item in descriptors])
    assert np.array_equal(arrays[0], before[0]) and np.array_equal(arrays[1], before[1])
    assert np.array_equal(
        arrays[2][7 : 7 + count].view(np.int8), source_table(**SOURCE).ravel()
    )
    assert np.array_equal(arrays[2][:7], before[2][:7])
    assert np.array_equal(arrays[2][7 + count :], before[2][7 + count :])
    assert bytes(returned) == bytes(descriptors[2])
