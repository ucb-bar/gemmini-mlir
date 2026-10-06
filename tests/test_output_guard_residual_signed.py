"""The three-argument provider also closes signed, non-ReLU source arithmetic."""

import ctypes
import hashlib
import subprocess

import numpy as np
import pytest
from merlin.llvmlower.quantized_affine_pair import derive, source_table
from test_joint_residual_catalog import fixture

from mlir_oot.captured_residual_bundle import binding_attributes, canonical
from mlir_oot.joint_residual_catalog import adapter, oracle, rewrite


@pytest.mark.parametrize("offset", [0, 3])
def test_signed_nonrelu_full_domain_and_ranked_guards(tmp_path, offset):
    module, original, record = fixture(m=1024, single_output_guard=True)
    route, old = record["routes"][0], original["routes"][0]
    contract = {**old["proof"]["source"], "relu": False}
    # Independently close the original synthetic provider's exact contract.
    control = derive(**contract, **old["proof"]["coefficients"])
    assert control["mismatched_pairs"] == 0
    old["proof"]["source"] = contract
    selected = derive(**contract, **route["proof"]["predictor"])
    assert selected["mismatched_pairs"] == 2
    route["proof"] = selected
    route["proof_sha256"] = hashlib.sha256(canonical(selected).encode()).hexdigest()
    module.body.block.last_op.attributes.update(
        binding_attributes(
            old, original["source_sha256"], original["capture_receipt_sha256"]
        )
    )
    rewrite(module, original, record)
    source, library = tmp_path / "adapter.c", tmp_path / "adapter.so"
    source.write_text(adapter(route, first_output_guard=False) + oracle(route))
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
            str(library),
        ],
        check=True,
    )

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * 2),
            ("strides", ctypes.c_int64 * 2),
        ]

    a = np.repeat(np.arange(-128, 128, dtype=np.int8), 256)
    b = np.tile(np.arange(-128, 128, dtype=np.int8), 256)
    saved_a, saved_b = a.copy(), b.copy()
    output = np.full(65536 + 17, 77, dtype=np.int8)
    descriptors = [
        Descriptor(x.ctypes.data, x.ctypes.data, o, (1024, 64), (64, 1))
        for x, o in ((a, 0), (b, 0), (output, offset))
    ]
    result = Descriptor()
    call = getattr(ctypes.CDLL(str(library)), "_mlir_ciface_" + route["symbol"])
    call(*map(ctypes.byref, (result, *descriptors)))
    np.testing.assert_array_equal(
        output[offset : offset + 65536], source_table(**contract).ravel()
    )
    assert np.all(output[:offset] == 77) and np.all(output[offset + 65536 :] == 77)
    np.testing.assert_array_equal(a, saved_a)
    np.testing.assert_array_equal(b, saved_b)
    assert result.aligned == output.ctypes.data and result.offset == offset
