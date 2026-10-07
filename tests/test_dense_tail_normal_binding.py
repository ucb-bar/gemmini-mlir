"""Normal option and accepted borrowed-view clone preserve the actual schedule."""

import hashlib
import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest
from xdsl.dialects.builtin import StringAttr

from mlir_oot.captured_requant_bundle import build
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.segmented_input_binding import derive


def test_nonnumeric_normal_option_refuses_before_reading_capture(tmp_path):
    with pytest.raises(ValueError, match="dense tail selection must be boolean"):
        build(
            Path("missing"),
            Path("missing"),
            tmp_path / "not_created",
            dense_tail_before_last_full=1,
        )
    assert not (tmp_path / "not_created").exists()


def test_acceptance_clone_preserves_tail_in_emitted_source_graph():
    spec = importlib.util.spec_from_file_location(
        "binding_fixture", Path(__file__).with_name("test_segmented_input_binding.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source = (
        helper.SOURCE.replace("1x3x5x2", "1x9x9x2")
        .replace("1x2x2x2", "1x5x7x2")
        .replace("tensor<8xi8>", "tensor<70xi8>")
        .replace("4x2", "35x2")
        .replace("4x3", "35x3")
        .replace("array<i64: 1,2,2,2>", "array<i64: 1,5,7,2>")
        .replace("array<i64: 1,2,2,1>", "array<i64: 1,2,1,1>")
        .replace("array<i64: 4,2>", "array<i64: 35,2>")
    )
    module = parse_module(source)
    s = Shape(35, 3, 2, bm=3, bn=1, cache_a=True, reuse_b=True, wide_b=True, scale=0.25)
    emitted = GoldenGemm(s, stationary_b_tail_before_last_full=True).build()
    emitted.body.block.first_op.properties["sym_name"] = StringAttr("consume_kernel")
    route = {
        "symbol": "consume",
        "kernel": "consume_kernel",
        "direct_conv": False,
        "schedule": asdict(s),
        "numeric_contract": {"max_output_lsb_error": 0},
        "dense_stationary_tail_decision": {"applied": True},
        "proof": {},
        "compilation": {
            "target_ir_sha256": hashlib.sha256(
                (str(emitted) + "\n").encode()
            ).hexdigest()
        },
    }
    bindings, refused = derive(module, [route])
    assert len(bindings) == 1 and not refused
    selected = bindings[0].generator
    assert selected.stationary_b_tail_before_last_full
    assert selected.input_view.rows == 35 and selected.input_view.cols == 2
    actual = selected.build()
    assert "gemmini.stationary_b_tail_before_last_full" in actual.attributes
    # A missing witness must refuse reconstruction of this actual admitted IR,
    # rather than silently compiling the default order under selected metadata.
    route = dict(route, dense_stationary_tail_decision={"applied": False})
    bindings, refused = derive(module, [route])
    assert not bindings and len(refused) == 1
    assert "reconstructed control differs" in refused[0]["reason"]

    for invalid in (1, "true", None):
        malformed = dict(route, dense_stationary_tail_decision={"applied": invalid})
        bindings, refused = derive(module, [malformed])
        assert not bindings and len(refused) == 1
        assert "boolean witness" in refused[0]["reason"]
