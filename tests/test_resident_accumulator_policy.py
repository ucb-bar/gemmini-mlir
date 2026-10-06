"""Only typed storage and source schedules admit the explicit stripe family."""

from pathlib import Path

import pytest

from mlir_oot.captured_requant_bundle import build
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.resident_accumulator_policy import select_resident_accumulator_stripes


def test_cached_input_with_narrow_stores_binds_exact_resource_plan():
    s = Shape(
        784,
        512,
        128,
        bm=49,
        bn=1,
        cache_a=True,
        prefetch_b=True,
        wide_store=True,
        wide_b=True,
        reuse_b=True,
        scale=0.0040965937077999115,
    )
    old = GoldenGemm(s)
    selected, d = select_resident_accumulator_stripes(old)
    assert selected is not old and d["applied"] and not d["automatic_policy"]
    assert d["input_rows"] == 6272 and d["weight_rows"] == 512
    assert selected.shape.bm == 16 and selected.shape.bn == 4
    for field in ["m", "n", "k", "scale", "relu", "output_dtype", "bias"]:
        assert getattr(s, field) == getattr(selected.shape, field)
    assert d["accumulator_rows"] == 1024
    selected.build().verify()


@pytest.mark.parametrize(
    "shape",
    [
        Shape(3136, 256, 64, bm=1, bn=16, wide_store=True),
        Shape(196, 1024, 256, bm=13, bn=4, cache_a=True, wide_store=True),
        Shape(784, 512, 256, bm=49, bn=1, cache_a=True, wide_store=True),
        Shape(37, 33, 33, bm=3, bn=1, cache_a=True, wide_store=True),
        Shape(37, 69, 33, bm=3, bn=1, cache_a=True, wide_store=False),
        Shape(37, 69, 33, bm=3, bn=1, cache_a=True, output_dtype="i32"),
    ],
)
def test_unproved_or_already_wide_contract_keeps_original_generator(shape):
    original = GoldenGemm(shape)
    selected, d = select_resident_accumulator_stripes(original)
    assert selected is original and not d["applied"] and d["refusal"]


@pytest.mark.parametrize("flag", [None, 1, "yes"])
def test_option_refuses_untyped_flag_before_source_io(flag):
    with pytest.raises(ValueError, match="boolean"):
        build(
            Path("absent"),
            Path("absent"),
            Path("absent"),
            dense_accumulator_stripes=flag,
        )


@pytest.mark.parametrize(
    "option",
    [{"dense_b_slot_policy": "remaining_rows"}, {"resident_a_load_coalescing": True}],
)
def test_conflicting_declared_slot_option_refuses_before_source_io(option):
    with pytest.raises(ValueError, match="conflict"):
        build(
            Path("absent"),
            Path("absent"),
            Path("absent"),
            dense_accumulator_stripes=True,
            **option,
        )
