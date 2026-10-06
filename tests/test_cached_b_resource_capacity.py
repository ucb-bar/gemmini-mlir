"""Full-weight resource admission and unchanged legacy policy boundaries."""

from dataclasses import replace

import pytest

from mlir_oot.dense_schedule import (
    choose_capacity_cached_b,
    select_capacity_cached_b,
    select_kernel,
)
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.tables import rtl_facts as F


def source_shape():
    return Shape(
        784,
        256,
        512,
        "i8",
        bm=8,
        bn=8,
        wide_b=True,
        wide_store=True,
        reuse_b=True,
        separate_b_bank=True,
        scale=0.002614574274048209,
        relu=True,
    )


def test_full_weight_capacity_exceeds_compile_budget_with_exact_resource_proof():
    original = GoldenGemm(source_shape())
    generator, proof = select_capacity_cached_b(original)
    s = generator.shape
    assert (s.m, s.n, s.k, s.output_dtype, s.bias, s.scale, s.relu) == (
        original.shape.m,
        original.shape.n,
        original.shape.k,
        original.shape.output_dtype,
        original.shape.bias,
        original.shape.scale,
        original.shape.relu,
    )
    assert s.bm == 1 and s.bn == 16 and s.cache_b and s.banked_m and s.prefetch_m
    assert proof["weight_reserved_interval"] == [8192, 16384]
    assert proof["input_reserved_intervals"] == [[0, 512], [4096, 4608]]
    assert proof["cached_weight_tiles"] == 512
    assert proof["accumulator_rows"] == 512
    assert proof["real_B_mesh_preloads"] == 25088
    with pytest.raises(ValueError, match="static or scratchpad"):
        s.validate()
    s.validate(cached_b_resource_capacity=True)
    generator.build().verify()


def test_independent_partial_shapes_and_reduction_tail():
    original = GoldenGemm(Shape(37, 69, 513, "i32", bm=2, bn=3, wide_b=True))
    generator, proof = select_capacity_cached_b(original)
    assert generator.shape.cache_b and not generator.shape.wide_a
    assert not generator.shape.pipeline_m
    assert proof["cached_weight_tiles"] == 165
    assert proof["weight_reserved_interval"] == [8192, 10832]
    assert proof["input_reserved_intervals"] == [[0, 16]]
    module = generator.build()
    module.verify()
    loads = [
        op
        for op in module.walk()
        if op.name == "gemmini.mvin" and op.attributes["load_id"].value.data == 1
    ]
    assert {op.attributes["rows"].value.data for op in loads} == {1, 16}
    assert {op.attributes["cols"].value.data for op in loads} == {5, 64}


def test_actual_placed_extent_and_accumulator_refuse_overflow():
    for n in (257, 512):
        original = GoldenGemm(replace(source_shape(), n=n))
        with pytest.raises(ValueError, match="cached B"):
            select_capacity_cached_b(original)
    with pytest.raises(ValueError, match="accumulator"):
        select_capacity_cached_b(GoldenGemm(source_shape()), row_tiles=3)
    with pytest.raises(ValueError, match="one stock scratchpad bank"):
        select_capacity_cached_b(GoldenGemm(Shape(513, 16, 4096)), row_tiles=2)


def test_legacy_default_and_policy_stay_bounded():
    s = source_shape()
    control, kind = select_kernel(s, transfer_command_policy=True)
    assert control.shape == s and kind == "dense_gemm"
    with pytest.raises(ValueError, match="static or scratchpad"):
        GoldenGemm(replace(s, bm=1, bn=16, cache_b=True))
    assert not control.cached_b_resource_capacity
    with pytest.raises(ValueError, match="requires cached B"):
        GoldenGemm(s, cached_b_resource_capacity=True)
    with pytest.raises(ValueError, match="boolean"):
        GoldenGemm(s, cached_b_resource_capacity=1)
    assert F.SPAD_ROWS == 16384 and F.ACC_ROWS == 1024


def test_explicit_choice_rejects_inert_or_unproved_contracts():
    with pytest.raises(ValueError, match="positive integer"):
        select_capacity_cached_b(GoldenGemm(source_shape()), row_tiles=True)
    with pytest.raises(ValueError, match="dense unseeded"):
        select_capacity_cached_b(GoldenGemm(replace(source_shape(), bias=True)))


def test_normal_admission_preserves_existing_residency_and_refuses_overflow():
    source = GoldenGemm(source_shape())
    candidate, decision = choose_capacity_cached_b(source)
    assert candidate.cached_b_resource_capacity and decision["applied"]
    for shape in [
        replace(source_shape(), n=512),
        Shape(37, 69, 65, "i32", bm=2, bn=3),
        Shape(196, 256, 1024, "i8", bm=13, bn=4, cache_a=True, reuse_b=True),
    ]:
        control = GoldenGemm(shape)
        unchanged, decision = choose_capacity_cached_b(control)
        assert unchanged is control and not decision["applied"]
        assert decision["performance"] == "UNKNOWN" and decision["refusal"]
