"""Complete constructor records survive real compiler factories and refusals."""

from dataclasses import FrozenInstanceError, fields, replace
from inspect import Parameter, signature

import pytest
from merlin.llvmlower.segmented_matrix_view import SegmentedRows

from mlir_oot.b_slot_placement import select_remaining_b_slots
from mlir_oot.dense_schedule import (
    select_coalesced_resident_a,
    select_stationary_b_spatial_tail,
)
from mlir_oot.golden_conv import ConvShape, GoldenConv
from mlir_oot.golden_flat_conv import GoldenFlatConv
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.golden_resident_conv import (
    GoldenResidentConv,
    place_resident_spatial_tail,
    retain_resident_commands,
)
from mlir_oot.golden_resident_stripe_conv import GoldenResidentStripeConv
from mlir_oot.paired_readout_binding import choose
from mlir_oot.readout_store_plan import PairedReadoutPlan
from mlir_oot.resident_stripe_reduction_policy import retain_reduction_commands

PLAN = PairedReadoutPlan((0.125,), (0.125, 0.125), -(1 << 31), (1 << 31) - 1)


def dense():
    return GoldenGemm(
        Shape(37, 35, 33, bm=3, bn=3, cache_a=True, reuse_b=True),
        resident_a_load_tiles=4,
        stationary_b_tail_before_last_full=True,
    )


def flat():
    return GoldenFlatConv(
        ConvShape(5, 7, 32, 35, bn=3, output_dtype="i32"),
        wide_a=True,
        separate_b_bank=True,
        band_rows=3,
        virtual_padding=True,
        pingpong_b=True,
        loop_spatial=True,
        store_plan=PLAN,
    )


def resident():
    return GoldenResidentConv(
        ConvShape(5, 7, 32, 35, bn=3, output_dtype="i32"),
        compact_commands=True,
        flat_spatial_planes=True,
        prefetch_b=True,
        weight_issue_tiles=2,
        store_plan=PLAN,
        tail_before_last_full=True,
    )


def stripe():
    return GoldenResidentStripeConv(
        ConvShape(8, 17, 16, 35, bn=3),
        stripe_rows=3,
        compact_inner_commands=True,
        compact_reduction_commands=True,
    )


def cached_dense():
    return GoldenGemm(
        Shape(37, 35, 64, bm=2, bn=3, cache_b=True, wide_a=True),
        cached_b_resource_capacity=True,
    )


@pytest.mark.parametrize("make", [dense, flat, resident, stripe, cached_dense])
def test_complete_immutable_record_and_clone_IR_identity(make):
    source = make()
    record = source.emission_options
    params = tuple(signature(type(source).__init__).parameters.values())[2:]
    assert [f.name for f in fields(record)] == [p.name for p in params]
    with pytest.raises(FrozenInstanceError):
        setattr(record, fields(record)[0].name, None)
    copied = source.with_emission_options()
    assert copied is not source and copied.emission_options == record
    assert copied.with_emission_options().emission_options == record
    assert str(source.build()) == str(copied.build())


def test_dense_factories_preserve_every_unmodified_fact_and_view():
    view = SegmentedRows(
        rows=37,
        cols=33,
        segment_rows=7,
        row_stride=66,
        segment_stride=1000,
        source_elements=6000,
        origin=3,
        element_bytes=1,
        dtype="i8",
    )
    g = GoldenGemm(
        Shape(37, 35, 33, bm=3, bn=3, cache_a=True, reuse_b=True),
        input_view=view,
    )
    coalesced, decision = select_coalesced_resident_a(g)
    assert decision["applied"]
    assert coalesced.emission_options == replace(
        g.emission_options, resident_a_load_tiles=4
    )
    tail, decision = select_stationary_b_spatial_tail(coalesced)
    assert decision["applied"]
    assert tail.emission_options == replace(
        coalesced.emission_options, stationary_b_tail_before_last_full=True
    )
    slotted, decision = select_remaining_b_slots(tail)
    assert decision["applied"]
    assert slotted.emission_options == replace(
        tail.emission_options, prefetch_b_rows=tuple(decision["prefetch_b_rows"])
    )
    assert slotted.input_view is view
    assert slotted.cached_b_resource_capacity == g.cached_b_resource_capacity
    slotted.build().verify()


@pytest.mark.parametrize("make", [flat, resident])
def test_paired_factory_changes_only_store_plan(make):
    source = make()
    proof = {
        "source_scales": [0.125],
        "accumulator_min": -(1 << 31),
        "accumulator_max": (1 << 31) - 1,
        "output_min": 0,
    }
    selected, plan, decision = choose(source, proof)
    assert decision["applied"]
    assert selected.emission_options == replace(
        source.emission_options, store_plan=plan
    )
    assert selected.store_plan is plan
    selected.build().verify()


def test_resident_modifiers_change_only_declared_facts():
    source = GoldenResidentConv(
        ConvShape(5, 7, 32, 35, bn=3, output_dtype="i32"),
        flat_spatial_planes=True,
        store_plan=PLAN,
    )
    compact, decision = retain_resident_commands(source)
    assert decision["applied"]
    assert compact.emission_options == replace(
        source.emission_options, compact_commands=True, loop_channels=True
    )
    tail, decision = place_resident_spatial_tail(compact)
    assert decision["applied"]
    assert tail.emission_options == replace(
        compact.emission_options, tail_before_last_full=True
    )
    tail.build().verify()
    source = GoldenResidentStripeConv(ConvShape(8, 17, 32, 35, bn=3), stripe_rows=3)
    selected, decision = retain_reduction_commands(source)
    assert decision["applied"]
    assert selected.emission_options == replace(
        source.emission_options,
        compact_inner_commands=True,
        compact_reduction_commands=True,
    )


def test_explicit_weight_alias_and_legacy_public_attribute_snapshot():
    g = GoldenResidentConv(ConvShape(5, 7, 16, 19, bn=2), weight_base=256)
    assert g.emission_options.weight_base == 256
    assert g.with_emission_options().explicit_weight_base == 256
    g = dense()
    g.resident_a_load_tiles = 2
    assert g.emission_options.resident_a_load_tiles == 2
    assert g.with_emission_options().resident_a_load_tiles == 2


def test_unknown_option_and_resource_failure_leave_source_unchanged():
    g = dense()
    old = g.emission_options
    with pytest.raises(ValueError, match="unknown emission option"):
        g.with_emission_options(future_option=True)
    with pytest.raises(ValueError, match="resident A DMA grouping"):
        g.with_emission_options(resident_a_load_tiles=5)
    assert g.emission_options == old
    with pytest.raises(ValueError, match="prefetched B slots overlap"):
        g.with_emission_options(
            shape=replace(g.shape, prefetch_b=True), prefetch_b_rows=(4096, 4096)
        )
    assert g.emission_options == old


def test_new_constructor_option_without_record_refuses(monkeypatch):
    sig = signature(GoldenGemm.__init__)
    new = sig.replace(
        parameters=[
            *sig.parameters.values(),
            Parameter("future_option", Parameter.KEYWORD_ONLY, default=False),
        ]
    )
    monkeypatch.setattr(GoldenGemm.__init__, "__signature__", new, raising=False)
    with pytest.raises(ValueError, match="complete constructor API"):
        dense().with_emission_options()


def test_changed_constructor_default_refuses(monkeypatch):
    sig = signature(GoldenGemm.__init__)
    new = sig.replace(
        parameters=[
            p.replace(default=2) if p.name == "resident_a_load_tiles" else p
            for p in sig.parameters.values()
        ]
    )
    monkeypatch.setattr(GoldenGemm.__init__, "__signature__", new, raising=False)
    with pytest.raises(ValueError, match="defaults differ"):
        dense().with_emission_options()


def test_undeclared_subclass_cannot_inherit_a_wrong_family_record():
    class Other(GoldenGemm):
        pass

    with pytest.raises(ValueError, match="no declared emission"):
        Other(Shape(1, 1, 1)).with_emission_options()
    # The ordinary row convolution is deliberately outside this replacement API.
    with pytest.raises(ValueError, match="no declared emission"):
        GoldenConv(ConvShape(3, 5, 16, 19)).with_emission_options()
