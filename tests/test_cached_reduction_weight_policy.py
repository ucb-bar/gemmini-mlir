"""Explicit normal factory selection closes family/resource/cost/refusal facts."""

import inspect
from dataclasses import asdict, replace

import pytest
from test_resident_conv_command_loops import executed_commands

from mlir_oot.captured_requant_bundle import build
from mlir_oot.conv_schedule import select_cached_reduction_weights, select_kernel
from mlir_oot.golden_conv import ConvShape, GoldenConv
from mlir_oot.golden_flat_conv import GoldenFlatConv, command_counts


@pytest.mark.parametrize(
    "s",
    [
        ConvShape(39, 35, 32, 67, stride=2, bn=4),
        ConvShape(
            56,
            56,
            128,
            128,
            stride=2,
            bn=4,
            output_dtype="i8",
            scale=0.0015212674625217915,
            relu=True,
            wide_b=True,
        ),
    ],
)
def test_selector_changes_only_explicit_weight_placement_and_requested_traffic(s):
    factory_control, _oldkind = select_kernel(
        s, flat_spatial=True, virtual_padding=True
    )
    control = factory_control.with_emission_options(separate_b_bank=False)
    before_options = control.emission_options
    candidate, d = select_cached_reduction_weights(control)
    assert candidate is not control and candidate.cached_reduction_weights
    assert candidate.emission_options == replace(
        before_options, cached_reduction_weights=True
    )
    assert control.emission_options == before_options
    assert (
        d["applied"]
        and not d["timing_claim"]
        and d["performance"] == "UNKNOWN"
        and not d["automatic_policy"]
    )
    assert asdict(control.conv) == asdict(candidate.conv)
    assert (
        control.band_rows == candidate.band_rows and control.wide_a == candidate.wide_a
    )
    assert d["candidate"]["weight_bytes"] < d["control"]["weight_bytes"]
    for m in ("compute", "preload", "mvin_a", "mvout", "padded_array_issue_cycles"):
        a = command_counts(
            control.conv,
            wide_a=control.wide_a,
            band_rows=control.band_rows,
            virtual_padding=True,
        )
        b = command_counts(
            candidate.conv,
            wide_a=candidate.wide_a,
            band_rows=candidate.band_rows,
            virtual_padding=True,
            cached_reduction_weights=True,
        )
        assert a[m] == b[m]


def test_actual_cfg_matches_requested_b_census():
    s = ConvShape(39, 35, 32, 67, stride=2, bn=4)
    factory_control, _ = select_kernel(s, flat_spatial=True, virtual_padding=True)
    control = factory_control.with_emission_options(separate_b_bank=False)
    c, decision = select_cached_reduction_weights(control)
    assert decision["applied"]
    commands = executed_commands(c.build())
    b = [row for row in commands if row[0] == "gemmini.mvin" and row[2][0].base == 1]
    counts = command_counts(
        c.conv,
        wide_a=c.wide_a,
        band_rows=c.band_rows,
        virtual_padding=True,
        cached_reduction_weights=True,
    )
    assert len(b) == counts["mvin_b"]
    assert (
        sum((row[1][2] >> 48) * ((row[1][2] >> 32) & 65535) for row in b)
        == counts["weight_bytes"]
    )


@pytest.mark.parametrize(
    "s", [ConvShape(7, 7, 512, 512, explicit_halo=True), ConvShape(5, 21, 17, 19)]
)
def test_illegal_full_weight_placement_preserves_default_family_and_ir(s):
    control, kind = select_kernel(
        s, flat_spatial=True, virtual_padding=not s.explicit_halo
    )
    candidate, selected = select_kernel(
        s,
        flat_spatial=True,
        virtual_padding=not s.explicit_halo,
        cached_reduction_weights=True,
    )
    assert (
        kind == selected and not candidate.cached_reduction_weight_decision["applied"]
    )
    assert str(control.build()) == str(candidate.build())


def test_one_band_without_lower_weight_work_retains_source():
    s = ConvShape(3, 7, 16, 16, bn=1, explicit_halo=True)
    control = GoldenFlatConv(s)
    candidate, d = select_cached_reduction_weights(control)
    assert candidate is control and not d["applied"] and "do not lower" in d["refusal"]


@pytest.mark.parametrize("loop_spatial", [False, True])
def test_separate_bank_selector_refuses_and_preserves_complete_control(loop_spatial):
    shape = ConvShape(39, 35, 32, 67, stride=2, bn=4)
    control, _kind = select_kernel(
        shape,
        flat_spatial=True,
        virtual_padding=True,
        spatial_command_loops=loop_spatial,
    )
    assert control.separate_b_bank
    before_options = control.emission_options
    before_ir = str(control.with_emission_options().build())
    selected, decision = select_cached_reduction_weights(control)
    assert selected is control and not decision["applied"]
    assert "exclusive scratchpad placement" in decision["refusal"]
    assert selected.emission_options == before_options
    assert str(selected.with_emission_options().build()) == before_ir


@pytest.mark.parametrize("loop_spatial", [False, True])
def test_normal_factory_does_not_override_selected_separate_bank(loop_spatial):
    shape = ConvShape(39, 35, 32, 67, stride=2, bn=4)
    control, kind = select_kernel(
        shape,
        flat_spatial=True,
        virtual_padding=True,
        spatial_command_loops=loop_spatial,
    )
    selected, selected_kind = select_kernel(
        shape,
        flat_spatial=True,
        virtual_padding=True,
        spatial_command_loops=loop_spatial,
        cached_reduction_weights=True,
    )
    assert selected_kind == kind
    assert not selected.cached_reduction_weight_decision["applied"]
    assert selected.separate_b_bank
    assert selected.emission_options == control.emission_options
    assert str(selected.build()) == str(control.build())


def test_other_resident_families_refuse_without_replacing():
    control = GoldenConv(ConvShape(5, 7, 16, 19))
    candidate, d = select_cached_reduction_weights(control)
    assert candidate is control and not d["applied"]
    c, _kind = select_kernel(
        ConvShape(28, 28, 256, 256, stride=2),
        flat_spatial=True,
        virtual_padding=True,
        source_stride_resident=True,
        cached_reduction_weights=True,
    )
    assert (
        not isinstance(c, GoldenFlatConv)
        and not c.cached_reduction_weight_decision["applied"]
    )


@pytest.mark.parametrize("value", [0, 1, None, "yes"])
def test_normal_entrypoints_refuse_untyped_choices_before_file_access(value, tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        select_kernel(
            ConvShape(5, 7, 16, 19), flat_spatial=True, cached_reduction_weights=value
        )
    with pytest.raises(ValueError, match="boolean"):
        build(
            tmp_path,
            tmp_path,
            tmp_path,
            flat_spatial=True,
            flat_cached_reduction_weights=value,
        )


def test_defaults_are_false_at_both_normal_entrypoints():
    assert (
        inspect.signature(select_kernel).parameters["cached_reduction_weights"].default
        is False
    )
    assert (
        inspect.signature(build).parameters["flat_cached_reduction_weights"].default
        is False
    )
