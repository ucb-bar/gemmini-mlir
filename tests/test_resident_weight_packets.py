"""Packet lookahead keeps output layout and original reduction semantics."""

import importlib.util
from pathlib import Path

import pytest

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_resident_conv import GoldenResidentConv

spec = importlib.util.spec_from_file_location(
    "packetproof", Path(__file__).with_name("resident_weight_packet_trace_probe.py")
)
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("tiles", [1, 2, 3, 4])
def test_packet_source_lifetime_and_all_output_tails(tiles, compact):
    s = ConvShape(5, 5, 32, 67, bn=4, output_dtype="i32")
    g = GoldenResidentConv(
        s,
        rows_per_tile=2,
        prefetch_b=True,
        weight_issue_tiles=tiles,
        compact_commands=compact,
    )
    result = proof.prove(g)
    assert result["all_output_cells_written_once"] == 1675
    assert result["requested_payload"] == {"load_bytes": 20096, "store_bytes": 6700}


@pytest.mark.parametrize("tiles", [0, 5, True, 2.0])
def test_packet_option_requires_typed_DMA_limit(tiles):
    with pytest.raises(ValueError, match="integer in 1..4"):
        GoldenResidentConv(
            ConvShape(3, 5, 16, 19), prefetch_b=True, weight_issue_tiles=tiles
        )


def test_packet_issue_refuses_incompatible_lifetime_and_loop_contract():
    s = ConvShape(3, 5, 16, 19)
    with pytest.raises(ValueError, match="bank-prefetched"):
        GoldenResidentConv(s, weight_issue_tiles=2)
    GoldenResidentConv(
        s, prefetch_b=True, compact_commands=True, weight_issue_tiles=2
    ).build().verify()


@pytest.mark.parametrize("cin", [48, 64, 80])
def test_retained_packet_pair_loop_closes_even_odd_K_and_N_groups(cin):
    shape = ConvShape(5, 5, cin, 67, bn=4, output_dtype="i32")
    generator = GoldenResidentConv(
        shape,
        rows_per_tile=2,
        prefetch_b=True,
        weight_issue_tiles=2,
        compact_commands=True,
    )
    actual = proof.prove(generator)
    assert actual["requested_payload"]["load_bytes"] == 628 * cin
    assert actual["all_output_cells_written_once"] == 1675


def test_normal_choice_keeps_semantics_and_refuses_other_families():
    from mlir_oot.golden_gemm import GoldenGemm, Shape
    from mlir_oot.golden_resident_conv import issue_resident_weight_packets

    s = ConvShape(5, 5, 32, 67, bn=4, output_dtype="i8", scale=0.013, relu=True)
    control = GoldenResidentConv(s, rows_per_tile=2)
    selected, decision = issue_resident_weight_packets(control)
    assert selected.conv == s and selected.rows_per_tile == 2
    assert (
        selected.weight_issue_tiles == 2
        and selected.compact_commands
        and selected.prefetch_b
    )
    assert decision["applied"] and decision["performance"] == "UNKNOWN"
    other = GoldenGemm(Shape(5, 67, 32))
    unchanged, refusal = issue_resident_weight_packets(other)
    assert unchanged is other and not refusal["applied"]
    for value in [0, True, 5]:
        with pytest.raises(ValueError):
            issue_resident_weight_packets(control, tiles=value)


def test_normal_packet_option_rejects_untyped_or_unproved_selection(tmp_path):
    from mlir_oot.captured_requant_bundle import build

    for value in [True, 0, 5, 2]:
        with pytest.raises(ValueError, match="resident weight"):
            build(
                tmp_path / "missing",
                tmp_path / "tools",
                tmp_path / "output",
                resident_weight_issue_tiles=value,
            )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("h,w,cin,cout", [(3, 5, 32, 67), (1, 17, 16, 19)])
def test_packet_layout_policy_preserves_flat_plan_without_claiming_illegality(
    h, w, cin, cout
):
    from merlin.xdsl_dialects._common import text

    from mlir_oot.golden_resident_conv import issue_resident_weight_packets

    shape = ConvShape(h, w, cin, cout, bn=4, output_dtype="i32")
    control = GoldenResidentConv(shape, flat_spatial_planes=True, compact_commands=True)
    unchanged, refusal = issue_resident_weight_packets(
        control, include_flat_planes=False
    )
    assert unchanged is control
    assert not refusal["applied"] and refusal["performance"] == "UNKNOWN"
    assert "layout policy" in refusal["refusal"]
    assert proof.prove(unchanged)["all_output_cells_written_once"] == h * w * cout

    default, default_decision = issue_resident_weight_packets(
        GoldenResidentConv(shape, flat_spatial_planes=True, compact_commands=True)
    )
    explicit, explicit_decision = issue_resident_weight_packets(
        GoldenResidentConv(shape, flat_spatial_planes=True, compact_commands=True),
        include_flat_planes=True,
    )
    assert default_decision == explicit_decision and default_decision["applied"]
    assert text(default.build(), generic=True) == text(explicit.build(), generic=True)


def test_packet_layout_policy_retains_channel_plane_choice_and_type_gate(tmp_path):
    from mlir_oot.captured_requant_bundle import build
    from mlir_oot.golden_resident_conv import issue_resident_weight_packets

    shape = ConvShape(5, 3, 32, 67, bn=4, output_dtype="i32")
    selected, decision = issue_resident_weight_packets(
        GoldenResidentConv(shape, rows_per_tile=2), include_flat_planes=False
    )
    assert decision["applied"] and not selected.flat_spatial_planes
    assert proof.prove(selected)["all_output_cells_written_once"] == 1005
    for value in [None, 0, 1, "channel_planes"]:
        with pytest.raises(ValueError, match="flat-plane"):
            issue_resident_weight_packets(selected, include_flat_planes=value)
        with pytest.raises(ValueError, match="flat-plane"):
            build(
                tmp_path / "missing",
                tmp_path / "tools",
                tmp_path / "output",
                resident_weight_issue_flat_planes=value,
            )
    assert not (tmp_path / "output").exists()
