"""Flat resident planes close source addresses across logical row boundaries."""

import importlib.util
from pathlib import Path

import pytest

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_resident_conv import GoldenResidentConv

spec = importlib.util.spec_from_file_location(
    "proof", Path(__file__).with_name("resident_weight_packet_trace_probe.py")
)
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


@pytest.mark.parametrize(
    "h,w,cin,cout",
    [(3, 5, 16, 19), (5, 3, 32, 67), (1, 1, 16, 1), (3, 15, 32, 23), (1, 17, 16, 19)],
)
@pytest.mark.parametrize(
    "compact,prefetch,packet",
    [(False, False, None), (True, False, None), (True, True, None), (True, True, 2)],
)
def test_all_source_cells_order_lifetimes_and_tails(
    h, w, cin, cout, compact, prefetch, packet
):
    s = ConvShape(h, w, cin, cout, bn=4, output_dtype="i32")
    g = GoldenResidentConv(
        s,
        flat_spatial_planes=True,
        compact_commands=compact,
        prefetch_b=prefetch,
        weight_issue_tiles=packet,
    )
    assert g.plane == 3 * (h + 2) * w
    result = proof.prove(g)
    assert result["all_output_cells_written_once"] == h * w * cout
    assert result["requested_payload"]["store_bytes"] == h * w * cout * 4


def test_resource_endpoints_and_incompatible_layout_refusals():
    g = GoldenResidentConv(
        ConvShape(7, 7, 512, 512, bn=16, output_dtype="i32"), flat_spatial_planes=True
    )
    assert g.plane * 32 == 6048 and len(g.row_tiles) * 16 * 16 == 1024
    with pytest.raises(ValueError, match="overlaps weight"):
        GoldenResidentConv(ConvShape(14, 14, 256, 256), flat_spatial_planes=True)
    for options in [
        {"rows_per_tile": 2},
        {"source_stride": True},
        {"flat_spatial_planes": 1},
    ]:
        selection = {"flat_spatial_planes": True}
        selection.update(options)
        with pytest.raises(ValueError, match="flat resident"):
            GoldenResidentConv(ConvShape(5, 5, 16, 19), **selection)


def test_normal_choice_preserves_declared_semantics_and_default_fallback():
    from mlir_oot.golden_flat_conv import GoldenFlatConv
    from mlir_oot.golden_resident_conv import select_flat_resident_planes

    s = ConvShape(7, 7, 512, 512, bn=16, output_dtype="i32")
    control = GoldenFlatConv(s, wide_a=True, separate_b_bank=True, virtual_padding=True)
    selected, decision = select_flat_resident_planes(control)
    assert (
        selected.conv == s
        and selected.flat_spatial_planes
        and selected.compact_commands
    )
    assert decision["input_reserved_interval"] == [0, 6048]
    assert decision["accumulator_rows"] == 1024
    assert decision["applied"] and decision["performance"] == "UNKNOWN"
    too_large = GoldenFlatConv(ConvShape(14, 14, 256, 256), virtual_padding=True)
    unchanged, refusal = select_flat_resident_planes(too_large)
    assert unchanged is too_large and not refusal["applied"]
    explicit = GoldenFlatConv(ConvShape(3, 5, 16, 19, explicit_halo=True))
    unchanged, refusal = select_flat_resident_planes(explicit)
    assert unchanged is explicit and not refusal["applied"]


def test_normal_option_requires_typed_proved_padding_before_mutation(tmp_path):
    from mlir_oot.captured_requant_bundle import build

    for value in (True, 1):
        with pytest.raises(ValueError, match="flat resident planes"):
            build(
                tmp_path / "missing",
                tmp_path / "tools",
                tmp_path / "out",
                flat_resident_planes=value,
            )
    assert not (tmp_path / "out").exists()
