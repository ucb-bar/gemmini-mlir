"""Source-stride paired stores obey both typed source and live resources."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

from mlir_oot.conv_schedule import source_stride_resource_layout
from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_resident_conv import GoldenResidentConv
from mlir_oot.paired_readout_binding import choose
from mlir_oot.readout_store_plan import PairedReadoutPlan
from mlir_oot.tables import rtl_facts as F

spec = importlib.util.spec_from_file_location(
    "paired_stride_proof",
    Path(__file__).with_name("resident_weight_packet_trace_probe.py"),
)
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)
PLAN = PairedReadoutPlan((0.125,), (0.125, 0.125), -(1 << 31), (1 << 31) - 1)


@pytest.mark.parametrize(
    "h,w,cin,cout", [(5, 7, 16, 19), (7, 5, 32, 67), (3, 27, 16, 129)]
)
@pytest.mark.parametrize("residue", [False, True])
def test_source_cells_order_ranges_and_both_store_lifetimes(h, w, cin, cout, residue):
    shape = ConvShape(h, w, cin, cout, stride=2, bn=4, output_dtype="i32")
    generator, decision = source_stride_resource_layout(shape, row_residue=residue)
    candidate = GoldenResidentConv(
        generator.conv,
        rows_per_tile=generator.rows_per_tile,
        source_stride=True,
        row_residue=generator.row_residue,
        weight_base=generator.explicit_weight_base,
        store_plan=PLAN,
    )
    module = candidate.build()
    module.verify()
    observed = proof.prove(candidate, module)
    assert observed["output_buffers"] == 2
    assert observed["both_stores_before_ACC_reuse_and_final_fence"]
    assert (
        observed["requested_payload"]["store_bytes"] == 2 * shape.oh * shape.ow * cout
    )
    assert decision["accumulator_rows"] <= F.ACC_ROWS


@pytest.mark.parametrize("residue", [False, True])
def test_source_bound_reconstruction_retains_layout_and_resource_facts(residue):
    original, resources = source_stride_resource_layout(
        ConvShape(28, 28, 256, 256, stride=2, bn=4, output_dtype="i32"),
        row_residue=residue,
    )
    original.source_stride_decision = resources
    candidate, plan, choice = choose(
        original,
        {
            "source_scales": [0.125],
            "accumulator_min": -(1 << 31),
            "accumulator_max": (1 << 31) - 1,
            "output_min": -128,
        },
    )
    assert choice["applied"] and plan.certificate() == PLAN.certificate()
    for name in (
        "conv",
        "source_stride",
        "row_residue",
        "rows_per_tile",
        "row_tiles",
        "plane",
        "bbase",
        "explicit_weight_base",
        "source_stride_decision",
    ):
        assert getattr(candidate, name) == getattr(original, name)
    assert not candidate.flat_spatial_planes
    assert candidate.bbase == 14400
    assert len(candidate.fb.entry.args) == 4


@pytest.mark.parametrize(
    "residue,compact", [(False, False), (True, False), (True, True)]
)
def test_default_source_stride_IR_byte_preservation(residue, compact):
    source = subprocess.check_output(
        ["git", "show", "d1d0e80:mlir_oot/golden_resident_conv.py"], text=True
    )
    namespace = {
        "__name__": "mlir_oot._paired_stride_baseline",
        "__package__": "mlir_oot",
    }
    exec(compile(source, "baseline.py", "exec"), namespace)  # noqa: S102 - pinned source
    shape = ConvShape(5, 7, 16, 19, stride=2, bn=4, output_dtype="i32")
    options = {
        "source_stride": True,
        "row_residue": residue,
        "compact_commands": compact,
    }
    assert str(GoldenResidentConv(shape, **options).build()) == str(
        namespace["GoldenResidentConv"](shape, **options).build()
    )


def test_unproved_and_unsupported_contracts_keep_refusals():
    with pytest.raises(ValueError, match="source-stride"):
        GoldenResidentConv(ConvShape(5, 7, 16, 19, output_dtype="i32"), store_plan=PLAN)
    with pytest.raises(ValueError, match="unscaled unactivated"):
        GoldenResidentConv(
            ConvShape(5, 7, 16, 19, stride=2, output_dtype="i8"),
            source_stride=True,
            store_plan=PLAN,
        )
    with pytest.raises(ValueError, match="producer interval"):
        GoldenResidentConv(
            ConvShape(5, 7, 16, 19, stride=2, output_dtype="i32"),
            source_stride=True,
            store_plan=PairedReadoutPlan((0.125,), (0.125, 0.125), -100, 100),
        )
    with pytest.raises(ValueError, match="source stride"):
        source_stride_resource_layout(
            ConvShape(5, 7, 17, 19, stride=2, output_dtype="i32")
        )
