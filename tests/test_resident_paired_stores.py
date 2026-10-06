"""Paired resident stores preserve completed ACC and the source domain."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_flat_conv import GoldenFlatConv
from mlir_oot.golden_resident_conv import (
    GoldenResidentConv,
    select_flat_resident_planes,
)
from mlir_oot.paired_readout_binding import choose
from mlir_oot.readout_store_plan import PairedReadoutPlan

spec = importlib.util.spec_from_file_location(
    "resident_proof", Path(__file__).with_name("resident_weight_packet_trace_probe.py")
)
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)
PLAN = PairedReadoutPlan((0.125,), (0.125, 0.125), -(1 << 31), (1 << 31) - 1)


@pytest.mark.parametrize(
    "h,w,cin,cout", [(3, 5, 16, 19), (5, 3, 32, 67), (1, 17, 16, 129)]
)
@pytest.mark.parametrize(
    "compact,prefetch,packet",
    [(False, False, None), (True, False, None), (True, True, None), (True, True, 2)],
)
def test_emitted_source_order_extents_and_both_store_lifetimes(
    h, w, cin, cout, compact, prefetch, packet
):
    g = GoldenResidentConv(
        ConvShape(h, w, cin, cout, bn=4, output_dtype="i32"),
        flat_spatial_planes=True,
        compact_commands=compact,
        prefetch_b=prefetch,
        weight_issue_tiles=packet,
        store_plan=PLAN,
    )
    module = g.build()
    module.verify()
    assert len(module.body.block.first_op.body.blocks.first.args) == 4
    result = proof.prove(g, module)
    assert result["output_buffers"] == 2
    assert result["both_stores_before_ACC_reuse_and_final_fence"]
    assert result["requested_payload"]["store_bytes"] == 2 * h * w * cout


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"compact_commands": True},
        {"flat_spatial_planes": True, "compact_commands": True},
        {"compact_commands": True, "prefetch_b": True, "weight_issue_tiles": 2},
    ],
)
def test_default_IR_unchanged(options):
    source = subprocess.check_output(
        ["git", "show", "4e1630b:mlir_oot/golden_resident_conv.py"], text=True
    )
    namespace = {"__name__": "mlir_oot._resident_baseline", "__package__": "mlir_oot"}
    exec(compile(source, "baseline.py", "exec"), namespace)  # noqa: S102 - pinned git source
    shape = ConvShape(3, 5, 32, 67, bn=4, output_dtype="i32")
    assert str(GoldenResidentConv(shape, **options).build()) == str(
        namespace["GoldenResidentConv"](shape, **options).build()
    )


def test_source_bound_selection_and_reconstruction_preserve_facts():
    shape = ConvShape(3, 5, 32, 67, bn=4, output_dtype="i32")
    flat = GoldenFlatConv(shape, virtual_padding=True, store_plan=PLAN)
    resident, decision = select_flat_resident_planes(flat)
    assert decision["applied"] and resident.store_plan == PLAN
    resident.flat_resident_decision = decision
    candidate, plan, paired = choose(
        resident,
        {
            "source_scales": [0.125],
            "accumulator_min": -(1 << 31),
            "accumulator_max": (1 << 31) - 1,
            "output_min": -128,
        },
    )
    assert paired["applied"] and plan.certificate() == PLAN.certificate()
    assert candidate.conv == resident.conv
    assert candidate.compact_commands == resident.compact_commands
    assert candidate.explicit_weight_base == resident.explicit_weight_base
    assert candidate.flat_resident_decision == decision


def test_unproved_domain_and_unsupported_layout_refuse():
    s = ConvShape(3, 5, 16, 19, output_dtype="i32")
    for plan in (True, PairedReadoutPlan((0.125,), (0.125, 0.125), -100, 100)):
        with pytest.raises(ValueError):
            GoldenResidentConv(s, flat_spatial_planes=True, store_plan=plan)
    with pytest.raises(ValueError, match="flat spatial"):
        GoldenResidentConv(s, store_plan=PLAN)
    with pytest.raises(ValueError, match="unscaled unactivated"):
        GoldenResidentConv(
            ConvShape(3, 5, 16, 19, output_dtype="i8"),
            flat_spatial_planes=True,
            store_plan=PLAN,
        )
    _unchanged, plan, decision = choose(GoldenResidentConv(s), {})
    assert plan is None and not decision["applied"]
    with pytest.raises(ValueError, match="i32.*overflow"):
        PLAN.require_conv_producer(ConvShape(1, 1, 16384, 1))
