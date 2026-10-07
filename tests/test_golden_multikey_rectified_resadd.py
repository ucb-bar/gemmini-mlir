"""Independent bounded two-fibre resource and primitive-order contracts."""

import copy

import pytest
from merlin.llvmlower import quantized_affine_pair as pair
from merlin.llvmlower import quantized_affine_rectifier as rectifier

from mlir_oot.golden_key_rectified_resadd import Capabilities, Plan, build
from mlir_oot.golden_rectified_resadd import Capabilities as BasicCapabilities
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.spad_fence_coalescing import OrderingContract
from mlir_oot.tables import isa, rtl_facts


@pytest.fixture(scope="module")
def certificate():
    proof = pair.derive(
        0.8350648283958435,
        0.8539831638336182,
        1.0693517923355103,
        p=309,
        q=316,
        scale=0.002527207136154175,
        relu=True,
    )
    return rectifier.derive(proof, max_pairs=2, indicator_family="predictor_key")


def supported():
    return Capabilities(BasicCapabilities(*([True] * 6)), True)


@pytest.mark.parametrize(
    "m,n,batch", [(16, 64, 1), (48, 128, 4), (80, 192, 4), (1568, 64, 4)]
)
def test_two_keys_reserve_complete_nonoverlapping_extents(certificate, m, n, batch):
    plan = Plan(m, n, certificate, supported(), batch, max_key_fibres=2)
    assert plan.compute_passes == 14
    intervals = list(plan.spad_intervals.values())
    for index, (base, rows) in enumerate(intervals):
        assert 0 <= base < base + rows <= rtl_facts.SPAD_ROWS
        assert all(
            base + rows <= other or other + length <= base
            for other, length in intervals[index + 1 :]
        )
    seeds = [relation["key"]["seed"] for relation in certificate["relation"]]
    assert plan.seed_delta(0) == seeds[0]
    assert plan.seed_delta(0) + plan.seed_delta(1) == seeds[1]
    assert plan.attributes()["correction_key_fibres"] == 2
    assert len(plan.tables()) == plan.acc_seed_offset + 2 * 16 * 4


@pytest.mark.parametrize("bound", [False, True, 0, 3, 4, "2"])
def test_unknown_or_unqualified_expansion_refuses(certificate, bound):
    with pytest.raises(ValueError, match="bounded"):
        Plan(16, 64, certificate, supported(), max_key_fibres=bound)


def test_default_single_fibre_refusal_is_retained(certificate):
    with pytest.raises(ValueError, match="exactly one"):
        Plan(16, 64, certificate, supported())


def test_second_key_witness_is_revalidated_before_emission(certificate):
    changed = copy.deepcopy(certificate)
    plan = Plan(16, 64, changed, supported(), max_key_fibres=2)
    changed["relation"][1]["key"]["seed"] += 1
    with pytest.raises(ValueError, match="changed"):
        build(plan, ordering_contract=OrderingContract("unreachable"))


def test_tail_emission_uses_only_identity_acc_rmw_and_no_negative_relu(certificate):
    module = build(
        Plan(80, 192, certificate, supported(), max_key_fibres=2),
        ordering_contract=OrderingContract(
            "/scratch2/agustin/wt/chipyard-stock/generators/gemmini/src/main/scala/gemmini"
        ),
    )
    acc_loads = [
        op
        for op in module.walk()
        if isinstance(op, G.MvinOp) and op.a("local") & isa.ACC_ADDR_BIT
    ]
    assert len(acc_loads) == 40  # Two seeds, four full panels and one row-panel tail.
    assert all(op.a("local") & (1 << 30) for op in acc_loads)
    negative_stores = [
        op
        for op in module.walk()
        if isinstance(op, G.ConfigStOp) and op.a("acc_scale") == -1.0
    ]
    assert negative_stores and all(
        op.a("acc_act") == isa.NO_ACTIVATION for op in negative_stores
    )
    assert all(
        not op.a("a") & isa.ACC_ADDR_BIT
        for op in module.walk()
        if isinstance(op, G.ComputeOp)
    )
