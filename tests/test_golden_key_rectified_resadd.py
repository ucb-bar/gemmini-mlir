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
        0.011258588172495365,
        0.00940733402967453,
        0.011643771082162857,
        p=298,
        q=249,
        scale=0.0032446938566863537,
        relu=True,
    )
    return rectifier.derive(proof, max_pairs=1, indicator_family="predictor_key")


def supported():
    return Capabilities(BasicCapabilities(*([True] * 6)), True)


@pytest.mark.parametrize(
    "m,n,batch", [(16, 64, 1), (48, 128, 4), (80, 192, 4), (12544, 64, 4),
                  (144, 128, 8), (272, 192, 16)]
)
def test_complete_nonsquare_shapes_and_partial_batch_resources(
    certificate, m, n, batch
):
    plan = Plan(m, n, certificate, supported(), batch)
    assert plan.compute_passes == 9
    assert plan.scratch_bytes == batch * 1024
    for i, (start, length) in enumerate(plan.spad_intervals.values()):
        assert 0 <= start < start + length <= rtl_facts.SPAD_ROWS
        assert all(
            start + length <= other or other + extent <= start
            for other, extent in list(plan.spad_intervals.values())[i + 1 :]
        )
    assert plan.weights == [127, 44, 122, 1, -1]
    assert len(plan.tables()) == 1472
    assert plan.acc_seed_offset % 4 == 0


@pytest.mark.parametrize("m,n", [(0, 64), (17, 64), (16, 65), (True, 64), (16, True)])
def test_partial_or_untyped_tiles_refuse(certificate, m, n):
    with pytest.raises(ValueError, match="complete DIM"):
        Plan(m, n, certificate, supported())


def test_missing_rmw_capability_and_too_small_spad_refuse(certificate, monkeypatch):
    with pytest.raises(ValueError, match="RMW"):
        Plan(16, 64, certificate, Capabilities(BasicCapabilities(*([True] * 6)), False))
    monkeypatch.setattr(rtl_facts, "SPAD_ROWS", 128)
    with pytest.raises(ValueError, match="capacity"):
        Plan(16, 64, certificate, supported())


@pytest.mark.parametrize("batch", [False, True, 0, -1, 1.5, "4"])
def test_untyped_or_empty_panel_batch_refuses(certificate, batch):
    with pytest.raises(ValueError, match="positive integral"):
        Plan(16, 64, certificate, supported(), batch)


def test_batch_capacity_is_derived_from_actual_live_extents(certificate, monkeypatch):
    Plan(272, 192, certificate, supported(), 16)
    with pytest.raises(ValueError, match="SPAD extents overlap|ACC reservation"):
        Plan(272, 192, certificate, supported(), 17)
    monkeypatch.setattr(rtl_facts, "ACC_ROWS", 512)
    Plan(144, 128, certificate, supported(), 8)
    with pytest.raises(ValueError, match="ACC reservation"):
        Plan(272, 192, certificate, supported(), 16)


def test_emission_rechecks_mutable_nested_key_witness(certificate):
    changed = copy.deepcopy(certificate)
    plan = Plan(48, 128, changed, supported())
    changed["relation"][0]["key"]["seed"] += 1
    with pytest.raises(ValueError, match="changed"):
        build(plan, ordering_contract=OrderingContract("unreachable"))


def test_multiple_or_empty_correction_fibres_stay_unsupported():
    proof = pair.derive(1, 1, 2, p=1, q=1, scale=0.25, relu=True)
    certificate = rectifier.derive(
        proof, max_pairs=65536, indicator_family="predictor_key"
    )
    with pytest.raises(ValueError, match="exactly one"):
        Plan(16, 64, certificate, supported())
    proof = pair.derive(1, 0.5, 1, p=2, q=1, scale=0.5)
    certificate = rectifier.derive(proof, max_pairs=0, indicator_family="predictor_key")
    with pytest.raises(ValueError, match="exactly one"):
        Plan(16, 64, certificate, supported())


def test_complete_key_bounds_must_admit_exact_float_conversion(
    monkeypatch, certificate
):
    changed = copy.deepcopy(certificate)
    # Actual validation is independent; simulate a wider validated source key
    # solely to exercise this provider's stricter conversion bound.
    monkeypatch.setattr(
        "mlir_oot.golden_key_rectified_resadd.validate", lambda value: value
    )
    changed["relation"][0]["key_range"] = [-(1 << 24), 1]
    with pytest.raises(ValueError, match="binary32"):
        Plan(16, 64, changed, supported())


def test_actual_primitive_stream_avoids_negative_scale_relu_mismatch(certificate):
    module = build(
        Plan(80, 192, certificate, supported()),
        ordering_contract=OrderingContract(
            "/scratch2/agustin/wt/chipyard-stock/generators/gemmini/src/main/scala/gemmini"
        ),
    )
    negative_stores = [
        op
        for op in module.walk()
        if isinstance(op, G.ConfigStOp) and op.a("acc_scale") == -1.0
    ]
    assert negative_stores
    assert all(op.a("acc_act") == isa.NO_ACTIVATION for op in negative_stores)
    # Reuse requires actual integer RMW, not an ACC execute read or input scale.
    acc_loads = [
        op
        for op in module.walk()
        if isinstance(op, G.MvinOp) and op.a("local") & isa.ACC_ADDR_BIT
    ]
    assert len(acc_loads) == 20  # Four complete panels and one independent tail.
    assert all(op.a("local") & (1 << 30) for op in acc_loads)
    assert all(
        not op.a("a") & isa.ACC_ADDR_BIT
        for op in module.walk()
        if isinstance(op, G.ComputeOp)
    )


@pytest.mark.parametrize("field", ["seed", "predictor_integer_range"])
def test_identity_dma_does_not_admit_unrepresentable_i32_literals(
    monkeypatch, certificate, field
):
    changed = copy.deepcopy(certificate)
    monkeypatch.setattr(
        "mlir_oot.golden_key_rectified_resadd.validate", lambda value: value
    )
    if field == "seed":
        changed["relation"][0]["key"][field] = (1 << 24) + 1
    else:
        changed["relation"][0][field] = [-(1 << 24) - 1, 1]
    with pytest.raises(ValueError, match="exact binary32 representation"):
        Plan(16, 64, changed, supported())
