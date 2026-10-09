import json

import pytest
from merlin.llvmlower.quantized_affine_pair import derive
from merlin.llvmlower.quantized_affine_rectifier import derive as synthesize

from mlir_oot.golden_rectified_resadd import Capabilities, Plan, build
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.rectifier_panel_batch import PanelBatch
from mlir_oot.spad_fence_coalescing import OrderingContract
from mlir_oot.tables import rtl_facts as F

SOURCE = "/scratch2/agustin/wt/chipyard-stock/generators/gemmini/src/main/scala/gemmini"


@pytest.fixture(scope="module")
def certificate():
    pair = derive(
        0.011258588172495365,
        0.00940733402967453,
        0.011643771082162857,
        p=298,
        q=249,
        scale=0.0032446938566863537,
        relu=True,
    )
    return synthesize(pair, max_pairs=1, indicator_family="axis_offsets")


def plan(certificate, m=64, n=64):
    return Plan(m, n, certificate, Capabilities(*([True] * 6)))


@pytest.mark.parametrize(
    "m,n", [(16, 64), (48, 128), (64, 64), (80, 192), (112, 128), (12544, 64)]
)
def test_full_and_partial_batch_resource_and_completion_boundaries(certificate, m, n):
    p = plan(certificate, m, n)
    contract = OrderingContract(SOURCE)
    module = build(
        p, coalesce_internal_spad=True, ordering_contract=contract, panel_batch=4
    )
    proof = json.loads(module.attributes["gemmini.rectifier_panel_batch"].data)
    assert proof["factor"] == 4 and proof["acc_reservation"] == [0, 256]
    assert not proof["DDR_alias_from_reservation_station"]
    intervals = list(PanelBatch().reservations(p).values())
    for index, (base, length) in enumerate(intervals):
        assert base + length <= F.SPAD_ROWS
        assert all(
            base + length <= other or other + size <= base
            for other, size in intervals[index + 1 :]
        )
    slots = 4 * bool(m // 64) + (m % 64) // 16
    assert sum(isinstance(op, G.ComputeOp) for op in module.walk()) == slots * 14 * 4
    # Setup plus function completion, and two mandatory fences per emitted
    # full/tail batch body. There are no per-panel publication fences.
    bodies = bool(m // 64) + bool(m % 64)
    assert sum(isinstance(op, G.FenceOp) for op in module.walk()) == 3 + 2 * bodies
    removed = json.loads(module.attributes["gemmini.spad_fence_coalescing"].data)
    assert removed["static_removed_fences"] == slots * 8


@pytest.mark.parametrize("factor", [0, 2, 3, 5, True, 4.0])
def test_unknown_batch_factors_refuse(certificate, factor):
    with pytest.raises(ValueError, match="explicit panel batch"):
        build(plan(certificate), panel_batch=factor)


def test_missing_ordering_and_acc_capacity_refuse(certificate, monkeypatch):
    p = plan(certificate)
    with pytest.raises(ValueError, match="pinned ordering"):
        build(p, panel_batch=4)
    monkeypatch.setattr(F, "ACC_ROWS", 255)
    with pytest.raises(ValueError, match="ACC reservation"):
        build(p, panel_batch=4, ordering_contract=OrderingContract(SOURCE))


def test_empty_relation_refuses_batch_but_preserves_original_policy():
    pair = derive(1, 0.5, 1, p=2, q=1, scale=0.5, relu=True)
    empty = synthesize(pair, max_pairs=0, indicator_family="axis_offsets")
    p = plan(empty)
    build(p).verify()
    with pytest.raises(ValueError, match="exact sparse"):
        build(p, panel_batch=4, ordering_contract=OrderingContract(SOURCE))


def test_explicit_default_is_identical(certificate):
    p = plan(certificate)
    assert str(build(p)) == str(build(p, panel_batch=1))
    assert "gemmini.rectifier_panel_batch" not in build(p).attributes
