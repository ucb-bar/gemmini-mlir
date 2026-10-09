import copy

import pytest
from merlin.llvmlower.quantized_affine_pair import derive
from merlin.llvmlower.quantized_affine_rectifier import derive as synthesize

from mlir_oot.golden_rectified_resadd import Capabilities, Plan, build
from mlir_oot.tables import rtl_facts


@pytest.fixture(scope="module")
def sparse():
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


def supported():
    return Capabilities(*([True] * 6))


@pytest.mark.parametrize("m,n", [(16, 64), (48, 128), (80, 192), (12544, 64)])
def test_nonsquare_complete_planes_and_full_private_resources(sparse, m, n):
    plan = Plan(m, n, sparse, supported())
    assert plan.compute_passes == 14
    intervals = list(plan.spad_intervals.values())
    for index, (start, rows) in enumerate(intervals):
        assert 0 <= start <= start + rows <= rtl_facts.SPAD_ROWS
        assert all(
            start + rows <= other or other + length <= start
            for other, length in intervals[index + 1 :]
        )
    assert plan.panel_rows <= rtl_facts.ACC_ROWS
    assert len(plan.tables()) == 1600
    build(plan).verify()


@pytest.mark.parametrize(
    "m,n", [(0, 64), (-16, 64), (17, 64), (16, 63), (32, 65), (True, 64), (16, True)]
)
def test_incomplete_or_untyped_extents_refuse(sparse, m, n):
    with pytest.raises(ValueError, match="complete DIM"):
        Plan(m, n, sparse, supported())


def test_unknown_capability_and_resource_capacity_refuse(sparse, monkeypatch):
    with pytest.raises(ValueError, match="capability contract"):
        Plan(16, 64, sparse, Capabilities(False, *([True] * 5)))
    monkeypatch.setattr(rtl_facts, "SPAD_ROWS", 100)
    with pytest.raises(ValueError, match="capacity"):
        Plan(16, 64, sparse, supported())


@pytest.mark.parametrize("relu", [False, True])
def test_empty_exact_relation_uses_no_replay_or_private_seed_storage(relu):
    pair = derive(1, 0.5, 1, p=2, q=1, scale=0.5, relu=relu)
    certificate = synthesize(pair, max_pairs=0, indicator_family="axis_offsets")
    plan = Plan(48, 128, certificate, supported())
    assert plan.relation is None and plan.seeds == []
    assert plan.compute_passes == 2
    assert plan.attributes()["predictor_store_and_reload_per_panel"] == 0
    build(plan).verify()


def test_generic_multiple_certificate_is_explicitly_unsupported_by_target():
    pair = derive(0.2, 0.3, 1, p=2, q=3, scale=0.1, relu=False)
    certificate = synthesize(pair, max_pairs=1000, indicator_family="axis_offsets")
    with pytest.raises(ValueError, match="multiple correction"):
        Plan(16, 64, certificate, supported())


def test_mutated_numeric_certificate_refuses_again_at_emission(sparse):
    certificate = copy.deepcopy(sparse)
    plan = Plan(16, 64, certificate, supported())
    certificate["relation"][0]["correction"] = 2
    with pytest.raises(ValueError, match="changed"):
        build(plan)
