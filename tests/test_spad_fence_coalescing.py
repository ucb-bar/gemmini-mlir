"""Actual operand/configuration proof must precede SPAD fence removal."""

import json

import pytest
from merlin.llvmlower.quantized_affine_pair import derive
from merlin.llvmlower.quantized_affine_rectifier import derive as synthesize
from xdsl.dialects.builtin import IntegerAttr, i64

from mlir_oot import spad_fence_coalescing as S
from mlir_oot.golden_rectified_resadd import Capabilities, Plan, build
from mlir_oot.ir import gemmini_dialect as G

DIRECTORY = (
    "/scratch2/agustin/wt/chipyard-stock/generators/gemmini/src/main/scala/gemmini"
)


@pytest.fixture(scope="module")
def certificate():
    proof = derive(
        0.011258588172495365,
        0.00940733402967453,
        0.011643771082162857,
        p=298,
        q=249,
        scale=0.0032446938566863537,
        relu=True,
    )
    return synthesize(proof, max_pairs=1, indicator_family="axis_offsets")


def plan(certificate, m=48, n=128):
    return Plan(m, n, certificate, Capabilities(*([True] * 6)))


def marked_module(plan, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(S, "coalesce", lambda *args: {})
        return build(
            plan,
            coalesce_internal_spad=True,
            ordering_contract=S.OrderingContract(DIRECTORY),
        )


@pytest.mark.parametrize("m,n", [(16, 64), (48, 128), (80, 192), (12544, 64)])
def test_exact_same_tile_proof_preserves_all_nonfence_commands(certificate, m, n):
    p = plan(certificate, m, n)
    original = build(p)
    updated = build(
        p, coalesce_internal_spad=True, ordering_contract=S.OrderingContract(DIRECTORY)
    )
    opcodes = lambda module: [
        (op.name, dict(op.attributes))
        for op in module.walk()
        if isinstance(op, G._GemminiOp) and not isinstance(op, G.FenceOp)
    ]
    assert opcodes(original) == opcodes(updated)
    assert sum(isinstance(op, G.FenceOp) for op in original.walk()) == 13
    assert sum(isinstance(op, G.FenceOp) for op in updated.walk()) == 5
    proof = json.loads(updated.attributes["gemmini.spad_fence_coalescing"].data)
    assert proof["static_removed_fences"] == proof["static_products"] == 8
    assert proof["exact_same_tile_raw_edges"] > 0
    assert proof["external_memory_fences_preserved"]


@pytest.mark.parametrize(
    "mutation", ["a", "c", "rows", "transpose", "stride", "side_effect", "unknown"]
)
def test_unproved_actual_operands_refuse_without_mutation(
    certificate, monkeypatch, mutation
):
    p = plan(certificate)
    module = marked_module(p, monkeypatch)
    fences = [
        op
        for op in module.walk()
        if isinstance(op, G.FenceOp) and op.a("internal_spad_stage", 0)
    ]
    first = fences[0]
    operations = list(first.parent.ops)
    at = operations.index(first)
    if mutation == "a":
        operations[at - 1].attributes["a"] = IntegerAttr(1, i64)
    elif mutation == "c":
        operations[at - 2].attributes["c"] = IntegerAttr(0, i64)
    elif mutation == "rows":
        operations[at - 1].attributes["a_rows"] = IntegerAttr(15, i64)
    elif mutation in ("transpose", "stride"):
        config = next(
            op for op in reversed(operations[:at]) if isinstance(op, G.ConfigExOp)
        )
        config.attributes["a_transpose" if mutation == "transpose" else "a_stride"] = (
            IntegerAttr(1 if mutation == "transpose" else 2, i64)
        )
    elif mutation == "side_effect":
        first.parent.insert_op_before(
            G.FlushOp(operands=[[]], result_types=[[]]), first
        )
    else:
        operations[at - 1].attributes["unknown"] = IntegerAttr(0, i64)
    with pytest.raises(ValueError):
        S.coalesce(module, S.OrderingContract(DIRECTORY), p.spad_intervals)
    assert all(fence.parent is not None for fence in fences)


def test_contract_missing_and_mutated_source_refuse(certificate, tmp_path):
    p = plan(certificate)
    with pytest.raises(ValueError, match="pinned ordering"):
        build(p, coalesce_internal_spad=True)
    with pytest.raises(ValueError, match="missing pinned"):
        build(
            p,
            coalesce_internal_spad=True,
            ordering_contract=S.OrderingContract(str(tmp_path)),
        )
    (tmp_path / "Configs.scala").write_text("unsupported config")
    with pytest.raises(ValueError, match="unsupported SPAD ordering"):
        S.OrderingContract(str(tmp_path)).require()


def test_overlapping_full_reservations_refuse(certificate, monkeypatch):
    p = plan(certificate)
    module = marked_module(p, monkeypatch)
    reservations = dict(p.spad_intervals)
    reservations["indicator0"] = reservations["temporary"]
    with pytest.raises(ValueError, match="overlapping full"):
        S.coalesce(module, S.OrderingContract(DIRECTORY), reservations)


def test_default_plan_metadata_bytes_are_unchanged(certificate):
    original = build(plan(certificate))
    explicit = build(plan(certificate), coalesce_internal_spad=False)
    assert str(original) == str(explicit)
    assert "gemmini.spad_fence_coalescing" not in original.attributes
