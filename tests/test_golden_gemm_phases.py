"""Independent resource admission and unchanged ordinary target default."""

from dataclasses import asdict

import pytest

from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.golden_gemm_phases import GoldenGemmPhases
from mlir_oot.tables import rtl_facts as F


def shape(m=7, n=64, k=65, columns=32):
    return Shape(
        m,
        n,
        k,
        "i32",
        bm=(m + F.DIM - 1) // F.DIM,
        bn=columns // F.DIM,
        cache_a=True,
        wide_b=True,
        prefetch_b=True,
    )


@pytest.mark.parametrize(
    "m,n,k,width", [(1, 32, 65, 16), (7, 64, 73, 32), (19, 96, 97, 16)]
)
def test_explicit_phases_resource_spans_and_tail_dimensions(m, n, k, width):
    s = shape(m, n, k, width)
    before = str(GoldenGemm(s).build())
    provider = GoldenGemmPhases(s, columns=width)
    module = provider.build(prefix="owned_pair")
    module.verify()
    r = asdict(provider.resources)
    assert r["total_accumulator_rows"] == 2 * s.bm * s.bn * F.DIM
    assert r["host_bytes_per_slot"] == 2 * m * width * 4
    assert r["a_rows"] == s.bm * ((k + F.DIM - 1) // F.DIM) * F.DIM
    assert r["total_accumulator_rows"] <= F.ACC_ROWS
    assert [op.sym_name.data for op in module.body.block.ops] == [
        "owned_pair_begin",
        "owned_pair_issue",
        "owned_pair_wait",
    ]
    assert str(GoldenGemm(s).build()) == before


@pytest.mark.parametrize(
    "change",
    [
        {"cache_a": False},
        {"prefetch_b": False},
        {"bias": True},
        {"scale": 0.5},
        {"relu": True},
        {"pipeline_m": True},
        {"cache_b": True},
        {"wide_a": True},
        {"output_dtype": "i8"},
    ],
)
def test_unknown_resource_or_numeric_variant_refuses(change):
    values = asdict(shape())
    values.update(change)
    with pytest.raises(ValueError):
        GoldenGemmPhases(Shape(**values), columns=32)


@pytest.mark.parametrize("n,width", [(73, 32), (64, 17), (64, 0), (64, True)])
def test_n_tail_unknown_width_and_capacity_refuse(n, width):
    with pytest.raises(ValueError):
        GoldenGemmPhases(shape(n=n), columns=width)
    with pytest.raises(ValueError, match="accumulator capacity"):
        GoldenGemmPhases(
            shape(m=F.DIM * 2, n=F.ACC_ROWS, k=2048, columns=F.ACC_ROWS // 2),
            columns=F.ACC_ROWS // 2,
        )


@pytest.mark.parametrize("name", ["9bad", "bad-name", "é", "", None])
def test_provider_symbol_refusal_before_emission(name):
    with pytest.raises(ValueError, match="ordinary symbol"):
        GoldenGemmPhases(shape(), columns=32).build(prefix=name)
