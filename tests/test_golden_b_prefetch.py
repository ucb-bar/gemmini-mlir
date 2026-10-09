"""B ping-pong reserves disjoint banks and drains exact tails using CPU loops."""

from dataclasses import asdict, replace
import hashlib
import json

import pytest
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from mlir_oot.contraction_patterns import IntegerGemm
from mlir_oot.golden_contraction_upstream import choose_shape
from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.golden_tuning import estimate
from mlir_oot.ir.gemmini_dialect import GEMMINI
from mlir_oot.tables import rtl_facts as F


@pytest.mark.parametrize("k", [17, 32, 48, 64, 80, 129, 2048, 5632])
def test_prefetch_has_exact_tail_rows_and_disjoint_banks(k):
    base = Shape(8, 1040, k, "i32", bm=1, bn=64, cache_a=True, wide_b=True)
    candidate = replace(base, prefetch_b=True)
    assert estimate(candidate) == estimate(base)
    module = GoldenGemm(candidate).build()
    b_loads = [op for op in module.walk() if op.name == "gemmini.mvin" and op.a("load_id") == 1]
    assert b_loads
    assert all(2 * F.SPAD_BANK_ROWS <= op.a("local") < F.SPAD_ROWS for op in b_loads)
    assert {op.a("local") // F.SPAD_BANK_ROWS for op in b_loads} == {2, 3}
    if k % F.DIM:
        assert any(op.a("rows") == k % F.DIM for op in b_loads)
    context = Context()
    for dialect in (Builtin, llvm.LLVM, GEMMINI):
        context.load_dialect(dialect)
    parsed = Parser(context, str(module)).parse_module()
    parsed.verify()
    lower(parsed)
    assert not any(op.name.startswith("gemmini.") for op in parsed.walk())


@pytest.mark.parametrize("shape,error", [
    (Shape(8, 32, 32, "i32", bm=1, bn=2), "cached A"),
    (Shape(8, 32, 16, "i32", bm=1, bn=2, cache_a=True), "multiple K panels"),
    (Shape(8, 32, 8208, "i32", bm=1, bn=2, cache_a=True), "lower two banks"),
    (Shape(8, 32, 32, "i32", bm=1, bn=2, cache_a=True, separate_b_bank=True), "own bank placement"),
])
def test_prefetch_refuses_unsupported_or_competing_resources(shape, error):
    with pytest.raises(ValueError, match=error):
        replace(shape, prefetch_b=True).validate()


def test_model_shape_selection_is_explicit_and_keeps_uncached_kernel():
    dims = IntegerGemm(batch=1, m=8, n=2048, k=2048)
    base = choose_shape(dims, large_n=True)
    assert base.cache_a and not base.prefetch_b
    assert choose_shape(dims, large_n=True, prefetch_b=True) == replace(base, prefetch_b=True)
    narrow = IntegerGemm(batch=1, m=8, n=256, k=2048)
    assert choose_shape(narrow, large_n=True, prefetch_b=True) == choose_shape(narrow, large_n=True)


def test_default_kernel_identity_is_unchanged_and_prefetch_is_distinct():
    from mlir_oot.golden_device_catalog import _symbol

    base = choose_shape(IntegerGemm(1, 8, 2048, 2048), large_n=True)
    key = {"batch": 0, "shape": asdict(base), "batched": False}
    previous_key = {**key, "shape": dict(key["shape"])}
    previous_key["shape"].pop("prefetch_b")
    previous_digest = hashlib.sha256(json.dumps(previous_key, sort_keys=True).encode()).hexdigest()[:16]
    assert _symbol(key) == "gemmini_golden_" + previous_digest
    assert _symbol({**key, "shape": asdict(replace(base, prefetch_b=True))}) != _symbol(key)


@pytest.mark.parametrize('m,n,k,bm,bn', [(17,73,65,2,4),(49,2048,512,4,16),(95,48,80,6,3)])
def test_multiple_cached_a_rows_have_disjoint_panels_and_exact_edges(m,n,k,bm,bn):
    shape=Shape(m,n,k,'i32',bm=bm,bn=bn,cache_a=True,wide_b=True,reuse_b=True,prefetch_b=True)
    module=GoldenGemm(shape).build()
    kt=(k+F.DIM-1)//F.DIM
    a_loads=[op for op in module.walk() if op.name=='gemmini.mvin' and op.a('load_id')==0]
    assert len(a_loads)==((m+F.DIM-1)//F.DIM)*kt
    assert {op.a('local') for op in a_loads}==set(range(0,len(a_loads)*F.DIM,F.DIM))
    assert all(op.a('local')+op.a('rows')<=2*F.SPAD_BANK_ROWS for op in a_loads)
    if m%F.DIM:
        assert len([op for op in a_loads if op.a('rows')==m%F.DIM])==kt
    computes=[op for op in module.walk() if op.name=='gemmini.compute']
    assert all(op.a('a_reserved_rows')==bm*kt*F.DIM for op in computes)
    assert max(op.a('a_max') for op in computes)==(((m+F.DIM-1)//F.DIM)*kt-1)*F.DIM
    module.verify();lower(module)
    assert not any(op.name.startswith('gemmini.') for op in module.walk())


def test_cached_a_requires_one_complete_m_block_and_full_reserved_bank_proof():
    with pytest.raises(ValueError,match='one complete M block'):
        Shape(33,64,64,'i32',bm=2,bn=4,cache_a=True).validate()
    # Each individual row fits, but the full cached A panel overlaps B banks.
    with pytest.raises(ValueError,match='lower two banks'):
        Shape(32,32,4112,'i32',bm=2,bn=2,cache_a=True,prefetch_b=True).validate()
