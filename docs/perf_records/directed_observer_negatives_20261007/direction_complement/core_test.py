"""One-edge observation with signed multipliers and exact original fallback."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from test_bounded_rne_word_cells import oracle, ordered, unorder, word
from test_source_expression_interval import EFFECTS, analyze, source

from merlin.llvmlower.bounded_rne_word_cells import prepare_bounded_rne_word_cells
from merlin.llvmlower.source_expression_interval import (
    build_source_interval_table,
    emit_source_interval_i8_lookup,
)
from merlin.llvmlower.toolchain import clang


def plan():
    module, proof = analyze(source())
    cells = prepare_bounded_rne_word_cells(proof, effects=EFFECTS)
    table = build_source_interval_table(proof.expression, effects=EFFECTS, leading_bits=9, max_table_bytes=4096)
    return module, cells, table


def emitted(cells, table, **kwargs):
    return emit_source_interval_i8_lookup(
        table_name="table",
        activation_name="activation",
        quantizer_name="quant",
        lookup_name="lookup",
        leading_bits=9,
        finite_table=table,
        ordered_observer_word_cells=(cells,),
        **kwargs,
    )


def test_one_far_edge_matches_independent_rational_bins_in_both_directions():
    _, cells, _ = plan()
    for q, (lower, upper) in zip(range(-128, 128), cells.ranges):
        points = [
            key
            for key in (lower - 1, lower, lower + 1, upper - 1, upper, upper + 1)
            if oracle(unorder(key)) is not None
        ]
        first = lower if q == 127 else upper if q == -128 else ordered(word(q))
        assert oracle(unorder(first)) == q
        for other in points:
            if other >= first:
                assert (other <= upper) == (oracle(unorder(other)) == q)
                assert (other > upper) == (oracle(unorder(other)) != q)
            if other <= first:
                assert (other >= lower) == (oracle(unorder(other)) == q)
                assert ((other ^ 0xFFFFFFFF) > (lower ^ 0xFFFFFFFF)) == (oracle(unorder(other)) != q)
    for raw in (0, 0x80000000, 1, 0x80000001):
        assert oracle(raw) == 0


def test_positive_factor_rederived_table_and_default_identity():
    _, cells, table = plan()
    ordinary = emit_source_interval_i8_lookup(
        table_name="table", activation_name="activation", quantizer_name="quant", lookup_name="lookup", leading_bits=9
    )
    code = emitted(cells, table)
    assert ordinary == emit_source_interval_i8_lookup(
        table_name="table", activation_name="activation", quantizer_name="quant", lookup_name="lookup", leading_bits=9
    )
    assert "high_word=quant" not in code and "bits(up)>>31" in code
    assert "low_product=lo*up,high_product=hi*up" in code
    assert "low_scaled=low_product*scale,high_scaled=high_product*scale" in code
    assert "if(!(scale>0.0f))goto source_fallback" in code
    assert "(other_key^(0u-direction))>bin[direction]" in code
    assert "return 0;" in emitted(cells, table, zero_observer_cells=(cells,))
    forged = bytes([table.data[0] ^ 1]) + table.data[1:]
    with pytest.raises(ValueError, match="rederived"):
        emitted(cells, replace(table, data=forged))
    for factor in ("-3.0", "0.0"):
        with pytest.raises(ValueError, match="positive"):
            analyze(source().replace("3.0:f32", factor + ":f32"))
    with pytest.raises(ValueError, match="choose one"):
        emitted(cells, table, observer_word_cells=(cells,))


@pytest.mark.parametrize("zero", [False, True])
def test_actual_compiled_complete_products_and_source_fallback(tmp_path, zero):
    _, cells, table = plan()
    code = emitted(cells, table, zero_observer_cells=(cells,) if zero else ())
    rows = np.frombuffer(table.data, "<f4").reshape(-1, 2)

    def literal(value):
        return ("-INFINITY" if value < 0 else "INFINITY") if np.isinf(value) else float(value).hex() + "f"

    declarations = ",".join("{" + literal(lo) + "," + literal(hi) + "}" for lo, hi in rows)
    text = (
        "#include <math.h>\n#include <fenv.h>\n"
        + code
        + "\nconst float table[512][2]={"
        + declarations
        + "};\n"
        + """
static unsigned fallback_count;
float activation(float x){fallback_count++;return x*2.0f;}
signed char quant(float x){if(x<-128.0f)x=-128.0f;if(x>127.0f)x=127.0f;
 int n=(int)x;float d=x-(float)n;float a=fabsf(d);
 if(a>0.5f||(a==0.5f&&(n&1)))n+=x<0?-1:1;return (signed char)n;}
signed char original(float x,float up,float scale){float a=x*2.0f;float p=a*up;float s=p*scale;return quant(s);}
float source_value(float x,float up,float scale){float a=x*2.0f;float p=a*up;return p*scale;}
signed char guarded(float x,float up,float scale){return fegetround()==FE_TONEAREST?lookup(x,up,scale):original(x,up,scale);}
unsigned fallbacks(void){return fallback_count;}
"""
    )
    c, so = tmp_path / "direction.c", tmp_path / "direction.so"
    c.write_text(text)
    subprocess.run(
        [str(clang()), "-O3", "-ffp-contract=off", "-frounding-math", "-fPIC", "-shared", str(c), "-lm", "-o", str(so)],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(so))
    for name in ("guarded", "original"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_float] * 3
        fn.restype = ctypes.c_int8
    lib.source_value.argtypes = [ctypes.c_float] * 3
    lib.source_value.restype = ctypes.c_float
    env = ctypes.CDLL(None)
    oldmode, oldflags = env.fegetround(), env.fetestexcept(61)
    pairs = [
        (x, up, scale)
        for x in (-0.0, 0.0, np.nextafter(np.float32(0), np.float32(1)), -0.5, 0.5, -127.5, 126.5, 1.0, -1.0)
        for up in (-0.0, 0.0, -1.0, 1.0, np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1)))
        for scale in (3.0, 0.0, -3.0)
    ]
    rng = np.random.default_rng(599)
    pairs.extend(zip(rng.uniform(-63, 63, 1025), rng.uniform(-19, 19, 1025), [3.0] * 1025))
    undefined_source_conversions = 0
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert env.fesetround(mode) == 0
            for sticky in (0, 1, 4, 8, 16, 32, 61):
                for x, up, scale in pairs:
                    # Inspect the actual source value under this mode before
                    # invoking its integer conversion. Overflow followed by
                    # zero can produce NaN; that source conversion has no
                    # defined parity oracle and is retained as a limitation.
                    if np.isnan(lib.source_value(x, up, scale)):
                        undefined_source_conversions += 1
                        continue
                    env.feclearexcept(61)
                    env.feraiseexcept(sticky)
                    got = lib.guarded(x, up, scale)
                    assert env.fetestexcept(61) & sticky == sticky
                    assert got == lib.original(x, up, scale)
                    if mode == 0:
                        with np.errstate(all="ignore"):
                            v = np.float32(np.float32(np.float32(x) * np.float32(2)) * np.float32(up))
                            v = np.float32(v * np.float32(scale))
                        assert got == (-128 if v <= -128 else 127 if v >= 127 else round(Fraction(float(v))))
    finally:
        env.fesetround(oldmode)
        env.feclearexcept(61)
        env.feraiseexcept(oldflags)
    assert lib.fallbacks() > 0
    assert undefined_source_conversions > 0
