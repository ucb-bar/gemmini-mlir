"""Exact source-polynomial BF16 preimages under a complete rounded theorem.

No approximation, sampled monotonicity inference or target implementation is
provided. A caller supplies the existing complete rounded-DAG/source-effect
theorem. Integer binary32 arithmetic derives each exact source endpoint; that
theorem covers every interior input. Returned F32 boxes can be wider than a
fresh endpoint evaluation, so consumers must retain refinement and price it.
"""

import hashlib
import json
from dataclasses import dataclass, field
from functools import cache

from .rounded_polynomial_monotonicity import (
    RoundedPolynomialMonotonicity,
    consume_rounded_polynomial_monotonicity,
)
from .source_numeric_capability import SourceNumericContract


def _word(value):
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("binary32 raw word required")
    return value


def _parts(word):
    word = _word(word)
    biased = (word >> 23) & 255
    if biased == 255:
        raise ValueError("finite source binary32 word required")
    coefficient = (word & 0x7FFFFF) | (0x800000 if biased else 0)
    return (-coefficient if word >> 31 else coefficient), biased - 150 if biased else -149


def _rne(coefficient, exponent, *, negative_zero=False):
    """Round the exact integer*2**exponent once, including gradual underflow."""
    if not coefficient:
        return 0x80000000 if negative_zero else 0
    negative = coefficient < 0
    magnitude = abs(coefficient)
    e = exponent + magnitude.bit_length() - 1
    target = max(e - 23, -149)
    shift = target - exponent
    if shift > 0:
        whole, remainder = divmod(magnitude, 1 << shift)
        half = 1 << (shift - 1)
        magnitude = whole + (remainder > half or (remainder == half and bool(whole & 1)))
    elif shift < 0:
        magnitude <<= -shift
    if not magnitude:
        return int(negative) << 31
    e = target + magnitude.bit_length() - 1
    if e > 127:
        raise ValueError("source binary32 overflow requires fallback")
    if e < -126:
        return (int(negative) << 31) | magnitude
    if magnitude.bit_length() > 24:
        magnitude >>= 1
    return (int(negative) << 31) | ((e + 127) << 23) | (magnitude & 0x7FFFFF)


def _multiply(a, b):
    av, ae = _parts(a)
    bv, be = _parts(b)
    return _rne(av * bv, ae + be, negative_zero=bool((a ^ b) >> 31))


def _add(a, b):
    av, ae = _parts(a)
    bv, be = _parts(b)
    e = min(ae, be)
    return _rne((av << (ae - e)) + (bv << (be - e)), e, negative_zero=bool(a >> 31 and b >> 31))


def _fma(a, b, c):
    av, ae = _parts(a)
    bv, be = _parts(b)
    cv, ce = _parts(c)
    pe = ae + be
    e = min(pe, ce)
    return _rne((av * bv << (pe - e)) + (cv << (ce - e)), e, negative_zero=bool((a ^ b) >> 31 and c >> 31))


def _floor(a):
    value, exponent = _parts(a)
    if not value:
        return a
    integer = value << exponent if exponent >= 0 else value // (1 << -exponent)
    return _rne(integer, 0)


def _key(word):
    return (~word & 0xFFFFFFFF) if word >> 31 else word ^ 0x80000000


def _from_key(key):
    return key ^ 0x80000000 if key >> 31 else ~key & 0xFFFFFFFF


def _bf16(word):
    if word >= 0x7F800000:
        raise ValueError("finite nonnegative source probability required")
    rounded = (word + 0x7FFF + ((word >> 16) & 1)) >> 16
    if rounded >= 0x7F80:
        raise ValueError("finite BF16 source probability required")
    return rounded


def source_polynomial_word(input_word, plan_words):
    """Independent integer evaluation of the existing exact typed source DAG.

    RNE multiply; exact finite floor; separate fraction subtraction; three
    single-rounding Horner FMAs; separate subtraction; single-rounding encoded
    FMA; truncation to signed i32 and positive float bitcast. The cutoff precedes
    arithmetic. This function grants no monotonicity or source/effect permission.
    """
    if type(plan_words) is not tuple or len(plan_words) != 8:
        raise ValueError("eight actual typed polynomial plan words required")
    for word in plan_words:
        _parts(word)
    input_word = _word(input_word)
    _parts(input_word)
    cutoff, scale, c0, c1, c2, c3, multiplier, bias = plan_words
    if _key(input_word) < _key(cutoff):
        return 0
    scaled = _multiply(input_word, scale)
    fraction = _add(scaled, _floor(scaled) ^ 0x80000000)
    polynomial = c0
    for coefficient in (c1, c2, c3):
        polynomial = _fma(fraction, polynomial, coefficient)
    q = _add(scaled, polynomial ^ 0x80000000)
    encoded = _fma(multiplier, q, bias)
    value, exponent = _parts(encoded)
    integer = abs(value) << exponent if exponent >= 0 else abs(value) >> -exponent
    if value < 0:
        integer = -integer
    if not 0 <= integer < 0x7F800000:
        raise ValueError("source exponent reconstruction requires finite positive float word")
    return integer


@dataclass(frozen=True)
class PolynomialBf16Cell:
    first_input_key: int
    last_input_key: int
    probability_bf16_word: int
    first_probability_word: int
    last_probability_word: int


@dataclass(frozen=True)
class PolynomialBf16Buckets:
    cells: tuple[PolynomialBf16Cell, ...]
    source_plan_words: tuple[int, ...]
    source_evaluations: int
    complete_domain: bool
    _proof: RoundedPolynomialMonotonicity = field(repr=False)
    _contract: SourceNumericContract = field(repr=False)
    _source_header: str = field(repr=False)

    @property
    def canonical_sha256(self):
        return hashlib.sha256(
            json.dumps(
                [self.source_plan_words, [vars(cell) for cell in self.cells]], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()


def prepare_polynomial_bf16_buckets(source_header, proof, contract, *, max_evaluations=2_000_000, max_cells=65_536):
    """Derive complete exact cells with a bounded native-independent feedback loop.

    Existing proof identities/effect fields are validated before work. Complete
    rounded monotonicity is an external theorem, never inferred from these
    evaluations. Dyadic subdivision finds transitions; contiguous equal BF16
    cells are merged. Every endpoint is integer-evaluated from the actual plan.
    Resource exhaustion refuses before returning a partial table.
    """
    for value in (max_evaluations, max_cells):
        if type(value) is not int or value <= 0:
            raise ValueError("positive integer preparation quotas required")
    consume_rounded_polynomial_monotonicity(source_header, proof, contract)
    if type(proof) is not RoundedPolynomialMonotonicity or type(contract) is not SourceNumericContract:
        raise ValueError("typed complete rounded source theorem/effect contract required")
    if proof.plan_words[1] >> 31 or not proof.plan_words[1] & 0x7FFFFFFF:
        raise ValueError("positive source scale required")
    visits = 0

    @cache
    def evaluate(key):
        nonlocal visits
        visits += 1
        if visits > max_evaluations:
            raise ValueError("source-polynomial preparation evaluation quota exceeded")
        word = source_polynomial_word(_from_key(key), proof.plan_words)
        return word, _bf16(word)

    cells = []

    def append(first, last, low, high, bf16):
        if cells and cells[-1].probability_bf16_word == bf16:
            previous = cells.pop()
            assert previous.last_input_key + 1 == first
            first, low = previous.first_input_key, previous.first_probability_word
        cells.append(PolynomialBf16Cell(first, last, bf16, low, high))
        if len(cells) > max_cells:
            raise ValueError("source-polynomial preparation cell quota exceeded")

    def partition(first, last):
        low, low_bf16 = evaluate(first)
        high, high_bf16 = evaluate(last)
        if low > high:
            raise ValueError("supplied rounded monotonicity contradicts source endpoint words")
        if low_bf16 == high_bf16:
            append(first, last, low, high, low_bf16)
            return
        if first == last:
            raise ValueError("one source word cannot produce two probabilities")
        mid = (first + last) // 2
        partition(first, mid)
        partition(mid + 1, last)

    partition(_key(proof.plan_words[0]), _key(0))
    result = PolynomialBf16Buckets(tuple(cells), proof.plan_words, visits, True, proof, contract, source_header)
    validate_polynomial_bf16_buckets(result)
    return result


def validate_polynomial_bf16_buckets(buckets):
    if type(buckets) is not PolynomialBf16Buckets or buckets.complete_domain is not True:
        raise ValueError("complete immutable source-polynomial BF16 buckets required")
    consume_rounded_polynomial_monotonicity(buckets._source_header, buckets._proof, buckets._contract)
    if buckets.source_plan_words != buckets._proof.plan_words or not buckets.cells:
        raise ValueError("source plan or complete bucket cover changed")
    first = _key(buckets.source_plan_words[0])
    for cell in buckets.cells:
        if type(cell) is not PolynomialBf16Cell or any(type(value) is not int for value in vars(cell).values()):
            raise ValueError("typed integral source bucket required")
        if cell.first_input_key != first or not first <= cell.last_input_key <= _key(0):
            raise ValueError("bucket cover is unordered, incomplete or overlaps")
        low = source_polynomial_word(_from_key(first), buckets.source_plan_words)
        high = source_polynomial_word(_from_key(cell.last_input_key), buckets.source_plan_words)
        if (low, high) != (cell.first_probability_word, cell.last_probability_word) or low > high:
            raise ValueError("source bucket endpoint witness changed")
        if _bf16(low) != cell.probability_bf16_word or _bf16(high) != cell.probability_bf16_word:
            raise ValueError("source bucket BF16 observation is not unique")
        first = cell.last_input_key + 1
    if first != _key(0) + 1:
        raise ValueError("source domain does not include both signed zeros")


def c_header(buckets):
    """Emit default-off interval membership; misses retain the checked source.

    The caller admits the immutable source/environment once per private epoch
    using the same source-domain/effect token as existing constants consumers.
    Each hit returns a source-exact BF16 word and a conservative ORIGINAL F32
    probability enclosure. It never substitutes BF16 P for the denominator.
    """
    validate_polynomial_bf16_buckets(buckets)
    expected = ",".join(f"UINT32_C(0x{x:08x})" for x in buckets.source_plan_words)
    table = ",\n".join(
        "{"
        + ",".join(
            f"UINT32_C(0x{x:08x})"
            for x in (
                cell.first_input_key,
                cell.last_input_key,
                cell.first_probability_word,
                cell.last_probability_word,
            )
        )
        + "}"
        for cell in buckets.cells
    )
    return (
        r"""
#ifndef MERLIN_POLYNOMIAL_BF16_BUCKETS_H
#define MERLIN_POLYNOMIAL_BF16_BUCKETS_H
#include "monotone_bit_polynomial.h"
#include <fenv.h>
#include <stddef.h>
static const uint32_t merlin_polynomial_bf16_cells[@COUNT@][4]={@TABLE@};
static inline int merlin_polynomial_bf16_bucket_admit(const merlin_monotone_bit_polynomial *p){
 if(!p||!p->fast_valid||p->word_budget||p->upper!=0.0f||fegetround()!=FE_TONEAREST)return 0;
 const merlin_bit_polynomial_plan *s=&p->checked.source;
 const float values[8]={s->cutoff,s->scale,s->coefficients[0],s->coefficients[1],
  s->coefficients[2],s->coefficients[3],s->bit_multiplier,s->bit_bias};
 const uint32_t expected[8]={@EXPECTED@};
 for(int i=0;i<8;i++)if(merlin_interval_bits(values[i])!=expected[i])return 0;
 return 1;
}
static inline int merlin_polynomial_bf16_bucket(merlin_f32_interval x,int admitted,
 const merlin_monotone_bit_polynomial *p,merlin_f32_interval *y){
 if(!admitted||!p||!y||!x.valid)return 0;
 uint32_t a=merlin_interval_bits(x.lo),b=merlin_interval_bits(x.hi);
 /* The existing representation-copy contract suffices. Do not introduce an
  * interposed classification call or a new floating flag observation. */
 if((a&UINT32_C(0x7fffffff))>=UINT32_C(0x7f800000)||
  (b&UINT32_C(0x7fffffff))>=UINT32_C(0x7f800000)||x.lo>x.hi||x.hi>0.0f)return 0;
 if(x.hi<p->checked.source.cutoff){*y=merlin_interval_point(0);return 1;}
 a=a>>31?~a:a^UINT32_C(0x80000000);b=b>>31?~b:b^UINT32_C(0x80000000);
 size_t first=0,last=@COUNT@;
 while(first<last){size_t mid=first+(last-first)/2;
  if(merlin_polynomial_bf16_cells[mid][1]<a)first=mid+1;else last=mid;}
 if(first==@COUNT@||a<merlin_polynomial_bf16_cells[first][0]||
  b>merlin_polynomial_bf16_cells[first][1])return 0;
 *y=(merlin_f32_interval){merlin_interval_float(merlin_polynomial_bf16_cells[first][2]),
  merlin_interval_float(merlin_polynomial_bf16_cells[first][3]),1};return 1;
}
#endif
""".replace("@COUNT@", str(len(buckets.cells)))
        .replace("@TABLE@", table)
        .replace("@EXPECTED@", expected)
    )
