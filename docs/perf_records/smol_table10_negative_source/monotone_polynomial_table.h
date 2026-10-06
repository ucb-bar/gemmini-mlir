#ifndef MERLIN_MONOTONE_POLYNOMIAL_TABLE_H
#define MERLIN_MONOTONE_POLYNOMIAL_TABLE_H
#include "monotone_bit_polynomial.h"
#include <limits.h>

/* Explicit private table of a proved monotone source polynomial's fractional
 * real transform. Preparation owns all writes. The original prepared plan and
 * knot storage must remain immutable and live through every application.
 * Linear interpolation has a proved curvature/quantization enclosure; it
 * never grants a new source approximation or consumer observation policy.
 */
typedef struct {
  const merlin_monotone_bit_polynomial *source;
  const int32_t *knots;
  unsigned bits, shift;
  uint32_t error;
  int valid;
} merlin_monotone_polynomial_table;

static inline merlin_polynomial_real_interval merlin_polynomial_fraction_e(
    double fraction, const merlin_bit_polynomial_plan *p) {
  merlin_polynomial_real_interval f = {fraction, fraction};
  merlin_polynomial_real_interval poly = {p->coefficients[0], p->coefficients[0]};
  for (int i = 1; i < 4; ++i) {
    poly = merlin_polynomial_real_product(f, poly);
    poly.lo = merlin_fma_next_down(poly.lo + (double)p->coefficients[i]);
    poly.hi = merlin_fma_next_up(poly.hi + (double)p->coefficients[i]);
  }
  return (merlin_polynomial_real_interval){
      merlin_fma_next_down(fraction - poly.hi),
      merlin_fma_next_up(fraction - poly.lo)};
}

static inline merlin_monotone_polynomial_table merlin_monotone_polynomial_table_prepare(
    const merlin_monotone_bit_polynomial *p, int32_t *storage,
    size_t capacity, unsigned bits) {
  merlin_monotone_polynomial_table out = {0};
  out.source = p;
  if (!p || !p->fast_valid || !storage || bits < 8 || bits > 16)
    return out;
  size_t count = ((size_t)1 << bits) + 1;
  if (capacity < count) return out;
  const merlin_bit_polynomial_plan *s = &p->checked.source;
  /* An integer base preserves the original positive IEEE word coordinates. */
  if (floor((double)s->bit_multiplier) != s->bit_multiplier ||
      floor((double)s->bit_bias) != s->bit_bias ||
      (double)s->bit_multiplier > INT32_MAX ||
      (double)s->bit_bias < INT32_MIN || (double)s->bit_bias > INT32_MAX)
    return out;
  double knot_error = 0;
  for (size_t i = 0; i < count; ++i) {
    double fraction = ldexp((double)i, -(int)bits);
    merlin_polynomial_real_interval real = merlin_polynomial_fraction_e(fraction, s);
    double low = merlin_fma_next_down(real.lo * (double)s->bit_multiplier);
    double high = merlin_fma_next_up(real.hi * (double)s->bit_multiplier);
    double knot = floor(low);
    if (!MERLIN_SOURCE_ISFINITE(knot) || knot < INT32_MIN || knot > INT32_MAX)
      return out;
    storage[i] = (int32_t)knot;
    knot_error = MERLIN_SOURCE_F64_MAX(knot_error, merlin_fma_up_add(high, -knot));
  }
  double curvature = merlin_fma_up_add(
      merlin_fma_up_mul(6, MERLIN_SOURCE_F64_ABS((double)s->coefficients[0])),
      merlin_fma_up_mul(2, MERLIN_SOURCE_F64_ABS((double)s->coefficients[1])));
  double derivative = 1;
  for (int i = 0; i < 3; ++i)
    derivative = merlin_fma_up_add(derivative,
        merlin_fma_up_mul(3-i, MERLIN_SOURCE_F64_ABS((double)s->coefficients[i])));
  /* Secant error <= |E''| h^2/8. Extracting the source fraction and truncating
   * its24-bit coordinate each contribute <=2^-24 in the fractional domain.
   * Integer interpolation truncation and source integer conversion add <2.
   * The prepared encoded error already encloses every original source FMA.
   */
  double interpolation = merlin_fma_up_mul(s->bit_multiplier,
      merlin_fma_up_add(merlin_fma_up_mul(curvature, ldexp(1., -2*(int)bits-3)),
                        merlin_fma_up_mul(derivative, 0x1p-23)));
  double error = ceil(merlin_fma_up_add(p->rounding_error,
      merlin_fma_up_add(knot_error, merlin_fma_up_add(interpolation, 2))));
  if (!MERLIN_SOURCE_ISFINITE(error) || error > INT32_MAX) return out;
  out.knots = storage; out.bits = bits; out.shift = 24-bits;
  out.error = (uint32_t)error; out.valid = 1;
  return out;
}

static inline int64_t merlin_monotone_polynomial_table_word(
    float x, const merlin_monotone_polynomial_table *p) {
  const merlin_bit_polynomial_plan *s = &p->source->checked.source;
  float scaled = x*s->scale, integral = MERLIN_MONOTONE_F32_FLOOR(scaled);
  float fraction = scaled-integral;
  uint32_t coordinate = (uint32_t)(fraction*0x1p24f);
  uint32_t index = coordinate >> p->shift;
  if (index == (UINT32_C(1) << p->bits)) --index;
  uint32_t residual = coordinate-(index << p->shift);
  int64_t difference = (int64_t)p->knots[index+1]-p->knots[index];
  int64_t value = p->knots[index] +
      difference*residual/(INT64_C(1) << p->shift);
  return (int64_t)(int32_t)s->bit_bias +
      (int64_t)(int32_t)integral*(int64_t)(int32_t)s->bit_multiplier + value;
}

static inline merlin_f32_interval merlin_monotone_polynomial_table_apply(
    merlin_f32_interval x, const merlin_monotone_polynomial_table *p) {
  if (!p || !p->source) return merlin_interval_bad();
  if (!p->valid || !x.valid || x.hi > p->source->upper)
    return merlin_monotone_bit_polynomial_apply(x, p->source);
  const merlin_bit_polynomial_plan *s = &p->source->checked.source;
  if (x.hi < s->cutoff) return merlin_interval_point(0);
  int includes_zero = x.lo < s->cutoff;
  float left = MERLIN_SOURCE_F32_MAX(x.lo, s->cutoff);
  if (left == x.hi && !includes_zero)
    return merlin_interval_point(merlin_interval_float(
        merlin_monotone_polynomial_source_word(left, s)));
  int64_t low = merlin_monotone_polynomial_table_word(left, p)-p->error;
  int64_t high = merlin_monotone_polynomial_table_word(x.hi, p)+p->error;
  if (includes_zero || low < 0) low = 0;
  if (high > INT64_C(2139095039)) high = INT64_C(2139095039);
  return merlin_interval(merlin_interval_float((uint32_t)low),
                         merlin_interval_float((uint32_t)high));
}
#endif
