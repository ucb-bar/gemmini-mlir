#ifndef MERLIN_ORDERED_FMA_BOUNDS_H
#define MERLIN_ORDERED_FMA_BOUNDS_H

/* A certificate for zero-seeded, increasing-K binary32 fused multiply-adds.
 * The caller supplies outward binary64 summaries of consecutive reconstructed
 * product chunks.  No target, tensor shape, or approximate-answer selector is
 * part of this contract.  A summary must enclose the EXACT reconstructed sum,
 * its absolute product sum, and sum(abs(original_product-reconstructed_product)).
 * This header checks arithmetic eligibility, not the origin of those summaries.
 * Source and certificate execution require IEEE RNE, gradual underflow, and no
 * fast-math/reassociation.  Failure must select exact source replay.
 */
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  double sum_lower, sum_upper;
  double absolute_upper, representation_error_upper;
  size_t length;
} merlin_fma_chunk;

typedef struct {
  double prefix_lower, prefix_upper;
  double representation_error, rounding_error;
  int valid;
} merlin_fma_bound;

/* IEEE binary64 adjacency. Unlike a math-library nextafter call, these helpers
 * preserve values without setting floating exception/errno flags. Certificates
 * observe values and eligibility, not those diagnostic side effects. */
static inline double merlin_fma_next_up(double value) {
  uint64_t raw; memcpy(&raw, &value, sizeof(raw));
  const uint64_t magnitude = raw & UINT64_C(0x7fffffffffffffff);
  if (magnitude > UINT64_C(0x7ff0000000000000) || raw == UINT64_C(0x7ff0000000000000)) return value;
  if (!magnitude) raw = 1;
  else if (raw >> 63) --raw;
  else ++raw;
  memcpy(&value, &raw, sizeof(raw)); return value;
}
static inline double merlin_fma_next_down(double value) {
  uint64_t raw; memcpy(&raw, &value, sizeof(raw));
  const uint64_t magnitude = raw & UINT64_C(0x7fffffffffffffff);
  if (magnitude > UINT64_C(0x7ff0000000000000) || raw == UINT64_C(0xfff0000000000000)) return value;
  if (!magnitude) raw = UINT64_C(0x8000000000000001);
  else if (raw >> 63) ++raw;
  else --raw;
  memcpy(&value, &raw, sizeof(raw)); return value;
}
static inline float merlin_fma_next_up_f32(float value) {
  uint32_t raw; memcpy(&raw, &value, sizeof(raw));
  const uint32_t magnitude = raw & UINT32_C(0x7fffffff);
  if (magnitude > UINT32_C(0x7f800000) || raw == UINT32_C(0x7f800000)) return value;
  if (!magnitude) raw = 1;
  else if (raw >> 31) --raw;
  else ++raw;
  memcpy(&value, &raw, sizeof(raw)); return value;
}
static inline float merlin_fma_next_down_f32(float value) {
  uint32_t raw; memcpy(&raw, &value, sizeof(raw));
  const uint32_t magnitude = raw & UINT32_C(0x7fffffff);
  if (magnitude > UINT32_C(0x7f800000) || raw == UINT32_C(0xff800000)) return value;
  if (!magnitude) raw = UINT32_C(0x80000001);
  else if (raw >> 31) ++raw;
  else --raw;
  memcpy(&value, &raw, sizeof(raw)); return value;
}
static inline double merlin_fma_half_ulp(double magnitude) {
  if (magnitude < 0x1p-126) return 0x1p-150;
  uint64_t raw; memcpy(&raw, &magnitude, sizeof(raw));
  const uint64_t exponent = (raw >> 52) & UINT64_C(0x7ff);
  raw = (exponent - 24) << 52;
  double result; memcpy(&result, &raw, sizeof(raw)); return result;
}

static inline double merlin_fma_up_add(double a, double b) {
  return merlin_fma_next_up(a + b);
}
static inline double merlin_fma_down_add(double a, double b) {
  return merlin_fma_next_down(a + b);
}
static inline double merlin_fma_up_mul(double a, double b) {
  return merlin_fma_next_up(a * b);
}

static inline merlin_fma_bound merlin_fma_bound_begin(void) {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
  return (merlin_fma_bound){0.0, 0.0, 0.0, 0.0, 0};
#else
  if (sizeof(double) != sizeof(uint64_t) || sizeof(float) != sizeof(uint32_t))
    return (merlin_fma_bound){0.0, 0.0, 0.0, 0.0, 0};
  const double one = 1.0;
  uint64_t one_bits; memcpy(&one_bits, &one, sizeof(one_bits));
  const float fone = 1.0f;
  uint32_t fone_bits; memcpy(&fone_bits, &fone, sizeof(fone_bits));
  volatile float tiny = 0x1p-149f;
  volatile double dtiny = 0x1p-1074;
  const int eligible = FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
      FLT_MAX_EXP == 128 && FLT_MIN_EXP == -125 && DBL_MANT_DIG == 53 &&
      DBL_MAX_EXP == 1024 && DBL_MIN_EXP == -1021 &&
      one_bits == UINT64_C(0x3ff0000000000000) &&
      fone_bits == UINT32_C(0x3f800000) &&
      fegetround() == FE_TONEAREST && tiny + tiny == 0x1p-148f &&
      dtiny + dtiny == 0x1p-1073;
  return (merlin_fma_bound){0.0, 0.0, 0.0, 0.0, eligible};
#endif
}

static inline int merlin_fma_bound_push(merlin_fma_bound *state,
                                      merlin_fma_chunk chunk) {
  if (!state->valid || chunk.length == 0 || chunk.length >= ((size_t)1 << 24) ||
      !isfinite(chunk.sum_lower) || !isfinite(chunk.sum_upper) ||
      !isfinite(chunk.absolute_upper) || !isfinite(chunk.representation_error_upper) ||
      chunk.sum_lower > chunk.sum_upper || chunk.absolute_upper < 0.0 ||
      chunk.representation_error_upper < 0.0 ||
      chunk.sum_lower > chunk.absolute_upper ||
      chunk.sum_upper < -chunk.absolute_upper) {
    state->valid = 0;
    return 0;
  }

  /* A partial sum of this chunk lies between its total negative and positive
   * variations: (T-A)/2 <= partial <= (T+A)/2.  Include the uncertain incoming
   * reconstructed prefix and all representation error accumulated so far. */
  const double middle_lower = merlin_fma_down_add(state->prefix_lower,
      merlin_fma_next_down(chunk.sum_lower * 0.5));
  const double middle_upper = merlin_fma_up_add(state->prefix_upper,
      merlin_fma_next_up(chunk.sum_upper * 0.5));
  const double repr = merlin_fma_up_add(state->representation_error,
                                       chunk.representation_error_upper);
  const double magnitude = merlin_fma_up_add(
      merlin_fma_up_add(fmax(fabs(middle_lower), fabs(middle_upper)),
                        merlin_fma_up_mul(0.5, chunk.absolute_upper)), repr);

  /* e[j] <= (1+u)e[j-1] + u*|exact_prefix[j]| + half_min_subnormal.
   * gamma_l >= (1+u)^l-1 gives an initial safe radius for every step. */
  const double lu = (double)chunk.length * 0x1p-24;
  const double gamma = merlin_fma_next_up(lu / (1.0 - lu));
  const double eta = merlin_fma_next_up((double)chunk.length * 0x1p-150 / (1.0 - lu));
  const double coarse = merlin_fma_up_add(
      merlin_fma_up_mul(merlin_fma_up_add(1.0, gamma), state->rounding_error),
      merlin_fma_up_add(merlin_fma_up_mul(gamma, magnitude), eta));
  const double rounded_input_max = merlin_fma_up_add(magnitude, coarse);
  if (!isfinite(rounded_input_max) || rounded_input_max >= (double)FLT_MAX) {
    state->valid = 0;
    return 0;
  }

  /* The coarse radius encloses every exact FMA input.  Its binade gives an
   * absolute half-ulp limit for every rounding, often tighter than u*|x|. */
  const double half_ulp = merlin_fma_half_ulp(rounded_input_max);
  const double sharp = merlin_fma_up_add(state->rounding_error,
      merlin_fma_up_mul((double)chunk.length, half_ulp));
  state->rounding_error = fmin(coarse, sharp);
  state->representation_error = repr;
  state->prefix_lower = merlin_fma_down_add(state->prefix_lower, chunk.sum_lower);
  state->prefix_upper = merlin_fma_up_add(state->prefix_upper, chunk.sum_upper);
  if (!isfinite(state->prefix_lower) || !isfinite(state->prefix_upper)) {
    state->valid = 0;
    return 0;
  }
  return 1;
}

static inline int merlin_fma_bound_finish(const merlin_fma_bound *state,
                                        float *lower, float *upper) {
  if (!state->valid) return 0;
  const double radius = merlin_fma_up_add(state->representation_error,
                                        state->rounding_error);
  const double lo = merlin_fma_down_add(state->prefix_lower, -radius);
  const double hi = merlin_fma_up_add(state->prefix_upper, radius);
  if (!isfinite(lo) || !isfinite(hi) || lo <= -(double)FLT_MAX || hi >= (double)FLT_MAX)
    return 0;
  *lower = (float)lo;
  *upper = (float)hi;
  if ((double)*lower > lo) *lower = merlin_fma_next_down_f32(*lower);
  if ((double)*upper < hi) *upper = merlin_fma_next_up_f32(*upper);
  return 1;
}

#endif
