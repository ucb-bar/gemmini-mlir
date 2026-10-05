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

static inline double merlin_fma_up_add(double a, double b) {
  return nextafter(a + b, INFINITY);
}
static inline double merlin_fma_down_add(double a, double b) {
  return nextafter(a + b, -INFINITY);
}
static inline double merlin_fma_up_mul(double a, double b) {
  return nextafter(a * b, INFINITY);
}

static inline merlin_fma_bound merlin_fma_bound_begin(void) {
  volatile float tiny = 0x1p-149f;
  volatile double dtiny = 0x1p-1074;
  const int eligible = FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
      FLT_MAX_EXP == 128 && FLT_MIN_EXP == -125 && DBL_MANT_DIG == 53 &&
      DBL_MAX_EXP == 1024 && DBL_MIN_EXP == -1021 &&
      fegetround() == FE_TONEAREST && tiny + tiny == 0x1p-148f &&
      dtiny + dtiny == 0x1p-1073;
  return (merlin_fma_bound){0.0, 0.0, 0.0, 0.0, eligible};
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
      nextafter(chunk.sum_lower * 0.5, -INFINITY));
  const double middle_upper = merlin_fma_up_add(state->prefix_upper,
      nextafter(chunk.sum_upper * 0.5, INFINITY));
  const double repr = merlin_fma_up_add(state->representation_error,
                                       chunk.representation_error_upper);
  const double magnitude = merlin_fma_up_add(
      merlin_fma_up_add(fmax(fabs(middle_lower), fabs(middle_upper)),
                        merlin_fma_up_mul(0.5, chunk.absolute_upper)), repr);

  /* e[j] <= (1+u)e[j-1] + u*|exact_prefix[j]| + half_min_subnormal.
   * gamma_l >= (1+u)^l-1 gives an initial safe radius for every step. */
  const double lu = (double)chunk.length * 0x1p-24;
  const double gamma = nextafter(lu / (1.0 - lu), INFINITY);
  const double eta = nextafter((double)chunk.length * 0x1p-150 / (1.0 - lu), INFINITY);
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
  double half_ulp = 0x1p-150;
  if (rounded_input_max > 0.0) {
    const double normal_half_ulp = scalbn(1.0, ilogb(rounded_input_max) - 24);
    half_ulp = fmax(half_ulp, normal_half_ulp);
  }
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
  if ((double)*lower > lo) *lower = nextafterf(*lower, -INFINITY);
  if ((double)*upper < hi) *upper = nextafterf(*upper, INFINITY);
  return 1;
}

#endif
