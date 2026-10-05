#ifndef MERLIN_FMA_PRODUCT_NORMS_H
#define MERLIN_FMA_PRODUCT_NORMS_H

/* Outward metadata for product-sum certificates. These row/column summaries
 * cost O(M*K+N*K); forming all pair bounds costs O(M*N), never O(M*N*K).
 * IEEE binary32 inputs, binary64 metadata, strict RNE and correctly rounded
 * binary64 sqrt are required. Callers use ordered_fma_bounds.h eligibility.
 */
#include "ordered_fma_bounds.h"

typedef struct {
  double absolute_sum, absolute_max, square_sum;
  double original_max, error_sum, error_max;
} merlin_fma_operand_norm;

static inline int merlin_fma_operand_summarize(
    const float *original, const float *reconstructed, size_t length,
    size_t original_stride, size_t reconstructed_stride,
    merlin_fma_operand_norm *out) {
  *out = (merlin_fma_operand_norm){0, 0, 0, 0, 0, 0};
  if (!length || length >= ((size_t)1 << 24)) return 0;
  for (size_t z = 0; z < length; ++z) {
    const double a = original[z * original_stride];
    const double ar = reconstructed[z * reconstructed_stride];
    if (!isfinite(a) || !isfinite(ar)) return 0;
    const double error = merlin_fma_next_up(fabs(a - ar));
    out->absolute_sum = merlin_fma_up_add(out->absolute_sum, fabs(ar));
    out->square_sum = merlin_fma_up_add(out->square_sum, merlin_fma_up_mul(ar, ar));
    out->absolute_max = fmax(out->absolute_max, fabs(ar));
    out->original_max = fmax(out->original_max, fabs(a));
    out->error_sum = merlin_fma_up_add(out->error_sum, error);
    out->error_max = fmax(out->error_max, error);
  }
  return 1;
}

static inline void merlin_fma_product_summarize(
    merlin_fma_operand_norm a, merlin_fma_operand_norm b,
    double *absolute_upper, double *representation_error_upper) {
  /* |ab-ar*br| <= |a-ar|*|b| + |ar|*|b-br|. */
  *representation_error_upper = merlin_fma_up_add(
      merlin_fma_up_mul(a.error_sum, b.original_max),
      merlin_fma_up_mul(a.absolute_sum, b.error_max));
  /* Three independently valid Holder bounds on sum(abs(ar*br)). */
  const double cs = merlin_fma_next_up(sqrt(merlin_fma_up_mul(a.square_sum, b.square_sum)));
  *absolute_upper = fmin(cs, fmin(
      merlin_fma_up_mul(a.absolute_sum, b.absolute_max),
      merlin_fma_up_mul(a.absolute_max, b.absolute_sum)));
}

#endif
