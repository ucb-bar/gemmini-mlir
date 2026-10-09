#ifndef MERLIN_EXACT_ABSOLUTE_DOT_BOUNDS_H
#define MERLIN_EXACT_ABSOLUTE_DOT_BOUNDS_H
#include "ordered_fma_bounds.h"

/* Internal producer contract: centers[i] is the exact real signed source dot;
 * absolute[i] is the exact sum of absolute source products for that same dot.
 * Both arrays have complete writes, distinct private owners, and stay immutable
 * through synchronous consumption. Representation equality and point source
 * envelopes must be proved by the canonical encoder, not asserted by a caller.
 * This is a source rounding enclosure, never a claim that ordered FMA is exact.
 */
typedef struct {
 const double *centers,*absolute;
 size_t cells,length;
 double gamma,subnormal;
 int valid;
} merlin_exact_absolute_dot_bounds;

static inline merlin_exact_absolute_dot_bounds merlin_exact_absolute_dot_prepare(
 const merlin_fma_zero_gamma_batch *gamma,const double *centers,
 const double *absolute,size_t cells,size_t length) {
 merlin_exact_absolute_dot_bounds p={0};
 if(!gamma||!gamma->valid||gamma->length!=length||!length||!cells||
    !centers||!absolute||!MERLIN_SOURCE_ISFINITE(gamma->gamma_upper)||
    gamma->gamma_upper<=0||!MERLIN_SOURCE_ISFINITE(gamma->subnormal_error_upper)||
    gamma->subnormal_error_upper<=0)return p;
 double maximum=0;
 for(size_t i=0;i<cells;i++){
  double t=absolute[i],s=centers[i];
  if(!MERLIN_SOURCE_ISFINITE(t)||!MERLIN_SOURCE_ISFINITE(s)||t<0||
     MERLIN_SOURCE_F64_ABS(s)>t)return p;
  maximum=MERLIN_SOURCE_F64_MAX(maximum,t);
 }
 double radius=merlin_fma_up_add(merlin_fma_up_mul(gamma->gamma_upper,maximum),
                               gamma->subnormal_error_upper);
 double envelope=merlin_fma_up_add(maximum,radius);
 if(!MERLIN_SOURCE_ISFINITE(envelope)||envelope>=(double)FLT_MAX)return p;
 return (merlin_exact_absolute_dot_bounds){centers,absolute,cells,length,
   gamma->gamma_upper,gamma->subnormal_error_upper,1};
}

/* |ordered_FMA-real_dot| <= gamma_k * sum |a_i*b_i| + eta_k.
 * Every real prefix is bounded by the same sum; uniform admission therefore
 * closes finite source prefixes and finite final binary32 endpoints. */
static inline int merlin_exact_absolute_dot_apply(
 const merlin_exact_absolute_dot_bounds *p,size_t cell,float *lower,float *upper) {
 if(!p||!p->valid||cell>=p->cells||!lower||!upper)return 0;
 double radius=merlin_fma_up_add(merlin_fma_up_mul(p->gamma,p->absolute[cell]),p->subnormal);
 double lo=merlin_fma_down_add(p->centers[cell],-radius);
 double hi=merlin_fma_up_add(p->centers[cell],radius);
#if defined(MERLIN_ENABLE_EXACT_BOUND_CONVERSION)
 *lower=MERLIN_F32_EXACT_FLOOR_FROM_F64(lo);
 *upper=MERLIN_F32_EXACT_CEIL_FROM_F64(hi);
#else
 *lower=(float)lo;*upper=(float)hi;
 if((double)*lower>lo)*lower=merlin_fma_next_down_f32(*lower);
 if((double)*upper<hi)*upper=merlin_fma_next_up_f32(*upper);
#endif
 return 1;
}
#endif
