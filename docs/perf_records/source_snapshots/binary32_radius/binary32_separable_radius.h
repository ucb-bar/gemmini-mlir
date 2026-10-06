#ifndef MERLIN_BINARY32_SEPARABLE_RADIUS_H
#define MERLIN_BINARY32_SEPARABLE_RADIUS_H
#include "separable_fma_radius.h"
/* Explicit IEEE directed capability, nontrapping/unobserved flags, stable
 * source RNE and gradual underflow. No ambient rounding-mode mutation.
 * Source equality, zero representation error and finite ordered source
 * prefixes are established by the unchanged private separable producer. */
#if !defined(MERLIN_F32_RADIUS_FMA_UP) || !defined(MERLIN_F32_RADIUS_ADD_UP) || !defined(MERLIN_F32_RADIUS_ADD_DOWN) || !defined(MERLIN_F32_EXACT_FLOOR_FROM_F64) || !defined(MERLIN_F32_EXACT_CEIL_FROM_F64)
#error "binary32 radius needs explicit directed arithmetic and exact casts"
#endif

typedef struct {
 const merlin_fma_separable_radius *source;
 float factor,subnormal;
 int valid;
} merlin_binary32_separable_radius;
static inline merlin_binary32_separable_radius merlin_binary32_radius_prepare(
 const merlin_fma_separable_radius *source) {
 merlin_binary32_separable_radius p={0};
 if(!source||!source->valid)return p;
 float factor=MERLIN_F32_EXACT_CEIL_FROM_F64(source->gamma_l1);
 float eta=MERLIN_F32_EXACT_CEIL_FROM_F64(source->subnormal);
 if(!MERLIN_SOURCE_ISFINITE(factor)||!MERLIN_SOURCE_ISFINITE(eta)||factor<0||eta<0)return p;
 return (merlin_binary32_separable_radius){source,factor,eta,1};
}
static inline int merlin_binary32_radius_apply(
 const merlin_binary32_separable_radius *p,size_t j,float *lower,float *upper) {
 if(!p||!p->valid)return 0;
 const merlin_fma_separable_radius *s=p->source;
 if(j>=s->columns->columns||!lower||!upper)return 0;
 /* The source-column maximum is an exact binary32 value: it is a maximum
  * over magnitudes of finite binary32 source operands. Upward cast is also
  * conservative if a future producer uses a wider admitted maximum. */
 float maximum=MERLIN_F32_EXACT_CEIL_FROM_F64(s->columns->source[j].maximum);
 float radius=MERLIN_F32_RADIUS_FMA_UP(p->factor,maximum,p->subnormal);
 float center_lo=MERLIN_F32_EXACT_FLOOR_FROM_F64(s->centers[j]);
 float center_hi=MERLIN_F32_EXACT_CEIL_FROM_F64(s->centers[j]);
 float lo=MERLIN_F32_RADIUS_ADD_DOWN(center_lo,-radius);
 float hi=MERLIN_F32_RADIUS_ADD_UP(center_hi,radius);
 /* Widening in binary32 may overflow despite the tighter original admission.
  * Preserve the complete checked binary64 path and do not publish infinities. */
 if(!MERLIN_SOURCE_ISFINITE(lo)||!MERLIN_SOURCE_ISFINITE(hi)||lo>hi)
  return merlin_fma_separable_radius_apply(s,j,lower,upper);
 *lower=lo;*upper=hi;return 1;
}
#endif
