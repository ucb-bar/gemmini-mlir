#ifndef MERLIN_SEPARABLE_FMA_RADIUS_H
#define MERLIN_SEPARABLE_FMA_RADIUS_H
#include "prepared_fma_product_bounds.h"
/* Source-zero-seeded ordered binary32 FMA, stable RNE/gradual underflow.
 * Admitted immutable error norms prove exact representation, not approximate
 * equality. The producer's binary64 center must be the exact real dot of those
 * same operands. No arbitrary center/public scalar admission is provided.
 */
typedef struct {
 const merlin_admitted_dot_norms *source;
 size_t columns,length;
 double maximum;
 int valid;
} merlin_fma_exact_columns;
static inline merlin_fma_exact_columns merlin_fma_exact_columns_prepare(
 const merlin_fma_product_columns *producer,
 const merlin_admitted_dot_norms *source,
 const merlin_admitted_dot_norms *error) {
 merlin_fma_exact_columns p={0,0,0,0,0};
 if(!producer||!producer->valid||!source||!error)return p;
 double maximum=0;
 for(size_t j=0;j<producer->columns;j++){
  if(error[j].l1!=0||error[j].maximum!=0||error[j].l2!=0||
     !MERLIN_SOURCE_ISFINITE(source[j].maximum)||source[j].maximum<0)return p;
  maximum=MERLIN_SOURCE_F64_MAX(maximum,source[j].maximum);
 }
 if(!producer->columns||!producer->length)return p;
 return (merlin_fma_exact_columns){source,producer->columns,producer->length,maximum,1};
}
typedef struct {
 const merlin_fma_exact_columns *columns;
 const double *centers;
 double gamma_l1,subnormal;
 int valid;
} merlin_fma_separable_radius;
static inline merlin_fma_separable_radius merlin_fma_separable_radius_prepare(
 const merlin_fma_product_row *producer,
 const merlin_fma_exact_columns *columns,
 const merlin_admitted_dot_norms *source,
 const merlin_admitted_dot_norms *error,size_t uncertain_positions) {
 merlin_fma_separable_radius p={0,0,0,0,0};
 if(!producer||!producer->valid||!columns||!columns->valid||!source||!error||
    uncertain_positions||producer->length!=columns->length||
    producer->columns!=columns->columns||!producer->centers||
    error->l1!=0||error->maximum!=0||error->l2!=0||
    !MERLIN_SOURCE_ISFINITE(source->l1)||source->l1<0)return p;
 double factor=merlin_fma_up_mul(producer->gamma_upper,source->l1);
 double absolute=merlin_fma_up_mul(source->l1,columns->maximum);
 double radius=merlin_fma_up_add(merlin_fma_up_mul(factor,columns->maximum),producer->subnormal_error_upper);
 double envelope=merlin_fma_up_add(absolute,radius);
 if(!MERLIN_SOURCE_ISFINITE(factor)||!MERLIN_SOURCE_ISFINITE(envelope)||envelope>=(double)FLT_MAX)return p;
 return (merlin_fma_separable_radius){columns,producer->centers,factor,producer->subnormal_error_upper,1};
}
/* |source_dot-real_dot| <= gamma_k sum_i |a_i b_i| + eta_k.
 * sum_i |a_i b_ij| <= L1(a) max_i |b_ij|. Both upward products
 * cover the real product regardless of reassociation. The row envelope above
 * also bounds every source prefix and final enclosure below FLT_MAX.
 * Only source-certified private exact products reach this internal consumer.
 */
static inline int merlin_fma_separable_radius_apply(
 const merlin_fma_separable_radius *p,size_t column,float *lower,float *upper) {
 if(!p||!p->valid||column>=p->columns->columns)return 0;
 double radius=merlin_fma_up_add(merlin_fma_up_mul(p->gamma_l1,
   p->columns->source[column].maximum),p->subnormal);
 double lo=merlin_fma_down_add(p->centers[column],-radius);
 double hi=merlin_fma_up_add(p->centers[column],radius);
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
