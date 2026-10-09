#ifndef MERLIN_POSITIVE_UNCERTAINTY_PRODUCT_H
#define MERLIN_POSITIVE_UNCERTAINTY_PRODUCT_H
#include "ordered_fma_bounds.h"
#include <stdint.h>
#include <limits.h>
/* Private finite binary32 operands, immutable through preparation and product.
 * Each nonnegative magnitude is rounded UP to an integer in [0,64] times an
 * exact binary64 power of two. Thus the exact integer dot times both scales
 * encloses the real sum of uncertainty * absolute(source RHS).
 * The callback must compute exactly K products, store all i32 outputs and
 * synchronize before consumption. It grants no source-FMA rounding shortcut.
 */
static inline int merlin_uncertainty_magnitude(float a,float lo,float hi,double *v){
 if(!MERLIN_SOURCE_ISFINITE(a)||!MERLIN_SOURCE_ISFINITE(lo)||!MERLIN_SOURCE_ISFINITE(hi)||!(lo<=a&&a<=hi))return 0;
 double d=MERLIN_SOURCE_F64_MAX(MERLIN_SOURCE_F64_ABS((double)lo-a),MERLIN_SOURCE_F64_ABS((double)hi-a));
 *v=d==0?0:merlin_fma_next_up(d);return MERLIN_SOURCE_ISFINITE(*v);
}
static inline int merlin_positive_scale(double maximum,double *step){
 if(!MERLIN_SOURCE_ISFINITE(maximum)||maximum<0)return 0;
 if(maximum==0){*step=1;return 1;}
 int exponent;frexp(maximum,&exponent);
 /* Binary32 endpoints and their binary64 outward differences fit this domain.
  * These bounds also keep rescaling exact, normal binary64, including zeros. */
 if(exponent < -160 || exponent > 130)return 0;
 *step=ldexp(1.0,exponent-6);return 1;
}
static inline int merlin_positive_ceil(double value,double step,int8_t *out){
 double q=value/step;
 if(!MERLIN_SOURCE_ISFINITE(q)||q<0||q>64)return 0;
 int n=(int)q;if((double)n<q)n++;
 if(n>64)return 0;*out=(int8_t)n;return 1;
}
static inline int merlin_uncertainty_pack(const float *a,const float *lo,const float *hi,
 int rows,int length,int8_t *plane,double *steps,int *nonzero){
 merlin_fma_bound env=merlin_fma_bound_begin();
 if(!env.valid||!a||!lo||!hi||!plane||!steps||!nonzero||rows<=0||length<=0||length>INT32_MAX/4096||
 (size_t)rows>SIZE_MAX/(size_t)length)return 0;
 *nonzero=0;
 for(int r=0;r<rows;r++){
  double maximum=0;
  for(int z=0;z<length;z++){double d;size_t t=(size_t)r*length+z;if(!merlin_uncertainty_magnitude(a[t],lo[t],hi[t],&d))return 0;maximum=MERLIN_SOURCE_F64_MAX(maximum,d);}
  if(!merlin_positive_scale(maximum,&steps[r]))return 0;
  *nonzero|=maximum!=0;
  for(int z=0;z<length;z++){double d;size_t t=(size_t)r*length+z;if(!merlin_uncertainty_magnitude(a[t],lo[t],hi[t],&d)||!merlin_positive_ceil(d,steps[r],&plane[t]))return 0;}
 }return 1;
}
static inline int merlin_absolute_rhs_pack(const float *b,int columns,int length,int8_t *plane,double *steps){
 if(!b||!plane||!steps||columns<=0||length<=0||length>INT32_MAX/4096||(size_t)columns>SIZE_MAX/(size_t)length)return 0;
 for(int c=0;c<columns;c++){
  double maximum=0;
  for(int z=0;z<length;z++){double v=b[(size_t)c*length+z];if(!MERLIN_SOURCE_ISFINITE(v))return 0;maximum=MERLIN_SOURCE_F64_MAX(maximum,MERLIN_SOURCE_F64_ABS(v));}
  if(!merlin_positive_scale(maximum,&steps[c]))return 0;
  for(int z=0;z<length;z++)if(!merlin_positive_ceil(MERLIN_SOURCE_F64_ABS((double)b[(size_t)c*length+z]),steps[c],&plane[(size_t)z*columns+c]))return 0;
 }return 1;
}
static inline double merlin_uncertainty_product_upper(int32_t sum,double a,double b){
 if(sum<0)return INFINITY;
 return merlin_fma_up_mul(merlin_fma_up_mul((double)sum,a),b);
}
#endif
