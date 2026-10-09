#ifndef MERLIN_ONE_ENDPOINT_WORD_POLYNOMIAL_H
#define MERLIN_ONE_ENDPOINT_WORD_POLYNOMIAL_H
#include "monotone_bit_polynomial.h"

/* Optional exact enclosure. The borrowed prepared polynomial must remain
 * immutable and live. Its original stable RNE/nontrapping/flags-unobserved
 * contract applies. On one floor segment, real E(s)=s-P(frac(s)) has derivative
 * at most L=1-min(P'). Preparation rounds bit_multiplier*L up to a power of two.
 * The distance between ordered binary32 words times their maximum ULP bounds
 * the real s distance, including subnormals and signed zero. One source word,
 * that Lipschitz distance and the original two-error budget enclose every
 * interior source word. Floor crossings use the existing enclosure.
 */
typedef struct {
  const merlin_monotone_bit_polynomial *polynomial;
  int word_distance_shift;
  uint32_t source_upper_word;
  int nonpositive_domain;
  int valid;
} merlin_one_endpoint_word_polynomial;

static inline merlin_one_endpoint_word_polynomial
merlin_one_endpoint_word_polynomial_prepare(
    const merlin_monotone_bit_polynomial *p) {
  merlin_one_endpoint_word_polynomial out={p,0,0,0,0};
  if(!p||!p->fast_valid)return out;
  const merlin_bit_polynomial_plan *s=&p->checked.source;
  double derivative_lo=merlin_fma_next_down(merlin_fma_next_down(
      (double)s->coefficients[2]+MERLIN_SOURCE_F64_MIN(0,2.0*s->coefficients[1]))
      +MERLIN_SOURCE_F64_MIN(0,3.0*s->coefficients[0]));
  double lipschitz=merlin_fma_next_up(1.0-derivative_lo);
  double encoded_lipschitz=merlin_fma_up_mul(s->bit_multiplier,lipschitz);
  if(!MERLIN_SOURCE_ISFINITE(encoded_lipschitz)||!(encoded_lipschitz>0))return out;
  (void)frexp(encoded_lipschitz,&out.word_distance_shift);
  merlin_polynomial_real_interval top=merlin_polynomial_real_e(p->upper*s->scale,s);
  double high=merlin_fma_next_up(merlin_fma_next_up(
      (double)s->bit_multiplier*top.hi)+s->bit_bias);
  high=merlin_fma_up_add(high,p->rounding_error);
  if(!MERLIN_SOURCE_ISFINITE(high)||high<0||high>=2139095040.0)return out;
  high=ceil(high);
  out.source_upper_word=high>=2139095039.0?UINT32_C(2139095039):(uint32_t)high;
  out.nonpositive_domain=p->upper<=0;
  out.valid=1;
  return out;
}

static inline uint32_t merlin_polynomial_ordered_word(float value) {
  uint32_t bits=merlin_interval_bits(value);
  return bits&UINT32_C(0x80000000)?~bits:bits|UINT32_C(0x80000000);
}

static inline uint32_t merlin_polynomial_word_distance_bound(uint32_t distance,
                                                            int shift) {
  if(!distance)return 0;
  if(shift>=31)return UINT32_C(2139095039);
  if(shift>=0){
    uint64_t wide=(uint64_t)distance<<shift;
    return wide>UINT32_C(2139095039)?UINT32_C(2139095039):(uint32_t)wide;
  }
  if(shift<=-32)return 1;
  unsigned right=(unsigned)-shift;
  uint32_t bound=(distance>>right)+((distance&((UINT32_C(1)<<right)-1))!=0);
  return bound>UINT32_C(2139095039)?UINT32_C(2139095039):bound;
}

static inline merlin_f32_interval merlin_one_endpoint_word_polynomial_apply(
    merlin_f32_interval x,const merlin_one_endpoint_word_polynomial *prepared) {
  if(!prepared||!prepared->polynomial)return merlin_interval_bad();
  const merlin_monotone_bit_polynomial *p=prepared->polynomial;
  if(!prepared->valid||!x.valid||x.hi>p->upper)
    return merlin_monotone_bit_polynomial_apply_words(x,p);
  const merlin_bit_polynomial_plan *s=&p->checked.source;
  if(x.hi<s->cutoff)return merlin_interval_point(0);
  int includes_zero=x.lo<s->cutoff;
  float first=MERLIN_SOURCE_F32_MAX(x.lo,s->cutoff)*s->scale;
  float last=x.hi*s->scale;
  float floor_first=MERLIN_MONOTONE_F32_FLOOR(first);
  /* Admitted floors lie in [-2**24,2**24-1]; the next integer is exact. */
  if(last>=floor_first+1.0f)
    return merlin_monotone_bit_polynomial_apply_words(x,p);
  float fraction=first-floor_first,poly=s->coefficients[0];
  for(int i=1;i<4;i++)poly=MERLIN_SOURCE_F32_FMA(fraction,poly,s->coefficients[i]);
  uint32_t word=(uint32_t)(int32_t)MERLIN_SOURCE_F32_FMA(
      s->bit_multiplier,first-poly,s->bit_bias);
  if(first==last&&!includes_zero)return merlin_interval_point(merlin_interval_float(word));
  uint32_t distance;unsigned exponent;
  if(prepared->nonpositive_domain){
    uint32_t a=merlin_interval_bits(first)&UINT32_C(0x7fffffff);
    uint32_t b=merlin_interval_bits(last)&UINT32_C(0x7fffffff);
    distance=a-b;exponent=(a>>23)&255;
  }else{
    distance=merlin_polynomial_ordered_word(last)-merlin_polynomial_ordered_word(first);
    float magnitude=MERLIN_SOURCE_F32_MAX(MERLIN_SOURCE_F32_ABS(first),MERLIN_SOURCE_F32_ABS(last));
    exponent=(merlin_interval_bits(magnitude)>>23)&255;
  }
  int ulp_shift=exponent?(int)exponent-150:-149;
  uint32_t extra=merlin_polynomial_word_distance_bound(distance,
      prepared->word_distance_shift+ulp_shift);
  uint32_t low=includes_zero||word<p->word_budget?0:word-p->word_budget;
  uint64_t high=(uint64_t)word+p->word_budget+extra;
  if(high>prepared->source_upper_word)high=prepared->source_upper_word;
  return merlin_interval(merlin_interval_float(low),merlin_interval_float((uint32_t)high));
}
#endif
