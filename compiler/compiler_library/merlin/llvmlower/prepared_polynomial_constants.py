"""Specialize immutable prepared source-polynomial constants under exact evidence.

The caller must retain the admitted context and RNE environment unchanged while
using it. Interval inputs retain the same finite, ordered private-span authority
as the existing batch helper. This does not authorize approximate endpoints or
change any F32 denominator observation. Runtime preparation refuses other plans.
"""

import struct

from merlin.llvmlower.rounded_polynomial_monotonicity import (
    RoundedPolynomialMonotonicity,
    consume_rounded_polynomial_monotonicity,
)
from merlin.llvmlower.source_numeric_capability import SourceNumericContract


def c_header(
    original_header: str,
    proof: RoundedPolynomialMonotonicity,
    contract: SourceNumericContract,
) -> str:
    """Emit a private immutable context consumer, never a default policy.

    Existing preparation proves source conversion/range safety. Complete rounded
    monotonicity supplies ordered exact endpoints, allowing a literal valid result
    after admission. Plan, RNE and zero-budget checks occur once at context entry;
    changed/unsupported preparation retains the original checked batch helper.
    Target evaluator equivalence remains the supplied theorem's obligation.
    """
    consume_rounded_polynomial_monotonicity(original_header, proof, contract)
    values = [struct.unpack("!f", struct.pack("!I", word))[0].hex() + "f" for word in proof.plan_words]
    cutoff, scale, *rest = values
    coefficients = rest[:4]
    multiplier, bias = rest[4:]
    words = ",".join(f"UINT32_C(0x{word:08x})" for word in proof.plan_words)
    template = r"""
#ifndef MERLIN_PREPARED_POLYNOMIAL_CONSTANTS_H
#define MERLIN_PREPARED_POLYNOMIAL_CONSTANTS_H
#include <fenv.h>
#include <string.h>
#include "prepared_polynomial_batch.h"
/* Consume only after ordinary source-domain/span admission. No callback may
 * mutate the plan, environment or borrowed endpoints in this private epoch. */
static inline int merlin_polynomial_constants_admit(
    const merlin_monotone_bit_polynomial *p) {
  if(!p || !p->fast_valid || p->upper!=0.0f || p->word_budget!=0 ||
     fegetround()!=FE_TONEAREST)return 0;
  const merlin_bit_polynomial_plan *s=&p->checked.source;
  const float actual[8]={s->cutoff,s->scale,s->coefficients[0],
    s->coefficients[1],s->coefficients[2],s->coefficients[3],
    s->bit_multiplier,s->bit_bias};
  const uint32_t expected[8]={@WORDS@};
  for(int i=0;i<8;i++){
    uint32_t bits;memcpy(&bits,actual+i,4);if(bits!=expected[i])return 0;
  }
  return 1;
}
static inline void merlin_polynomial_constants_four(
    const merlin_f32_interval x[4],const merlin_monotone_bit_polynomial *p,
    int admitted,merlin_f32_interval out[4]) {
  if(!admitted)goto checked;
  for(int i=0;i<4;i++)if(!x[i].valid || x[i].hi>0.0f)goto checked;
  float scaled[8],fraction[8],poly[8];
  for(int i=0;i<4;i++){
    scaled[2*i]=MERLIN_SOURCE_F32_MAX(x[i].lo,@CUTOFF@)*@SCALE@;
    scaled[2*i+1]=MERLIN_SOURCE_F32_MAX(x[i].hi,@CUTOFF@)*@SCALE@;
  }
  for(int i=0;i<8;i++){
#if defined(MERLIN_POLYNOMIAL_BOUNDED_FLOOR)
    float f=scaled[i]==0.0f?scaled[i]:(float)(int32_t)__builtin_floorf(scaled[i]);
#else
    float f=MERLIN_MONOTONE_F32_FLOOR(scaled[i]);
#endif
    fraction[i]=scaled[i]-f;poly[i]=@C0@;
  }
  const float coefficients[3]={@C1@,@C2@,@C3@};
  for(int c=0;c<3;c++){
#if defined(MERLIN_ENABLE_SOURCE_FMA_BATCH_8)
    MERLIN_SOURCE_F32_FMA_EIGHT(fraction,poly,coefficients[c]);
#else
    for(int i=0;i<8;i++)poly[i]=MERLIN_SOURCE_F32_FMA(fraction[i],poly[i],coefficients[c]);
#endif
  }
  for(int i=0;i<4;i++){
    uint32_t first=(uint32_t)(int32_t)MERLIN_SOURCE_F32_FMA(@MULTIPLIER@,scaled[2*i]-poly[2*i],@BIAS@);
    uint32_t last=(uint32_t)(int32_t)MERLIN_SOURCE_F32_FMA(@MULTIPLIER@,scaled[2*i+1]-poly[2*i+1],@BIAS@);
    if(x[i].hi<@CUTOFF@){out[i]=(merlin_f32_interval){0,0,1};continue;}
    float lo=x[i].lo<@CUTOFF@?0:merlin_interval_float(first);
    out[i]=(merlin_f32_interval){lo,merlin_interval_float(last),1};
  }
  return;
checked:
  merlin_polynomial_words_four(x,p,out);
}
#endif
"""
    substitutions = {
        "WORDS": words,
        "CUTOFF": cutoff,
        "SCALE": scale,
        "MULTIPLIER": multiplier,
        "BIAS": bias,
        **{f"C{i}": value for i, value in enumerate(coefficients)},
    }
    for name, value in substitutions.items():
        template = template.replace("@" + name + "@", value)
    return template
