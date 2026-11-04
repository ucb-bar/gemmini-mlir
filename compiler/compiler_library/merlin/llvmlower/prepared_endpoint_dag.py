"""Optional finite endpoint-DAG preparation for closed private consumers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedEndpointContract:
    stable_rne_gradual_underflow: bool
    nontrapping_effects_unobserved: bool
    immutable_private_epoch: bool
    disjoint_complete_output_spans: bool
    record_all_parts_before_consumption: bool
    invalidate_before_any_refinement: bool
    original_operation_order_retained: bool
    checked_fallback_retained: bool
    closed_bf16_consumer_uses: bool

    def validate(self) -> None:
        if any(type(x) is not bool or not x for x in vars(self).values()):
            raise ValueError("complete prepared endpoint ownership/effect contract required")


def c_header(contract: PreparedEndpointContract) -> str:
    """Emit a producer and bounded row evaluator; no new numeric permission.

    Layout is explicitly [part,row,column]. A producer-issued private owner
    records every value while copying it. Any refinement invalidates the row
    before mutation; prepared evaluation then refuses that row for the rest of
    the epoch. Checked evaluation remains available. A row magnitude recurrence
    follows the exact positive source multiply/add order under RNE; monotonicity
    and sign symmetry bound all signed endpoints and centers. Every bound
    prefix is finite, and final magnitude is no greater than largest finite
    BF16, before validation-free point arithmetic is used. This is a range
    theorem about the input intervals, not about their relation to source truth.
    Caller arrays and metadata are disjoint, correctly sized and exclusively
    owned throughout this synchronous epoch. No certificate survives mutation.
    """
    if not isinstance(contract, PreparedEndpointContract):
        raise ValueError("typed prepared endpoint contract required")
    contract.validate()
    return r"""#ifndef MERLIN_PREPARED_ENDPOINT_DAG_H
#define MERLIN_PREPARED_ENDPOINT_DAG_H
#include "positive_scalar_interval.h"
#include "bf16_quant_frontier.h"
#include <stdint.h>
#include <stddef.h>
typedef struct {
 float *lower,*upper,*center;
 uint32_t *maximum_bits;
 unsigned char *dirty;
 size_t rows,columns,parts,recorded;
 const void *epoch;
 int valid;
} merlin_endpoint_span_owner;
static inline merlin_endpoint_span_owner merlin_endpoint_span_begin(
 const merlin_fma_bound *environment,float *lower,float *upper,float *center,
 uint32_t *maximum_bits,unsigned char *dirty,size_t capacity,
 size_t rows,size_t columns,size_t parts,const void *epoch){
 merlin_endpoint_span_owner bad={0};
 if(!environment||!environment->valid||!lower||!upper||!center||!maximum_bits||!dirty||
    !epoch||!rows||!columns||!parts||capacity<rows||rows>SIZE_MAX/columns||
    rows*columns>SIZE_MAX/parts||rows*columns*parts>SIZE_MAX/sizeof(float))return bad;
 for(size_t r=0;r<rows;r++){maximum_bits[r]=0;dirty[r]=0;}
 return (merlin_endpoint_span_owner){lower,upper,center,maximum_bits,dirty,
   rows,columns,parts,0,epoch,1};
}
static inline int merlin_endpoint_span_record(
 merlin_endpoint_span_owner *owner,const void *epoch,size_t part,
 const float *lower,const float *upper,const double *center){
 if(!owner||!owner->valid||owner->epoch!=epoch||part!=owner->recorded||
    part>=owner->parts||!lower||!upper||!center)return 0;
 const size_t offset=part*owner->rows*owner->columns;
 for(size_t r=0;r<owner->rows;r++){
  uint32_t maximum=owner->maximum_bits[r];
  for(size_t c=0;c<owner->columns;c++){
   size_t t=r*owner->columns+c;float l=lower[t],h=upper[t],v=(float)center[t];
   uint32_t lb=merlin_interval_bits(l)&UINT32_C(0x7fffffff);
   uint32_t hb=merlin_interval_bits(h)&UINT32_C(0x7fffffff);
   if(lb>=UINT32_C(0x7f800000)||hb>=UINT32_C(0x7f800000)||
      !(l<=v&&v<=h)){owner->valid=0;return 0;}
   if(lb>maximum)maximum=lb;if(hb>maximum)maximum=hb;
   owner->lower[offset+t]=l;owner->upper[offset+t]=h;owner->center[offset+t]=v;
  }
  owner->maximum_bits[r]=maximum;
 }
 owner->recorded++;return 1;
}
static inline int merlin_endpoint_span_invalidate_row(
 merlin_endpoint_span_owner *owner,const void *epoch,size_t row){
 if(!owner||!owner->valid||owner->epoch!=epoch||row>=owner->rows)return 0;
 owner->dirty[row]=1;return 1;
}
/* Return 0 for the original checked row path, without writing outputs. */
static inline int merlin_endpoint_prepared_row(
 const merlin_endpoint_span_owner *owner,const void *epoch,size_t row,
 const float *factors,size_t factor_count,size_t parts_per_factor,
 float denominator_low,float denominator_high,float *lower,float *upper,float *candidate){
 if(!owner||!owner->valid||owner->epoch!=epoch||owner->recorded!=owner->parts||
    row>=owner->rows||owner->dirty[row]||!factors||!factor_count||!parts_per_factor||
    factor_count>SIZE_MAX/parts_per_factor||factor_count*parts_per_factor!=owner->parts||
    !lower||!upper||!candidate||fegetround()!=FE_TONEAREST)return 0;
 merlin_f32_interval den=merlin_interval(denominator_low,denominator_high);
 if(denominator_low==0&&denominator_high==0)den=merlin_interval_point(1);
 merlin_f32_interval reciprocal=merlin_interval_recip_positive(den);
 if(!reciprocal.valid)return 0;
 float de=(float)(((double)denominator_low+denominator_high)*.5);if(de==0)de=1;
 float point_reciprocal=1.0f/de;
 if(!(reciprocal.lo<=point_reciprocal&&point_reciprocal<=reciprocal.hi))return 0;
 float maximum=merlin_interval_float(owner->maximum_bits[row]),bound=0;
 for(size_t t=0;t<factor_count;t++){
  if(!MERLIN_SOURCE_ISFINITE(factors[t])||!(factors[t]>=0))return 0;
  bound=bound*factors[t];if(!MERLIN_SOURCE_ISFINITE(bound))return 0;
  for(size_t z=0;z<parts_per_factor;z++){
   bound=bound+maximum;if(!MERLIN_SOURCE_ISFINITE(bound))return 0;
  }
 }
 bound=bound*reciprocal.hi;
 if(!MERLIN_SOURCE_ISFINITE(bound)||bound>merlin_interval_float(UINT32_C(0x7f7f0000)))return 0;
 for(size_t c=0;c<owner->columns;c++){
  float lo=0,hi=0,point=0;
  for(size_t t=0;t<factor_count;t++){
   lo=lo*factors[t];hi=hi*factors[t];point=point*factors[t];
   for(size_t z=0;z<parts_per_factor;z++){
    size_t ix=((t*parts_per_factor+z)*owner->rows+row)*owner->columns+c;
    float l=owner->lower[ix],h=owner->upper[ix];
    lo=lo+l;hi=hi+h;point=point+(l==h?l:owner->center[ix]);
   }
  }
  lo=lo*(lo<0?reciprocal.hi:reciprocal.lo);
  hi=hi*(hi<0?reciprocal.lo:reciprocal.hi);point=point*point_reciprocal;
  float l=merlin_interval_bf16(lo),h=merlin_interval_bf16(hi),v=merlin_interval_bf16(point);
  lower[c]=l;upper[c]=h;candidate[c]=MERLIN_SOURCE_F32_MAX(l,MERLIN_SOURCE_F32_MIN(h,v));
 }
 return 1;
}
#endif
"""
