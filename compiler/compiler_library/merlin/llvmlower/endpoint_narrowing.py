"""Explicit interval narrowing and dependency-local private observer reuse."""

from dataclasses import dataclass

from .prepared_endpoint_dag import PreparedEndpointContract
from .prepared_endpoint_dag import c_header as prepared_header


@dataclass(frozen=True)
class EndpointNarrowingContract:
    """Additional caller proofs; a subset check grants no numerical policy."""

    stable_rne_gradual_underflow: bool
    nontrapping_effects_unobserved: bool
    pure_scalar_math_return_observations: bool
    private_source_epoch: bool
    immutable_owner_descriptors_and_magnitude_fact: bool
    disjoint_owned_storage: bool
    complete_parts_recorded: bool
    exclusive_mutation_api: bool
    unchanged_source_and_numeric_policy: bool
    source_membership_checked_before_refinement: bool
    original_operation_order_retained: bool
    closed_row_column_dependencies: bool
    no_publication_before_complete_success: bool
    checked_fallback_retained: bool

    def validate(self) -> None:
        if any(type(value) is not bool or not value for value in vars(self).values()):
            raise ValueError("complete endpoint narrowing source/ownership/effect contract required")


def c_header(prepared: PreparedEndpointContract, narrowing: EndpointNarrowingContract) -> str:
    """Compose an optional private observer with checked narrowing operations.

    The original producer and evaluator are unchanged. This additional owner
    preserves their magnitude fact only through finite interval subsets in the
    same private epoch. Source membership remains the caller's original proof.
    Replacement, widening, unknown writes or context changes invalidate the row.
    Partial narrowing refreshes one output column; denominator narrowing
    refreshes every column in that row. Other rows and columns keep their exact
    previously evaluated words. No source matching or automatic selection is
    performed, and no public result may be published on refusal.
    """
    if not isinstance(narrowing, EndpointNarrowingContract):
        raise ValueError("typed endpoint narrowing contract required")
    narrowing.validate()
    return prepared_header(prepared) + _NARROWING_HEADER


_NARROWING_HEADER = r"""
#ifndef MERLIN_ENDPOINT_NARROWING_H
#define MERLIN_ENDPOINT_NARROWING_H
/* Caller proves the source epoch, immutable scalar policy/factors, original
 * source membership of refinements, and exclusive mutation through this API.
 * Owner descriptors themselves are disjoint from every represented span.
 * Checked subset tests preserve a range fact, not source truth or approximation
 * permission. The supplied interval policy is unchanged. */
typedef struct {
 merlin_endpoint_span_owner *span;
 float *source_lower,*source_upper,*source_center;
 uint32_t *source_maximum_bits;unsigned char *source_dirty;
 size_t rows,columns,parts;
 const float *factors;
 float *denominator_low,*denominator_high;
 uint32_t *factor_words,*denominator_low_words,*denominator_high_words;
 uint64_t *generation,*observed_generation;
 size_t *pending_columns;
 unsigned char *row_state;
 float *cached_lower,*cached_upper,*cached_candidate;
 size_t factor_count,parts_per_factor;
 const void *epoch;
 int valid;
} merlin_endpoint_narrowing_owner;
typedef struct { const void *address;size_t bytes,alignment; } merlin_endpoint_owned_span;

static inline int merlin_endpoint_owned_spans_disjoint(
 const merlin_endpoint_owned_span *spans,size_t count){
 for(size_t i=0;i<count;i++){
  uintptr_t low=(uintptr_t)spans[i].address;
  if(!low||!spans[i].bytes||!spans[i].alignment||low%spans[i].alignment||
     spans[i].bytes>UINTPTR_MAX-low)return 0;
  for(size_t j=0;j<i;j++){
   uintptr_t previous=(uintptr_t)spans[j].address;
   if(!(low+spans[i].bytes<=previous||previous+spans[j].bytes<=low))return 0;
  }
 }
 return 1;
}
static inline int merlin_endpoint_narrowing_environment(void){
 return fegetround()==FE_TONEAREST;
}
static inline int merlin_endpoint_narrowing_subset(
 float old_low,float old_high,float low,float high){
 uint32_t ol=merlin_interval_bits(old_low),oh=merlin_interval_bits(old_high);
 uint32_t l=merlin_interval_bits(low),h=merlin_interval_bits(high);
 if((ol&UINT32_C(0x7fffffff))>=UINT32_C(0x7f800000)||
    (oh&UINT32_C(0x7fffffff))>=UINT32_C(0x7f800000)||
    (l&UINT32_C(0x7fffffff))>=UINT32_C(0x7f800000)||
    (h&UINT32_C(0x7fffffff))>=UINT32_C(0x7f800000)||
    !(old_low<=low&&low<=high&&high<=old_high))return 0;
 /* A numeric zero equality does not establish signed-zero word membership.
  * Admit either zero only in a strict sign-crossing interval, or when that
  * exact zero word is already an old endpoint. Conservative refusal is safe. */
 if(!(old_low<0&&old_high>0)){
  if((l&UINT32_C(0x7fffffff))==0&&l!=ol&&l!=oh)return 0;
  if((h&UINT32_C(0x7fffffff))==0&&h!=ol&&h!=oh)return 0;
 }
 return 1;
}
static inline int merlin_endpoint_narrowing_invalidate_row(
 merlin_endpoint_narrowing_owner *owner,const void *epoch,size_t row){
 if(!owner||!owner->valid||!owner->span||owner->epoch!=epoch||
    owner->span->epoch!=epoch||row>=owner->rows)return 0;
 owner->source_dirty[row]=1;owner->row_state[row]=2;return 1;
}
static inline int merlin_endpoint_narrowing_context(
 merlin_endpoint_narrowing_owner *owner,const void *epoch,size_t row){
 if(!owner||!owner->valid||!owner->span||owner->epoch!=epoch||row>=owner->rows)return 0;
 if(!owner->span->valid||owner->span->epoch!=epoch||
    owner->span->recorded!=owner->span->parts)goto global_invalid;
 if(owner->span->lower!=owner->source_lower||owner->span->upper!=owner->source_upper||
    owner->span->center!=owner->source_center||
    owner->span->maximum_bits!=owner->source_maximum_bits||
    owner->span->dirty!=owner->source_dirty||owner->span->rows!=owner->rows||
    owner->span->columns!=owner->columns||owner->span->parts!=owner->parts)goto global_invalid;
 if(owner->row_state[row]>=2)return 0;
 if(owner->source_dirty[row]!=(owner->row_state[row]&1))goto invalid;
 if(!merlin_endpoint_narrowing_environment())goto global_invalid;
 for(size_t t=0;t<owner->factor_count;t++){
  size_t index=row*owner->factor_count+t;
  if(merlin_interval_bits(owner->factors[index])!=owner->factor_words[index])goto invalid;
 }
 if(merlin_interval_bits(owner->denominator_low[row])!=owner->denominator_low_words[row]||
    merlin_interval_bits(owner->denominator_high[row])!=owner->denominator_high_words[row])
  goto invalid;
 return 1;
invalid:
 merlin_endpoint_narrowing_invalidate_row(owner,epoch,row);return 0;
global_invalid:
 owner->source_dirty[row]=1;owner->row_state[row]=2;owner->valid=0;return 0;
}
static inline merlin_endpoint_narrowing_owner merlin_endpoint_narrowing_begin(
 merlin_endpoint_span_owner *span,const void *epoch,const float *factors,
 size_t factor_count,size_t parts_per_factor,float *denominator_low,
 float *denominator_high,uint32_t *factor_words,size_t factor_capacity,
 uint32_t *denominator_low_words,uint32_t *denominator_high_words,size_t row_capacity,
 uint64_t *generation,uint64_t *observed_generation,size_t *pending_columns,
 unsigned char *row_state,float *cached_lower,
 float *cached_upper,float *cached_candidate,size_t cell_capacity){
 merlin_endpoint_narrowing_owner bad={0};
 if(!span||!span->valid||span->epoch!=epoch||!epoch||
    span->recorded!=span->parts||!span->rows||!span->columns||!span->parts||
    span->rows>SIZE_MAX/span->columns||!factor_count||!parts_per_factor||
    factor_count>SIZE_MAX/parts_per_factor||factor_count*parts_per_factor!=span->parts||
    !merlin_endpoint_narrowing_environment()||span->rows>SIZE_MAX/factor_count)return bad;
 size_t cells=span->rows*span->columns,factor_cells=span->rows*factor_count;
 if(cell_capacity<cells||row_capacity<span->rows||factor_capacity<factor_cells||
    cells>SIZE_MAX/sizeof(uint64_t)||cells>SIZE_MAX/span->parts||
    cells*span->parts>SIZE_MAX/sizeof(float)||
    span->rows>SIZE_MAX/sizeof(size_t)||
    factor_cells>SIZE_MAX/sizeof(uint32_t))return bad;
 merlin_endpoint_owned_span spans[]={
  {span->lower,cells*span->parts*sizeof(float),_Alignof(float)},
  {span->upper,cells*span->parts*sizeof(float),_Alignof(float)},
  {span->center,cells*span->parts*sizeof(float),_Alignof(float)},
  {span->maximum_bits,span->rows*sizeof(uint32_t),_Alignof(uint32_t)},
  {span->dirty,span->rows,_Alignof(unsigned char)},
  {factors,factor_cells*sizeof(float),_Alignof(float)},
  {denominator_low,span->rows*sizeof(float),_Alignof(float)},
  {denominator_high,span->rows*sizeof(float),_Alignof(float)},
  {factor_words,factor_cells*sizeof(uint32_t),_Alignof(uint32_t)},
  {denominator_low_words,span->rows*sizeof(uint32_t),_Alignof(uint32_t)},
  {denominator_high_words,span->rows*sizeof(uint32_t),_Alignof(uint32_t)},
  {generation,cells*sizeof(uint64_t),_Alignof(uint64_t)},
  {observed_generation,cells*sizeof(uint64_t),_Alignof(uint64_t)},
  {pending_columns,span->rows*sizeof(size_t),_Alignof(size_t)},
  {row_state,span->rows,_Alignof(unsigned char)},
  {cached_lower,cells*sizeof(float),_Alignof(float)},
  {cached_upper,cells*sizeof(float),_Alignof(float)},
  {cached_candidate,cells*sizeof(float),_Alignof(float)}};
 if(!merlin_endpoint_owned_spans_disjoint(spans,sizeof(spans)/sizeof(spans[0])))return bad;
 for(size_t row=0;row<span->rows;row++){
  if(span->dirty[row]||!MERLIN_SOURCE_ISFINITE(denominator_low[row])||
     !MERLIN_SOURCE_ISFINITE(denominator_high[row])||
     !(0<=denominator_low[row]&&denominator_low[row]<=denominator_high[row]))return bad;
  for(size_t t=0;t<factor_count;t++){
   float value=factors[row*factor_count+t];
   if(!MERLIN_SOURCE_ISFINITE(value)||!(value>=0))return bad;
  }
 }
 for(size_t i=0;i<factor_cells;i++)factor_words[i]=merlin_interval_bits(factors[i]);
 for(size_t row=0;row<span->rows;row++){
  denominator_low_words[row]=merlin_interval_bits(denominator_low[row]);
  denominator_high_words[row]=merlin_interval_bits(denominator_high[row]);
  pending_columns[row]=span->columns;
  row_state[row]=0;
 }
 for(size_t i=0;i<cells;i++){generation[i]=1;observed_generation[i]=0;}
 return (merlin_endpoint_narrowing_owner){span,span->lower,span->upper,span->center,
  span->maximum_bits,span->dirty,span->rows,span->columns,span->parts,
  factors,denominator_low,denominator_high,
  factor_words,denominator_low_words,denominator_high_words,generation,observed_generation,
  pending_columns,row_state,
  cached_lower,cached_upper,cached_candidate,factor_count,parts_per_factor,epoch,1};
}
static inline int merlin_endpoint_narrowing_partial(
 merlin_endpoint_narrowing_owner *owner,const void *epoch,size_t part,size_t row,
 size_t column,float low,float high){
 if(!merlin_endpoint_narrowing_context(owner,epoch,row))return 0;
 merlin_endpoint_span_owner *span=owner->span;
 if(part>=span->parts||column>=span->columns)goto invalid;
 size_t cell=row*span->columns+column,index=part*span->rows*span->columns+cell;
 if(!merlin_endpoint_narrowing_subset(span->lower[index],span->upper[index],low,high)||
    owner->generation[cell]==UINT64_MAX)goto invalid;
 /* A point ignores the old center in BOTH original evaluators. A nonpoint
  * interval must keep its unchanged center inside, and inside the old cap. */
 float center=span->center[index];
 uint32_t magnitude=merlin_interval_bits(center)&UINT32_C(0x7fffffff);
 if(magnitude>=UINT32_C(0x7f800000)||magnitude>span->maximum_bits[row]||
    (low!=high&&!(low<=center&&center<=high)))goto invalid;
 if(merlin_interval_bits(low)==merlin_interval_bits(span->lower[index])&&
    merlin_interval_bits(high)==merlin_interval_bits(span->upper[index]))return 1;
 /* The original producer contract still invalidates before every mutation.
  * Only this separate owner preserves its own subset-derived range fact. */
 span->dirty[row]=1;owner->row_state[row]=1;
 span->lower[index]=low;span->upper[index]=high;
 if(owner->observed_generation[cell]==owner->generation[cell])owner->pending_columns[row]++;
 owner->generation[cell]++;return 1;
invalid:
 merlin_endpoint_narrowing_invalidate_row(owner,epoch,row);return 0;
}
static inline int merlin_endpoint_narrowing_denominator(
 merlin_endpoint_narrowing_owner *owner,const void *epoch,size_t row,float low,float high){
 if(!merlin_endpoint_narrowing_context(owner,epoch,row))return 0;
 merlin_endpoint_span_owner *span=owner->span;
 if(!merlin_endpoint_narrowing_subset(owner->denominator_low[row],
       owner->denominator_high[row],low,high)||!(low>=0))goto invalid;
 if(merlin_interval_bits(low)==owner->denominator_low_words[row]&&
    merlin_interval_bits(high)==owner->denominator_high_words[row])return 1;
 for(size_t c=0;c<span->columns;c++)
  if(owner->generation[row*span->columns+c]==UINT64_MAX)goto invalid;
 span->dirty[row]=1;owner->row_state[row]=1;
 owner->denominator_low[row]=low;owner->denominator_high[row]=high;
 owner->denominator_low_words[row]=merlin_interval_bits(low);
 owner->denominator_high_words[row]=merlin_interval_bits(high);
 for(size_t c=0;c<span->columns;c++)owner->generation[row*span->columns+c]++;
 owner->pending_columns[row]=span->columns;
 return 1;
invalid:
 merlin_endpoint_narrowing_invalidate_row(owner,epoch,row);return 0;
}
/* Only internal row/column dependencies are cached. A later cross-row/head
 * quantizer remains unchanged and must reevaluate its complete observations.
 * `evaluated` counts recomputed columns, not instructions, cycles or gains. */
static inline int merlin_endpoint_narrowing_refresh_row(
 merlin_endpoint_narrowing_owner *owner,const void *epoch,size_t row,size_t *evaluated){
 if(!evaluated||!merlin_endpoint_narrowing_context(owner,epoch,row))return 0;
 if(!owner->pending_columns[row]){*evaluated=0;return 1;}
 merlin_endpoint_span_owner *span=owner->span;
 const float *factors=owner->factors+row*owner->factor_count;
 float denominator_low=owner->denominator_low[row],denominator_high=owner->denominator_high[row];
 merlin_f32_interval den=merlin_interval(denominator_low,denominator_high);
 if(denominator_low==0&&denominator_high==0)den=merlin_interval_point(1);
 merlin_f32_interval reciprocal=merlin_interval_recip_positive(den);
 if(!reciprocal.valid)goto invalid;
 float de=(float)(((double)denominator_low+denominator_high)*.5);if(de==0)de=1;
 float point_reciprocal=1.0f/de;
 if(!(reciprocal.lo<=point_reciprocal&&point_reciprocal<=reciprocal.hi))goto invalid;
 float maximum=merlin_interval_float(span->maximum_bits[row]),bound=0;
 for(size_t t=0;t<owner->factor_count;t++){
  bound=bound*factors[t];if(!MERLIN_SOURCE_ISFINITE(bound))goto invalid;
  for(size_t z=0;z<owner->parts_per_factor;z++){
   bound=bound+maximum;if(!MERLIN_SOURCE_ISFINITE(bound))goto invalid;
  }
 }
 bound=bound*reciprocal.hi;
 if(!MERLIN_SOURCE_ISFINITE(bound)||bound>merlin_interval_float(UINT32_C(0x7f7f0000)))
  goto invalid;
 size_t count=0;
 for(size_t c=0;c<span->columns;c++){
  size_t cell=row*span->columns+c;
  if(owner->observed_generation[cell]==owner->generation[cell])continue;
  float lo=0,hi=0,point=0;
  for(size_t t=0;t<owner->factor_count;t++){
   lo=lo*factors[t];hi=hi*factors[t];point=point*factors[t];
   for(size_t z=0;z<owner->parts_per_factor;z++){
    size_t index=((t*owner->parts_per_factor+z)*span->rows+row)*span->columns+c;
    float l=span->lower[index],h=span->upper[index];
    lo=lo+l;hi=hi+h;point=point+(l==h?l:span->center[index]);
   }
  }
  lo=lo*(lo<0?reciprocal.hi:reciprocal.lo);
  hi=hi*(hi<0?reciprocal.lo:reciprocal.hi);point=point*point_reciprocal;
  float l=merlin_interval_bf16(lo),h=merlin_interval_bf16(hi),v=merlin_interval_bf16(point);
  /* C min/max zero ties can choose a different zero sign in another compiled
   * context. This capability supplies no stronger min/max effect theorem.
   * Retain the original source on every zero boundary/candidate; never publish
   * the private cache after refusal. Earlier private columns may be written. */
  if(l==0.0f||h==0.0f||v==0.0f)goto invalid;
  owner->cached_lower[cell]=l;owner->cached_upper[cell]=h;
  owner->cached_candidate[cell]=MERLIN_SOURCE_F32_MAX(l,MERLIN_SOURCE_F32_MIN(h,v));
  owner->observed_generation[cell]=owner->generation[cell];count++;
 }
 owner->pending_columns[row]=0;*evaluated=count;return 1;
invalid:
 merlin_endpoint_narrowing_invalidate_row(owner,epoch,row);return 0;
}
#endif
"""
