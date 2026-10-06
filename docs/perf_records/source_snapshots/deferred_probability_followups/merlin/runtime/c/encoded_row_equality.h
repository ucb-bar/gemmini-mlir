#ifndef MERLIN_ENCODED_ROW_EQUALITY_H
#define MERLIN_ENCODED_ROW_EQUALITY_H
#include "ordered_fma_bounds.h"
/* Produced during the mandatory float->double reconstruction write. Source,
 * reconstructed output, optional endpoint spans and flags are distinct private
 * owners. The compiler must prove all remain unchanged through synchronous
 * bounds consumption. No proof survives the next encoding or owner mutation.
 * Equality is numeric (signed zeros have identical absolute/error norms).
 * This object proves representation equality, never source-FMA exactness. */
typedef struct {
 const float *source,*lower,*upper;
 const double *reconstructed;
 const unsigned char *exact;
 size_t rows,length;
 int valid;
} merlin_encoded_row_equality;
static inline merlin_encoded_row_equality merlin_encoded_rows_widen(
 const merlin_fma_bound *environment,const float *source,const float *encoded,
 const float *lower,const float *upper,double *reconstructed,
 unsigned char *exact,size_t rows,size_t length) {
 merlin_encoded_row_equality p={0};
 if(!environment||!environment->valid||!source||!encoded||!reconstructed||!exact||
    !rows||!length||rows>SIZE_MAX/length||(lower==0)!=(upper==0))return p;
 for(size_t row=0;row<rows;row++){
  int same=1;
  for(size_t z=0;z<length;z++){
   size_t index=row*length+z;float original=source[index],value=encoded[index];
   reconstructed[index]=(double)value;
   int equal=MERLIN_SOURCE_ISFINITE(original)&&MERLIN_SOURCE_ISFINITE(value)&&original==value;
   if(lower)equal=equal&&lower[index]==original&&upper[index]==original;
   same= same && equal;
  }
  exact[row]=(unsigned char)same;
 }
 return (merlin_encoded_row_equality){source,lower,upper,reconstructed,exact,rows,length,1};
}
static inline int merlin_encoded_row_matches(
 const merlin_encoded_row_equality *p,size_t row,const float *source,
 const double *reconstructed,const float *lower,const float *upper,size_t length) {
 if(!p||!p->valid||row>=p->rows||length!=p->length||!p->exact[row])return 0;
 size_t offset=row*length;
 return source==p->source+offset&&reconstructed==p->reconstructed+offset&&
  lower==(p->lower?p->lower+offset:0)&&upper==(p->upper?p->upper+offset:0);
}

/* Separate private facts: bit 1 proves finite center reconstruction equality;
 * bit 0 additionally proves both source endpoints equal that center. Neither
 * fact removes source-FMA rounding or uncertain input contributions. Owners,
 * dimensions and the synchronous immutable epoch have the same requirements
 * as the original producer; a distinct type prevents interpreting bit 1 as a
 * point-span admission. */
typedef struct {
 const float *source,*lower,*upper;
 const double *reconstructed;
 const unsigned char *facts;
 size_t rows,length;
 int valid;
} merlin_encoded_row_facts;
static inline merlin_encoded_row_facts merlin_encoded_rows_widen_facts(
 const merlin_fma_bound *environment,const float *source,const float *encoded,
 const float *lower,const float *upper,double *reconstructed,
 unsigned char *facts,size_t rows,size_t length) {
 merlin_encoded_row_facts p={0};
 if(!environment||!environment->valid||!source||!encoded||!reconstructed||!facts||
    !rows||!length||rows>SIZE_MAX/length||(lower==0)!=(upper==0))return p;
 for(size_t row=0;row<rows;row++){
  int same=1,point=1;
  for(size_t z=0;z<length;z++){
   size_t index=row*length+z;float original=source[index],value=encoded[index];
   reconstructed[index]=(double)value;
   same=same&&(MERLIN_SOURCE_ISFINITE(original)&&MERLIN_SOURCE_ISFINITE(value)&&original==value);
   if(lower)point=point&&(lower[index]==original&&upper[index]==original);
  }
  facts[row]=(unsigned char)((same?2:0)+(same&&point?1:0));
 }
 return (merlin_encoded_row_facts){source,lower,upper,reconstructed,facts,rows,length,1};
}
static inline unsigned merlin_encoded_row_bound_facts(
 const merlin_encoded_row_facts *p,size_t row,const float *source,
 const double *reconstructed,const float *lower,const float *upper,size_t length) {
 if(!p||!p->valid||row>=p->rows||length!=p->length||!p->facts)return 0;
 size_t offset=row*length;
 if(source!=p->source+offset||reconstructed!=p->reconstructed+offset||
    lower!=(p->lower?p->lower+offset:0)||upper!=(p->upper?p->upper+offset:0))return 0;
 return p->facts[row];
}
static inline int merlin_encoded_row_point_matches(
 const merlin_encoded_row_facts *p,size_t row,const float *source,
 const double *reconstructed,const float *lower,const float *upper,size_t length) {
 return (merlin_encoded_row_bound_facts(p,row,source,reconstructed,lower,upper,length)&1)!=0;
}
static inline int merlin_encoded_row_reconstruction_matches(
 const merlin_encoded_row_facts *p,size_t row,const float *source,
 const double *reconstructed,const float *lower,const float *upper,size_t length) {
 return (merlin_encoded_row_bound_facts(p,row,source,reconstructed,lower,upper,length)&2)!=0;
}
#endif
