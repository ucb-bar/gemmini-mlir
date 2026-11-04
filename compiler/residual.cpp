#include "compiler.h"
namespace backend {
// Concatenated after deviceRuntime(): shares its instruction/configuration helpers.
std::string residualRuntime() { return R"C(
/* Positive finite scales up to 16 are selected by the compiler. Narrow scaled
   loads saturate to i8, so split s=floor(s)+fraction(s). Each integer part is
   accumulated from unscaled i8 loads; the fractional part cannot clip. */
DG_INLINE void dg_residual(const int8_t *A,const int8_t *B,void *C,
 uint64_t M,uint64_t N,uint32_t lhsScaleBits,uint32_t rhsScaleBits,
 unsigned out32,unsigned relu) {
 union { uint32_t u; float f; } ls,rs,lf,rf;
 ls.u=lhsScaleBits; rs.u=rhsScaleBits;
 unsigned li=(unsigned)ls.f,ri=(unsigned)rs.f;
 lf.f=ls.f-(float)li; rf.f=rs.f-(float)ri;
 dg_fence();
 dg_load_config(N,0,1,lf.u);
 dg_load_config(N,1,1,rf.u);
 dg_load_config(N,2,1,0x3f800000U);
 dg_store_config(N*(out32?4:1),0x3f800000U,out32?0:relu);
 for(uint64_t i=0;i<M;i+=16) {
  uint64_t m=M-i;if(m>16)m=16;
  for(uint64_t j=0;j<N;j+=16) {
   uint64_t n=N-j;if(n>16)n=16;
   const int8_t *a=A+i*N+j,*b=B+i*N+j;
   uint64_t overwrite=dg_desc(0x80000000U,m,n);
   uint64_t add=dg_desc(0xc0000000U,m,n);
   /* Start with an integer load when possible, avoiding a zero fractional load. */
   if(li) {
    DG_CMD(14,a,overwrite);
    for(unsigned r=1;r<li;r++) { DG_CMD(14,a,add); }
    if(lf.u) { DG_CMD(2,a,add); }
   } else { DG_CMD(2,a,overwrite); }
   for(unsigned r=0;r<ri;r++) { DG_CMD(14,b,add); }
   if(rf.u) { DG_CMD(1,b,add); }
   DG_CMD(3,(int8_t*)C+(i*N+j)*(out32?4:1),
    dg_desc(out32?0xa0000000U:0x80000000U,m,n));
  }
 }
 dg_fence();
 /* Full accumulator readout bypasses the scaled activation path. */
 if(out32 && relu) {
  int32_t *p=(int32_t*)C;
  for(uint64_t k=0;k<M*N;k++) if(p[k]<0)p[k]=0;
 }
}
)C"; }
}
