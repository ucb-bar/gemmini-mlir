#include "compiler.h"
namespace backend {
std::string convPoolHardwareRuntime();
std::string poolRuntime() { return poolHardwareRuntime()+convPoolHardwareRuntime()+R"C(
/* The caller proves neutral pool padding and at least one valid tap per
   window. Input taps occupy acc0..15. Acc16 retains the maximum;
   acc32 retains the difference. The final output is transient byte storage.
   No full pre-pool tensor, helper global, or partial buffer is required. */
DG_INLINE void dg_pool_fold_acc0(int8_t *dst,uint64_t n,unsigned first,uint32_t scale,unsigned relu,unsigned inputrow) {
 dg_store_config(16,scale,relu);
 DG_CMD(3,dst,dg_desc(0x80000000U | inputrow,1,n));
 dg_fence();
 dg_load_config(0,0,1,0x3f800000U);
 if(first) {
  DG_CMD(2,dst,dg_desc(0x80000010U,1,n));
 } else {
  DG_CMD(2,dst,dg_desc(0x80000020U,1,n));
  dg_store_config(16,0x3f800000U,0);
  DG_CMD(3,dst,dg_desc(0x80000010U,1,n));
  dg_fence();
  /* For every i8 x, RNE(x*(-.5-eps))+RNE(x*(-.5+eps))=-x.
     The individual products are in [-64.01,64.01], so no clipping occurs. */
  dg_load_config(0,1,1,0xbf000001U);
  dg_load_config(0,2,1,0xbefffffeU);
  DG_CMD(1,dst,dg_desc(0xc0000020U,1,n));
  DG_CMD(14,dst,dg_desc(0xc0000020U,1,n));
  dg_fence();
  /* With ReLU and the caller's positive scale, x,y are in [0,127].
     Then clamp(x-y,0,127)=max(x-y,0), so one correction suffices.
     Signed i8 taps allow x-y up to 255 and retain three corrections. */
  unsigned rounds=relu?1:3;
  for(unsigned r=0;r<rounds;r++) {
   dg_store_config(16,0x3f800000U,1);
   DG_CMD(3,dst,dg_desc(0x80000020U,1,n));
   dg_fence();
   DG_CMD(2,dst,dg_desc(0xc0000010U,1,n));
   if(r+1<rounds) {
    DG_CMD(1,dst,dg_desc(0xc0000020U,1,n));
    DG_CMD(14,dst,dg_desc(0xc0000020U,1,n));
   }
   dg_fence();
  }
 }
 dg_store_config(16,0x3f800000U,0);
 DG_CMD(3,dst,dg_desc(0x80000010U,1,n));
 dg_fence();
}
DG_INLINE void dg_readout_pool_ex(const int32_t *A,int8_t *C,
 uint64_t batch,uint64_t H,uint64_t width,uint64_t N,uint64_t OH,uint64_t OW,
 uint64_t PH,uint64_t PW,uint64_t SH,uint64_t SW,int64_t PT,int64_t PL,
 int8_t pad,uint32_t scale,unsigned relu) {
 if(dg_readout_pool_hw(A,C,batch,H,width,N,OH,OW,PH,PW,SH,SW,PT,PL,pad,scale,relu))return;
 (void)pad;
 dg_fence();
 uint64_t totalM=batch*H*width;
 uint64_t cachedBase=~(uint64_t)0,cachedJ=~(uint64_t)0,cachedN=0;
 for(uint64_t b=0;b<batch;b++)for(uint64_t y=0;y<OH;y++)for(uint64_t x=0;x<OW;x++)for(uint64_t j=0;j<N;j+=16) {
  uint64_t n=N-j;if(n>16)n=16;
  int8_t *dst=C+((b*OH+y)*OW+x)*N+j;
  unsigned first=1;
  for(uint64_t py=0;py<PH;py++)for(uint64_t px=0;px<PW;px++) {
   int64_t iy=(int64_t)(y*SH+py)-PT,ix=(int64_t)(x*SW+px)-PL;
   if(iy<0 || ix<0 || (uint64_t)iy>=H || (uint64_t)ix>=width)continue;
   uint64_t p=(b*H+(uint64_t)iy)*width+(uint64_t)ix;
   uint64_t tileBase=(p/16)*16;
   if(tileBase!=cachedBase || j!=cachedJ || n!=cachedN) {
    uint64_t rows=totalM-tileBase;if(rows>16)rows=16;
    /* Full-width rows remain intact in acc0..15 while folding touches
       only acc16/acc32. The stride is in bytes, as in dg_readout. */
    dg_fence();
    dg_load_config(N*4,0,0,0x3f800000U);
    DG_CMD(2,A+tileBase*N+j,dg_desc(0x80000000U,rows,n));
    dg_fence();
    cachedBase=tileBase;cachedJ=j;cachedN=n;
   }
   dg_pool_fold_acc0(dst,n,first,scale,relu,(unsigned)(p-tileBase));
   first=0;
  }
 }
 dg_fence();
}
DG_INLINE void dg_matmul_pool_ex(const int8_t *A,const int8_t *W,const int32_t *D,int8_t *C,
 uint64_t M,uint64_t K,uint64_t N,uint64_t batch,uint64_t H,uint64_t width,
 uint64_t OH,uint64_t OW,uint64_t PH,uint64_t PW,uint64_t SH,uint64_t SW,
 int64_t PT,int64_t PL,int8_t pad,uint32_t scale,unsigned relu) {
 (void)M;(void)pad;
 dg_fence();
 DG_CMD(0,(1ULL<<16)|4,1ULL<<48);
 for(uint64_t b=0;b<batch;b++)for(uint64_t y=0;y<OH;y++)for(uint64_t x=0;x<OW;x++)for(uint64_t j=0;j<N;j+=16) {
  uint64_t n=N-j;if(n>16)n=16;
  int8_t *dst=C+((b*OH+y)*OW+x)*N+j;
  unsigned first=1;
  for(uint64_t py=0;py<PH;py++)for(uint64_t px=0;px<PW;px++) {
   int64_t iy=(int64_t)(y*SH+py)-PT,ix=(int64_t)(x*SW+px)-PL;
   if(iy<0 || ix<0 || (uint64_t)iy>=H || (uint64_t)ix>=width)continue;
   uint64_t p=(b*H+(uint64_t)iy)*width+(uint64_t)ix;
   dg_load_config(K,0,0,0x3f800000U);
   dg_load_config(N,1,0,0x3f800000U);
   dg_load_config(0,2,0,0x3f800000U);
   if(D || K==0) { DG_CMD(14,D?D+j:(const int32_t*)0,dg_desc(0x80000000U,1,n)); }
   for(uint64_t k=0;k<K;k+=16) {
    uint64_t z=K-k;if(z>16)z=16;
    DG_CMD(2,A+p*K+k,dg_desc(0,1,z));
    DG_CMD(1,W+k*N+j,dg_desc(16,z,n));
    DG_CMD(6,dg_desc(16,z,n),dg_desc((D || k)?0xc0000000U:0x80000000U,1,n));
    DG_CMD(4,dg_desc(0,1,z),dg_desc(0xffffffffU,16,16));
    dg_fence();
   }
   dg_pool_fold_acc0(dst,n,first,scale,relu,0);
   first=0;
  }
 }
 dg_fence();
}
DG_INLINE void dg_conv2d_pool_ex(const int8_t *A,const int8_t *W,const int32_t *D,int8_t *C,
 uint64_t batch,uint64_t H,uint64_t width,uint64_t CI,uint64_t CO,
 uint64_t convOH,uint64_t convOW,uint64_t KH,uint64_t KW,uint64_t convSH,uint64_t convSW,
 uint64_t DH,uint64_t DW,int64_t convPT,int64_t convPL,
 uint64_t OH,uint64_t OW,uint64_t PH,uint64_t PW,uint64_t SH,uint64_t SW,
 int64_t PT,int64_t PL,int8_t pad,uint32_t scale,unsigned relu) {
 if(dg_conv2d_pool_hw(A,W,D,C,batch,H,width,CI,CO,convOH,convOW,KH,KW,convSH,convSW,DH,DW,convPT,convPL,OH,OW,PH,PW,SH,SW,PT,PL,pad,scale,relu))return;
 (void)pad;
 dg_fence();
 DG_CMD(0,(1ULL<<16)|4,1ULL<<48);
 for(uint64_t b=0;b<batch;b++)for(uint64_t y=0;y<OH;y++)for(uint64_t x=0;x<OW;x++)for(uint64_t j=0;j<CO;j+=16) {
  uint64_t n=CO-j;if(n>16)n=16;
  int8_t *dst=C+((b*OH+y)*OW+x)*CO+j;
  unsigned first=1;
  for(uint64_t py=0;py<PH;py++)for(uint64_t px=0;px<PW;px++) {
   int64_t oy=(int64_t)(y*SH+py)-PT,ox=(int64_t)(x*SW+px)-PL;
   if(oy<0 || ox<0 || (uint64_t)oy>=convOH || (uint64_t)ox>=convOW)continue;
   dg_load_config(0,0,0,0x3f800000U);
   dg_load_config(CO,1,0,0x3f800000U);
   dg_load_config(0,2,0,0x3f800000U);
   if(D || CI==0 || KH==0 || KW==0) { DG_CMD(14,D?D+j:(const int32_t*)0,dg_desc(0x80000000U,1,n)); }
   unsigned productFirst=1;
   for(uint64_t ky=0;ky<KH;ky++)for(uint64_t kx=0;kx<KW;kx++)for(uint64_t ci=0;ci<CI;ci+=16) {
    uint64_t z=CI-ci;if(z>16)z=16;
    int64_t iy=oy*(int64_t)convSH+(int64_t)(ky*DH)-convPT;
    int64_t ix=ox*(int64_t)convSW+(int64_t)(kx*DW)-convPL;
    const int8_t *src=(const int8_t*)0;
    if(iy>=0 && ix>=0 && (uint64_t)iy<H && (uint64_t)ix<width)
     src=A+((b*H+(uint64_t)iy)*width+(uint64_t)ix)*CI+ci;
    DG_CMD(2,src,dg_desc(0,1,z));
    DG_CMD(1,W+((ky*KW+kx)*CI+ci)*CO+j,dg_desc(16,z,n));
    DG_CMD(6,dg_desc(16,z,n),dg_desc((D || !productFirst)?0xc0000000U:0x80000000U,1,n));
    DG_CMD(4,dg_desc(0,1,z),dg_desc(0xffffffffU,16,16));
    productFirst=0;
    dg_fence();
   }
   dg_pool_fold_acc0(dst,n,first,scale,relu,0);
   first=0;
  }
 }
 dg_fence();
}
)C"; }
}
