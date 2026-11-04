#include "compiler.h"
namespace backend {
std::string deviceRuntime() { return std::string("#define DG_BLOCKING ")+(compilerSettingBool("matmul_blocking",true)?"1":"0")+"\n#define DG_COLUMN_BATCHING "+(compilerSettingBool("column_batching",true)?"1":"0")+"\n#define DG_COLUMN_TILE_MODE "+(compilerSettingString("column_readout_tile","64x8")=="128x4"?"1":compilerSettingString("column_readout_tile","64x8")=="auto"?"2":"0")+"\n"+R"C(
#ifndef DG_COLUMN_TILE_MODE
#define DG_COLUMN_TILE_MODE 0
#endif
#define DG_INLINE static __attribute__((always_inline)) inline
#define DG_CMD(F,A,B) __asm__ volatile(".insn r 0x7b, 3, " #F ", x0, %0, %1" :: "r"((uint64_t)(A)), "r"((uint64_t)(B)) : "memory")
DG_INLINE void dg_fence(void) { __asm__ volatile("fence" ::: "memory"); }
/* Entry-only translation flush, before this invocation starts DMA.
 * FIR327163,327166-327171: funct7, rs1.bit0=skip; zero selects retry.
 * FIR198108-198115: retry|skip asserts global TLB sfence (rs1=rs2=0).
 * FIR198644-198646 invalidates the last-translation cache; FIR326576-326577
 * also forwards flush to scratchpad DMA. rs2 is unused by flush decode.
 * Call once at kernel entry; this is separate from the completion fence.
 */
DG_INLINE void dg_flush(void) {
 dg_fence();
 DG_CMD(7,0,0);
 dg_fence();
}
DG_INLINE uint64_t dg_desc(uint32_t addr, uint64_t rows, uint64_t cols) {
 return (uint64_t)addr | (cols << 32) | (rows << 48);
}
DG_INLINE void dg_load_config(uint64_t stride, unsigned id, unsigned narrow, uint32_t scale) {
 DG_CMD(0, ((uint64_t)scale<<32) | (16ULL<<16) | (id<<3) | (narrow<<2) | 1, stride);
}
DG_INLINE void dg_store_config(uint64_t stride, uint32_t scale, unsigned relu) {
 DG_CMD(0, 2 | (relu<<2), ((uint64_t)scale<<32) | stride);
}
)C" + blockedRuntime() + R"C(
DG_INLINE void dg_matmul_ex(const int8_t *A, const int8_t *B, const int32_t *D, void *C,
 uint64_t M, uint64_t K, uint64_t N, unsigned out32, uint32_t scale, unsigned relu) {
 if(DG_BLOCKING && K>=16 && (M>16 || N>16 || K>=128)) {
  dg_matmul_blocked(A,B,D,C,M,K,N,out32,scale,relu);return;
 }
 dg_fence();
 DG_CMD(0, (1ULL<<16) | 4, 1ULL<<48);
 dg_load_config(K,0,0,0x3f800000U);
 dg_load_config(N,1,0,0x3f800000U);
 dg_load_config(0,2,0,0x3f800000U);
 dg_store_config(N*(out32?4:1),scale,relu);
 for(uint64_t i=0;i<M;i+=16) {
  uint64_t m=M-i;if(m>16)m=16;
  for(uint64_t j=0;j<N;j+=16) {
   uint64_t n=N-j;if(n>16)n=16;
   if(D || K==0) {
    DG_CMD(14,D?D+j:(const int32_t*)0,dg_desc(0x80000000U,m,n));
   }
   for(uint64_t k=0;k<K;k+=16) {
    uint64_t z=K-k;if(z>16)z=16;
    DG_CMD(2,A+i*K+k,dg_desc(0,m,z));
    DG_CMD(1,B+k*N+j,dg_desc(16,z,n));
    uint32_t dest=(k!=0 || D)?0xc0000000U:0x80000000U;
    DG_CMD(6,dg_desc(16,z,n),dg_desc(dest,m,n));
    DG_CMD(4,dg_desc(0,m,z),dg_desc(0xffffffffU,16,16));
    dg_fence();
   }
   DG_CMD(3,(int8_t*)C+(i*N+j)*(out32?4:1),dg_desc(out32?0xa0000000U:0x80000000U,m,n));
   dg_fence();
  }
 }
 dg_fence();
}
DG_INLINE void dg_matmul(const int8_t *A,const int8_t *B,int32_t *C,uint64_t M,uint64_t K,uint64_t N) {
 dg_matmul_ex(A,B,(const int32_t*)0,C,M,K,N,1,0x3f800000U,0);
}
DG_INLINE void dg_movement(const int8_t *A,void *C,uint64_t M,uint64_t N,unsigned out32) {
 dg_fence();
 dg_load_config(N,0,out32,0x3f800000U);
 dg_store_config(N*(out32?4:1),0x3f800000U,0);
 for(uint64_t i=0;i<M;i+=16) {
  uint64_t m=M-i;if(m>16)m=16;
  for(uint64_t j=0;j<N;j+=16) {
   uint64_t n=N-j;if(n>16)n=16;
   DG_CMD(2,A+i*N+j,dg_desc(out32?0x80000000U:0,m,n));
   DG_CMD(3,(int8_t*)C+(i*N+j)*(out32?4:1),dg_desc(out32?0xa0000000U:0,m,n));
  }
 }
 dg_fence();
}
DG_INLINE void dg_readout(const int32_t *A,int8_t *C,uint64_t M,uint64_t N,uint32_t scale,unsigned relu) {
 dg_fence();
 dg_load_config(N*4,0,0,0x3f800000U);
 dg_store_config(N,scale,relu);
 for(uint64_t i=0;i<M;i+=16) {
  uint64_t m=M-i;if(m>16)m=16;
  for(uint64_t j=0;j<N;j+=16) {
   uint64_t n=N-j;if(n>16)n=16;
   DG_CMD(2,A+i*N+j,dg_desc(0x80000000U,m,n));
   DG_CMD(3,C+i*N+j,dg_desc(0x80000000U,m,n));
   dg_fence();
  }
 }
 dg_fence();
}
/* Retained serial implementation, validated before the batched variant. */
DG_INLINE void dg_readout_cols_serial(const int32_t *A,int8_t *C,uint64_t M,uint64_t N,const uint32_t *scales,unsigned relu) {
 dg_fence();
 dg_load_config(N*4,0,0,0x3f800000U);
 for(uint64_t j=0;j<N;j++) {
  dg_store_config(N,scales[j],relu);
  for(uint64_t i=0;i<M;i+=16) {
   uint64_t m=M-i;if(m>16)m=16;
   DG_CMD(2,A+i*N+j,dg_desc(0x80000000U,m,1));
   DG_CMD(3,C+i*N+j,dg_desc(0x80000000U,m,1));
   dg_fence();
  }
 }
 dg_fence();
}
DG_INLINE void dg_readout_cols(const int32_t *A,int8_t *C,uint64_t M,uint64_t N,const uint32_t *scales,unsigned relu) {
 if(!DG_COLUMN_BATCHING || M<=16) { dg_readout_cols_serial(A,C,M,N,scales,relu);return; }
 dg_fence();
 dg_load_config(N*4,0,0,0x3f800000U);
 /* Load each column into its own 64 accumulator rows. FIR229230 and
    FIR229264: local row advances by one, host row by configured stride.
    FIR229836-229848: mvout selects local rows, not a column within a row.
    Thus a column stays in lane0, with no transpose or host tensor math.
    The selected64x8 or128x4 layout occupies rows0..511; all loads finish before readout.
    A scale remains fixed for all stores of that column; finish those
    stores before changing scale or reusing any of the accumulator rows. */
 const uint64_t DG_COLUMN_TILE_ROWS=(DG_COLUMN_TILE_MODE==1||(DG_COLUMN_TILE_MODE==2&&M>64&&N>64))?128:64;
 const uint64_t DG_COLUMN_TILE_COLS=512/DG_COLUMN_TILE_ROWS;
 for(uint64_t i=0;i<M;i+=DG_COLUMN_TILE_ROWS) {
  uint64_t rows=M-i;if(rows>DG_COLUMN_TILE_ROWS)rows=DG_COLUMN_TILE_ROWS;
  for(uint64_t j=0;j<N;j+=DG_COLUMN_TILE_COLS) {
   uint64_t cols=N-j;if(cols>DG_COLUMN_TILE_COLS)cols=DG_COLUMN_TILE_COLS;
   for(uint64_t c=0;c<cols;c++) {
    for(uint64_t r=0;r<rows;r+=16) {
     uint64_t m=rows-r;if(m>16)m=16;
     uint32_t addr=0x80000000U+(uint32_t)(c*DG_COLUMN_TILE_ROWS+r);
     DG_CMD(2,A+(i+r)*N+j+c,dg_desc(addr,m,1));
    }
   }
   dg_fence();
   for(uint64_t c=0;c<cols;c++) {
    dg_store_config(N,scales[j+c],relu);
    for(uint64_t r=0;r<rows;r+=16) {
     uint64_t m=rows-r;if(m>16)m=16;
     uint32_t addr=0x80000000U+(uint32_t)(c*DG_COLUMN_TILE_ROWS+r);
     DG_CMD(3,C+(i+r)*N+j+c,dg_desc(addr,m,1));
    }
    dg_fence();
   }
  }
 }
 dg_fence();
}
DG_INLINE void dg_pool_config(unsigned size) {
 (void)size;
 dg_store_config(16,0x3f800000U,0);
}
DG_INLINE void dg_pool_i8(const int8_t *A,int8_t *C,uint64_t B,uint64_t H,uint64_t W,uint64_t N,
 uint64_t OH,uint64_t OW,uint64_t PH,uint64_t PW,uint64_t SH,uint64_t SW,int64_t PT,int64_t PL,const int8_t *pad,int8_t *partial) {
 (void)partial;
 dg_fence();
 dg_load_config(0,0,1,0x3f800000U);
 /* Adjacent symmetric perturbations of -1/2: independently rounded
    products sum exactly to -x for every signed8 x, without i8 overflow. */
 dg_load_config(0,1,1,0xbf000001U);
 dg_load_config(0,2,1,0xbefffffeU);
 dg_pool_config(0);
 for(uint64_t b=0;b<B;b++)for(uint64_t y=0;y<OH;y++)for(uint64_t x=0;x<OW;x++)for(uint64_t c=0;c<N;c+=16) {
  uint64_t n=N-c;if(n>16)n=16;
  int8_t *dst=C+((b*OH+y)*OW+x)*N+c;
  for(uint64_t ky=0;ky<PH;ky++)for(uint64_t kx=0;kx<PW;kx++) {
   int64_t iy=(int64_t)(y*SH+ky)-PT,ix=(int64_t)(x*SW+kx)-PL;
   const int8_t *src=(iy>=0 && ix>=0 && (uint64_t)iy<H && (uint64_t)ix<W)?A+((b*H+iy)*W+ix)*N+c:pad;
   if(ky==0 && kx==0) {
    DG_CMD(2,src,dg_desc(0x80000010U,1,n));
   } else {
    /* d=x-y in row0, preserving y in row16. Consume src before dst
       becomes transient correction storage, including when src==dst. */
    DG_CMD(2,src,dg_desc(0x80000000U,1,n));
    DG_CMD(1,dst,dg_desc(0xc0000000U,1,n));
    DG_CMD(14,dst,dg_desc(0xc0000000U,1,n));
    dg_fence();
    /* Each correction is clamp(d,0,127). Three rounds cover d<=255. */
    for(unsigned round=0;round<3;round++) {
     dg_store_config(16,0x3f800000U,1);
     DG_CMD(3,dst,dg_desc(0x80000000U,1,n));
     dg_fence();
     DG_CMD(2,dst,dg_desc(0xc0000010U,1,n));
     if(round!=2) {
      DG_CMD(1,dst,dg_desc(0xc0000000U,1,n));
      DG_CMD(14,dst,dg_desc(0xc0000000U,1,n));
     }
     dg_fence();
    }
   }
   dg_store_config(16,0x3f800000U,0);
   DG_CMD(3,dst,dg_desc(0x80000010U,1,n));
   dg_fence();
  }
 }
 dg_store_config(N,0x3f800000U,0);
 dg_fence();
}
DG_INLINE void dg_bias(const int32_t *A,const int32_t *B,int32_t *C,uint64_t M,uint64_t N) {
 dg_fence();
 dg_load_config(N*4,0,0,0x3f800000U);
 dg_load_config(0,1,0,0x3f800000U);
 dg_store_config(N*4,0x3f800000U,0);
 for(uint64_t i=0;i<M;i+=16) {
  uint64_t m=M-i;if(m>16)m=16;
  for(uint64_t j=0;j<N;j+=16) {
   uint64_t n=N-j;if(n>16)n=16;
   DG_CMD(2,A+i*N+j,dg_desc(0x80000000U,m,n));
   DG_CMD(1,B+j,dg_desc(0xc0000000U,m,n));
   DG_CMD(3,C+i*N+j,dg_desc(0xa0000000U,m,n));
   dg_fence();
  }
 }
 dg_fence();
}

)C"; }
std::string deviceMatmul(const std::string&A,const std::string&B,const std::string&C,int64_t M,int64_t K,int64_t N) {
 return "dg_matmul((const int8_t*)("+A+"),(const int8_t*)("+B+"),(int32_t*)("+C+"),"+std::to_string(M)+","+std::to_string(K)+","+std::to_string(N)+");\n";
}
std::string deviceMovement(const std::string&A,const std::string&C,int64_t M,int64_t N,const std::string&outDtype) {
 if(outDtype!="i8" && outDtype!="i32")throw std::runtime_error("unsupported movement output dtype");
 return "dg_movement((const int8_t*)("+A+"),(void*)("+C+"),"+std::to_string(M)+","+std::to_string(N)+","+(outDtype=="i32"?"1":"0")+");\n";
}
}
