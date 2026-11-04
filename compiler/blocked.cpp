#include "compiler.h"
namespace backend {
static std::string blockedTileRuntime(const std::string &tile) {
 int bm=64,bn=64,bk=128;
 if(tile=="64x128x64") { bn=128;bk=64; }
 else if(tile=="128x64x64") { bm=128;bk=64; }
 else if(tile=="64x64x64") { bk=64; }
 else if(tile!="64x64x128") throw std::runtime_error("unsupported matmul_tile: "+tile);
 return "enum { DG_BM="+std::to_string(bm)+", DG_BN="+std::to_string(bn)+", DG_BK="+std::to_string(bk)+", DG_BBASE=DG_BM*DG_BK/16 };\n"+R"C(
/* Raw FIR evidence for this layout:
 * LoadController 229264-229275: source row uses configured byte stride;
 * local row uses base+row and passes configured block_stride to DMA.
 * BeatMerger 144950-144958: each next 16-byte i8 column block advances
 * local row address by block_stride. Descriptor cols is 7 bits (229199).
 * Thus A tile(it,kt) starts kt*BM+it*16; B tile(kt,jt) starts
 * BBASE+jt*BK+kt*16, where BBASE=BM*BK/16. SP footprint is
 * (BM+BN)*BK/16 rows, ACC footprint BM*BN/16 rows. Supported geometries:
 * 64x64x128:1024SP/256ACC, 64x128x64:768SP/512ACC,
 * 128x64x64:768SP/512ACC, 64x64x64:512SP/256ACC.
 * Accumulator tile(it,jt) starts (it*(BN/16)+jt)*16.
 * Execute command operands follow LoopMatmulExecute 226854-227153:
 * preload6(B,C), compute4(A,garbage), with true tail dimensions.
 */
DG_INLINE void dg_matmul_blocked(const int8_t *A,const int8_t *B,const int32_t *D,void *C,
 uint64_t M,uint64_t K,uint64_t N,unsigned out32,uint32_t scale,unsigned relu) {
 dg_fence();
 DG_CMD(0,(1ULL<<16)|4,1ULL<<48);
 DG_CMD(0,(0x3f800000ULL<<32)|((uint64_t)DG_BM<<16)|1,K);
 DG_CMD(0,(0x3f800000ULL<<32)|((uint64_t)DG_BK<<16)|(1ULL<<3)|1,N);
 dg_load_config(0,2,0,0x3f800000U);
 dg_store_config(N*(out32?4:1),scale,relu);
 for(uint64_t ib=0;ib<M;ib+=DG_BM) {
  uint64_t mb=M-ib;if(mb>DG_BM)mb=DG_BM;
  for(uint64_t jb=0;jb<N;jb+=DG_BN) {
   uint64_t nb=N-jb;if(nb>DG_BN)nb=DG_BN;
   if(D || K==0) {
    for(uint64_t ii=0;ii<mb;ii+=16) {
     uint64_t mr=mb-ii;if(mr>16)mr=16;
     for(uint64_t jj=0;jj<nb;jj+=16) {
      uint64_t nr=nb-jj;if(nr>16)nr=16;
      uint32_t ca=(uint32_t)((ii/16*(DG_BN/16)+jj/16)*16);
      DG_CMD(14,D?D+jb+jj:(const int32_t*)0,dg_desc(0x80000000U|ca,mr,nr));
     }
    }
   }
   for(uint64_t kb=0;kb<K;kb+=DG_BK) {
    uint64_t kbsize=K-kb;if(kbsize>DG_BK)kbsize=DG_BK;
    /* A: each command gathers up to four column tiles across one row tile. */
    for(uint64_t ii=0;ii<mb;ii+=16) {
     uint64_t mr=mb-ii;if(mr>16)mr=16;
     for(uint64_t ks=0;ks<kbsize;ks+=64) {
      uint64_t cols=kbsize-ks;if(cols>64)cols=64;
      uint32_t aa=(uint32_t)((ks/16)*DG_BM+ii);
      DG_CMD(2,A+(ib+ii)*K+kb+ks,dg_desc(aa,mr,cols));
     }
    }
    /* B: each command gathers up to four output column tiles. */
    for(uint64_t kk=0;kk<kbsize;kk+=16) {
     uint64_t kr=kbsize-kk;if(kr>16)kr=16;
     for(uint64_t ns=0;ns<nb;ns+=64) {
      uint64_t cols=nb-ns;if(cols>64)cols=64;
      uint32_t ba=(uint32_t)(DG_BBASE+(ns/16)*DG_BK+kk);
      DG_CMD(1,B+(kb+kk)*N+jb+ns,dg_desc(ba,kr,cols));
     }
    }
    dg_fence();
    for(uint64_t kk=0;kk<kbsize;kk+=16) {
     uint64_t kr=kbsize-kk;if(kr>16)kr=16;
     for(uint64_t jj=0;jj<nb;jj+=16) {
      uint64_t nr=nb-jj;if(nr>16)nr=16;
      uint32_t ba=(uint32_t)(DG_BBASE+(jj/16)*DG_BK+kk);
      for(uint64_t ii=0;ii<mb;ii+=16) {
       uint64_t mr=mb-ii;if(mr>16)mr=16;
       uint32_t aa=(uint32_t)((kk/16)*DG_BM+ii);
       uint32_t ca=(uint32_t)((ii/16*(DG_BN/16)+jj/16)*16);
       uint32_t acc=(kb!=0 || kk!=0 || D)?0xc0000000U:0x80000000U;
       DG_CMD(6,dg_desc(ba,kr,nr),dg_desc(acc|ca,mr,nr));
       DG_CMD(4,dg_desc(aa,mr,kr),dg_desc(0xffffffffU,16,16));
      }
     }
    }
    dg_fence();
   }
   for(uint64_t ii=0;ii<mb;ii+=16) {
    uint64_t mr=mb-ii;if(mr>16)mr=16;
    for(uint64_t jj=0;jj<nb;jj+=16) {
     uint64_t nr=nb-jj;if(nr>16)nr=16;
     uint32_t ca=(uint32_t)((ii/16*(DG_BN/16)+jj/16)*16);
     DG_CMD(3,(int8_t*)C+((ib+ii)*N+jb+jj)*(out32?4:1),dg_desc((out32?0xa0000000U:0x80000000U)|ca,mr,nr));
    }
   }
   dg_fence();
  }
 }
 dg_fence();
}
)C"; }
std::string blockedRuntime() {
 const std::string tile=compilerSettingString("matmul_tile","64x64x128");
 if(tile!="auto")return blockedTileRuntime(tile);
 std::string result;
 const char *tiles[]={"64x64x128","64x128x64","128x64x64"};
 const char *suffixes[]={"default","wide","tall"};
 for(unsigned i=0;i<3;i++) {
  std::string code=blockedTileRuntime(tiles[i]);
  const char *names[]={"DG_BBASE","DG_BM","DG_BN","DG_BK","dg_matmul_blocked"};
  for(const char *name:names) {
   const std::string oldName=name,newName=oldName+"_"+suffixes[i];
   size_t pos=0;
   while((pos=code.find(oldName,pos))!=std::string::npos) {
    code.replace(pos,oldName.size(),newName);pos+=newName.size();
   }
  }
  result+=code;
 }
 return result+R"C(
/* These costs count shape-dependent DMA commands and the two fences for
   each K block. Ceil division avoids integer addition overflow; double
   products cannot overflow even for uint64_t dimensions. Tensor values
   never enter the scheduling decision. Strict comparisons preserve the
   incumbent geometry on ties. */
DG_INLINE double dg_matmul_blocked_cost(uint64_t M,uint64_t K,uint64_t N,
 uint64_t bm,uint64_t bn,uint64_t bk) {
 double mt=(double)(M/16+(M%16!=0));
 double nt=(double)(N/64+(N%64!=0));
 double kt=(double)(K/16+(K%16!=0));
 double ks=(double)(K/64+(K%64!=0));
 double mb=(double)(M/bm+(M%bm!=0));
 double nb=(double)(N/bn+(N%bn!=0));
 double kb=(double)(K/bk+(K%bk!=0));
 return mt*nb*ks+mb*kt*nt+2.0*mb*nb*kb;
}
DG_INLINE void dg_matmul_blocked(const int8_t *A,const int8_t *B,const int32_t *D,void *C,
 uint64_t M,uint64_t K,uint64_t N,unsigned out32,uint32_t scale,unsigned relu) {
 double best=dg_matmul_blocked_cost(M,K,N,64,64,128);
 double wide=dg_matmul_blocked_cost(M,K,N,64,128,64);
 double tall=dg_matmul_blocked_cost(M,K,N,128,64,64);
 unsigned choice=0;
 if(wide<best) { best=wide;choice=1; }
 if(tall<best) { choice=2; }
 if(choice==1)dg_matmul_blocked_wide(A,B,D,C,M,K,N,out32,scale,relu);
 else if(choice==2)dg_matmul_blocked_tall(A,B,D,C,M,K,N,out32,scale,relu);
 else dg_matmul_blocked_default(A,B,D,C,M,K,N,out32,scale,relu);
}
)C";
}
}
