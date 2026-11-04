"""Authored 64-by-112 panel Gemmini weight-stationary contraction lowering.

Register fields follow rtl_facts/facts.json; no hardware-loop descriptor is used.
"""

def runtime_header():
    return r'''
typedef unsigned long gm_u64;
typedef unsigned int gm_u32;
#define GM_INLINE static __attribute__((always_inline)) inline
#define GM_INSN(F,A,B) __asm__ volatile (".insn r 0x7b, 3, " #F ", x0, %0, %1" :: "r"((gm_u64)(A)), "r"((gm_u64)(B)) : "memory")
GM_INLINE gm_u32 gm_float_bits(float x) { union {float f; gm_u32 u;} v; v.f=x; return v.u; }
GM_INLINE gm_u64 gm_shape(gm_u32 addr, int rows, int cols) {
  return (gm_u64)addr | ((gm_u64)(unsigned)cols << 32) | ((gm_u64)(unsigned)rows << 48);
}
GM_INLINE void gm_fence(void) { __asm__ volatile("fence" ::: "memory"); }
/* Controller.scala 477,489-493: funct7, rs1[0]=skip.  Zero selects
   flush_retry.  FrontendTLB.scala 52-58 clears the whole DMA TLB (sfence
   rs1=rs2=0), preserving requests rather than selecting flush_skip. */
GM_INLINE void gm_flush(void) { GM_INSN(7,0,0); }
GM_INLINE void gm_config_ex(void) {
  GM_INSN(0, ((gm_u64)0x3f800000 << 32) | ((gm_u64)1 << 16) | 4, (gm_u64)1 << 48);
}
/* Stride is in bytes; state IDs 0,1,2 select LOAD,LOAD2,LOAD3. */
GM_INLINE void gm_config_ld(gm_u64 stride, int state, int shrink) {
  GM_INSN(0, ((gm_u64)0x3f800000 << 32) | ((gm_u64)16 << 16) | ((gm_u64)state << 3) | ((gm_u64)shrink << 2) | 1, stride);
}
GM_INLINE void gm_config_st(gm_u64 stride, float scale, int relu) {
  GM_INSN(0, 2 | ((gm_u64)(relu!=0) << 2), stride | ((gm_u64)gm_float_bits(scale) << 32));
}
GM_INLINE void gm_mvin(const void *src, gm_u32 addr, int rows, int cols, int state) {
  gm_u64 shape=gm_shape(addr,rows,cols);
  if(state==0) { GM_INSN(2,src,shape); }
  else if(state==1) { GM_INSN(1,src,shape); }
  else { GM_INSN(14,src,shape); }
}
GM_INLINE void gm_mvout(void *dst, gm_u32 addr, int rows, int cols) {
  GM_INSN(3,dst,gm_shape(addr,rows,cols));
}
GM_INLINE void gm_preload(gm_u32 b, gm_u32 c, int k, int n, int m) {
  GM_INSN(6,gm_shape(b,k,n),gm_shape(c,m,n));
}
GM_INLINE void gm_compute(gm_u32 a, int m, int k) {
  GM_INSN(4,gm_shape(a,m,k),gm_shape(0xffffffffu,16,16));
}
GM_INLINE int gm_min16(int n) { return n<16?n:16; }
/* A 64x112 output panel occupies 448 accumulator rows in bank zero.
   K panels contain up to 256 elements. A occupies rows 0..1023; B
   occupies rows 8192..9983 (16 K tiles times 112 rows). LoadController's
   seven-bit column field admits 112 columns, with block stride 16. */
GM_INLINE void gm_matmul_strided(const signed char *A,const signed char *B,void *Y,
    int M,int K,int N,int lda,int ldb,int ldy,int out_bytes,float scale,int relu,const int *bias) {
  gm_config_ex();
  gm_config_ld(lda,0,0);
  gm_config_ld(ldb,1,0);
  gm_config_ld(0,2,0);
  gm_config_st((gm_u64)ldy*out_bytes,scale,relu);
  /* A single output tile has no cross-tile operand reuse.  Start execution
     after four K tiles rather than queuing a full 256-element panel first. */
  int k_panel=(M<=16 && N<=16)?64:256;
  if(M<=16 && N<=16 && K<=16) {
    if(bias) gm_mvin(bias,0x80000000u,M,N,2);
    if(K>0) {
      gm_mvin(A,0,M,K,0);
      gm_mvin(B,8192,K,N,1);
      gm_preload(8192,bias?0xc0000000u:0x80000000u,K,N,M);
      gm_compute(0,M,K);
    } else if(!bias) gm_mvin((const void*)0,0x80000000u,M,N,2);
    gm_mvout(Y,out_bytes==4?0xa0000000u:0x80000000u,M,N);
    gm_fence();
    return;
  }
  for(int ib=0;ib<M;ib+=64) {
    int mb=M-ib; if(mb>64) mb=64;
    for(int jb=0;jb<N;jb+=112) {
      int nb=N-jb; if(nb>112) nb=112;
      if(bias || K==0) {
        for(int ii=0;ii<mb;ii+=16) {
          int mr=gm_min16(mb-ii);
          /* DMA.scala 267-274 bypasses block-stride address advance for
             full-width accumulator loads. Load each bias chunk separately. */
          for(int jj=0;jj<nb;jj+=16) {
            int nr=gm_min16(nb-jj);
            const int *bp=bias?bias+jb+jj:(const int *)0;
            gm_mvin(bp,0x80000000u+(ii/16)*112+jj,mr,nr,2);
          }
        }
      }
      for(int kb=0;kb<K;kb+=k_panel) {
        int klen=K-kb; if(klen>k_panel) klen=k_panel;
        /* A: local tile row = M_tile*256 + K_tile*16. */
        for(int ii=0;ii<mb;ii+=16) {
          int mr=gm_min16(mb-ii);
          for(int kk=0;kk<klen;kk+=64) {
            int kc=klen-kk; if(kc>64) kc=64;
            gm_mvin(A+(gm_u64)(ib+ii)*lda+kb+kk,(ii/16)*256+kk,mr,kc,0);
          }
        }
        /* B: local tile row = 8192 + K_tile*112 + N_tile*16. */
        for(int kk=0;kk<klen;kk+=16) {
          int kr=gm_min16(klen-kk);
          gm_mvin(B+(gm_u64)(kb+kk)*ldb+jb,8192+(kk/16)*112,kr,nb,1);
        }
        for(int kk=0;kk<klen;kk+=16) {
          int kr=gm_min16(klen-kk);
          for(int jj=0;jj<nb;jj+=16) {
            int nr=gm_min16(nb-jj);
            gm_u32 ba=8192+(kk/16)*112+jj;
            for(int ii=0;ii<mb;ii+=16) {
              int mr=gm_min16(mb-ii);
              gm_u32 ca=((kb || kk || bias)?0xc0000000u:0x80000000u)+(ii/16)*112+jj;
              gm_preload(ba,ca,kr,nr,mr);
              gm_compute((ii/16)*256+kk,mr,kr);
            }
          }
        }
      }
      gm_u32 read_base=out_bytes==4?0xa0000000u:0x80000000u;
      for(int ii=0;ii<mb;ii+=16) {
        int mr=gm_min16(mb-ii);
        if(out_bytes==4) {
          /* StoreController.scala 225 forbids multi-block full-accumulator
             readout.  Each command reads at most one 16-lane block. */
          for(int jj=0;jj<nb;jj+=16) {
            int nr=gm_min16(nb-jj);
            gm_mvout((unsigned char *)Y+((gm_u64)(ib+ii)*ldy+jb+jj)*4,
                read_base+(ii/16)*112+jj,mr,nr);
          }
        } else {
          /* Keep narrow readout within the already qualified 64-column
             DMA extent, while the resident panel spans seven tiles. */
          for(int jj=0;jj<nb;jj+=64) {
            int nr=nb-jj; if(nr>64) nr=64;
            gm_mvout((unsigned char *)Y+((gm_u64)(ib+ii)*ldy+jb+jj)*out_bytes,
                read_base+(ii/16)*112+jj,mr,nr);
          }
        }
      }
    }
  }
  gm_fence();
}
GM_INLINE void gm_matmul(const signed char *A,const signed char *B,void *Y,
    int M,int K,int N,int out_bytes,float scale,int relu,const int *bias) {
  gm_matmul_strided(A,B,Y,M,K,N,K,N,N,out_bytes,scale,relu,bias);
}
/* Max reduction uses the StoreController's 3x3 pooling path.  Every loaded
   lane is an input byte or the neutral -128 constant; the host only gathers. */
GM_INLINE void gm_maxpool2d(const signed char *X,signed char *Y,
    int B,int H,int W,int C,int OH,int OW,int KH,int KW,int SH,int SW,int PH,int PW) {
  static const signed char neutral[16] __attribute__((aligned(64))) = {
    -128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128};
  gm_config_ld(C,0,1);
  gm_u64 pool_cfg=2 | ((gm_u64)1 << 4) | ((gm_u64)3 << 6)
      | ((gm_u64)1 << 24) | ((gm_u64)1 << 32) | ((gm_u64)1 << 40)
      | ((gm_u64)3 << 48) | ((gm_u64)3 << 56);
  GM_INSN(0,pool_cfg,((gm_u64)0x3f800000 << 32) | (gm_u64)C);
  for(int b=0;b<B;b++) for(int y=0;y<OH;y++) for(int x=0;x<OW;x++)
    for(int c=0;c<C;c+=16) {
      int nc=gm_min16(C-c), q=0;
      signed char *out=Y+(((gm_u64)b*OH+y)*OW+x)*C+c;
      do {
        int slot=0;
        if(q) { gm_mvin(out,0x80000000u,1,nc,0); slot=1; }
        for(;slot<9;slot++) {
          const signed char *src=neutral;
          if(q<KH*KW) {
            int iy=y*SH-PH+q/KW, ix=x*SW-PW+q%KW;
            if(iy>=0 && iy<H && ix>=0 && ix<W)
              src=X+(((gm_u64)b*H+iy)*W+ix)*C+c;
            q++;
          }
          gm_mvin(src,0x80000000u+slot,1,nc,0);
        }
        gm_mvout(out,0x80000000u,1,nc);
        gm_fence();
      } while(q<KH*KW);
    }
  gm_config_st(C,1.0f,0);
}

'''
