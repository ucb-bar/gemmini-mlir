"""Optional thin-M lowering which keeps all LHS tiles in scratchpad.

K tiles use M local rows, rather than reserving 16 rows for each short tile.
The layout is expressed through the public CONFIG_MVIN block-stride field.
"""


def runtime_header():
    from device_optimized import runtime_header as panel_header
    header = panel_header()
    definition = 'GM_INLINE void gm_matmul_strided('
    assert header.count(definition) == 1
    header = header.replace(definition, 'GM_INLINE void gm_matmul_panel_strided(', 1)
    wrapper = r'''
/* LoadController.scala 150 retains CONFIG_MVIN.rs1[29:16] as block_stride.
   BeatMerger.scala 76-83 adds block_stride for each 16 input bytes.  Thus
   block_stride=M packs each K tile into exactly its M live local rows.
   ReservationStation.scala 294-295 tracks this same packed footprint. */
GM_INLINE void gm_resident_config_ld_a(gm_u64 stride, int rows) {
  GM_INSN(0, ((gm_u64)0x3f800000 << 32) | ((gm_u64)rows << 16) | 1, stride);
}
GM_INLINE void gm_matmul_strided(const signed char *A,const signed char *B,void *Y,
    int M,int K,int N,int lda,int ldb,int ldy,int out_bytes,float scale,int relu,const int *bias) {
  gm_u64 a_rows=(((gm_u64)(unsigned)K+15)/16)*(unsigned)M;
  if(M<1 || M>16 || N<=64 || K<=16 || a_rows>8192) {
    gm_matmul_panel_strided(A,B,Y,M,K,N,lda,ldb,ldy,out_bytes,scale,relu,bias);
    return;
  }
  gm_config_ex();
  gm_resident_config_ld_a(lda,M);
  gm_config_ld(ldb,1,0);
  gm_config_ld(0,2,0);
  gm_config_st((gm_u64)ldy*out_bytes,scale,relu);
  /* The complete packed LHS occupies [0,a_rows); RHS panels use bank2. */
  for(int k=0;k<K;k+=64) {
    int kc=K-k; if(kc>64) kc=64;
    gm_mvin(A+k,(k/16)*M,M,kc,0);
  }
  for(int jb=0;jb<N;jb+=64) {
    int nb=N-jb; if(nb>64) nb=64;
    /* Full-width DMA cannot advance between accumulator column blocks. */
    if(bias) for(int jj=0;jj<nb;jj+=16) {
      int nr=gm_min16(nb-jj);
      gm_mvin(bias+jb+jj,0x80000000u+jj,M,nr,2);
    }
    for(int kb=0;kb<K;kb+=256) {
      int klen=K-kb; if(klen>256) klen=256;
      for(int kk=0;kk<klen;kk+=16) {
        int kr=gm_min16(klen-kk);
        gm_mvin(B+(gm_u64)(kb+kk)*ldb+jb,8192+(kk/16)*64,kr,nb,1);
      }
      for(int kk=0;kk<klen;kk+=16) {
        int kr=gm_min16(klen-kk);
        for(int jj=0;jj<nb;jj+=16) {
          int nr=gm_min16(nb-jj);
          gm_u32 ca=((kb || kk || bias)?0xc0000000u:0x80000000u)+jj;
          gm_preload(8192+(kk/16)*64+jj,ca,kr,nr,M);
          gm_compute(((kb+kk)/16)*M,M,kr);
        }
      }
    }
    if(out_bytes==4) {
      for(int jj=0;jj<nb;jj+=16) {
        int nr=gm_min16(nb-jj);
        gm_mvout((unsigned char*)Y+(gm_u64)(jb+jj)*4,0xa0000000u+jj,M,nr);
      }
    } else {
      gm_mvout((unsigned char*)Y+(gm_u64)jb*out_bytes,0x80000000u,M,nb);
    }
  }
  gm_fence();
}
'''
    insertion = 'GM_INLINE void gm_matmul('
    assert header.count(insertion) == 1
    return header.replace(insertion, wrapper + '\n' + insertion, 1)
