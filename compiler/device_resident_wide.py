"""Optional thin-M lowering with legal seven-tile N panels.

K tiles use M local rows, rather than reserving 16 rows for each short tile.
The layout is expressed through the public CONFIG_MVIN block-stride field.
"""


def runtime_header():
    from device_resident import runtime_header as panel_header
    header = panel_header()
    definition = 'GM_INLINE void gm_matmul_strided('
    assert header.count(definition) == 1
    header = header.replace(definition, 'GM_INLINE void gm_matmul_resident64_strided(', 1)
    wrapper = r'''
/* Seven 16-column blocks fit the public 7-bit MVIN/MVOUT column field.
   B uses [8192,9984), and output accumulators use [0,112). */
GM_INLINE void gm_matmul_strided(const signed char *A,const signed char *B,void *Y,
    int M,int K,int N,int lda,int ldb,int ldy,int out_bytes,float scale,int relu,const int *bias) {
  gm_u64 a_rows=(((gm_u64)(unsigned)K+15)/16)*(unsigned)M;
  if(M<1 || M>16 || N<=64 || K<=16 || a_rows>8192) {
    gm_matmul_resident64_strided(A,B,Y,M,K,N,lda,ldb,ldy,out_bytes,scale,relu,bias);
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
  for(int jb=0;jb<N;jb+=112) {
    int nb=N-jb; if(nb>112) nb=112;
    /* Full-width input elements use one physical accumulator row per load. */
    if(bias) for(int jj=0;jj<nb;jj+=16)
      gm_mvin(bias+jb+jj,0x80000000u+jj,M,gm_min16(nb-jj),2);
    for(int kb=0;kb<K;kb+=256) {
      int klen=K-kb; if(klen>256) klen=256;
      for(int kk=0;kk<klen;kk+=16) {
        int kr=gm_min16(klen-kk);
        gm_mvin(B+(gm_u64)(kb+kk)*ldb+jb,8192+(kk/16)*112,kr,nb,1);
      }
      for(int kk=0;kk<klen;kk+=16) {
        int kr=gm_min16(klen-kk);
        for(int jj=0;jj<nb;jj+=16) {
          int nr=gm_min16(nb-jj);
          gm_u32 ca=((kb || kk || bias)?0xc0000000u:0x80000000u)+jj;
          gm_preload(8192+(kk/16)*112+jj,ca,kr,nr,M);
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
