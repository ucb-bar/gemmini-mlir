#include "compiler.h"
namespace backend {
std::string convRuntime() {
 const std::string rowBatch=compilerSettingString("conv_row_batch","64");
 if(rowBatch!="64" && rowBatch!="128")
  throw std::runtime_error("unsupported conv_row_batch: "+rowBatch);
 return std::string("#define DG_CONV_BLOCKING ")+(compilerSettingBool("conv_blocking",false)?"1":"0")+"\n#define DG_CONV_COORDINATE_LOOPS "+(compilerSettingBool("conv_coordinate_loops",false)?"1":"0")+"\n#define DG_CONV_GATHER_BATCHING "+(compilerSettingBool("conv_gather_batching",false)?"1":"0")+"\n#define DG_CONV_GATHER_OUTER_COORDINATES "+(compilerSettingBool("conv_gather_outer_coordinates",false)?"1":"0")+"\n#define DG_CONV_ROW_BATCH "+rowBatch+"\n"+R"C(
#ifndef DG_CONV_ROW_BATCH
#define DG_CONV_ROW_BATCH 64
#endif
/* Spatial convolution uses only address arithmetic on the scalar core.
 * LoadController FIR229229: source pointer zero enables all_zeros DMA.
 * FIR229264-229275: source base+row*stride, local base+row, scale1 i8.
 * LoopMatmulExecute FIR226854-227153 gives direct preload6(B,C) and
 * compute4(A,garbage), with rows/columns describing each true tail.
 * Gathered A rows0..15 contain sixteen output positions and <=16 CI lanes.
 * W occupies scratch rows16..31. C occupies accumulator rows0..15.
 */
/* Optional row blocking reuses one DMA weight tile across four (64 rows)
 * or eight (128 rows) output tiles. A occupies0..ROW_BATCH-1, W occupies
 * ROW_BATCH..ROW_BATCH+15, and accumulator C occupies0..ROW_BATCH-1.
 * Thus128 rows use144 SP rows and128 ACC rows, within1024SP/512ACC;
 * FIR184670 gives the accumulator memory a9-bit row address.
 * This is below the incumbent1024-row blocked-matmul scratch footprint;
 * raw LoopMatmul half ranges are0..8191 and8192..16383 (FIR228917-228934).
 * Each accumulator tile keeps identical ky/kx/CI reduction order and tails.
 * Only source address arithmetic and DMA/execute dispatch run on the host.
 */
#if DG_CONV_GATHER_BATCHING
/* FIR229197-229198: DMA descriptor rows uses bits52:48 (5 bits).
 * Runs use at most16 rows, as in the incumbent tiled loads. Source rows
 * advance by configured SW*CI bytes (FIR229264-229268), local rows by1
 * (FIR229230-229234); source0 enables zero DMA (FIR229229).
 * A run never crosses an output row or a valid/padding transition. */
DG_INLINE void dg_conv_gather_runs(const int8_t *A,uint64_t p,uint64_t rows,
 uint64_t H,uint64_t width,uint64_t CI,uint64_t OH,uint64_t OW,
 uint64_t SH,uint64_t SW,uint64_t DH,uint64_t DW,int64_t PT,int64_t PL,
 uint64_t ky,uint64_t kx,uint64_t ci,uint64_t kr
#if DG_CONV_GATHER_OUTER_COORDINATES
 ,uint64_t initial_b,uint64_t initial_oy,uint64_t initial_ox
#endif
 ) {
#if DG_CONV_GATHER_OUTER_COORDINATES
 uint64_t b=initial_b,oy=initial_oy,ox=initial_ox;
#else
 uint64_t b=p/(OH*OW),oy=(p/OW)%OH,ox=p%OW;
#endif
 for(uint64_t r=0;r<rows;) {
  uint64_t run=rows-r;if(run>16)run=16;
  uint64_t rowleft=OW-ox;if(run>rowleft)run=rowleft;
  int64_t iy=(int64_t)(oy*SH+ky*DH)-PT;
  int64_t ix=(int64_t)(ox*SW+kx*DW)-PL;
  const int8_t *src=(const int8_t*)0;
  if(iy>=0 && (uint64_t)iy<H) {
   if(ix<0) {
    if(SW) {
     uint64_t left=0-(uint64_t)ix;
     uint64_t padding=left/SW+(left%SW!=0);
     if(run>padding)run=padding;
    }
   } else if((uint64_t)ix<width) {
    if(SW) {
     uint64_t valid=(width-1-(uint64_t)ix)/SW+1;
     if(run>valid)run=valid;
    }
    src=A+((b*H+(uint64_t)iy)*width+(uint64_t)ix)*CI+ci;
   }
  }
  DG_CMD(2,src,dg_desc((uint32_t)r,run,kr));
  r+=run;ox+=run;
  if(ox==OW) { ox=0;if(++oy==OH) { oy=0;++b; } }
 }
}
#endif
DG_INLINE void dg_conv2d_blocked(const int8_t *A,const int8_t *W,const int32_t *D,void *C,
 uint64_t batch,uint64_t H,uint64_t width,uint64_t CI,uint64_t CO,
 uint64_t OH,uint64_t OW,uint64_t KH,uint64_t KW,uint64_t SH,uint64_t SW,
 uint64_t DH,uint64_t DW,int64_t PT,int64_t PL,unsigned out32,uint32_t scale,unsigned relu) {
 dg_fence();
 DG_CMD(0,(1ULL<<16)|4,1ULL<<48);
#if DG_CONV_GATHER_BATCHING
 dg_load_config(SW*CI,0,0,0x3f800000U);
#else
 dg_load_config(0,0,0,0x3f800000U);
#endif
 dg_load_config(CO,1,0,0x3f800000U);
 dg_load_config(0,2,0,0x3f800000U);
 dg_store_config(CO*(out32?4:1),scale,relu);
 uint64_t M=batch*OH*OW;
 for(uint64_t p=0;p<M;p+=DG_CONV_ROW_BATCH) {
  uint64_t rows=M-p;if(rows>DG_CONV_ROW_BATCH)rows=DG_CONV_ROW_BATCH;
#if DG_CONV_GATHER_BATCHING && DG_CONV_GATHER_OUTER_COORDINATES
  /* Every ky/kx/CI gather starts at this same flat output block. */
  uint64_t initial_b=p/(OH*OW),initial_oy=(p/OW)%OH,initial_ox=p%OW;
#endif
  for(uint64_t j=0;j<CO;j+=16) {
   uint64_t nr=CO-j;if(nr>16)nr=16;
   if(D || CI==0 || KH==0 || KW==0) {
    for(uint64_t ii=0;ii<rows;ii+=16) {
     uint64_t mr=rows-ii;if(mr>16)mr=16;
     DG_CMD(14,D?D+j:(const int32_t*)0,dg_desc(0x80000000U|(uint32_t)ii,mr,nr));
    }
   }
   unsigned first=1;
   for(uint64_t ky=0;ky<KH;ky++)for(uint64_t kx=0;kx<KW;kx++)for(uint64_t ci=0;ci<CI;ci+=16) {
    uint64_t kr=CI-ci;if(kr>16)kr=16;
#if DG_CONV_GATHER_BATCHING
#if DG_CONV_GATHER_OUTER_COORDINATES
    dg_conv_gather_runs(A,p,rows,H,width,CI,OH,OW,SH,SW,DH,DW,PT,PL,ky,kx,ci,kr,initial_b,initial_oy,initial_ox);
#else
    dg_conv_gather_runs(A,p,rows,H,width,CI,OH,OW,SH,SW,DH,DW,PT,PL,ky,kx,ci,kr);
#endif
#else
#if DG_CONV_COORDINATE_LOOPS
    /* Decode the group's first flat NHWC position once. Advancing the
       mixed-radix counters below preserves pos=(b*OH+oy)*OW+ox=p+r. */
    uint64_t b=p/(OH*OW),oy=(p/OW)%OH,ox=p%OW;
#endif
    for(uint64_t r=0;r<rows;r++) {
#if !DG_CONV_COORDINATE_LOOPS
     uint64_t pos=p+r;
     uint64_t b=pos/(OH*OW),oy=(pos/OW)%OH,ox=pos%OW;
#endif
     int64_t iy=(int64_t)(oy*SH+ky*DH)-PT;
     int64_t ix=(int64_t)(ox*SW+kx*DW)-PL;
     const int8_t *src=(const int8_t*)0;
     if(iy>=0 && ix>=0 && (uint64_t)iy<H && (uint64_t)ix<width)
      src=A+((b*H+(uint64_t)iy)*width+(uint64_t)ix)*CI+ci;
     DG_CMD(2,src,dg_desc((uint32_t)r,1,kr));
#if DG_CONV_COORDINATE_LOOPS
     if(++ox==OW) { ox=0;if(++oy==OH) { oy=0;++b; } }
#endif
    }
#endif
    DG_CMD(1,W+((ky*KW+kx)*CI+ci)*CO+j,dg_desc(DG_CONV_ROW_BATCH,kr,nr));
    uint32_t acc=(!first || D)?0xc0000000U:0x80000000U;
    for(uint64_t ii=0;ii<rows;ii+=16) {
     uint64_t mr=rows-ii;if(mr>16)mr=16;
     DG_CMD(6,dg_desc(DG_CONV_ROW_BATCH,kr,nr),dg_desc(acc|(uint32_t)ii,mr,nr));
     DG_CMD(4,dg_desc((uint32_t)ii,mr,kr),dg_desc(0xffffffffU,16,16));
    }
    first=0;
    dg_fence();
   }
   for(uint64_t ii=0;ii<rows;ii+=16) {
    uint64_t mr=rows-ii;if(mr>16)mr=16;
    DG_CMD(3,(int8_t*)C+((p+ii)*CO+j)*(out32?4:1),dg_desc((out32?0xa0000000U:0x80000000U)|(uint32_t)ii,mr,nr));
   }
   dg_fence();
  }
 }
 dg_fence();
}

DG_INLINE void dg_conv2d_ex(const int8_t *A,const int8_t *W,const int32_t *D,void *C,
 uint64_t batch,uint64_t H,uint64_t width,uint64_t CI,uint64_t CO,
 uint64_t OH,uint64_t OW,uint64_t KH,uint64_t KW,uint64_t SH,uint64_t SW,
 uint64_t DH,uint64_t DW,int64_t PT,int64_t PL,unsigned out32,uint32_t scale,unsigned relu) {
 if(DG_CONV_BLOCKING && batch*OH*OW>16) {
  dg_conv2d_blocked(A,W,D,C,batch,H,width,CI,CO,OH,OW,KH,KW,SH,SW,DH,DW,PT,PL,out32,scale,relu);
  return;
 }
 dg_fence();
 DG_CMD(0,(1ULL<<16)|4,1ULL<<48);
#if DG_CONV_GATHER_BATCHING
 dg_load_config(SW*CI,0,0,0x3f800000U);
#else
 dg_load_config(0,0,0,0x3f800000U);
#endif
 dg_load_config(CO,1,0,0x3f800000U);
 dg_load_config(0,2,0,0x3f800000U);
 dg_store_config(CO*(out32?4:1),scale,relu);
 uint64_t M=batch*OH*OW;
 for(uint64_t p=0;p<M;p+=16) {
  uint64_t mr=M-p;if(mr>16)mr=16;
#if DG_CONV_GATHER_BATCHING && DG_CONV_GATHER_OUTER_COORDINATES
  /* Every ky/kx/CI gather starts at this same flat output block. */
  uint64_t initial_b=p/(OH*OW),initial_oy=(p/OW)%OH,initial_ox=p%OW;
#endif
  for(uint64_t j=0;j<CO;j+=16) {
   uint64_t nr=CO-j;if(nr>16)nr=16;
   if(D || CI==0 || KH==0 || KW==0) {
    DG_CMD(14,D?D+j:(const int32_t*)0,dg_desc(0x80000000U,mr,nr));
   }
   unsigned first=1;
   for(uint64_t ky=0;ky<KH;ky++)for(uint64_t kx=0;kx<KW;kx++)for(uint64_t ci=0;ci<CI;ci+=16) {
    uint64_t kr=CI-ci;if(kr>16)kr=16;
#if DG_CONV_GATHER_BATCHING
#if DG_CONV_GATHER_OUTER_COORDINATES
    dg_conv_gather_runs(A,p,mr,H,width,CI,OH,OW,SH,SW,DH,DW,PT,PL,ky,kx,ci,kr,initial_b,initial_oy,initial_ox);
#else
    dg_conv_gather_runs(A,p,mr,H,width,CI,OH,OW,SH,SW,DH,DW,PT,PL,ky,kx,ci,kr);
#endif
#else
#if DG_CONV_COORDINATE_LOOPS
    /* Decode the group's first flat NHWC position once. Advancing the
       mixed-radix counters below preserves pos=(b*OH+oy)*OW+ox=p+r. */
    uint64_t b=p/(OH*OW),oy=(p/OW)%OH,ox=p%OW;
#endif
    for(uint64_t r=0;r<mr;r++) {
#if !DG_CONV_COORDINATE_LOOPS
     uint64_t pos=p+r;
     uint64_t b=pos/(OH*OW),oy=(pos/OW)%OH,ox=pos%OW;
#endif
     int64_t iy=(int64_t)(oy*SH+ky*DH)-PT;
     int64_t ix=(int64_t)(ox*SW+kx*DW)-PL;
     const int8_t *src=(const int8_t*)0;
     if(iy>=0 && ix>=0 && (uint64_t)iy<H && (uint64_t)ix<width)
      src=A+((b*H+(uint64_t)iy)*width+(uint64_t)ix)*CI+ci;
     DG_CMD(2,src,dg_desc((uint32_t)r,1,kr));
#if DG_CONV_COORDINATE_LOOPS
     if(++ox==OW) { ox=0;if(++oy==OH) { oy=0;++b; } }
#endif
    }
#endif
    DG_CMD(1,W+((ky*KW+kx)*CI+ci)*CO+j,dg_desc(16,kr,nr));
    uint32_t acc=(!first || D)?0xc0000000U:0x80000000U;
    DG_CMD(6,dg_desc(16,kr,nr),dg_desc(acc,mr,nr));
    DG_CMD(4,dg_desc(0,mr,kr),dg_desc(0xffffffffU,16,16));
    first=0;
    dg_fence();
   }
   DG_CMD(3,(int8_t*)C+(p*CO+j)*(out32?4:1),dg_desc(out32?0xa0000000U:0x80000000U,mr,nr));
   dg_fence();
  }
 }
 dg_fence();
}
)C"; }
}
