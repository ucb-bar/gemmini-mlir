#include <string>
namespace backend {
std::string poolHardwareRuntime() { return R"C(
/* Native pooling is a StoreController address walk followed by signed-byte
   StreamWriter max. FIR229759-229788 defines the configuration fields;
   FIR229625-229641 and229855-229880 define input/output row walks.
   Accumulator readout applies scale/activation before StreamWriter pooling.
   The 15-bit request tracker (FIR229521) counts OH*OW*PH*PW taps.
   Padded addresses produce zero (FIR186056-186072), so padding is admitted
   only with ReLU and a neutral pad. Unpadded signed-byte pooling also works. */
DG_INLINE unsigned dg_readout_pool_hw(const int32_t *A,int8_t *C,
 uint64_t batch,uint64_t H,uint64_t width,uint64_t N,uint64_t OH,uint64_t OW,
 uint64_t PH,uint64_t PW,uint64_t SH,uint64_t SW,int64_t PT,int64_t PL,
 int8_t pad,uint32_t scale,unsigned relu) {
 /* All rejection precedes commands or writes. Bit tests establish finite,
    strictly positive binary32 scale without performing tensor arithmetic. */
 if(!batch || !H || !width || !N || N>16 || !OH || !OW || relu>1)return 0;
 if(H>255 || width>255 || OH>255 || OW>255 || H>512/width)return 0;
 if(!PH || PH>3 || PH!=PW || !SH || SH>3 || SH!=SW)return 0;
 if(PT<0 || PT>3 || PL<0 || PL>3)return 0;
 if((scale&0x80000000U) || !(scale&0x7fffffffU) || (scale&0x7f800000U)==0x7f800000U)return 0;
 if(OH*OW*PH*PW>=32768)return 0;
 /* First and last windows intersect along each axis; intermediate starts
    are monotonic. This also excludes entirely padded output windows. */
 if((uint64_t)PT>=PH || (uint64_t)PL>=PW)return 0;
 if((OH-1)*SH>=(uint64_t)PT+H || (OW-1)*SW>=(uint64_t)PL+width)return 0;
 unsigned hasPadding=PT!=0 || PL!=0 || (OH-1)*SH+PH>H+(uint64_t)PT || (OW-1)*SW+PW>width+(uint64_t)PL;
 if(hasPadding && (!relu || pad>0))return 0;
 uint64_t rows=H*width;
 uint64_t cfg=2 | ((uint64_t)relu<<2) | (SH<<4) | (PH<<6)
  | ((uint64_t)PT<<8) | ((uint64_t)PL<<10) | (OW<<24)
  | (OH<<32) | (OW<<40) | (H<<48) | (width<<56);
 dg_fence();
 dg_store_config(N,0x3f800000U,0);
 for(uint64_t b=0;b<batch;b++) {
  dg_load_config(N*4,0,0,0x3f800000U);
  for(uint64_t r=0;r<rows;r+=16) {
   uint64_t m=rows-r;if(m>16)m=16;
   DG_CMD(2,A+(b*rows+r)*N,dg_desc(0x80000000U | (uint32_t)r,m,N));
   dg_fence();
  }
  /* FIR229990 selects pool_total_rows rather than descriptor rows when
     pooling is enabled; cols<=16 ensures a single legal mvout block. */
  DG_CMD(0,cfg,((uint64_t)scale<<32) | N);
  DG_CMD(3,C+b*OH*OW*N,dg_desc(0x80000000U,1,N));
  dg_fence();
 }
 /* Clear all native pool state before another helper performs mvout. */
 dg_store_config(N,0x3f800000U,0);
 dg_fence();
 return 1;
}
)C"; }
}
