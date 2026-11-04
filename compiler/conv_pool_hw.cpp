#include <string>
#include <vector>
#include <cstdint>
#include <sstream>
namespace backend {
bool compilerSettingBool(const std::string&name,bool fallback);
std::string convPoolHardwareRuntime() { return std::string("#define DG_POOL_STRIPE_BLOCKING ")+
 (compilerSettingBool("pool_stripe_blocking",false)?"1":"0")+"\n#define DG_POOL_GATHER_BATCHING "+
 (compilerSettingBool("pool_gather_batching",false)?"1":"0")+"\n"+R"C(
/* A spatial stripe occupies accumulator rows0..511. Each row corresponds
   to one valid convolution output position, in row-major order. Native
   StoreController pooling reads base+orow*ocols+ocol (FIR229855-229861)
   and writes ((porow*outdim+pocol)*stride) bytes (FIR229873-229880).
   For one pooled output row, porows=1 and the stripe's top clipping becomes
   native upad. The source row is therefore y*SH-PT+wrow, exactly as in
   the full convolution image. Scratch A0..15/W16..31 (default) or
   A0..63/W64..79 (optional blocking) is separate from accumulator storage.
   Each weight DMA serves up to four 16-row output tiles. Accumulator
   addresses stay p+ii, preserving stripe layout and ky,kx,CI order. */
#if DG_POOL_GATHER_BATCHING
/* FIR229264-229275: each DMA source row is base+row*configuredStride,
   and its scratch destination is base+row. A run has fixed input height
   and horizontal increment convSW*CI bytes. Split at output-row and padding
   boundaries so every run is wholly valid or wholly hardware-zero padding.
   At most16 rows per DMA fit the descriptor's array-height bound. */
DG_INLINE void dg_pool_conv_gather_runs(const int8_t *A,uint64_t b,uint64_t lo,
 uint64_t p,uint64_t rows,uint64_t H,uint64_t width,uint64_t CI,uint64_t convOW,
 uint64_t convSH,uint64_t convSW,uint64_t DH,uint64_t DW,int64_t convPT,int64_t convPL,
 uint64_t ky,uint64_t kx,uint64_t ci,uint64_t kr) {
 uint64_t oy=lo+p/convOW,ox=p%convOW;
 for(uint64_t r=0;r<rows;) {
  uint64_t run=rows-r;if(run>16)run=16;
  uint64_t rowleft=convOW-ox;if(run>rowleft)run=rowleft;
  int64_t iy=(int64_t)(oy*convSH+ky*DH)-convPT;
  int64_t ix=(int64_t)(ox*convSW+kx*DW)-convPL;
  const int8_t *src=(const int8_t*)0;
  if(iy>=0 && (uint64_t)iy<H) {
   if(ix<0) {
    if(convSW) {
     uint64_t left=0-(uint64_t)ix;
     uint64_t padding=left/convSW+(left%convSW!=0);
     if(run>padding)run=padding;
    }
   } else if((uint64_t)ix<width) {
    if(convSW) {
     uint64_t valid=(width-1-(uint64_t)ix)/convSW+1;
     if(run>valid)run=valid;
    }
    src=A+((b*H+(uint64_t)iy)*width+(uint64_t)ix)*CI+ci;
   }
  }
  DG_CMD(2,src,dg_desc((uint32_t)r,run,kr));
  r+=run;ox+=run;
  if(ox==convOW) { ox=0;++oy; }
 }
}
#endif
DG_INLINE unsigned dg_conv2d_pool_hw(const int8_t *A,const int8_t *W,const int32_t *D,int8_t *C,
 uint64_t batch,uint64_t H,uint64_t width,uint64_t CI,uint64_t CO,
 uint64_t convOH,uint64_t convOW,uint64_t KH,uint64_t KW,uint64_t convSH,uint64_t convSW,
 uint64_t DH,uint64_t DW,int64_t convPT,int64_t convPL,
 uint64_t OH,uint64_t OW,uint64_t PH,uint64_t PW,uint64_t SH,uint64_t SW,
 int64_t PT,int64_t PL,int8_t pad,uint32_t scale,unsigned relu) {
 /* Rejection is effect-free. Native stride/size/pad are two-bit fields;
    pocols/outdim/ocols are eight-bit fields (FIR229759-229788).
    The native DMA tracker counts taps in fifteen bits (FIR229521). */
 if(!batch || !H || !width || !CO || !convOH || !convOW || !OH || !OW || relu>1)return 0;
 if(CO>0xffffffffULL || convOW>255 || OW>255 || !PH || PH>3 || PH!=PW || !SH || SH>3 || SH!=SW)return 0;
 if(PT<0 || PT>3 || PL<0 || PL>3)return 0;
 if((scale&0x80000000U) || !(scale&0x7fffffffU) || (scale&0x7f800000U)==0x7f800000U)return 0;
 if(OW*PH*PW>=32768)return 0;
 /* Bounds protect the signed spatial coordinate arithmetic below. */
 if(convOH>0x7fffffffULL || OH>0x7fffffffULL || KH>0x7fffffffULL || KW>0x7fffffffULL ||
    convSH>0x7fffffffULL || convSW>0x7fffffffULL || DH>0x7fffffffULL || DW>0x7fffffffULL ||
    convPT<(-2147483647LL-1) || convPT>2147483647LL || convPL<(-2147483647LL-1) || convPL>2147483647LL)return 0;
 if((uint64_t)PT>=PH || (uint64_t)PL>=PW)return 0;
 if((OH-1)*SH>=(uint64_t)PT+convOH || (OW-1)*SW>=(uint64_t)PL+convOW)return 0;
 unsigned hasPadding=PT!=0 || PL!=0 || (OH-1)*SH+PH>convOH+(uint64_t)PT || (OW-1)*SW+PW>convOW+(uint64_t)PL;
 /* FIR186056-186072 converts native out-of-bounds addresses to zero.
    ReLU with positive scale makes that neutral for every nonempty window. */
 if(hasPadding && (!relu || pad>0))return 0;
 for(uint64_t y=0;y<OH;y++) {
  int64_t start=(int64_t)(y*SH)-PT;
  uint64_t lo=start>0?(uint64_t)start:0;
  uint64_t hi=(uint64_t)(start+(int64_t)PH);if(hi>convOH)hi=convOH;
  if(hi<=lo || hi-lo>512/convOW)return 0;
 }
 dg_fence();
 DG_CMD(0,(1ULL<<16)|4,1ULL<<48);
 dg_store_config(CO,0x3f800000U,0);
 for(uint64_t b=0;b<batch;b++)for(uint64_t y=0;y<OH;y++) {
  int64_t start=(int64_t)(y*SH)-PT;
  uint64_t lo=start>0?(uint64_t)start:0;
  uint64_t hi=(uint64_t)(start+(int64_t)PH);if(hi>convOH)hi=convOH;
  uint64_t stripeH=hi-lo,rows=stripeH*convOW;
  uint64_t upad=start<0?(uint64_t)(-start):0;
  for(uint64_t j=0;j<CO;j+=16) {
   uint64_t nr=CO-j;if(nr>16)nr=16;
#if DG_POOL_GATHER_BATCHING
   dg_load_config(convSW*CI,0,0,0x3f800000U);
#else
   dg_load_config(0,0,0,0x3f800000U);
#endif
   dg_load_config(CO,1,0,0x3f800000U);
   dg_load_config(0,2,0,0x3f800000U);
   const uint64_t rowBatch=DG_POOL_STRIPE_BLOCKING?64:16;
   const uint32_t weightBase=DG_POOL_STRIPE_BLOCKING?64:16;
   for(uint64_t p=0;p<rows;p+=rowBatch) {
    uint64_t groupRows=rows-p;if(groupRows>rowBatch)groupRows=rowBatch;
    if(D || CI==0 || KH==0 || KW==0) {
#if DG_POOL_STRIPE_BLOCKING
     for(uint64_t ii=0;ii<groupRows;ii+=16) {
      uint64_t mr=groupRows-ii;if(mr>16)mr=16;
      DG_CMD(14,D?D+j:(const int32_t*)0,dg_desc(0x80000000U | (uint32_t)(p+ii),mr,nr));
     }
#else
     DG_CMD(14,D?D+j:(const int32_t*)0,dg_desc(0x80000000U | (uint32_t)p,groupRows,nr));
#endif
    }
    unsigned first=1;
    for(uint64_t ky=0;ky<KH;ky++)for(uint64_t kx=0;kx<KW;kx++)for(uint64_t ci=0;ci<CI;ci+=16) {
     uint64_t kr=CI-ci;if(kr>16)kr=16;
#if DG_POOL_GATHER_BATCHING
     dg_pool_conv_gather_runs(A,b,lo,p,groupRows,H,width,CI,convOW,
      convSH,convSW,DH,DW,convPT,convPL,ky,kx,ci,kr);
#else
     for(uint64_t r=0;r<groupRows;r++) {
      uint64_t pos=p+r,oy=lo+pos/convOW,ox=pos%convOW;
      int64_t iy=(int64_t)(oy*convSH+ky*DH)-convPT;
      int64_t ix=(int64_t)(ox*convSW+kx*DW)-convPL;
      const int8_t *src=(const int8_t*)0;
      if(iy>=0 && ix>=0 && (uint64_t)iy<H && (uint64_t)ix<width)
       src=A+((b*H+(uint64_t)iy)*width+(uint64_t)ix)*CI+ci;
      /* LoadController source pointer zero supplies convolution padding. */
      DG_CMD(2,src,dg_desc((uint32_t)r,1,kr));
     }
#endif
     DG_CMD(1,W+((ky*KW+kx)*CI+ci)*CO+j,dg_desc(weightBase,kr,nr));
     uint32_t acc=(!first || D)?0xc0000000U:0x80000000U;
#if DG_POOL_STRIPE_BLOCKING
     for(uint64_t ii=0;ii<groupRows;ii+=16) {
      uint64_t mr=groupRows-ii;if(mr>16)mr=16;
      DG_CMD(6,dg_desc(weightBase,kr,nr),dg_desc(acc | (uint32_t)(p+ii),mr,nr));
      DG_CMD(4,dg_desc((uint32_t)ii,mr,kr),dg_desc(0xffffffffU,16,16));
     }
#else
     DG_CMD(6,dg_desc(weightBase,kr,nr),dg_desc(acc | (uint32_t)p,groupRows,nr));
     DG_CMD(4,dg_desc(0,groupRows,kr),dg_desc(0xffffffffU,16,16));
#endif
     first=0;
     dg_fence();
    }
   }
   dg_fence();
   uint64_t cfg=2 | ((uint64_t)relu<<2) | (SH<<4) | (PH<<6)
    | (upad<<8) | ((uint64_t)PL<<10) | (OW<<24)
    | (1ULL<<32) | (OW<<40) | (stripeH<<48) | (convOW<<56);
   /* N<=16 per channel tile satisfies StoreController's single-block
      pooling restriction, FIR230352-230360. CO is the DRAM pixel stride. */
   DG_CMD(0,cfg,((uint64_t)scale<<32) | CO);
   DG_CMD(3,C+((b*OH+y)*OW)*CO+j,dg_desc(0x80000000U,1,nr));
   dg_fence();
  }
 }
 dg_store_config(CO,0x3f800000U,0);
 dg_fence();
 return 1;
}
)C"; }
}
namespace backend {
/* geometry: batch,H,W,CI,CO,convOH,convOW,KH,KW,convSH,convSW,DH,DW,
   convPT,convPL. pool: OH,OW,PH,PW,SH,SW,PT,PL.
   An empty result means the ordinary guarded runtime must be used. */
std::string convPoolStaticRuntime(const std::string& helperName,
 const std::vector<int64_t>& geometry,const std::vector<int64_t>& pool,
 int64_t pad,uint32_t scale,unsigned relu,bool hasBias) {
 if(!compilerSettingBool("pool_static_control",false))return "";
 if(geometry.size()!=15 || pool.size()!=8 || helperName.empty())return "";
 for(size_t i=0;i<helperName.size();i++) {
  char c=helperName[i];
  if(!((c>='a'&&c<='z') || (c>='A'&&c<='Z') || c=='_' || (i && c>='0'&&c<='9')))return "";
 }
 for(unsigned i=0;i<13;i++)if(geometry[i]<0 || geometry[i]>2147483647LL)return "";
 for(unsigned i=13;i<15;i++)if(geometry[i]<(-2147483647LL-1) || geometry[i]>2147483647LL)return "";
 for(unsigned i=0;i<6;i++)if(pool[i]<=0 || pool[i]>2147483647LL)return "";
 if(pool[6]<0 || pool[6]>3 || pool[7]<0 || pool[7]>3)return "";
 uint64_t batch=geometry[0],H=geometry[1],width=geometry[2],CI=geometry[3],CO=geometry[4],
  convOH=geometry[5],convOW=geometry[6],KH=geometry[7],KW=geometry[8],
  convSH=geometry[9],convSW=geometry[10],DH=geometry[11],DW=geometry[12],
  OH=pool[0],OW=pool[1],PH=pool[2],PW=pool[3],SH=pool[4],SW=pool[5];
 int64_t convPT=geometry[13],convPL=geometry[14],PT=pool[6],PL=pool[7];
 if(!batch || !H || !width || !CO || !convOH || !convOW || relu>1 || pad< -128 || pad>127)return "";
 if(convOW>255 || OW>255 || PH>3 || PH!=PW || SH>3 || SH!=SW)return "";
 if((scale&0x80000000U) || !(scale&0x7fffffffU) || (scale&0x7f800000U)==0x7f800000U)return "";
 if(OW*PH*PW>=32768 || (uint64_t)PT>=PH || (uint64_t)PL>=PW)return "";
 if((OH-1)*SH>=(uint64_t)PT+convOH || (OW-1)*SW>=(uint64_t)PL+convOW)return "";
 bool hasPadding=PT || PL || (OH-1)*SH+PH>convOH+(uint64_t)PT || (OW-1)*SW+PW>convOW+(uint64_t)PL;
 if(hasPadding && (!relu || pad>0))return "";
 struct Stripe {uint64_t begin,end,height,upad;bool zeroLo;};
 std::vector<Stripe> stripes;
 for(uint64_t y=0;y<OH;) {
  int64_t start=(int64_t)(y*SH)-PT;
  uint64_t lo=start>0?(uint64_t)start:0,hi=(uint64_t)(start+(int64_t)PH);
  if(hi>convOH)hi=convOH;
  if(hi<=lo || hi-lo>512/convOW)return "";
  uint64_t end=y+1;
  /* Every interior stripe has the same height/upad. Group this entire
     interval into one loop with literal initial/limit/step. Only clipped
     boundary stripes need their own (also literal-bound) loops. */
  if(start>=0 && (uint64_t)start+PH<=convOH) {
   uint64_t last=(convOH+(uint64_t)PT-PH)/SH;
   end=last+1;if(end>OH)end=OH;
  }
  stripes.push_back({y,end,hi-lo,start<0?(uint64_t)(-start):0,start<0});
  if(stripes.size()>8)return "";
  y=end;
 }
 const uint64_t rowBatch=compilerSettingBool("pool_stripe_blocking",false)?64:16;
 const uint64_t weightBase=rowBatch;
 /* A runtime run-length gather loop has a data-dependent step. Keep this
    specialization restricted to fixed-position gather; the ordinary runtime
    remains the safe fallback when run batching is selected. */
 if(compilerSettingBool("pool_gather_batching",false))return "";
 const bool gather=false;
 std::ostringstream out;
 unsigned operations=0;
 auto line=[&](const std::string& s){
  out<<s<<'\n';
  for(const char* token:{"DG_CMD(","dg_load_config(","dg_store_config(","dg_fence(","for(","if("}) {
   size_t at=0;while((at=s.find(token,at))!=std::string::npos) {++operations;++at;}
  }
 };
 auto u=[](uint64_t n){return std::to_string(n)+"ULL";};
 auto si=[](int64_t n){return std::to_string(n)+"LL";};
 line("DG_INLINE void "+helperName+"(const int8_t *A,const int8_t *W,const int32_t *D,int8_t *C) {");
 line("dg_fence(); DG_CMD(0,(1ULL<<16)|4,1ULL<<48);");
 line("dg_store_config("+u(CO)+",0x3f800000U,0);");
 line("for(uint64_t b=0;b<"+u(batch)+";b++) {");
 for(const auto& stripe:stripes) {
  const uint64_t rows=stripe.height*convOW;
  line("for(uint64_t y="+u(stripe.begin)+";y<"+u(stripe.end)+";y++) {");
  line("uint64_t lo="+(stripe.zeroLo?std::string("0ULL"):"y*"+u(SH)+"-"+u(PT))+";");
  auto channel=[&](uint64_t nr) {
   line("dg_load_config("+u(gather?convSW*CI:0)+",0,0,0x3f800000U);");
   line("dg_load_config("+u(CO)+",1,0,0x3f800000U);");
   line("dg_load_config(0,2,0,0x3f800000U);");
   for(uint64_t p=0;p<rows;p+=rowBatch) {
    uint64_t groupRows=rows-p;if(groupRows>rowBatch)groupRows=rowBatch;
    /* A known bias state avoids a runtime first-reduction flag. With no
       bias, hardware source-zero initializes the same accumulator to 0;
       all subsequent products use the same ky/kx/CI accumulation order. */
    for(uint64_t ii=0;ii<groupRows;ii+=16) {
     uint64_t mr=groupRows-ii;if(mr>16)mr=16;
     line("DG_CMD(14,"+std::string(hasBias?"D+j":"(const int32_t*)0")+",dg_desc("+u(0x80000000ULL+p+ii)+","+u(mr)+","+u(nr)+"));");
    }
    if(CI && KH && KW) {
     line("for(uint64_t ky=0;ky<"+u(KH)+";ky++)for(uint64_t kx=0;kx<"+u(KW)+";kx++) {");
     auto reduction=[&](uint64_t kr) {
      if(gather) {
       line("dg_pool_conv_gather_runs(A,b,lo,"+u(p)+","+u(groupRows)+","+u(H)+","+u(width)+","+u(CI)+","+u(convOW)+","+u(convSH)+","+u(convSW)+","+u(DH)+","+u(DW)+","+si(convPT)+","+si(convPL)+",ky,kx,ci,"+u(kr)+");");
      } else {
       line("for(uint64_t r=0;r<"+u(groupRows)+";r++) {");
       line("uint64_t pos="+u(p)+"+r,oy=lo+pos/"+u(convOW)+",ox=pos%"+u(convOW)+";");
       line("int64_t iy=(int64_t)(oy*"+u(convSH)+"+ky*"+u(DH)+")-("+si(convPT)+");");
       line("int64_t ix=(int64_t)(ox*"+u(convSW)+"+kx*"+u(DW)+")-("+si(convPL)+");");
       line("const int8_t *src=(const int8_t*)0;");
       line("if(iy>=0 && ix>=0 && (uint64_t)iy<"+u(H)+" && (uint64_t)ix<"+u(width)+")src=A+((b*"+u(H)+"+(uint64_t)iy)*"+u(width)+"+(uint64_t)ix)*"+u(CI)+"+ci;");
       line("DG_CMD(2,src,dg_desc((uint32_t)r,1,"+u(kr)+")); }");
      }
      line("DG_CMD(1,W+((ky*"+u(KW)+"+kx)*"+u(CI)+"+ci)*"+u(CO)+"+j,dg_desc("+u(weightBase)+","+u(kr)+","+u(nr)+"));");
      for(uint64_t ii=0;ii<groupRows;ii+=16) {
       uint64_t mr=groupRows-ii;if(mr>16)mr=16;
       line("DG_CMD(6,dg_desc("+u(weightBase)+","+u(kr)+","+u(nr)+"),dg_desc("+u(0xc0000000ULL+p+ii)+","+u(mr)+","+u(nr)+"));");
       line("DG_CMD(4,dg_desc("+u(ii)+","+u(mr)+","+u(kr)+"),dg_desc(0xffffffffU,16,16));");
      }
      line("dg_fence();");
     };
     if(CI>=16) {
      line("for(uint64_t ci=0;ci<"+u(CI/16*16)+";ci+=16) {");reduction(16);line("}");
     }
     if(CI%16) {
      line("{ uint64_t ci="+u(CI/16*16)+";");reduction(CI%16);line("}");
     }
     line("}");
    }
   }
   uint64_t cfg=2 | ((uint64_t)relu<<2) | (SH<<4) | (PH<<6) | (stripe.upad<<8) |
    ((uint64_t)PL<<10) | (OW<<24) | (1ULL<<32) | (OW<<40) | (stripe.height<<48) | (convOW<<56);
   line("dg_fence(); DG_CMD(0,"+u(cfg)+","+u(((uint64_t)scale<<32)|CO)+");");
   line("DG_CMD(3,C+((b*"+u(OH)+"+y)*"+u(OW)+")*"+u(CO)+"+j,dg_desc(0x80000000U,1,"+u(nr)+")); dg_fence();");
  };
  if(CO>=16) {
   line("for(uint64_t j=0;j<"+u(CO/16*16)+";j+=16) {");channel(16);line("}");
  }
  if(CO%16) {
   line("{ uint64_t j="+u(CO/16*16)+";");channel(CO%16);line("}");
  }
  line("}");
  if(operations>4096)return "";
 }
 line("} dg_store_config("+u(CO)+",0x3f800000U,0); dg_fence(); }");
 if(operations>4096)return "";
 return out.str();
}
}
