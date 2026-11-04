#include "compiler.h"
#include <cmath>
#include <iomanip>
namespace mc {
std::string deviceSupport(){return R"C(
/* Instruction fields and geometry from the supplied raw FIRRTL:
   RoccCommandRouter opcode 0x7b; Load/Store/ExecuteController packed fields. */
#define GM_INLINE static __attribute__((always_inline)) inline
#define GM_INS(F,A,B) __asm__ volatile(".insn r 0x7b, 3, " #F ", x0, %0, %1" :: "r"((unsigned long)(A)), "r"((unsigned long)(B)) : "memory")
GM_INLINE unsigned long gm_pack(unsigned addr,int cols,int rows){return (unsigned long)addr|((unsigned long)cols<<32)|((unsigned long)rows<<48);}
GM_INLINE unsigned gm_fbits(float f){union{float f;unsigned u;}v;v.f=f;return v.u;}
GM_INLINE void gm_fence(void){__asm__ volatile("fence":::"memory");}
/* Controller funct=7 flushes address translation; rs1 bit0=0 retries,
   rather than discarding, any faulted DMA. This is not a completion fence. */
GM_INLINE void gm_flush(void){GM_INS(7,0UL,0UL);}
GM_INLINE void gm_config_load(unsigned long stride,float scale,int shrink){GM_INS(0,((unsigned long)gm_fbits(scale)<<32)|(16UL<<16)|((unsigned long)shrink<<2)|1UL,stride);}
GM_INLINE void gm_mvin(const void*p,unsigned addr,int cols,int rows){GM_INS(2,p,gm_pack(addr,cols,rows));}
/* State0 wide i8 rows: config block_stride bits29:16 specify the
   scratch row distance between consecutive 16-column blocks. */
GM_INLINE void gm_config_load_blocks(unsigned long stride,unsigned block_stride){GM_INS(0,((unsigned long)gm_fbits(1.0f)<<32)|((unsigned long)block_stride<<16)|1UL,stride);}
GM_INLINE void gm_config_load_a(unsigned long stride){GM_INS(0,((unsigned long)gm_fbits(1.0f)<<32)|(16UL<<16)|(1UL<<3)|1UL,stride);}
GM_INLINE void gm_mvin_a(const void*p,unsigned addr,int cols,int rows){GM_INS(1,p,gm_pack(addr,cols,rows));}
GM_INLINE void gm_config_store(unsigned long stride,float scale,int relu){GM_INS(0,2UL|((unsigned long)relu<<2),((unsigned long)gm_fbits(scale)<<32)|stride);}
GM_INLINE void gm_mvout(void*p,unsigned addr,int cols,int rows){GM_INS(3,p,gm_pack(addr,cols,rows));}
GM_INLINE void gm_config_execute(void){GM_INS(0,(1UL<<16)|4UL,1UL<<48);}
GM_INLINE void gm_preload(unsigned long src,unsigned long dst){GM_INS(6,src,dst);}
GM_INLINE void gm_compute(unsigned long a,unsigned long b){GM_INS(4,a,b);}
/* The store controller's pool stride/size fields are two bits; reducing four
   rows as one 2x2 window gives a reusable exact signed-i8 maximum primitive. */
GM_INLINE void gm_pool4(const signed char*in,signed char*out,int cols){
 gm_config_load(cols,1.0f,1);gm_mvin(in,0x80000000U,cols,4);
 unsigned long cfg=2UL|(2UL<<4)|(2UL<<6)|(1UL<<24)|(1UL<<32)|(1UL<<40)|(2UL<<48)|(2UL<<56);
 GM_INS(0,cfg,((unsigned long)gm_fbits(1.0f)<<32)|(unsigned long)cols);
 gm_mvout(out,0x80000000U,cols,1);gm_fence();
}
/* Three is representable in both two-bit pooling fields. One 3x3 output
   visits accumulator rows 0..8 and commits only its final maximum. */
GM_INLINE void gm_pool9(const signed char*in,signed char*out,int cols){
 gm_config_load(cols,1.0f,1);gm_mvin(in,0x80000000U,cols,9);
 unsigned long cfg=2UL|(3UL<<4)|(3UL<<6)|(1UL<<24)|(1UL<<32)|(1UL<<40)|(3UL<<48)|(3UL<<56);
 GM_INS(0,cfg,((unsigned long)gm_fbits(1.0f)<<32)|(unsigned long)cols);
 gm_mvout(out,0x80000000U,cols,1);gm_fence();
}
)C";}
static std::string flit(double n){std::ostringstream s;s<<std::setprecision(9)<<(float)n;std::string v=s.str();if(v.find_first_of(".eE")==std::string::npos)v+=".0";return v+"f";}
void emitMatmul(Context&c,const std::string&A,const std::string&W,const std::string&Y,int64_t M,int64_t K,int64_t N,const Obj&attrs,const std::string&bias,bool transpose){
 const int id=c.serial++;std::string tag="mm"+std::to_string(id);
 auto &b=c.body;
 std::string dtype=str(attrs,"output_dtype","i32");bool narrow=dtype=="i8",pool=hasStage(attrs,"maxpool");
 bool relu=hasStage(attrs,"relu");
 double scale=hasStage(attrs,"acc_scale")?real(attrs,"acc_scale",1.0):1.0;
 if(hasStage(attrs,"requant")){
  if(num(attrs,"requant_shift")!=0)throw std::runtime_error("target cannot implement integer requant shift");
  scale=1.0;
 }
 std::string initial=str(attrs,"initial_tensor");bool scalarInit=attrs.getInteger("initial_value").has_value()&&num(attrs,"initial_value")!=0;
 bool init=!initial.empty()||scalarInit;
 if(scalarInit)c.globals<<"static const int "<<tag<<"_init[16]={";
 if(scalarInit){for(int i=0;i<16;i++)c.globals<<num(attrs,"initial_value")<<",";c.globals<<"};\n";}
 const Arr*scales=attrs.getArray("acc_scale_by_column");
 bool uniformTiles=scales!=nullptr;
 if(scales){
  if(scales->size()!=static_cast<size_t>(N))throw std::runtime_error("acc_scale_by_column count does not match output columns");
  for(int64_t j=0;j<N;j++)if((*scales)[j].getAsNumber()!=(*scales)[j/16*16].getAsNumber())uniformTiles=false;
 }
 // For a single M tile, retain its complete K slice and stream four W
 // column tiles together. Each output occupies a disjoint sixteen-row
 // accumulator slot; the final scratchpad bank needs only 64 streamed rows.
 if(M>0&&M<=16&&K>0&&K<=12288&&N>16&&!transpose&&!pool&&!init&&!scales&&
    !hasStage(attrs,"requant")&&(narrow||(dtype=="i32"&&scale==1.0&&!relu))){
  b<<"{/* Thin-M four-column DMA/accumulator block. */\n"
   <<"gm_config_execute();gm_config_load_a("<<K<<");\n"
   <<"for(long kk=0;kk<"<<K<<";kk+=16){int kr=("<<K<<"-kk<16?"<<K<<"-kk:16);gm_mvin_a(((const signed char*)("<<A<<"))+kk,(unsigned)((kk/16)*"<<M<<"),kr,"<<M<<");}\n"
   <<"gm_config_store("<<N*(narrow?1:4)<<","<<flit(scale)<<","<<relu<<");\n"
   <<"for(long nb=0;nb<"<<N<<";nb+=64){int nc=("<<N<<"-nb<64?"<<N<<"-nb:64);\n";
  if(!bias.empty())b<<"gm_config_load(0,1.0f,0);for(int cj=0;cj*16<nc;cj++){int nr=(nc-cj*16<16?nc-cj*16:16);gm_mvin(((const int*)("<<bias<<"))+nb+cj*16,0x80000000U+cj*16,nr,"<<M<<");}\n";
  b<<"gm_config_load_blocks("<<N<<",16);\n"
   <<"for(long kk=0;kk<"<<K<<";kk+=16){int kr=("<<K<<"-kk<16?"<<K<<"-kk:16);gm_mvin(((const signed char*)("<<W<<"))+kk*"<<N<<"+nb,12288,nc,kr);\n"
   <<"for(int cj=0;cj*16<nc;cj++){int nr=(nc-cj*16<16?nc-cj*16:16);gm_preload(gm_pack(12288+cj*16,nr,kr),gm_pack((kk"<<(bias.empty()?"!=0":">=0")<<"?0xc0000000U:0x80000000U)+cj*16,nr,"<<M<<"));\n"
   <<"gm_compute(gm_pack((unsigned)((kk/16)*"<<M<<"),kr,"<<M<<"),gm_pack(0xffffffffU,nr,"<<M<<"));}}\n"
   <<"for(int cj=0;cj*16<nc;cj++){int nr=(nc-cj*16<16?nc-cj*16:16);gm_mvout((("<<(narrow?"signed char":"int")<<"*)("<<Y<<"))+nb+cj*16,"<<(narrow?"0x80000000U":"0xa0000000U")<<"+cj*16,nr,"<<M<<");}\n"
   <<"}gm_fence();}\n";
  return;
 }
 std::string target=Y;
 // Four scratchpad banks each contain 4096 rows of sixteen bytes in the
 // supplied FIRRTL.  Reserve the last bank for the streamed operand and
 // retain a complete K slice of the operand with more inter-tile reuse.
 // A tiles need only mr rows (including the M=1 case); W tiles need kr rows.
 const int64_t mt=(M+15)/16,nt=(N+15)/16,kt=(K+15)/16;
 const bool fitA=kt*std::min<int64_t>(M,16)<=12288;
 const bool fitW=kt*16<=12288;
 const int64_t savedA=fitA&&nt>1?mt*(nt-1)*kt:0;
 const int64_t savedW=fitW&&mt>1?nt*(mt-1)*kt:0;
 // Search rectangular complete-K operand caches. Both operands reserve 16
 // rows per K tile, including tails, so addresses never depend on tail sizes.
 // Minimize exact DMA instruction count over partial boundary blocks.
 int64_t blockI=0,blockJ=0;
 int64_t bestLoads=2*mt*nt*kt-std::max(savedA,savedW);
 const int64_t cacheSlots=kt>0?16384/(16*kt):0;
 if(!transpose&&!pool&&!init&&(!scales||uniformTiles)&&mt>1&&nt>1){
  for(int64_t i=1;i<=mt&&i<cacheSlots;i++)
   for(int64_t j=1;j<=nt&&i+j<=cacheSlots;j++){
    // A is loaded once per M tile and N cache block. W loads up to four
    // adjacent column tiles per command; each N block restarts its grouping.
    // Count full blocks and the last partial block separately, then multiply
    // by the number of M cache blocks. All legal cache sizes are <=1024 slots.
    const int64_t nblocks=(nt+j-1)/j,mblocks=(mt+i-1)/i;
    const int64_t wgroups=(nt/j)*((j+3)/4)+((nt%j)+3)/4;
    const int64_t loads=kt*(mt*nblocks+mblocks*wgroups);
    if(loads<bestLoads){bestLoads=loads;blockI=i;blockJ=j;}
   }
 }
 const bool cache2d=blockI>0;
 const bool cacheA=!cache2d&&!transpose&&savedA>0&&savedA>=savedW;
 const bool cacheW=!cache2d&&!transpose&&savedW>0&&!cacheA;
 if(pool){target=tag+"_unpooled";c.globals<<"static signed char "<<target<<"["<<M*N<<"] __attribute__((aligned(64)));\n";}
 if(transpose)c.globals<<"static signed char "<<tag<<"_w[256] __attribute__((aligned(64)));\n";
 b<<"{\n gm_config_execute();gm_config_load_a("<<K<<");\n";
 if(scales){c.globals<<"static const float "<<tag<<"_scales["<<N<<"]={";for(auto&v:*scales)c.globals<<flit(v.getAsNumber().value_or(1.0))<<",";c.globals<<"};\n";}
 if(cache2d){
  const int64_t weightBase=blockI*kt*16;
  b<<"/* Complete-K two-operand cache: "<<blockI<<" x "<<blockJ<<" output tiles. */\n"
   <<"for(long bi=0;bi<"<<mt<<";bi+="<<blockI<<")for(long bj=0;bj<"<<nt<<";bj+="<<blockJ<<"){\n"
   <<"for(long ci=0;ci<"<<blockI<<"&&bi+ci<"<<mt<<";ci++){long mi=(bi+ci)*16;int mr=("<<M<<"-mi<16?"<<M<<"-mi:16);\n"
   <<"for(long ck=0;ck<"<<K<<";ck+=16){int cr=("<<K<<"-ck<16?"<<K<<"-ck:16);gm_mvin_a(((const signed char*)("<<A<<"))+mi*"<<K<<"+ck,(unsigned)((ci*"<<kt<<"+ck/16)*16),cr,mr);}}\n"
   // The native reader emits up to four i8 blocks per row (64 bytes).
   // A block_stride of kt*16 lands neighboring columns in their existing
   // complete-K cache slots; tail width never crosses the logical N bound.
   <<"gm_config_load_blocks("<<N<<","<<kt*16<<");for(long cj=0;cj<"<<blockJ<<"&&bj+cj<"<<nt<<";cj+=4){long nj=(bj+cj)*16;int nc=("<<blockJ<<"-cj<4?("<<blockJ<<"-cj)*16:64);if(nc>"<<N<<"-nj)nc="<<N<<"-nj;\n"
   <<"for(long ck=0;ck<"<<K<<";ck+=16){int cr=("<<K<<"-ck<16?"<<K<<"-ck:16);gm_mvin(((const signed char*)("<<W<<"))+ck*"<<N<<"+nj,(unsigned)("<<weightBase<<"+(cj*"<<kt<<"+ck/16)*16),nc,cr);}}\n"
   <<"for(long ci=0;ci<"<<blockI<<"&&bi+ci<"<<mt<<";ci++){long mi=(bi+ci)*16;int mr=("<<M<<"-mi<16?"<<M<<"-mi:16);\n"
   <<"for(long cj=0;cj<"<<blockJ<<"&&bj+cj<"<<nt<<";cj++){long nj=(bj+cj)*16;int nr=("<<N<<"-nj<16?"<<N<<"-nj:16);\n";
 }else if(cacheW){
  b<<"for(long nj=0;nj<"<<N<<";nj+=16){int nr=("<<N<<"-nj<16?"<<N<<"-nj:16);\ngm_config_load("<<N<<",1.0f,0);\nfor(long ck=0;ck<"<<K<<";ck+=16){int cr=("<<K<<"-ck<16?"<<K<<"-ck:16);gm_mvin(((const signed char*)("<<W<<"))+ck*"<<N<<"+nj,(unsigned)ck,nr,cr);}\nfor(long mi=0;mi<"<<M<<";mi+=16){int mr=("<<M<<"-mi<16?"<<M<<"-mi:16);\n";
 }else{
  b<<"for(long mi=0;mi<"<<M<<";mi+=16){int mr=("<<M<<"-mi<16?"<<M<<"-mi:16);\n";
  if(cacheA)b<<"for(long ck=0;ck<"<<K<<";ck+=16){int cr=("<<K<<"-ck<16?"<<K<<"-ck:16);gm_mvin_a(((const signed char*)("<<A<<"))+mi*"<<K<<"+ck,(unsigned)((ck/16)*mr),cr,mr);}\n";
  b<<"for(long nj=0;nj<"<<N<<";nj+=16){int nr=("<<N<<"-nj<16?"<<N<<"-nj:16);\n";
 }
 if(!initial.empty())b<<"gm_config_load("<<N*4<<",1.0f,0);gm_mvin(((const int*)("<<c.ptr(initial)<<"))+mi*"<<N<<"+nj,0x80000000U,nr,mr);\n";
 if(scalarInit)b<<"gm_config_load(0,1.0f,0);gm_mvin("<<tag<<"_init,"<<(initial.empty()?"0x80000000U":"0xc0000000U")<<",nr,mr);\n";
 if(!bias.empty())b<<"gm_config_load(0,1.0f,0);gm_mvin(((const int*)("<<bias<<"))+nj,"<<(init?"0xc0000000U":"0x80000000U")<<",nr,mr);\n";
 if(!cacheW&&!cache2d)b<<"gm_config_load("<<(transpose?16:N)<<",1.0f,0);\n";
 b<<"for(long kk=0;kk<"<<K<<";kk+=16){int kr=("<<K<<"-kk<16?"<<K<<"-kk:16);\n";
 if(!cacheA&&!cache2d)b<<"gm_mvin_a(((const signed char*)("<<A<<"))+mi*"<<K<<"+kk,"<<(cacheW?12288:0)<<",kr,mr);\n";
 if(transpose){b<<"gm_fence();for(int x=0;x<kr;x++)for(int y=0;y<nr;y++)"<<tag<<"_w[x*16+y]=((const signed char*)("<<W<<"))[(nj+y)*"<<K<<"+kk+x];\ngm_mvin("<<tag<<"_w,16,nr,kr);\n";}
 else if(!cacheW&&!cache2d)b<<"gm_mvin(((const signed char*)("<<W<<"))+kk*"<<N<<"+nj,"<<(cacheA?12288:16)<<",nr,kr);\n";
 b<<"gm_preload(gm_pack("<<(cache2d?"(unsigned)("+std::to_string(blockI*kt*16)+"+(cj*"+std::to_string(kt)+"+kk/16)*16)":cacheW?"(unsigned)kk":cacheA?"12288":"16")<<",nr,kr),gm_pack((kk"<<(bias.empty()&&!init?"!=0":">=0")<<"?0xc0000000U:0x80000000U),nr,mr));\ngm_compute(gm_pack("<<(cache2d?"(unsigned)((ci*"+std::to_string(kt)+"+kk/16)*16)":cacheA?"(unsigned)((kk/16)*mr)":cacheW?"12288":"0")<<",kr,mr),gm_pack(0xffffffffU,nr,mr));\n";
 // The transpose tile lives in DRAM and is reused; wait before overwriting it.
 if(transpose)b<<"gm_fence();\n";
 b<<"}\n";
 if(scales&&!uniformTiles){
   // Local-address offsets select rows, not columns. Materialize the full
   // tile once, then gather each column with a 64-byte DMA row stride and
   // scatter its scaled i8 readout directly with the output's N-byte stride.
   // The ROB tracks overlap on accumulator column zero while columns queue;
   // the existing tile-final fence completes every column before _full reuse.
   c.globals<<"static int "<<tag<<"_full[256] __attribute__((aligned(64)));\n";
   b<<"gm_config_store(64,1.0f,0);gm_mvout("<<tag<<"_full,0xa0000000U,nr,mr);gm_fence();\ngm_config_load(64,1.0f,0);\nfor(int j=0;j<nr;j++){gm_mvin("<<tag<<"_full+j,0x80000000U,1,mr);gm_config_store("<<N<<","<<tag<<"_scales[nj+j],"<<relu<<");gm_mvout(((signed char*)("<<target<<"))+mi*"<<N<<"+nj+j,0x80000000U,1,mr); }\n";
 }else{
   b<<"gm_config_store("<<N*(narrow?1:4)<<","<<(uniformTiles?tag+"_scales[nj]":flit(scale))<<","<<relu<<");gm_mvout((("<<(narrow?"signed char":"int")<<"*)("<<target<<"))+mi*"<<N<<"+nj,"<<(narrow?"0x80000000U":"0xa0000000U")<<",nr,mr);\n";
 }
 // ReservationStation tracks overlapping accumulator read/write and
 // scratchpad load/execute ranges until completion. With no CPU temporary
 // consumer, queued output tiles therefore need only one completion fence.
 const bool queueTiles=!transpose&&!pool&&(!scales||uniformTiles);
 if(!queueTiles)b<<"gm_fence();\n";
 b<<(cache2d?"}}}\n":"}}\n");
 if(queueTiles)b<<"gm_fence();\n";
 b<<"}\n";
 if(pool)emitPool(c,target,Y,M,N,attrs,tag);
}
}
