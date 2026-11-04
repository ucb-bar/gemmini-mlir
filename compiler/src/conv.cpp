#include "compiler.h"
#include <limits>
#include <cmath>
#include <iomanip>

namespace mc {
namespace {
std::vector<int64_t> geometry(const Obj &a, const char *name,
                              std::initializer_list<int64_t> defaults) {
  auto v = ints(a, name);
  if (v.empty()) v.assign(defaults);
  if (v.size() != defaults.size())
    throw std::runtime_error(std::string("CONV2D: invalid ") + name);
  return v;
}
int64_t product(int64_t a, int64_t b) {
  if (a < 1 || b < 1 || a > std::numeric_limits<int64_t>::max() / b)
    throw std::runtime_error("CONV2D: invalid or overflowing shape");
  return a*b;
}
bool emitDmaConv(Context &c,const std::string &src,const std::string &weight,
                 const std::string &dst,const std::string &bias,
                 int64_t batches,int64_t H,int64_t W,int64_t C,int64_t Ho,
                 int64_t Wo,int64_t kh,int64_t kw,int64_t co,
                 const std::vector<int64_t>&stride,const std::vector<int64_t>&padding,
                 const std::vector<int64_t>&dilation,const Obj&attrs){
 const int64_t K=kh*kw*C;
 const auto dtype=str(attrs,"output_dtype");
 if(K>12288||hasStage(attrs,"maxpool")||hasStage(attrs,"requant")||
    attrs.getArray("acc_scale_by_column")||!str(attrs,"initial_tensor").empty()||
    attrs.getInteger("initial_value").has_value()||(dtype!="i8"&&dtype!="i32"))return false;
 int last=-1;
 if(auto stages=attrs.getArray("epilogue"))for(const auto &v:*stages){
  auto name=v.getAsString();if(!name)return false;
  int order=(*name=="bias"||*name=="bias_add")?0:*name=="acc_scale"?1:*name=="relu"?2:-1;
  if(order<0||order<=last)return false;last=order;
 }
 const double scale=hasStage(attrs,"acc_scale")?real(attrs,"acc_scale",1.0):1.0;
 if(!std::isfinite(scale)||!std::isfinite((float)scale))return false;
 std::ostringstream lit;lit<<std::setprecision(9)<<(float)scale;
 std::string scaleLit=lit.str();if(scaleLit.find_first_of(".eE")==std::string::npos)scaleLit+=".0";scaleLit+="f";
 if(C%16==0){
 // Compare complete-K cache schedules using public geometry. A linear tile's
 // DMA may split at an output-row boundary; each partial padding tile also
 // needs one ZeroWriter command per K tile. These are issue-count estimates.
 const int64_t totalM=batches*Ho*Wo,mt=(totalM+15)/16,nt=(co+15)/16,kt=K/16;
 const int64_t oldMt=batches*Ho*((Wo+15)/16);
 // Count actual padded DMA commands and source-row segments, rather than
 // treating every padded tile as zero-filled or every linear tile as split.
 // One source segment has fixed address/range setup; its C/16 chunk loads
 // share that setup in emitted code below.
 struct LoadWork{long double moves=0,segments=0,windows=0,validSegments=0,bytes=0;};
 auto tileWork=[&](int64_t mi,int64_t mr,bool linear){
  LoadWork w;
  for(int64_t ph=0;ph<kh;ph++)for(int64_t pw=0;pw<kw;pw++){
   bool clear=false;int64_t valid=0;
   for(int64_t rr=0;rr<mr;){
    const int64_t pos=mi+rr,oh=linear?(pos/Wo)%Ho:mi/Wo;
    const int64_t ow=linear?pos%Wo:mi%Wo;
    const int64_t count=std::min<int64_t>(mr-rr,Wo-ow);
    const int64_t ih=oh*stride[0]+ph*dilation[0]-padding[0];
    const int64_t iw=ow*stride[1]+pw*dilation[1]-padding[1];
    clear|=ih<0||ih>=H||iw<0||iw+(count-1)*stride[1]>=W;
    const int64_t lo=std::min<int64_t>(count,iw<0?(-iw+stride[1]-1)/stride[1]:0);
    const int64_t hi=std::min<int64_t>(count,W-iw<=0?0:(W-iw+stride[1]-1)/stride[1]);
    if(ih>=0&&ih<H&&hi>lo){valid++;w.bytes+=(hi-lo)*C;}
    w.segments++;rr+=count;
   }
   w.moves+=(valid+(clear?1:0))*(C/16);w.validSegments+=valid;w.windows++;
  }
  return w;
 };
 LoadWork linearWork,rowWork;
 auto addWork=[](LoadWork &a,const LoadWork &b){a.moves+=b.moves;a.segments+=b.segments;a.windows+=b.windows;a.validSegments+=b.validSegments;a.bytes+=b.bytes;};
 for(int64_t mi=0;mi<totalM;mi+=16)addWork(linearWork,tileWork(mi,std::min<int64_t>(16,totalM-mi),true));
 for(int64_t oh=0;oh<Ho;oh++)for(int64_t ow=0;ow<Wo;ow+=16)addWork(rowWork,tileWork(oh*Wo+ow,std::min<int64_t>(16,Wo-ow),false));
 rowWork.moves*=batches;rowWork.segments*=batches;rowWork.windows*=batches;rowWork.validSegments*=batches;rowWork.bytes*=batches;
 // Issue-work units count DMA/preload/compute instructions and address/control
 // expressions. They are a planner heuristic, not a cycle estimate. Segmented
 // gather costs four scalar array/pointer expressions per valid channel chunk;
 // expensive coordinate divisions occur once per segment, not once per chunk.
 const long double rowCpu=rowWork.segments*16+rowWork.windows*4+rowWork.moves*2;
 const long double linearCpu=linearWork.segments*16+linearWork.windows*4+
    linearWork.validSegments*(C/16)*4+linearWork.moves*2;
 const long double oldComputes=(long double)nt*oldMt*kt;
 const long double newComputes=(long double)nt*mt*kt;
 // Valid A rows determine DRAM traffic: zero DMA never reads DRAM, and a
 // short spatial tile transfers fewer bytes per command than a full W tile.
 // Require retained-operand schedules to avoid increasing total source bytes.
 // This conservative guard avoids trading narrow padded A reads for repeated
 // full-height W streams solely because the latter issue fewer commands.
 const long double weightBytes=(long double)K*co;
 const long double baselineBytes=nt*rowWork.bytes+weightBytes;
 long double best=(long double)nt*kt+nt*(rowWork.moves+rowCpu)+2*oldComputes+baselineBytes/16;
 int schedule=0;int64_t blockI=1,blockJ=1;
 if(K<=12288){
  const long double bytes=linearWork.bytes+mt*weightBytes;
  const long double cost=linearWork.moves+linearCpu+(long double)mt*nt*kt+2*newComputes+bytes/16;
  if(bytes<=baselineBytes&&cost<best){best=cost;schedule=1;}
 }
 const int64_t slots=16384/K;
 for(int64_t i=1;i<=std::min<int64_t>(mt,slots-1);i++)
  for(int64_t j=1;j<=std::min<int64_t>(nt,slots-i);j++){
   const long double bytes=((nt+j-1)/j)*linearWork.bytes+((mt+i-1)/i)*weightBytes;
   const long double cost=((nt+j-1)/j)*(linearWork.moves+linearCpu)+
                    (long double)((mt+i-1)/i)*nt*kt+2*newComputes+bytes/16;
   if(bytes<=baselineBytes&&cost<best){best=cost;schedule=2;blockI=i;blockJ=j;}
  }
 if(schedule){
  auto &b=c.body;
  // mi,mr name one linear output tile; abase names its first scratch row.
  // Split at Wo boundaries before deriving batch and input row. CPU code
  // computes only addresses/ranges; ZeroWriter supplies every padding value.
  auto loadA=[&](){
   b<<"for(long ph=0;ph<"<<kh<<";ph++)for(long pw=0;pw<"<<kw<<";pw++){int clear=0,segments=0;const signed char*ap[16];int ao[16],an[16];\n"
    <<"for(int rr=0;rr<mr;){long pos=mi+rr,nb=pos/"<<Ho*Wo<<",oh=(pos/"<<Wo<<")%"<<Ho<<",ow=pos%"<<Wo<<";int count=("<<Wo<<"-ow<mr-rr?"<<Wo<<"-ow:mr-rr);\n"
    <<"long ih=oh*"<<stride[0]<<"+ph*"<<dilation[0]<<"-"<<padding[0]<<",iw=ow*"<<stride[1]<<"+pw*"<<dilation[1]<<"-"<<padding[1]<<";\n"
    <<"if(ih<0||ih>="<<H<<"||iw<0||iw+(count-1)*"<<stride[1]<<">="<<W<<")clear=1;\n"
    <<"int lo=iw<0?(-iw+"<<stride[1]-1<<")/"<<stride[1]<<":0;if(lo>count)lo=count;long end="<<W<<"-iw;int hi=end<=0?0:(end+"<<stride[1]-1<<")/"<<stride[1]<<";if(hi>count)hi=count;if(hi<lo)hi=lo;\n"
    <<"if(ih>=0&&ih<"<<H<<"&&hi>lo){ap[segments]=((const signed char*)("<<src<<"))+((nb*"<<H<<"+ih)*"<<W<<"+iw+lo*"<<stride[1]<<")*"<<C<<";ao[segments]=rr+lo;an[segments]=hi-lo;segments++;}rr+=count;}\n"
    <<"for(long ci=0;ci<"<<C<<";ci+=16){long ck=(ph*"<<kw<<"+pw)*"<<C<<"+ci;unsigned ad=(unsigned)(abase+ck);if(clear)gm_mvin_a((const void*)0,ad,16,mr);\n"
    <<"for(int seg=0;seg<segments;seg++)gm_mvin_a(ap[seg]+ci,ad+ao[seg],16,an[seg]);}}\n";
  };
  b<<"{/* Direct DMA complete-K cache schedule "<<schedule<<", block "<<blockI<<"x"<<blockJ<<". */\n"
   <<"gm_config_execute();gm_config_load_a("<<C*stride[1]<<");gm_config_store("<<co*(dtype=="i8"?1:4)<<","<<scaleLit<<","<<hasStage(attrs,"relu")<<");\n";
  if(schedule==1){
   b<<"for(long mi=0;mi<"<<totalM<<";mi+=16){int mr=("<<totalM<<"-mi<16?"<<totalM<<"-mi:16);long abase=0;\n";
   loadA();
   b<<"for(long nj=0;nj<"<<co<<";nj+=16){int nr=("<<co<<"-nj<16?"<<co<<"-nj:16);\n";
  }else{
   b<<"for(long mb=0;mb<"<<totalM<<";mb+="<<blockI*16<<")for(long nbj=0;nbj<"<<co<<";nbj+="<<blockJ*16<<"){\n"
    <<"for(long ai=0;ai<"<<blockI<<"&&mb+ai*16<"<<totalM<<";ai++){long mi=mb+ai*16,abase=ai*"<<K<<";int mr=("<<totalM<<"-mi<16?"<<totalM<<"-mi:16);\n";
   loadA();b<<"}\ngm_config_load("<<co<<",1.0f,0);\n"
    <<"for(long wj=0;wj<"<<blockJ<<"&&nbj+wj*16<"<<co<<";wj++){long nj=nbj+wj*16;int nr=("<<co<<"-nj<16?"<<co<<"-nj:16);for(long ck=0;ck<"<<K<<";ck+=16)gm_mvin(((const signed char*)("<<weight<<"))+ck*"<<co<<"+nj,(unsigned)("<<blockI*K<<"+wj*"<<K<<"+ck),nr,16);}\n"
    <<"for(long ai=0;ai<"<<blockI<<"&&mb+ai*16<"<<totalM<<";ai++)for(long wj=0;wj<"<<blockJ<<"&&nbj+wj*16<"<<co<<";wj++){long mi=mb+ai*16,nj=nbj+wj*16;int mr=("<<totalM<<"-mi<16?"<<totalM<<"-mi:16),nr=("<<co<<"-nj<16?"<<co<<"-nj:16);\n";
  }
  if(!bias.empty())b<<"gm_config_load(0,1.0f,0);gm_mvin(((const int*)("<<bias<<"))+nj,0x80000000U,nr,mr);\n";
  if(schedule==1)b<<"gm_config_load("<<co<<",1.0f,0);\n";
  b<<"for(long ck=0;ck<"<<K<<";ck+=16){\n";
  if(schedule==1)b<<"gm_mvin(((const signed char*)("<<weight<<"))+ck*"<<co<<"+nj,12288,nr,16);\n";
  b<<"gm_preload(gm_pack("<<(schedule==1?"12288":"(unsigned)("+std::to_string(blockI*K)+"+wj*"+std::to_string(K)+"+ck)")<<",nr,16),gm_pack((ck"<<(bias.empty()?"!=0":">=0")<<"?0xc0000000U:0x80000000U),nr,mr));gm_compute(gm_pack("<<(schedule==1?"(unsigned)ck":"(unsigned)(ai*"+std::to_string(K)+"+ck)")<<",16,mr),gm_pack(0xffffffffU,nr,mr));}\n"
   <<"gm_mvout((("<<(dtype=="i8"?"signed char":"int")<<"*)("<<dst<<"))+mi*"<<co<<"+nj,"<<(dtype=="i8"?"0x80000000U":"0xa0000000U")<<",nr,mr);\n"
   <<"}}gm_fence();}\n";
  return true;
 }
 }

 // A row-oriented schedule retains several full-K W tiles and uses disjoint
 // 16-row accumulator slots. Every input window DMA feeds all retained tiles.
 const int64_t rowJ=C%16==0?std::min<int64_t>({(co+15)/16,12288/K,4}):1;
 if(rowJ>1){
  auto &b=c.body;
  b<<"{/* Direct row window DMA with "<<rowJ<<" retained W tiles. */\n"
   <<"gm_config_execute();gm_config_load_a("<<C*stride[1]<<");gm_config_store("<<co*(dtype=="i8"?1:4)<<","<<scaleLit<<","<<hasStage(attrs,"relu")<<");\n"
   <<"for(long nbj=0;nbj<"<<co<<";nbj+="<<rowJ*16<<"){gm_config_load_blocks("<<co<<","<<K<<");int nc=("<<co<<"-nbj<"<<rowJ*16<<"?"<<co<<"-nbj:"<<rowJ*16<<");\n"
   <<"for(long ck=0;ck<"<<K<<";ck+=16)gm_mvin(((const signed char*)("<<weight<<"))+ck*"<<co<<"+nbj,(unsigned)ck,nc,16);\n"
   <<"for(long nb=0;nb<"<<batches<<";nb++)for(long oh=0;oh<"<<Ho<<";oh++)for(long ow=0;ow<"<<Wo<<";ow+=16){int mr=("<<Wo<<"-ow<16?"<<Wo<<"-ow:16);\n";
  if(!bias.empty())b<<"gm_config_load(0,1.0f,0);for(long wj=0;wj<"<<rowJ<<"&&nbj+wj*16<"<<co<<";wj++){long nj=nbj+wj*16;int nr=("<<co<<"-nj<16?"<<co<<"-nj:16);gm_mvin(((const int*)("<<bias<<"))+nj,0x80000000U+(unsigned)wj*16,nr,mr);}\n";
  b<<"for(long ph=0;ph<"<<kh<<";ph++)for(long pw=0;pw<"<<kw<<";pw++){long ih=oh*"<<stride[0]<<"+ph*"<<dilation[0]<<"-"<<padding[0]<<",iw=ow*"<<stride[1]<<"+pw*"<<dilation[1]<<"-"<<padding[1]<<";\n"
   <<"int lo=iw<0?(-iw+"<<stride[1]-1<<")/"<<stride[1]<<":0;if(lo>mr)lo=mr;long end="<<W<<"-iw;int hi=end<=0?0:(end+"<<stride[1]-1<<")/"<<stride[1]<<";if(hi>mr)hi=mr;if(hi<lo)hi=lo;\n"
   <<"for(long ci=0;ci<"<<C<<";ci+=16){if(ih<0||ih>="<<H<<"||lo!=0||hi!=mr)gm_mvin_a((const void*)0,12288,16,mr);\n"
   <<"if(ih>=0&&ih<"<<H<<"&&hi>lo)gm_mvin_a(((const signed char*)("<<src<<"))+((nb*"<<H<<"+ih)*"<<W<<"+iw+lo*"<<stride[1]<<")*"<<C<<"+ci,12288+lo,16,hi-lo);long ck=(ph*"<<kw<<"+pw)*"<<C<<"+ci;\n"
   <<"for(long wj=0;wj<"<<rowJ<<"&&nbj+wj*16<"<<co<<";wj++){long nj=nbj+wj*16;int nr=("<<co<<"-nj<16?"<<co<<"-nj:16);gm_preload(gm_pack((unsigned)(wj*"<<K<<"+ck),nr,16),gm_pack((ck"<<(bias.empty()?"!=0":">=0")<<"?0xc0000000U:0x80000000U)+(unsigned)wj*16,nr,mr));gm_compute(gm_pack(12288,16,mr),gm_pack(0xffffffffU,nr,mr));}}}\n"
   <<"for(long wj=0;wj<"<<rowJ<<"&&nbj+wj*16<"<<co<<";wj++){long nj=nbj+wj*16;int nr=("<<co<<"-nj<16?"<<co<<"-nj:16);gm_mvout((("<<(dtype=="i8"?"signed char":"int")<<"*)("<<dst<<"))+((nb*"<<Ho<<"+oh)*"<<Wo<<"+ow)*"<<co<<"+nj,"<<(dtype=="i8"?"0x80000000U":"0xa0000000U")<<"+(unsigned)wj*16,nr,mr);}\n"
   <<"}}gm_fence();}\n";
  return true;
 }
 // Raw FIRRTL LoadController (229229,229261..229281): rs1=0 selects
 // all_zeros, otherwise each DMA row reads rs1+row_counter*strides[state_id].
 // Scratchpad (186128..186132) routes all_zeros to ZeroWriter, bypassing DRAM.
 // Cache full K rows of W below12288; reserve16 rows from12288 for A.
 const int64_t interiorStart=std::min<int64_t>(Wo,(padding[1]+stride[1]-1)/stride[1]);
 const int64_t interiorEnd=std::max<int64_t>(interiorStart,std::min<int64_t>(Wo,
    W-kw+padding[1]>=0?(W-kw+padding[1])/stride[1]+1:0));
 const int64_t segmentSteps=kw*((C+15)/16),rowSteps=(kw*C+15)/16;
 const int64_t oldSteps=((Wo+15)/16)*segmentSteps;
 const int64_t groupedSteps=((interiorStart+15)/16+(Wo-interiorEnd+15)/16)*segmentSteps+
    ((interiorEnd-interiorStart+15)/16)*rowSteps;
 // Splitting at padding boundaries adds output tiles. Use row groups only
 // when their exact per-output-row compute/preload count still decreases.
 const bool rowGroups=dilation[1]==1&&interiorEnd>interiorStart&&groupedSteps<oldSteps;
 auto &b=c.body;
 b<<"{ /* Direct NHWC window DMA; tensor arithmetic remains on the array. */\n"
  <<"gm_config_execute();gm_config_load_a("<<C*stride[1]<<");\n"
  <<"for(long nj=0;nj<"<<co<<";nj+=16){int nr=("<<co<<"-nj<16?"<<co<<"-nj:16);\n"
  <<"gm_config_load("<<co<<",1.0f,0);for(long ck=0;ck<"<<K<<";ck+=16){int kr=("<<K<<"-ck<16?"<<K<<"-ck:16);gm_mvin(((const signed char*)("<<weight<<"))+ck*"<<co<<"+nj,(unsigned)ck,nr,kr); }\n"
  <<"for(long nb=0;nb<"<<batches<<";nb++)for(long oh=0;oh<"<<Ho<<";oh++)for(long ow=0;ow<"<<Wo<<";){int mr=("<<Wo<<"-ow<16?"<<Wo<<"-ow:16);\n";
 if(rowGroups)b<<"int interior=ow>="<<interiorStart<<"&&ow<"<<interiorEnd<<";long stop=ow<"<<interiorStart<<"?"<<interiorStart<<":interior?"<<interiorEnd<<":"<<Wo<<";if(mr>stop-ow)mr=stop-ow;\n";
 if(!bias.empty())b<<"gm_config_load(0,1.0f,0);gm_mvin(((const int*)("<<bias<<"))+nj,0x80000000U,nr,mr);\n";
 if(rowGroups){
  b<<"if(interior){for(long ph=0;ph<"<<kh<<";ph++){long ih=oh*"<<stride[0]<<"+ph*"<<dilation[0]<<"-"<<padding[0]<<";long iw=ow*"<<stride[1]<<"-"<<padding[1]<<";\n"
   <<"for(long z=0;z<"<<kw*C<<";z+=16){int kr=("<<kw*C<<"-z<16?"<<kw*C<<"-z:16);if(ih<0||ih>="<<H<<")gm_mvin_a((const void*)0,12288,kr,mr);else gm_mvin_a(((const signed char*)("<<src<<"))+((nb*"<<H<<"+ih)*"<<W<<"+iw)*"<<C<<"+z,12288,kr,mr);\n"
   <<"long ck=ph*"<<kw*C<<"+z;gm_preload(gm_pack((unsigned)ck,nr,kr),gm_pack((ck"<<(bias.empty()?"!=0":">=0")<<"?0xc0000000U:0x80000000U),nr,mr));gm_compute(gm_pack(12288,kr,mr),gm_pack(0xffffffffU,nr,mr));}}}else{\n";
 }
 b<<"for(long ph=0;ph<"<<kh<<";ph++)for(long pw=0;pw<"<<kw<<";pw++){long ih=oh*"<<stride[0]<<"+ph*"<<dilation[0]<<"-"<<padding[0]<<";long iw=ow*"<<stride[1]<<"+pw*"<<dilation[1]<<"-"<<padding[1]<<";\n"
  <<"int lo=iw<0?(-iw+"<<stride[1]-1<<")/"<<stride[1]<<":0;if(lo>mr)lo=mr;long end="<<W<<"-iw;int hi=end<=0?0:(end+"<<stride[1]-1<<")/"<<stride[1]<<";if(hi>mr)hi=mr;if(hi<lo)hi=lo;\n"
  <<"for(long ci=0;ci<"<<C<<";ci+=16){int kr=("<<C<<"-ci<16?"<<C<<"-ci:16);\n"
  <<"if(ih<0||ih>="<<H<<"||lo!=0||hi!=mr)gm_mvin_a((const void*)0,12288,kr,mr);\n"
  <<"if(ih>=0&&ih<"<<H<<"&&hi>lo)gm_mvin_a(((const signed char*)("<<src<<"))+((nb*"<<H<<"+ih)*"<<W<<"+iw+lo*"<<stride[1]<<")*"<<C<<"+ci,12288+lo,kr,hi-lo);\n"
  <<"long ck=(ph*"<<kw<<"+pw)*"<<C<<"+ci;gm_preload(gm_pack((unsigned)ck,nr,kr),gm_pack((ck"<<(bias.empty()?"!=0":">=0")<<"?0xc0000000U:0x80000000U),nr,mr));gm_compute(gm_pack(12288,kr,mr),gm_pack(0xffffffffU,nr,mr));\n}}\n";
 if(rowGroups)b<<"}\n";
 b<<"gm_config_store("<<co*(dtype=="i8"?1:4)<<","<<scaleLit<<","<<hasStage(attrs,"relu")<<");gm_mvout((("<<(dtype=="i8"?"signed char":"int")<<"*)("<<dst<<"))+((nb*"<<Ho<<"+oh)*"<<Wo<<"+ow)*"<<co<<"+nj,"<<(dtype=="i8"?"0x80000000U":"0xa0000000U")<<",nr,mr);\n"
  <<"ow+=mr;}}gm_fence();}\n";
 return true;
}

}

void emitConv(Context &c, const Obj &command) {
  const Obj &operands = command.getObject("operands") ? *command.getObject("operands") : command;
  const Obj &attributes = command.getObject("attributes") ? *command.getObject("attributes") : command;
  auto inputName = str(operands, "ifm", str(operands, "src"));
  auto weightName = str(operands, "weight");
  auto outputName = str(operands, "dst");
  const Tensor &input = c.tensor(inputName);
  const Tensor &weight = c.tensor(weightName);
  const Tensor &output = c.tensor(outputName);
  if (str(attributes, "layout", "nhwc") != "nhwc")
    throw std::runtime_error("CONV2D: only nhwc layout is defined");
  if (input.shape.size() != 4 || input.dtype != "i8" || weight.dtype != "i8")
    throw std::runtime_error("CONV2D: expected rank-four NHWC i8 input and i8 weight");
  auto kernel = ints(attributes, "kernel");
  if (kernel.size() != 4)
    throw std::runtime_error("CONV2D: kernel [kh, kw, ci, co] is required");
  auto stride = geometry(attributes, "stride", {1, 1});
  auto padding = geometry(attributes, "padding", {0, 0, 0, 0});
  auto dilation = geometry(attributes, "dilation", {1, 1});
  for (auto x : input.shape) if (x < 1) throw std::runtime_error("CONV2D: nonpositive input shape");
  for (auto x : kernel) if (x < 1) throw std::runtime_error("CONV2D: nonpositive kernel");
  for (auto x : stride) if (x < 1) throw std::runtime_error("CONV2D: nonpositive stride");
  for (auto x : dilation) if (x < 1) throw std::runtime_error("CONV2D: nonpositive dilation");
  for (auto x : padding) if (x < 0) throw std::runtime_error("CONV2D: negative padding");
  const int64_t batches = input.shape[0], H = input.shape[1], W = input.shape[2], C = input.shape[3];
  const int64_t kh = kernel[0], kw = kernel[1], co = kernel[3];
  if (kernel[2] != C) throw std::runtime_error("CONV2D: kernel input channels disagree with input");
  const int64_t eh = product(dilation[0], kh-1 ? kh-1 : 1) * (kh > 1) + 1;
  const int64_t ew = product(dilation[1], kw-1 ? kw-1 : 1) * (kw > 1) + 1;
  const int64_t nh = H + padding[0] + padding[2] - eh;
  const int64_t nw = W + padding[1] + padding[3] - ew;
  if (nh < 0 || nw < 0) throw std::runtime_error("CONV2D: kernel exceeds padded input");
  const int64_t Ho = nh / stride[0] + 1, Wo = nw / stride[1] + 1;
  const int64_t M = product(Ho, Wo), K = product(product(kh, kw), C);
  if (weight.shape != std::vector<int64_t>{K, co})
    throw std::runtime_error("CONV2D: expected pre-im2col weight [kh*kw*ci,co]");
  int64_t rowsOut = M;
  if (hasStage(attributes, "maxpool")) {
    if (ints(attributes, "pool_in_dims") != std::vector<int64_t>{Ho, Wo})
      throw std::runtime_error("CONV2D: pool_in_dims disagrees with convolution geometry");
    auto ps = ints(attributes, "pool_size"), st = ints(attributes, "pool_stride");
    auto pd = geometry(attributes, "pool_padding", {0, 0, 0, 0});
    if (ps.size() != 2 || st.size() != 2 || ps[0] < 1 || ps[1] < 1 || st[0] < 1 || st[1] < 1)
      throw std::runtime_error("CONV2D: positive pool_size and pool_stride are required");
    for (auto x : pd) if (x < 0) throw std::runtime_error("CONV2D: negative pool padding");
    int64_t ph = Ho + pd[0] + pd[2] - ps[0], pw = Wo + pd[1] + pd[3] - ps[1];
    if (ph < 0 || pw < 0) throw std::runtime_error("CONV2D: pool window exceeds padded input");
    rowsOut = product(ph / st[0] + 1, pw / st[1] + 1);
  }
  if (output.count() != product(product(batches, rowsOut), co))
    throw std::runtime_error("CONV2D: output shape disagrees with convolution geometry");
  Obj attrs;
  for (const auto &entry : attributes) attrs[entry.first] = copy(entry.second);
  if (!attrs.getString("output_dtype")) attrs["output_dtype"] = output.dtype;
  if (str(attrs, "output_dtype") != output.dtype)
    throw std::runtime_error("CONV2D: output_dtype disagrees with destination");
  std::string bias;
  if (hasStage(attrs, "bias") || hasStage(attrs, "bias_add")) {
    auto bn = str(attrs, "bias", str(operands, "bias"));
    if (bn.empty()) throw std::runtime_error("CONV2D: bias stage requires bias tensor");
    const Tensor &b = c.tensor(bn);
    if (b.shape != std::vector<int64_t>{co} || b.dtype != "i32")
      throw std::runtime_error("CONV2D: expected i32 channel bias");
    bias = c.ptr(bn);
  }
  // Pooling consumes the narrowed convolution result. Preserve every earlier
  // epilogue stage and only split a final maxpool from the accelerator conv.
  if(hasStage(attrs,"maxpool")&&output.dtype=="i8"){
    const Arr *stages=attrs.getArray("epilogue");
    bool finalPool=stages&&!stages->empty()&&stages->back().getAsString()&&
      *stages->back().getAsString()=="maxpool";
    if(finalPool){
      Obj convAttrs;for(const auto &entry:attrs)convAttrs[entry.first]=copy(entry.second);
      Arr convStages;
      for(size_t i=0;i+1<stages->size();i++)convStages.push_back(copy((*stages)[i]));
      convAttrs["epilogue"]=std::move(convStages);
      const std::string poolTag="cvpool"+std::to_string(c.serial);
      const std::string prepool=poolTag+"_unpooled";
      // emitDmaConv checks all eligibility guards before writing output code.
      if(emitDmaConv(c,c.ptr(inputName),c.ptr(weightName),prepool,bias,
                     batches,H,W,C,Ho,Wo,kh,kw,co,stride,padding,dilation,convAttrs)){
        ++c.serial;
        c.globals<<"static signed char "<<prepool<<"["<<product(product(batches,M),co)
                 <<"] __attribute__((aligned(64)));\n";
        emitPool(c,prepool,c.ptr(outputName),product(batches,M),co,attrs,poolTag);
        return;
      }
    }
  }
  if(emitDmaConv(c,c.ptr(inputName),c.ptr(weightName),c.ptr(outputName),bias,
                 batches,H,W,C,Ho,Wo,kh,kw,co,stride,padding,dilation,attrs))return;
  const std::string tag = "cv" + std::to_string(c.serial++);
  const std::string bn = tag + "b";
  const std::string src = c.ptr(inputName), rhs = c.ptr(weightName), dst = c.ptr(outputName);
  // A pointwise unit-stride convolution is already an ordinary row-major matrix.
  const bool direct = kh == 1 && kw == 1 && stride[0] == 1 && stride[1] == 1 &&
    padding == std::vector<int64_t>{0, 0, 0, 0};
  std::string lhs;
  if (direct) {
    lhs = "(" + src + "+" + bn + "*" + std::to_string(product(M, K)) + ")";
  } else {
    lhs = tag + "col";
    c.globals << "static signed char " << lhs << "[" << product(M, K) << "] __attribute__((aligned(64)));\n";
  }
  c.body << "{ for (long " << bn << "=0; " << bn << "<" << batches << "; ++" << bn << ") {\n";
  if (!direct) {
    // Host work only forms the contraction's input windows; all tensor arithmetic
    // and narrowing remain in emitMatmul's accelerator instructions.
    c.body << "for(long oh=0;oh<" << Ho << ";++oh) for(long ow=0;ow<" << Wo
      << ";++ow) for(long kh=0;kh<" << kh << ";++kh) for(long kw=0;kw<" << kw
      << ";++kw) { long ih=oh*" << stride[0] << "+kh*" << dilation[0] << "-" << padding[0]
      << ";long iw=ow*" << stride[1] << "+kw*" << dilation[1] << "-" << padding[1] << ";\n"
      << "long cd=((oh*" << Wo << "+ow)*" << kh << "*" << kw << "+kh*" << kw << "+kw)*" << C << ";\n"
      << "if(ih>=0 && ih<" << H << " && iw>=0 && iw<" << W << ") {\n"
      << "long si=((" << bn << "*" << H << "+ih)*" << W << "+iw)*" << C << ";\n"
      << "for(long ci=0;ci<" << C << ";++ci) " << lhs << "[cd+ci]=" << src << "[si+ci];\n"
      << "}else for(long ci=0;ci<" << C << ";++ci) " << lhs << "[cd+ci]=0; }\n";
  }
  emitMatmul(c, lhs, rhs, "(" + dst + "+" + bn + "*" + std::to_string(product(rowsOut, co)) + ")",
             M, K, co, attrs, bias);
  c.body << "} }\n";
}
}
