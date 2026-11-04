#include "compiler.h"
namespace mc {
void emitPool(Context&c,const std::string&src,const std::string&dst,int64_t M,int64_t N,const Obj&attrs,const std::string&tag){
 auto &b=c.body;
 const std::string &target=src,&Y=dst;
  auto dims=ints(attrs,"pool_in_dims"),sz=ints(attrs,"pool_size"),st=ints(attrs,"pool_stride"),pad=ints(attrs,"pool_padding");
  if(dims.size()!=2||sz.size()!=2||st.size()!=2)throw std::runtime_error("maxpool requires dimensions, size and stride");
  if(pad.empty())pad={0,0,0,0};int64_t H=dims[0],WW=dims[1],OH=(H+pad[0]+pad[2]-sz[0])/st[0]+1,OW=(WW+pad[1]+pad[3]-sz[1])/st[1]+1;
  if((sz[0]==2||sz[0]==3)&&sz[1]==sz[0]){
   // Load each spatial row segment directly into contiguous accumulator rows.
   // Explicit edge initialization preserves arbitrary signed-i8 pad values;
   // the ZeroWriter's zero fill cannot represent the usual minimum pad.
   const int64_t side=sz[0],nodes=side*side;
   c.globals<<"static const signed char "<<tag<<"_pool_pad[16] __attribute__((aligned(64)))={";
   for(int i=0;i<16;i++)c.globals<<num(attrs,"pool_pad_value",0)<<",";
   c.globals<<"};\n";
   b<<"{for(long bn=0;bn<"<<M/(H*WW)<<";bn++)for(long oh=0;oh<"<<OH<<";oh++)for(long ow=0;ow<"<<OW<<";ow++)for(long cj=0;cj<"<<N<<";cj+=16){int nc=("<<N<<"-cj<16?"<<N<<"-cj:16);long ih0=oh*"<<st[0]<<"-"<<pad[0]<<",iw0=ow*"<<st[1]<<"-"<<pad[1]<<";\n";
   b<<"if(ih0<0||ih0+"<<side<<">"<<H<<"||iw0<0||iw0+"<<side<<">"<<WW<<"){gm_config_load(0,1.0f,1);gm_mvin("<<tag<<"_pool_pad,0x80000000U,nc,"<<nodes<<");}\n";
   b<<"gm_config_load("<<N<<",1.0f,1);for(int ph=0;ph<"<<side<<";ph++){long ih=ih0+ph;if(ih>=0&&ih<"<<H<<"){int pbegin=iw0<0?-iw0:0,pend=(iw0+"<<side<<">"<<WW<<"?"<<WW<<"-iw0:"<<side<<");if(pbegin<pend)gm_mvin("<<target<<"+((bn*"<<H<<"+ih)*"<<WW<<"+iw0+pbegin)*"<<N<<"+cj,0x80000000U+ph*"<<side<<"+pbegin,nc,pend-pbegin);}}\n";
   // Native pooling records the entire accumulator bank as its read range
   // in ReservationStation, so later window loads wait for this readout.
   b<<"GM_INS(0,2UL|("<<side<<"UL<<4)|("<<side<<"UL<<6)|(1UL<<24)|(1UL<<32)|(1UL<<40)|("<<side<<"UL<<48)|("<<side<<"UL<<56),((unsigned long)gm_fbits(1.0f)<<32)|"<<N<<"UL);gm_mvout(((signed char*)("<<Y<<"))+((bn*"<<OH<<"+oh)*"<<OW<<"+ow)*"<<N<<"+cj,0x80000000U,nc,1);\n";
   b<<"}gm_fence();}\n";
   return;
  }
  int64_t nodes=sz[0]*sz[1],alloc=((nodes+3)/4)*4;
  c.globals<<"static signed char "<<tag<<"_pool["<<alloc*16<<"] __attribute__((aligned(64)));\nstatic signed char "<<tag<<"_reduce["<<alloc*16<<"] __attribute__((aligned(64)));\n";
  b<<"{for(long bn=0;bn<"<<M/(H*WW)<<";bn++)for(long oh=0;oh<"<<OH<<";oh++)for(long ow=0;ow<"<<OW<<";ow++)for(long cj=0;cj<"<<N<<";cj+=16){int nc=("<<N<<"-cj<16?"<<N<<"-cj:16);\n";
  b<<"for(int ph=0;ph<"<<sz[0]<<";ph++)for(int pw=0;pw<"<<sz[1]<<";pw++){long ih=oh*"<<st[0]<<"+ph-"<<pad[0]<<",iw=ow*"<<st[1]<<"+pw-"<<pad[1]<<";for(int ch=0;ch<nc;ch++)"<<tag<<"_pool[(ph*"<<sz[1]<<"+pw)*nc+ch]=(ih<0||ih>="<<H<<"||iw<0||iw>="<<WW<<")?"<<num(attrs,"pool_pad_value",0)<<":"<<target<<"[((bn*"<<H<<"+ih)*"<<WW<<"+iw)*"<<N<<"+cj+ch];}\n";
  if(sz[0]==3&&sz[1]==3){
   b<<"gm_pool9("<<tag<<"_pool,"<<tag<<"_reduce,nc);for(int ch=0;ch<nc;ch++)((signed char*)("<<Y<<"))[((bn*"<<OH<<"+oh)*"<<OW<<"+ow)*"<<N<<"+cj+ch]="<<tag<<"_reduce[ch];}}\n";
  }else b<<"int count="<<nodes<<";while(count>1){int groups=(count+3)/4;for(int p=count;p<groups*4;p++)for(int ch=0;ch<nc;ch++)"<<tag<<"_pool[p*nc+ch]=-128;for(int g=0;g<groups;g++)gm_pool4("<<tag<<"_pool+g*4*nc,"<<tag<<"_reduce+g*nc,nc);for(int p=0;p<groups*nc;p++)"<<tag<<"_pool[p]="<<tag<<"_reduce[p];count=groups;}for(int ch=0;ch<nc;ch++)((signed char*)("<<Y<<"))[((bn*"<<OH<<"+oh)*"<<OW<<"+ow)*"<<N<<"+cj+ch]="<<tag<<"_pool[ch];}}\n";
}
}
