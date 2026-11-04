#include "compiler.h"
#include <cmath>
#include <iomanip>
namespace mc {
namespace {
std::string fl(float x) { std::ostringstream s; s<<std::setprecision(9)<<std::scientific<<x<<"f"; return s.str(); }
const Obj& obj(const Obj&c,const char*k) {auto p=c.getObject(k);if(!p)throw std::runtime_error(std::string("missing command ")+k);return *p;}
int bytes(const std::string &d){if(d=="i8")return 1;if(d=="i32")return 4;throw std::runtime_error("unsupported movement/add dtype "+d);}
void checkMatrix(const Tensor&t) {if(t.shape.empty()||t.count()<1)throw std::runtime_error("expected positive static tensor");}
}
// Queue DMA-only tiles: the reservation station tracks accumulator read/write
// overlap, and one final fence completes all stores before a CPU consumer.
void emitReadout(Context&c,const std::string&src,const std::string&dst,int64_t rows,int64_t cols,const Obj&attrs){
 if(rows<1||cols<1)throw std::runtime_error("accelerator readout requires a positive static matrix");
 const std::string dtype=str(attrs,"output_dtype","i8");
 const float scale=(float)real(attrs,"acc_scale",1.0);const bool relu=hasStage(attrs,"relu");
 if(dtype!="i8"&&dtype!="i32")throw std::runtime_error("accelerator readout supports i8 or i32 output");
 if(!std::isfinite(scale))throw std::runtime_error("accelerator readout requires finite scale");
 // The full-width accumulator DMA path bypasses both the narrowing scaler
 // and activation. Reject a request it cannot execute rather than silently
 // treating full-width readout as a scaled or activated operation.
 if(dtype=="i32"&&(scale!=1.0f||relu))throw std::runtime_error("full i32 readout cannot scale or activate");
 std::string id=std::to_string(c.serial++),mi="rdm"+id,ni="rdn"+id,mr="rdrows"+id,nr="rdcols"+id;
 c.body<<"{\ngm_config_execute();\ngm_config_load("<<cols*4<<",1.0f,0);\ngm_config_store("<<cols*(dtype=="i8"?1:4)<<","<<fl(scale)<<","<<relu<<");\n";
 c.body<<"for(long "<<mi<<"=0;"<<mi<<"<"<<rows<<";"<<mi<<"+=16)for(long "<<ni<<"=0;"<<ni<<"<"<<cols<<";"<<ni<<"+=16){\n";
 c.body<<"int "<<mr<<"=("<<rows<<"-"<<mi<<"<16?"<<rows<<"-"<<mi<<":16),"<<nr<<"=("<<cols<<"-"<<ni<<"<16?"<<cols<<"-"<<ni<<":16);\n";
 c.body<<"gm_mvin(((const int*)("<<src<<"))+"<<mi<<"*"<<cols<<"+"<<ni<<",0x80000000u,"<<nr<<","<<mr<<");\n";
 c.body<<"gm_mvout((("<<(dtype=="i8"?"signed char":"int")<<"*)("<<dst<<"))+"<<mi<<"*"<<cols<<"+"<<ni<<","<<(dtype=="i8"?"0x80000000u":"0xa0000000u")<<","<<nr<<","<<mr<<");\n}\ngm_fence();\n}\n";
}
bool emitOther(Context&c,const Obj&command){
 const std::string op=str(command,"opcode");
 if(op=="ROW_MAJOR_VIEW")return true;
 const Obj&o=obj(command,"operands"); const Obj empty; const Obj&a=command.getObject("attributes")?*command.getObject("attributes"):empty;
 if(op=="ATTENTION_QK"||op=="ATTENTION_PV"){
  const bool tr=op=="ATTENTION_QK";std::string an=str(o,tr?"q":"p"),wn=str(o,tr?"k":"v"),yn=str(o,"dst");
  auto &A=c.tensor(an),&W=c.tensor(wn);if(A.shape.size()!=2||W.shape.size()!=2)throw std::runtime_error("attention expects rank two");
  int64_t M=A.shape[0],K=A.shape[1],N=tr?W.shape[0]:W.shape[1];if(K!=(tr?W.shape[1]:W.shape[0]))throw std::runtime_error("attention contraction mismatch");
  emitMatmul(c,c.ptr(an),c.ptr(wn),c.ptr(yn),M,K,N,a,"",tr);return true;
 }
 if(op=="BATCHED_MATMUL"){
  std::string an=str(o,"a"),wn=str(o,"w"),yn=str(o,"dst");auto A=c.tensor(an),W=c.tensor(wn),Y=c.tensor(yn);size_t r=A.shape.size();
  if(r<3||W.shape.size()!=r||Y.shape.size()!=r)throw std::runtime_error("batched matmul rank mismatch");
  int64_t B=1;for(size_t i=0;i+2<r;++i){if(A.shape[i]!=W.shape[i]||A.shape[i]!=Y.shape[i]||A.shape[i]<1)throw std::runtime_error("batched matmul batch mismatch");B*=A.shape[i];}
  int64_t M=A.shape[r-2],K=A.shape[r-1],N=W.shape[r-1];if(W.shape[r-2]!=K||Y.shape[r-2]!=M||Y.shape[r-1]!=N)throw std::runtime_error("batched contraction mismatch");
  std::string b="b"+std::to_string(c.serial++);c.body<<"for(long "<<b<<"=0;"<<b<<"<"<<B<<";++"<<b<<"){\n";
  emitMatmul(c,"("+c.ptr(an)+"+"+b+"*"+std::to_string(M*K)+")","("+c.ptr(wn)+"+"+b+"*"+std::to_string(K*N)+")","("+c.ptr(yn)+"+"+b+"*"+std::to_string(M*N)+")",M,K,N,a);
  c.body<<"}\n";return true;
 }
 if(op!="MOVEMENT"&&op!="BIAS_ADD"&&op!="RESIDUAL_ADD")return false;
 std::string xn=str(o,op=="RESIDUAL_ADD"?"lhs":"src"),yn=str(o,"dst");auto X=c.tensor(xn),Y=c.tensor(yn);checkMatrix(X);checkMatrix(Y);
 if(X.count()!=Y.count())throw std::runtime_error("elementwise output shape mismatch");
 int64_t N=X.shape.back(),M=X.count()/N;int xb=bytes(X.dtype),yb=bytes(Y.dtype);bool residual=op=="RESIDUAL_ADD",bias=op=="BIAS_ADD";
 std::string zn=residual?str(o,"rhs"):(bias?str(o,"bias"):"");Tensor Z;if(!zn.empty())Z=c.tensor(zn);
 if(bias&&(X.dtype!="i32"||Z.dtype!="i32"||Y.dtype!="i32"||Z.count()!=N))throw std::runtime_error("bias add requires matching i32 accumulator tensors");
 if(residual&&(X.dtype!="i8"||Z.dtype!="i8"||Y.dtype!="i8"||Z.count()!=X.count()))throw std::runtime_error("residual requires matching i8 tensors");
 if(!residual&&!(Y.dtype=="i32"||(X.dtype=="i8"&&Y.dtype=="i8")))throw std::runtime_error("movement may only preserve or widen values");
 float sx=residual?(float)real(a,"lhs_scale"):1.0f,sz=residual?(float)real(a,"rhs_scale"):1.0f;
 if(residual){if(!std::isfinite(sx)||!std::isfinite(sz)||sx<0||sz<0||sx>256||sz>256)throw std::runtime_error("residual scale unsupported");
  if(num(a,"bound_lsb",-1)<1&&(sx!=std::floor(sx)||sz!=std::floor(sz)))throw std::runtime_error("exact noninteger residual requires joint floating rounding");}
 std::string id=std::to_string(c.serial++),mi="rm"+id,ni="rn"+id,rr="rr"+id,cc="cc"+id;
 c.body<<"{\ngm_config_execute();\ngm_config_store("<<N*yb<<",1.0f,"<<(hasStage(a,"relu")?1:0)<<");\n";
 c.body<<"for(long "<<mi<<"=0;"<<mi<<"<"<<M<<";"<<mi<<"+=16)for(long "<<ni<<"=0;"<<ni<<"<"<<N<<";"<<ni<<"+=16){\n";
 c.body<<"int "<<rr<<"=("<<M<<"-"<<mi<<"<16?"<<M<<"-"<<mi<<":16),"<<cc<<"=("<<N<<"-"<<ni<<"<16?"<<N<<"-"<<ni<<":16);\n";
 bool initialized=false;
 auto load=[&](const std::string&p,float scale,int b,int64_t stride){
  c.body<<"gm_config_load("<<stride<<","<<fl(scale)<<","<<(b==1?1:0)<<");\ngm_mvin("<<p<<","<<(initialized?"0xc0000000u":"0x80000000u")<<","<<cc<<","<<rr<<");\n";initialized=true;
 };
 auto scaled=[&](const std::string&p,float scale){int n=(int)std::floor(scale);for(int i=0;i<n;++i)load(p,1.0f,1,N);float f=scale-n;if(f>0)load(p,f,1,N);};
 std::string xp="("+c.ptr(xn)+"+"+mi+"*"+std::to_string(N)+"+"+ni+")";
 if(residual){scaled(xp,sx);scaled("("+c.ptr(zn)+"+"+mi+"*"+std::to_string(N)+"+"+ni+")",sz);if(!initialized)load(xp,0.0f,1,N);}
 else{load(xp,1.0f,xb,N*xb);if(bias)load("("+c.ptr(zn)+"+"+ni+")",1.0f,4,0);}
 c.body<<"gm_mvout("<<c.ptr(yn)<<"+"<<mi<<"*"<<N<<"+"<<ni<<","<<(yb==4?"0xa0000000u":"0x80000000u")<<","<<cc<<","<<rr<<");\n}\ngm_fence();\n}\n";return true;
}
}
