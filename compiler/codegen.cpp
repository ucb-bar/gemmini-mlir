#include "compiler.h"
#include <sstream>
#include <iomanip>
#include <set>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <optional>
namespace backend {
std::string convPoolStaticRuntime(const std::string&,const std::vector<int64_t>&,const std::vector<int64_t>&,int64_t,uint32_t,unsigned,bool);
namespace {
std::string ct(std::string d){if(d=="i1")return "unsigned char";if(d=="i8")return "signed char";if(d=="i16")return "short";if(d=="i32")return "int";if(d=="i64"||d=="index")return "long";if(d=="f32")return "float";if(d=="f64")return "double";throw std::runtime_error("C dtype "+d);}
std::string ut(std::string d){if(d=="i1"||d=="i8")return "unsigned char";if(d=="i16")return "unsigned short";if(d=="i32")return "unsigned int";return "unsigned long";}
std::string val(const Value&v){if(auto x=v.getAsInteger())return std::to_string(*x);if(auto x=v.getAsNumber()){if(std::isnan(*x))return "__builtin_nan(\"\")";if(std::isinf(*x))return *x<0?"(-__builtin_inf())":"__builtin_inf()";std::ostringstream o;o<<std::setprecision(17)<<*x;std::string s=o.str();if(s.find_first_of(".eE")==std::string::npos)s+=".0";return s;}if(auto x=v.getAsBoolean())return *x?"1":"0";throw std::runtime_error("non scalar literal");}
std::vector<int64_t> vec(const Object&o,llvm::StringRef k,std::vector<int64_t>d={}){auto*a=o.getArray(k);if(!a)return d;std::vector<int64_t>r;for(auto&v:*a)r.push_back(*v.getAsInteger());return r;}
std::string ix(const Value&v){if(auto i=v.getAsInteger())return "d"+std::to_string(*i);if(auto s=v.getAsString())return s->str();auto*o=v.getAsObject();if(!o)throw std::runtime_error("invalid affine expression");if(auto c=o->getInteger("constant"))return std::to_string(*c);std::string op=str(*o,"op");auto*l=o->get("lhs"),*r=o->get("rhs");if(op=="dim")return "d"+std::to_string(integer(*o,"index"));if(!l||!r)throw std::runtime_error("affine AST missing operands");std::string a=ix(*l);if(op=="floordiv"||op=="ceildiv"||op=="mod"){auto*constant=r->getAsObject();auto divisor=constant?constant->getInteger("constant"):std::nullopt;if(!divisor||*divisor<=0)throw std::runtime_error("affine "+op+" requires a positive constant divisor");return std::string(op=="floordiv"?"cg_affine_floordiv(":op=="ceildiv"?"cg_affine_ceildiv(":"cg_affine_mod(")+a+","+std::to_string(*divisor)+"L)";}if(op!="add"&&op!="mul"&&op!="sub")throw std::runtime_error("unsupported affine operator "+op);return "("+a+(op=="add"?"+":op=="mul"?"*":"-")+ix(*r)+")";}

std::string flat(const Array&m,const std::vector<int64_t>&s){if(m.empty())return "0";if(m.size()!=s.size())throw std::runtime_error("index map rank mismatch");std::string a="0";for(size_t i=0;i<s.size();i++)a="("+a+"*"+std::to_string(s[i])+"+"+ix(m[i])+")";return a;}
}
std::string emitC(const Object &program,const Object &cb){
 std::ostringstream decl,body;std::map<std::string,const Object*>objs;std::map<std::string,std::string>ptr;std::map<std::string,int>groups;std::map<std::string,int>uses;std::set<std::string>observable;
 auto*meta=cb.getObject("_compiler");auto*mapping=meta?meta->getObject("values"):nullptr;
 const Object*params=cb.getObject("params");const Object*abi=params?params->getObject("kernel_abi"):nullptr;if(!abi)abi=cb.getObject("kernel_abi");if(!abi)throw std::runtime_error("missing kernel ABI");
 auto&tensors=*cb.getObject("tensors");int ai=0;std::map<std::string,std::string>tensorPtr;std::string signature="void kernel_entry(";
 for(auto&av:arr(*abi,"args")){auto&a=*av.getAsObject();auto name=str(a,"tensor");auto*t=tensors.getObject(name);if(!t)throw std::runtime_error("missing tensor "+name);if(ai)signature+=",";std::string p="p"+std::to_string(ai++);signature+="void* "+p;tensorPtr[name]="(("+ct(str(*t,"dtype"))+"*)"+p+")";}signature+=")";
 if(cb.getObject("declined"))return "__attribute__((noinline,used)) void kernel_complete(void){__asm__ volatile(\"fence\" ::: \"memory\");}\n"+signature+"{}\n";
 for(auto&iv:arr(program,"inputs")){auto&i=*iv.getAsObject();auto id=str(i,"id");objs[id]=&i;auto p=tensorPtr.find(str(i,"name"));if(p==tensorPtr.end())throw std::runtime_error("missing input pointer "+str(i,"name"));ptr[id]=p->second;}
 auto&exprs=arr(program,"expressions");int ei=0;for(auto&ev:exprs){auto&e=*ev.getAsObject();auto id=str(e,"id");objs[id]=&e;std::string p;if(mapping)if(auto n=mapping->getString(id))if(tensorPtr.count(n->str()))p=tensorPtr[n->str()];if(p.empty()){p="v"+std::to_string(ei);decl<<"static "<<ct(str(e,"dtype"))<<" "<<p<<"["<<std::max<int64_t>(1,elements(e))<<"] __attribute__((aligned(64)));\n";}ptr[id]=p;ei++;if(auto*oa=e.getArray("operands"))for(auto&x:*oa)uses[x.getAsString()->str()]++;}
 for(auto&ov:arr(program,"outputs"))observable.insert(str(*ov.getAsObject(),"value"));
 if(meta)if(auto*ga=meta->getArray("groups")){int g=0;for(auto&gv:*ga){auto&go=*gv.getAsObject();if(auto*es=go.getArray("expression_indices"))for(auto&i:*es){auto n=*i.getAsInteger();if(n>=0&&n<(long)exprs.size())groups[str(*exprs[n].getAsObject(),"id")]=g;}g++;}}
 std::map<std::string,size_t> position;for(size_t i=0;i<exprs.size();i++)position[str(*exprs[i].getAsObject(),"id")]=i;
 auto sourceLine=[&](const Object&e){if(auto index=e.getInteger("source_op_index"))return *index+100;if(auto*a=e.getArray("source_op_indices"))if(!a->empty())return *a->back().getAsInteger()+100;return (int64_t)position.at(str(e,"id"))+100;};
 std::set<std::string> fusedSkip;
 struct Fusion{std::string dst,bias,biasId,precast;const Object*pool=nullptr;float scale=1;bool relu=false;bool out32=false;bool devicePool=false;};std::map<std::string,Fusion> fusion;
 for(size_t i=0;i<exprs.size();i++){auto&e=*exprs[i].getAsObject();auto op=str(e,"operator");if(op!="matmul"&&op!="conv2d")continue;
   auto&initialOperands=arr(e,"operands");if(initialOperands.size()<2||str(*objs.at(initialOperands[0].getAsString()->str()),"dtype")!="i8"||str(*objs.at(initialOperands[1].getAsString()->str()),"dtype")!="i8")continue;
   std::string cur=str(e,"id");Fusion f;std::vector<std::string> chain;bool scaled=false,activated=false,biased=false;
   while(uses[cur]==1&&!observable.count(cur)){
     size_t next=exprs.size();for(size_t j=i+1;j<exprs.size();j++){auto&x=*exprs[j].getAsObject();if(auto*oa=x.getArray("operands"))for(auto&v:*oa)if(v.getAsString()&&v.getAsString()->str()==cur)next=j;}
     if(next==exprs.size())break;auto&x=*exprs[next].getAsObject();auto xo=str(x,"operator"),xid=str(x,"id");if(groups.count(cur)&&groups.count(xid)&&groups[cur]!=groups[xid])break;const Object ep;const Object&xp=x.getObject("parameters")?*x.getObject("parameters"):ep;auto&oa=arr(x,"operands");bool accept=false;
     if(xo=="add_bias"&&!f.pool&&!scaled&&!activated&&!biased&&oa.size()==2){f.biasId=oa[1].getAsString()->str();f.bias=ptr.at(f.biasId);biased=true;accept=true;}
     if(xo=="scale_round"&&!f.pool&&!scaled&&!activated&&!xp.getArray("factors")){if(auto v=xp.getNumber("factor")){if(*v>0){f.scale=(float)*v;scaled=true;accept=true;}}}
     if(xo=="relu"&&!f.pool&&!activated){f.relu=true;activated=true;accept=true;}
     if(xo=="max_pool"&&!f.pool&&str(x,"dtype")=="i32"){f.pool=&x;accept=true;}
     if(xo=="cast"&&str(x,"dtype")=="i8"){chain.push_back(xid);f.dst=ptr.at(xid);if(f.pool){if(meta)if(auto*map=meta->getObject("pool_device_fused"))f.devicePool=map->getBoolean(str(e,"id")).value_or(false);if(!f.devicePool){if(meta)if(auto*map=meta->getObject("pool_precast"))if(auto name=map->getString(str(e,"id")))if(tensorPtr.count(name->str()))f.precast=tensorPtr[name->str()];if(f.precast.empty())throw std::runtime_error("missing declared pool precast scratch for "+str(e,"id"));}}fusion[str(e,"id")]=f;for(auto&z:chain)fusedSkip.insert(z);break;}
     if(!accept)break;chain.push_back(xid);cur=xid;
   }
   if(!fusion.count(str(e,"id"))&&!f.pool&&biased&&!scaled&&!activated&&!chain.empty()){f.dst=ptr.at(cur);f.out32=true;fusion[str(e,"id")]=f;for(auto&z:chain)fusedSkip.insert(z);}
 }
 std::map<std::string,Fusion> readoutFusion;std::map<std::string,std::string> columnScales;std::map<std::string,std::vector<uint32_t>> literalColumnScales;
 for(size_t i=0;i<exprs.size();i++){auto&e=*exprs[i].getAsObject();auto root=str(e,"operator"),id=str(e,"id");if(root!="scale_round"&&root!="relu")continue;if(fusedSkip.count(id))continue;auto&oa=arr(e,"operands");if(oa.empty()||str(*objs.at(oa[0].getAsString()->str()),"dtype")!="i32")continue;Fusion f;f.relu=root=="relu";const Object ep;auto&pr=e.getObject("parameters")?*e.getObject("parameters"):ep;std::string cols;
 if(root=="scale_round"){if(auto*fs=pr.getArray("factors")){bool okay=true;for(auto&v:*fs){auto z=v.getAsNumber();if(!z||*z<=0)okay=false;}if(!okay)continue;cols="cs"+std::to_string(i);auto es=shape(e);bool literal=compilerSettingBool("constant_column_scales",false)&&!es.empty()&&es.back()>0&&es.back()<=64&&fs->size()==size_t(es.back());std::vector<uint32_t> scaleBits;for(auto&value:*fs){auto number=*value.getAsNumber();float z=(float)number;if(!std::isfinite(number)||!std::isfinite(z)||z<=0)literal=false;uint32_t b;std::memcpy(&b,&z,4);scaleBits.push_back(b);}if(literal)literalColumnScales[id]=scaleBits;else{decl<<"static const uint32_t "<<cols<<"[]={";for(size_t j=0;j<scaleBits.size();j++){if(j)decl<<",";decl<<scaleBits[j]<<"U";}decl<<"};\n";}}else{auto z=pr.getNumber("factor");if(!z||*z<=0)continue;f.scale=(float)*z;}}
 std::string cur=id;std::vector<std::string> chain;while(uses[cur]==1&&!observable.count(cur)){size_t next=exprs.size();for(size_t j=i+1;j<exprs.size();j++){auto&x=*exprs[j].getAsObject();if(auto*aa=x.getArray("operands"))if(!aa->empty()&&(*aa)[0].getAsString()&&(*aa)[0].getAsString()->str()==cur)next=j;}if(next==exprs.size())break;auto&x=*exprs[next].getAsObject();auto xid=str(x,"id"),xo=str(x,"operator");if(groups.count(cur)&&groups.count(xid)&&groups[cur]!=groups[xid])break;if(xo=="relu"&&!f.pool&&!f.relu){f.relu=true;chain.push_back(xid);cur=xid;continue;}if(xo=="max_pool"&&!f.pool&&cols.empty()&&str(x,"dtype")=="i32"){f.pool=&x;chain.push_back(xid);cur=xid;continue;}if(xo=="cast"&&str(x,"dtype")=="i8"){f.dst=ptr.at(xid);if(f.pool){if(meta)if(auto*map=meta->getObject("pool_device_fused"))f.devicePool=map->getBoolean(id).value_or(false);if(!f.devicePool){if(meta)if(auto*map=meta->getObject("pool_precast"))if(auto name=map->getString(id))if(tensorPtr.count(name->str()))f.precast=tensorPtr[name->str()];if(f.precast.empty())throw std::runtime_error("missing declared pool precast scratch for "+id);}}chain.push_back(xid);readoutFusion[id]=f;if(!cols.empty())columnScales[id]=cols;for(auto&z:chain)fusedSkip.insert(z);}break;}
 }
 struct Residual{std::string dst;uint32_t lhs,rhs;bool out32,relu;};std::map<std::string,Residual> residuals;
 std::map<std::string,std::pair<const Object*,std::string>> movementPools;
 for(size_t i=0;i<exprs.size();i++){auto&e=*exprs[i].getAsObject();auto id=str(e,"id"),op=str(e,"operator");if(op=="scaled_add"){
 auto&oa=arr(e,"operands");if(oa.size()!=2||str(*objs.at(oa[0].getAsString()->str()),"dtype")!="i8"||str(*objs.at(oa[1].getAsString()->str()),"dtype")!="i8")continue;const Object ep;auto&pr=e.getObject("parameters")?*e.getObject("parameters"):ep;auto lv=pr.getNumber("lhs_factor"),rv=pr.getNumber("rhs_factor");if(!lv||!rv||!std::isfinite(*lv)||!std::isfinite(*rv)||*lv<0||*lv>16||*rv<0||*rv>16)continue;auto bound=integer(pr,"allowed_error_lsb");bool integral=*lv==(int)*lv&&*rv==(int)*rv;if(!integral&&bound<((*lv<=1&&*rv<=1)?1:2))continue;
 float lf=(float)*lv,rf=(float)*rv;Residual r;r.relu=false;r.out32=str(e,"dtype")=="i32";r.dst=ptr.at(id);std::memcpy(&r.lhs,&lf,4);std::memcpy(&r.rhs,&rf,4);std::vector<std::string> owned;for(size_t j=i+1;j<exprs.size();j++){auto&x=*exprs[j].getAsObject();auto xid=str(x,"id");if(!groups.count(id)||!groups.count(xid)||groups[id]!=groups[xid])continue;auto xo=str(x,"operator");if(xo=="relu")r.relu=true;else if(xo!="cast")break;owned.push_back(xid);r.dst=ptr.at(xid);r.out32=str(x,"dtype")=="i32";}if(r.out32&&r.relu)continue;residuals[id]=r;for(auto&z:owned)fusedSkip.insert(z);
 }else if(op=="cast"&&str(e,"dtype")=="i32"){
 auto&oa=arr(e,"operands");if(oa.empty()||str(*objs.at(oa[0].getAsString()->str()),"dtype")!="i8")continue;const Object*pool=nullptr;std::string end=id;std::vector<std::string>owned;for(size_t j=i+1;j<exprs.size();j++){auto&x=*exprs[j].getAsObject();auto xid=str(x,"id");if(!groups.count(id)||!groups.count(xid)||groups[id]!=groups[xid])continue;auto xo=str(x,"operator");if(xo=="max_pool"&&!pool)pool=&x;else if(xo!="cast")break;owned.push_back(xid);end=xid;}if(pool&&str(*objs.at(end),"dtype")=="i8"){movementPools[id]={pool,end};for(auto&z:owned)fusedSkip.insert(z);}
 }}
 ei=0;for(auto&ev:exprs){auto&e=*ev.getAsObject();auto id=str(e,"id");if(fusedSkip.count(id)){ei++;continue;}auto op=str(e,"operator"),out=ptr.at(id);const Object empty;const Object&p=e.getObject("parameters")?*e.getObject("parameters"):empty;std::vector<std::string>a,ids;if(auto*oa=e.getArray("operands"))for(auto&x:*oa){ids.push_back(x.getAsString()->str());a.push_back(ptr.at(ids.back()));}auto n=elements(e);auto s=shape(e);auto dtype=str(e,"dtype");int tagGroup=groups.count(id)?groups[id]:ei;body<<"\n#line "<<sourceLine(e)<<" \"merlin_task_"<<tagGroup<<"\"\n{\n";ei++;
 auto emitDevicePool=[&](const Fusion&f,const std::string&A,const std::string&B,int64_t M,int64_t K,int64_t N,const std::vector<int64_t>&convGeometry){if(!f.bias.empty()){auto&biasObject=*objs.at(f.biasId);if(str(biasObject,"dtype")!="i32"||shape(biasObject)!=std::vector<int64_t>{N}||(position.count(f.biasId)&&position.at(f.biasId)>=position.at(id)))throw std::runtime_error("fused contraction pool bias must be an available i32 channel vector");}uint32_t bits;std::memcpy(&bits,&f.scale,4);const Object emptyPool;auto&pp=f.pool->getObject("parameters")?*f.pool->getObject("parameters"):emptyPool;auto spatial=vec(pp,"spatial_shape"),window=vec(pp,"window"),stride=vec(pp,"stride"),padding=vec(pp,"padding",{0,0,0,0});if(spatial.size()!=2||window.size()!=2||stride.size()!=2||padding.size()!=4)throw std::runtime_error("fused matmul pool geometry");auto H=spatial[0],W=spatial[1],PH=window[0],PW=window[1],SH=stride[0],SW=stride[1];if(H<=0||W<=0||PH<=0||PW<=0||SH<=0||SW<=0)throw std::runtime_error("fused matmul pool geometry");auto OH=(H+padding[0]+padding[2]-PH)/SH+1,OW=(W+padding[1]+padding[3]-PW)/SW+1;int64_t pad=-128;if(auto*pv=pp.get("padding_value"))if(pv->kind()!=Value::Null){auto value=pv->getAsInteger();if(!value)throw std::runtime_error("fused matmul pool padding");pad=std::max<int64_t>(-128,std::min<int64_t>(127,*value));}if(M%(H*W)||OH<=0||OW<=0||PH-1-padding[0]<0||PW-1-padding[1]<0||(OH-1)*SH-padding[0]>=H||(OW-1)*SW-padding[1]>=W||(pad!=-128&&(!f.relu||pad>0)))throw std::runtime_error("unsupported fused matmul pool padding");

 if(!convGeometry.empty()) {
  std::string helperName="dg_static_pool_"+std::to_string(ei);
  auto specialized=convPoolStaticRuntime(helperName,convGeometry,{OH,OW,PH,PW,SH,SW,padding[0],padding[1]},pad,bits,f.relu?1:0,!f.bias.empty());
  if(!specialized.empty()) {
   decl<<specialized;
   body<<helperName<<"((const int8_t*)("<<A<<"),(const int8_t*)("<<B<<"),"<<(f.bias.empty()?"(const int32_t*)0":"(const int32_t*)("+f.bias+")")<<",(int8_t*)("<<f.dst<<"));\n";
   return;
  }
 }
 if(B.empty())body<<"dg_readout_pool_ex((const int32_t*)("<<A<<"),(int8_t*)("<<f.dst<<"),"<<M/H/W<<","<<H<<","<<W<<","<<N;else{body<<(convGeometry.empty()?"dg_matmul_pool_ex(":"dg_conv2d_pool_ex(")<<"(const int8_t*)("<<A<<"),(const int8_t*)("<<B<<"),"<<(f.bias.empty()?"(const int32_t*)0":"(const int32_t*)("+f.bias+")")<<",(int8_t*)("<<f.dst<<"),";
 if(convGeometry.empty())body<<M<<","<<K<<","<<N<<","<<M/H/W<<","<<H<<","<<W;else for(size_t k=0;k<convGeometry.size();k++){if(k)body<<",";body<<convGeometry[k];}}
 body<<","<<OH<<","<<OW<<","<<PH<<","<<PW<<","<<SH<<","<<SW<<","<<padding[0]<<","<<padding[1]<<","<<pad<<","<<bits<<"U,"<<(f.relu?1:0)<<");\n";
 };
 auto emitMatmul=[&](const std::string&A,const std::string&B,const std::string&C,int64_t M,int64_t K,int64_t N){
   auto fit=fusion.find(id);if(fit==fusion.end()){body<<deviceMatmul(A,B,C,M,K,N);return;}
   auto&f=fit->second;auto destination=f.pool?f.precast:f.dst;uint32_t bits;std::memcpy(&bits,&f.scale,4);
   if(f.devicePool){emitDevicePool(f,A,B,M,K,N,{});return;}
   body<<"dg_matmul_ex((const int8_t*)("<<A<<"),(const int8_t*)("<<B<<"),"<<(f.bias.empty()?"(const int32_t*)0":"(const int32_t*)("+f.bias+")")<<",(void*)("<<destination<<"+((int32_t*)("<<C<<")-(int32_t*)("<<ptr.at(id)<<"))),"<<M<<","<<K<<","<<N<<","<<(f.out32?1:0)<<","<<bits<<"U,"<<(f.relu?1:0)<<");\n";
 };
 auto emitLiteralColumnReadout=[&](const Fusion&f,const std::string&source,int64_t M,int64_t N){auto&scales=literalColumnScales.at(id);const auto tileChoice=compilerSettingString("column_readout_tile","64x8");const bool tallTile=tileChoice=="128x4"||(tileChoice=="auto"&&M>64&&N>64);const int64_t tileRows=tallTile?128:64,tileCols=tallTile?4:8;body<<"dg_fence();dg_load_config("<<N*4<<",0,0,1065353216U);\n";
 if(!compilerSettingBool("column_batching",true)||M<=16){for(size_t c=0;c<scales.size();c++)body<<"dg_store_config("<<N<<","<<scales[c]<<"U,"<<(f.relu?1:0)<<");for(uint64_t i=0;i<"<<M<<";i+=16){uint64_t m="<<M<<"-i;if(m>16)m=16;DG_CMD(2,(const int32_t*)("<<source<<")+i*"<<N<<"+"<<c<<",dg_desc(0x80000000U,m,1));DG_CMD(3,(int8_t*)("<<f.dst<<")+i*"<<N<<"+"<<c<<",dg_desc(0x80000000U,m,1));dg_fence();}\n";}
 else{body<<"for(uint64_t i=0;i<"<<M<<";i+="<<tileRows<<"){uint64_t rows="<<M<<"-i;if(rows>"<<tileRows<<")rows="<<tileRows<<";\n";for(size_t j=0;j<scales.size();j+=tileCols){auto end=std::min(j+size_t(tileCols),scales.size());for(size_t c=j;c<end;c++)body<<"for(uint64_t r=0;r<rows;r+=16){uint64_t m=rows-r;if(m>16)m=16;DG_CMD(2,(const int32_t*)("<<source<<")+(i+r)*"<<N<<"+"<<c<<",dg_desc("<<(0x80000000U+uint32_t((c-j)*tileRows))<<"U+(uint32_t)r,m,1));}\n";body<<"dg_fence();\n";for(size_t c=j;c<end;c++)body<<"dg_store_config("<<N<<","<<scales[c]<<"U,"<<(f.relu?1:0)<<");for(uint64_t r=0;r<rows;r+=16){uint64_t m=rows-r;if(m>16)m=16;DG_CMD(3,(int8_t*)("<<f.dst<<")+(i+r)*"<<N<<"+"<<c<<",dg_desc("<<(0x80000000U+uint32_t((c-j)*tileRows))<<"U+(uint32_t)r,m,1));}dg_fence();\n";}body<<"}\n";}
 body<<"dg_fence();\n";
 };
 auto emitPool=[&](const Object&pe,const std::string&source,const std::string&destination,const Object&sourceObject){body<<"\n#line "<<sourceLine(pe)<<" \"merlin_task_"<<tagGroup<<"\"\n";const Object ep;auto&pr=pe.getObject("parameters")?*pe.getObject("parameters"):ep;auto as=shape(sourceObject),sp=vec(pr,"spatial_shape"),win=vec(pr,"window"),st=vec(pr,"stride"),pa=vec(pr,"padding",{0,0,0,0});if(sp.size()!=2||win.size()!=2||st.size()!=2)throw std::runtime_error("pool geometry");auto H=sp[0],W=sp[1],C=as.back(),OH=(H+pa[0]+pa[2]-win[0])/st[0]+1,OW=(W+pa[1]+pa[3]-win[1])/st[1]+1,B=elements(sourceObject)/H/W/C;auto*pv=pr.get("padding_value");std::string pad="-128";if(pv&&pv->kind()!=Value::Null){auto z=pv->getAsInteger();if(!z)throw std::runtime_error("noninteger pool padding");pad=std::to_string(std::max<int64_t>(-128,std::min<int64_t>(127,*z)));}std::string padn="pad"+std::to_string(ei);decl<<"static const int8_t "<<padn<<"[16]={";for(int k=0;k<16;k++){if(k)decl<<",";decl<<pad;}decl<<"};\n";body<<"dg_pool_i8((const int8_t*)("<<source<<"),(int8_t*)("<<destination<<"),"<<B<<","<<H<<","<<W<<","<<C<<","<<OH<<","<<OW<<","<<win[0]<<","<<win[1]<<","<<st[0]<<","<<st[1]<<","<<pa[0]<<","<<pa[1]<<","<<padn<<",";std::string partial="(int8_t*)0";if(win[0]>3||win[1]>3){if(meta)if(auto*map=meta->getObject("pool_partial"))if(auto name=map->getString(str(pe,"id")))if(tensorPtr.count(name->str()))partial="(int8_t*)("+tensorPtr[name->str()]+")";if(partial=="(int8_t*)0")throw std::runtime_error("missing declared partial pool scratch for "+str(pe,"id"));}body<<partial<<");\n";};
 auto loop=[&](std::string rhs){body<<"for(long i=0;i<"<<n<<";++i) "<<out<<"[i]="<<rhs<<";\n";};
 auto coordinateLoops=[&](const std::vector<int64_t>&bounds){std::vector<std::string>coordinates;for(size_t k=0;k<bounds.size();k++){if(bounds[k]==1){coordinates.push_back("0");continue;}auto coordinate="c"+std::to_string(k);coordinates.push_back(coordinate);body<<"for(long "<<coordinate<<"=0;"<<coordinate<<"<"<<bounds[k]<<";++"<<coordinate<<")";}return coordinates;};
 auto affineOffset=[&](const std::vector<int64_t>&dimensions,const std::vector<std::string>&coordinates,const std::vector<int64_t>&offsets,const std::vector<int64_t>&strides){if(coordinates.size()!=dimensions.size())throw std::runtime_error("layout coordinate rank");int64_t suffix=1,base=0;std::vector<int64_t>coefficients(dimensions.size());for(size_t k=dimensions.size();k-->0;){base+=(offsets.empty()?0:offsets[k])*suffix;coefficients[k]=(strides.empty()?1:strides[k])*suffix;suffix*=dimensions[k];}std::string index=std::to_string(base);for(size_t k=0;k<dimensions.size();k++)if(coordinates[k]!="0"&&coefficients[k])index+="+"+coordinates[k]+(coefficients[k]==1?"":"*"+std::to_string(coefficients[k]));return "("+index+")";};
 if(residuals.count(id)){auto&r=residuals[id];body<<"dg_residual((const int8_t*)("<<a[0]<<"),(const int8_t*)("<<a[1]<<"),(void*)("<<r.dst<<"),"<<n/s.back()<<","<<s.back()<<","<<r.lhs<<"U,"<<r.rhs<<"U,"<<(r.out32?1:0)<<","<<(r.relu?1:0)<<");\n";}
 else if(movementPools.count(id)){auto&f=movementPools[id];emitPool(*f.first,a[0],ptr.at(f.second),*objs.at(ids[0]));}
 else if(readoutFusion.count(id)){auto&f=readoutFusion[id];auto destination=f.pool?f.precast:f.dst;uint32_t bits;std::memcpy(&bits,&f.scale,4);if(f.devicePool){if(columnScales.count(id))throw std::runtime_error("fused readout pool requires uniform scale");emitDevicePool(f,a[0],"",n/s.back(),0,s.back(),{});}else if(literalColumnScales.count(id))emitLiteralColumnReadout(f,a[0],n/s.back(),s.back());else if(columnScales.count(id))body<<"dg_readout_cols((const int32_t*)("<<a[0]<<"),(int8_t*)("<<destination<<"),"<<n/s.back()<<","<<s.back()<<","<<columnScales[id]<<","<<(f.relu?1:0)<<");\n";else body<<"dg_readout((const int32_t*)("<<a[0]<<"),(int8_t*)("<<destination<<"),"<<n/s.back()<<","<<s.back()<<","<<bits<<"U,"<<(f.relu?1:0)<<");\n";}
 else if(op=="empty"){loop("0");}
 else if(op=="constant"){auto*v=p.get("value");if(!v)v=e.get("value");if(!v)throw std::runtime_error("constant lacks value");if(auto*ar=v->getAsArray()){decl<<"static const "<<ct(dtype)<<" c"<<ei<<"[]={";for(size_t j=0;j<ar->size();j++){if(j)decl<<",";decl<<val((*ar)[j]);}decl<<"};\n";loop("c"+std::to_string(ei)+"[i%"+std::to_string(ar->size())+"]");}else loop(val(*v));}
 else if(op=="fill"||op=="splat")loop(a.at(0)+"[0]");
 else if(op=="concat"){
 auto axis=integer(p,"dimension",integer(p,"axis"));int64_t outer=1,inner=1;for(size_t k=0;k<size_t(axis);k++)outer*=s[k];for(size_t k=axis+1;k<s.size();k++)inner*=s[k];int64_t offset=0;for(size_t j=0;j<a.size();j++){auto as=shape(*objs.at(ids[j]));auto span=as[axis]*inner;body<<"for(long b=0;b<"<<outer<<";++b)for(long i=0;i<"<<span<<";++i)"<<out<<"[b*"<<s[axis]*inner<<"+"<<offset<<"+i]="<<a[j]<<"[b*"<<span<<"+i];\n";offset+=span;}
 }
 else if(op=="dequantize_per_tensor")loop("cg_mul((float)((long)"+a.at(0)+"[i]-(long)"+a.at(2)+"[0]),(float)"+a.at(1)+"[0])");
 else if(op=="quantize_per_tensor"){
 body<<"for(long i=0;i<"<<n<<";++i){long v=cg_round((float)"<<a.at(0)<<"[i]/(float)"<<a.at(1)<<"[0])+(long)"<<a.at(2)<<"[0];if(v<"<<integer(p,"quant_min",-128)<<")v="<<integer(p,"quant_min",-128)<<";if(v>"<<integer(p,"quant_max",127)<<")v="<<integer(p,"quant_max",127)<<";"<<out<<"[i]=v;}\n";
 }
 else if(op=="matmul"){
 auto as=shape(*objs.at(ids.at(0))),bs=shape(*objs.at(ids.at(1)));auto M=as[as.size()-2],K=as.back(),N=bs.back();int64_t batches=n/M/N;
 if(str(*objs.at(ids[0]),"dtype")=="i8"&&str(*objs.at(ids[1]),"dtype")=="i8")for(int64_t b=0;b<batches;b++)emitMatmul(a[0]+"+"+std::to_string(b*M*K),a[1]+"+"+std::to_string(b*K*N),out+"+"+std::to_string(b*M*N),M,K,N);
 else {body<<"for(long b=0;b<"<<batches<<";++b)for(long m=0;m<"<<M<<";++m)for(long j=0;j<"<<N<<";++j){unsigned int z=0;for(long k=0;k<"<<K<<";++k)z+=(unsigned int)((long)"<<a[0]<<"[(b*"<<M<<"+m)*"<<K<<"+k]*(long)"<<a[1]<<"[(b*"<<K<<"+k)*"<<N<<"+j]);"<<out<<"[(b*"<<M<<"+m)*"<<N<<"+j]=(int)z;}\n";}}
 else if(op=="conv2d"){
 auto as=shape(*objs.at(ids.at(0))),bs=shape(*objs.at(ids.at(1)));auto ker=vec(p,"kernel"),st=vec(p,"stride",{1,1}),pa=vec(p,"padding",{0,0,0,0}),di=vec(p,"dilation",{1,1});if(as.size()!=4||ker.size()!=2||st.size()!=2||pa.size()!=4||di.size()!=2||st[0]<=0||st[1]<=0||di[0]<=0||di[1]<=0||bs.empty())throw std::runtime_error("conv geometry");auto H=as[1],W=as[2],CI=as[3],KH=ker[0],KW=ker[1],CO=bs.back(),OH=(H+pa[0]+pa[2]-di[0]*(KH-1)-1)/st[0]+1,OW=(W+pa[1]+pa[3]-di[1]*(KW-1)-1)/st[1]+1;auto M=as[0]*OH*OW,K=KH*KW*CI;bool direct=KH==1&&KW==1&&st==std::vector<int64_t>{1,1}&&pa==std::vector<int64_t>{0,0,0,0};if(direct){if(bs.size()!=2||bs[0]!=CI||elements(e)!=M*CO)throw std::runtime_error("1x1 convolution matrix geometry");emitMatmul(a[0],a[1],out,M,K,CO);}else if(str(*objs.at(ids[0]),"dtype")=="i8"&&str(*objs.at(ids[1]),"dtype")=="i8"&&dtype=="i32"){
 auto fit=fusion.find(id);std::string destination=out,bias="(const int32_t*)0";bool out32=true,relu=false;float scale=1.0f;if(fit!=fusion.end()){auto&f=fit->second;destination=f.pool?f.precast:f.dst;out32=f.out32;relu=f.relu;scale=f.scale;if(!f.bias.empty())bias="(const int32_t*)("+f.bias+")";}uint32_t bits;std::memcpy(&bits,&scale,4);
 if(fit!=fusion.end()&&fit->second.devicePool)emitDevicePool(fit->second,a[0],a[1],M,K,CO,{as[0],H,W,CI,CO,OH,OW,KH,KW,st[0],st[1],di[0],di[1],pa[0],pa[1]});else body<<"dg_conv2d_ex((const int8_t*)("<<a[0]<<"),(const int8_t*)("<<a[1]<<"),"<<bias<<",(void*)("<<destination<<"),"<<as[0]<<","<<H<<","<<W<<","<<CI<<","<<CO<<","<<OH<<","<<OW<<","<<KH<<","<<KW<<","<<st[0]<<","<<st[1]<<","<<di[0]<<","<<di[1]<<","<<pa[0]<<","<<pa[1]<<","<<(out32?1:0)<<","<<bits<<"U,"<<(relu?1:0)<<");\n";
 }else{std::string tmp;if(meta)if(auto*im=meta->getObject("conv_im2col"))if(auto name=im->getString(id))if(tensorPtr.count(name->str()))tmp=tensorPtr[name->str()];if(tmp.empty())throw std::runtime_error("missing declared im2col scratch for "+id);
 body<<"for(long b=0;b<"<<as[0]<<";++b)for(long y=0;y<"<<OH<<";++y)for(long x=0;x<"<<OW<<";++x)for(long ky=0;ky<"<<KH<<";++ky)for(long kx=0;kx<"<<KW<<";++kx)for(long c=0;c<"<<CI<<";++c){long iy=y*"<<st[0]<<"+ky*"<<di[0]<<"-"<<pa[0]<<",ix=x*"<<st[1]<<"+kx*"<<di[1]<<"-"<<pa[1]<<";"<<tmp<<"[((b*"<<OH<<"+y)*"<<OW<<"+x)*"<<K<<"+(ky*"<<KW<<"+kx)*"<<CI<<"+c]=(iy>=0&&iy<"<<H<<"&&ix>=0&&ix<"<<W<<")?"<<a[0]<<"[((b*"<<H<<"+iy)*"<<W<<"+ix)*"<<CI<<"+c]:0;}\n";emitMatmul(tmp,a[1],out,M,K,CO);}
 }
 else if(op=="add_bias"){if(dtype=="i32"){body<<"dg_bias((const int32_t*)("<<a[0]<<"),(const int32_t*)("<<a[1]<<"),(int32_t*)("<<out<<"),"<<n/s.back()<<","<<s.back()<<");\n";}else loop("(int)((unsigned int)"+a.at(0)+"[i]+(unsigned int)"+a.at(1)+"[i%"+std::to_string(s.back())+"])");}
 else if(op=="scale_round"){
 std::string factor;auto*f=p.get("factor");if(auto*fs=p.getArray("factors")){std::string fpn="f"+std::to_string(ei);decl<<"static const float "<<fpn<<"[]={";for(size_t j=0;j<fs->size();j++){if(j)decl<<",";decl<<"(float)"<<val((*fs)[j]);}decl<<"};\n";factor=fpn+"[i%"+std::to_string(fs->size())+"]";}else if(f)factor="(float)"+val(*f);else throw std::runtime_error("missing scale");loop("cg_round(cg_mul((float)"+a.at(0)+"[i],"+factor+"))");}
 else if(op=="scaled_add"){auto*l=p.get("lhs_factor"),*r=p.get("rhs_factor");if(!l||!r)throw std::runtime_error("scaled_add factors");loop("cg_round(cg_add(cg_mul((float)"+a.at(0)+"[i],(float)"+val(*l)+"),cg_mul((float)"+a.at(1)+"[i],(float)"+val(*r)+")))");}
 else if(op=="round_shift_right"){auto k=integer(p,"shift");loop(k>0?"(((long)"+a.at(0)+"[i]+"+std::to_string(1LL<<(k-1))+")>>"+std::to_string(k)+")":a.at(0)+"[i]");}
 else if(op=="relu")loop("("+a.at(0)+"[i]>0?"+a.at(0)+"[i]:0)");
 else if(op=="cast"){if(str(*objs.at(ids[0]),"dtype")=="i8"&&(dtype=="i8"||dtype=="i32")){body<<deviceMovement(a[0],out,n/s.back(),s.back(),dtype);}else if(dtype=="i8"&&str(*objs.at(ids[0]),"dtype")=="i32")body<<"dg_readout((const int32_t*)("<<a[0]<<"),(int8_t*)("<<out<<"),"<<n/s.back()<<","<<s.back()<<",1065353216U,0);\n";else if(dtype=="i8")loop("cg_sat8("+a.at(0)+"[i])");else loop("("+ct(dtype)+")"+a.at(0)+"[i]");}
 else if(op=="xor_low_bit")loop("("+a.at(0)+"[i]^1)");
 else if(op=="identity"||op=="movement"){if(str(*objs.at(ids[0]),"dtype")=="i8"&&(dtype=="i8"||dtype=="i32"))body<<deviceMovement(a[0],out,n/s.back(),s.back(),dtype);else loop(a.at(0)+"[i]");}
 else if(op=="reshape"||op=="copy")loop(a.at(0)+"[i]");
 else if(op=="extract_slice"||op=="insert_slice"){
 auto offsets=vec(p,"static_offsets"),strides=vec(p,"static_strides");auto srcShape=shape(*objs.at(ids[0]));auto large=op=="insert_slice"?s:srcShape;auto small=op=="insert_slice"?srcShape:s;if(offsets.size()!=large.size()||small.size()!=large.size())throw std::runtime_error("slice rank");if(strides.empty())strides.assign(large.size(),1);if(strides.size()!=large.size())throw std::runtime_error("slice stride rank");
 if(op=="insert_slice")loop(a.at(1)+"[i]");auto coordinates=coordinateLoops(small);auto smallIndex=affineOffset(small,coordinates,{},{}),largeIndex=affineOffset(large,coordinates,offsets,strides);body<<"{"<<out<<"["<<(op=="insert_slice"?largeIndex:smallIndex)<<"]="<<a[0]<<"["<<(op=="insert_slice"?smallIndex:largeIndex)<<"];}\n";
 }
 else if(op=="transpose"){
 auto per=vec(p,"permutation"),as=shape(*objs.at(ids.at(0)));if(per.empty())for(size_t k=0;k<s.size();k++)per.push_back(s.size()-1-k);if(per.size()!=s.size()||as.size()!=s.size())throw std::runtime_error("transpose rank");std::vector<bool>seen(s.size(),false);for(size_t j=0;j<per.size();j++){auto k=per[j];if(k<0||k>=int64_t(s.size())||seen[k]||s[j]!=as[k])throw std::runtime_error("transpose permutation");seen[k]=true;}auto coordinates=coordinateLoops(s);std::vector<std::string>sourceCoordinates(as.size());for(size_t j=0;j<per.size();j++)sourceCoordinates[per[j]]=coordinates[j];body<<"{"<<out<<"["<<affineOffset(s,coordinates,{},{})<<"]="<<a[0]<<"["<<affineOffset(as,sourceCoordinates,{},{})<<"];}\n";
 }
 else if(op=="max_pool"&&dtype=="i8"&&str(*objs.at(ids[0]),"dtype")=="i8")emitPool(e,a[0],out,*objs.at(ids[0]));
 else if(op=="max_pool"){
 auto as=shape(*objs.at(ids.at(0))),sp=vec(p,"spatial_shape"),win=vec(p,"window"),st=vec(p,"stride"),pa=vec(p,"padding",{0,0,0,0});if(sp.size()!=2||win.size()!=2||st.size()!=2)throw std::runtime_error("pool geometry");auto H=sp[0],W=sp[1],C=as.back(),OH=(H+pa[0]+pa[2]-win[0])/st[0]+1,OW=(W+pa[1]+pa[3]-win[1])/st[1]+1,B=elements(*objs.at(ids[0]))/H/W/C;auto*pv=p.get("padding_value");std::string pad=pv&&pv->kind()!=Value::Null?val(*pv):"(-2147483647-1)";
 body<<"for(long b=0;b<"<<B<<";++b)for(long y=0;y<"<<OH<<";++y)for(long x=0;x<"<<OW<<";++x)for(long c=0;c<"<<C<<";++c){int z=(-2147483647-1);for(long ky=0;ky<"<<win[0]<<";++ky)for(long kx=0;kx<"<<win[1]<<";++kx){long iy=y*"<<st[0]<<"+ky-"<<pa[0]<<",ix=x*"<<st[1]<<"+kx-"<<pa[1]<<";int v=(iy>=0&&iy<"<<H<<"&&ix>=0&&ix<"<<W<<")?"<<a[0]<<"[((b*"<<H<<"+iy)*"<<W<<"+ix)*"<<C<<"+c]:"<<pad<<";if(v>z)z=v;}"<<out<<"[((b*"<<OH<<"+y)*"<<OW<<"+x)*"<<C<<"+c]=z;}\n";
 }
 else if(op=="generic"){
 auto loops=vec(p,"loop_shape");auto&maps=arr(p,"indexing_maps");auto*sp=p.getObject("scalar_program");if(!sp)throw std::runtime_error("generic scalar_program missing");auto inputCount=integer(p,"input_count",a.size()-1);auto destinationIndex=integer(p,"destination_index",inputCount);bool elidedDestination=p.getBoolean("elided_destination").value_or(false);if(!elidedDestination&&a.size()>size_t(destinationIndex))loop(a[destinationIndex]+"[i]");
 for(size_t j=0;j<loops.size();j++)body<<"for(long d"<<j<<"=0;d"<<j<<"<"<<loops[j]<<";++d"<<j<<")";body<<"{\n";std::map<std::string,std::string>sv,stypes;int si=0;auto&args=arr(*sp,"args");for(size_t j=0;j<args.size();j++){auto sid=args[j].getAsString()->str();if(elidedDestination&&j==size_t(destinationIndex)){sv[sid]="0";stypes[sid]=dtype;continue;}auto operandIndex=j-(elidedDestination&&j>size_t(destinationIndex)?1:0);auto idx=flat(*maps[j].getAsArray(),shape(*objs.at(ids[operandIndex])));std::string v="s"+std::to_string(si++);sv[sid]=v;stypes[sid]=str(*objs.at(ids[operandIndex]),"dtype");body<<ct(str(*objs.at(ids[operandIndex]),"dtype"))<<" "<<v<<"="<<(j>=size_t(inputCount)?out:a[operandIndex])<<"["<<idx<<"];\n";}
 for(auto&iv:arr(*sp,"instructions")){auto&i=*iv.getAsObject();auto sid=str(i,"id"),opr=str(i,"op"),dt=str(i,"dtype","i32"),v="s"+std::to_string(si++);std::vector<std::string>x;if(auto*oa=i.getArray("operands"))for(auto&ov:*oa){auto name=ov.getAsString()->str();x.push_back(sv.count(name)?sv[name]:ptr.at(name)+"[0]");}auto operandType=[&](size_t z){auto&oa=arr(i,"operands");auto name=oa[z].getAsString()->str();return stypes.count(name)?stypes.at(name):str(*objs.at(name),"dtype");};
 auto unsignedOperand=[&](size_t z){auto ty=operandType(z);auto result="(("+ut(ty)+")("+x.at(z)+"))";if(ty=="i1")result="("+result+"&1U)";return result;};
 auto signedOperand=[&](size_t z){return operandType(z)=="i1"?"(-(int)("+x.at(z)+"&1U))":x.at(z);};
 std::string rhs;
 if(opr=="linalg.index")rhs="d"+std::to_string(integer(i,"dimension"));
 else if(opr=="tensor.extract"){auto&oo=arr(i,"operands");auto tid=oo[0].getAsString()->str();auto ts=shape(*objs.at(tid));std::string idx="0";if(x.size()!=ts.size()+1)throw std::runtime_error("tensor.extract rank");for(size_t z=0;z<ts.size();z++)idx="("+idx+"*"+std::to_string(ts[z])+"+"+x[z+1]+")";rhs=ptr.at(tid)+"["+idx+"]";}
 else if(opr=="arith.constant"||opr=="constant")rhs=val(*i.get("value"));
 else if(opr=="arith.addi"||opr=="arith.addf")rhs="("+x.at(0)+"+"+x.at(1)+")";
 else if(opr=="arith.subi"||opr=="arith.subf")rhs="("+x.at(0)+"-"+x.at(1)+")";
 else if(opr=="arith.muli"||opr=="arith.mulf")rhs="("+x.at(0)+"*"+x.at(1)+")";
 else if(opr=="arith.divf")rhs="("+x.at(0)+"/"+x.at(1)+")";
 else if(opr=="arith.divsi")rhs="("+signedOperand(0)+"/"+signedOperand(1)+")";
 else if(opr=="arith.divui")rhs="("+unsignedOperand(0)+"/"+unsignedOperand(1)+")";
 else if(opr=="arith.maximumf"||opr=="arith.maxnumf"||opr=="arith.minimumf"||opr=="arith.minnumf")rhs="cg_minmax("+x.at(0)+","+x.at(1)+","+(opr=="arith.minimumf"||opr=="arith.minnumf"?"1":"0")+","+(opr=="arith.maxnumf"||opr=="arith.minnumf"?"1":"0")+")";
 else if(opr=="arith.maxsi")rhs="("+signedOperand(0)+">"+signedOperand(1)+"?"+x[0]+":"+x[1]+")";
 else if(opr=="arith.minsi")rhs="("+signedOperand(0)+"<"+signedOperand(1)+"?"+x[0]+":"+x[1]+")";
 else if(opr=="arith.xori"||opr=="arith.andi"||opr=="arith.ori")rhs="("+x.at(0)+(opr=="arith.xori"?"^":opr=="arith.andi"?"&":"|")+x.at(1)+")";
 else if(opr=="arith.shli")rhs="("+x.at(0)+"<<"+x.at(1)+")";
 else if(opr=="arith.shrsi")rhs="("+signedOperand(0)+">>"+unsignedOperand(1)+")";
 else if(opr=="arith.shrui")rhs="("+unsignedOperand(0)+">>"+unsignedOperand(1)+")";
 else if(opr=="arith.negf")rhs="(-"+x.at(0)+")";
 else if(opr=="arith.select")rhs="("+x.at(0)+"?"+x.at(1)+":"+x.at(2)+")";
 else if(opr=="arith.cmpi"||opr=="arith.cmpf"){auto pred=str(i,"predicate");std::string cmp=pred=="eq"||pred=="oeq"?"==":pred=="ne"||pred=="one"?"!=":pred=="slt"||pred=="ult"||pred=="olt"?"<":pred=="sle"||pred=="ule"||pred=="ole"?"<=":pred=="sgt"||pred=="ugt"||pred=="ogt"?">":">=";rhs="("+x.at(0)+cmp+x.at(1)+")";if(opr=="arith.cmpi"){bool uns=pred=="ult"||pred=="ule"||pred=="ugt"||pred=="uge";rhs="("+(uns?unsignedOperand(0):signedOperand(0))+cmp+(uns?unsignedOperand(1):signedOperand(1))+")";}if(opr=="arith.cmpf"){
 std::string nan="(("+x[0]+"!="+x[0]+")||("+x[1]+"!="+x[1]+"))";
 if(pred=="uno")rhs=nan;else if(pred=="ord")rhs="(!"+nan+")";else if(pred=="false")rhs="0";else if(pred=="true")rhs="1";
 else if(pred=="ueq"||pred=="ugt"||pred=="uge"||pred=="ult"||pred=="ule"||pred=="une"){std::string cp=pred=="ueq"?"==":pred=="une"?"!=":pred=="ugt"?">":pred=="uge"?">=":pred=="ult"?"<":"<=";rhs="("+nan+"||("+x[0]+cp+x[1]+"))";}
 else rhs="(!"+nan+"&&"+rhs+")";
 }}
 else if(opr=="math.sin")rhs="cg_sincos("+x.at(0)+",0)";
 else if(opr=="math.cos")rhs="cg_sincos("+x.at(0)+",1)";
 else if(opr=="math.tanh")rhs="cg_tanh("+x.at(0)+")";
 else if(opr=="math.absf")rhs=(dt=="f64"?"__builtin_fabs(":"__builtin_fabsf(")+x.at(0)+")";
 else if(opr=="math.rsqrt")rhs="(1.0f/cg_sqrt("+x.at(0)+"))";
 else if(opr=="math.sqrt")rhs="cg_sqrt("+x.at(0)+")";
 else if(opr=="math.exp")rhs="cg_exp("+x.at(0)+")";
 else if(opr=="math.roundeven"||opr=="math.round")rhs=(dt=="f64"?"cg_roundd(":"cg_roundf(")+x.at(0)+","+(opr=="math.roundeven"?"1":"0")+")";
 else if(opr=="arith.extui"||opr=="arith.uitofp")rhs="("+ct(dt)+")"+unsignedOperand(0);
 else if(opr=="arith.fptoui")rhs="("+ut(dt)+")"+x.at(0);
 else if(opr=="arith.extsi"||opr=="arith.sitofp"||opr=="arith.index_cast")rhs="("+ct(dt)+")"+signedOperand(0);
 else if(opr=="arith.trunci"||opr=="arith.fptosi"||opr=="arith.extf"||opr=="arith.truncf")rhs="("+ct(dt)+")"+x.at(0);
 else throw std::runtime_error("unsupported generic scalar op "+opr);
 if(dt=="i1")rhs="(("+rhs+")&1U)";sv[sid]=v;stypes[sid]=dt;body<<ct(dt)<<" "<<v<<"="<<rhs<<";\n";
 }auto y=str(*sp,"yield");body<<out<<"["<<flat(*maps.back().getAsArray(),s)<<"]="<<sv.at(y)<<";\n}\n";
 }
 else throw std::runtime_error("unsupported C operator "+op);
 if(auto fit=fusion.find(id);fit!=fusion.end()&&fit->second.pool&&!fit->second.devicePool){auto&f=fit->second;auto&poolInputs=arr(*f.pool,"operands");emitPool(*f.pool,f.precast,f.dst,*objs.at(poolInputs[0].getAsString()->str()));}
 if(auto fit=readoutFusion.find(id);fit!=readoutFusion.end()&&fit->second.pool&&!fit->second.devicePool){auto&f=fit->second;auto&poolInputs=arr(*f.pool,"operands");emitPool(*f.pool,f.precast,f.dst,*objs.at(poolInputs[0].getAsString()->str()));}
 body<<"}\n";
 }
 body<<"\n#line 99999 \"merlin_task\"\n";for(auto&ov:arr(program,"outputs")){auto&o=*ov.getAsObject();auto id=str(o,"value"),name=str(o,"name");if(!tensorPtr.count(name))throw std::runtime_error("missing output pointer "+name);auto dst=tensorPtr[name],src=ptr.at(id);if(dst!=src)body<<"for(long i=0;i<"<<elements(o)<<";++i)"<<dst<<"[i]="<<src<<"[i];\n";}
 std::string helpers=R"C(
static __attribute__((always_inline)) inline long cg_affine_floordiv(long a,long b){long q=a/b,r=a%b;return q-(r<0);}
static __attribute__((always_inline)) inline long cg_affine_ceildiv(long a,long b){long q=a/b,r=a%b;return q+(r>0);}
static __attribute__((always_inline)) inline long cg_affine_mod(long a,long b){long r=a%b;return r<0?r+b:r;}
static __attribute__((always_inline)) inline signed char cg_sat8(long x){return x>127?127:x< -128?-128:(signed char)x;}
static __attribute__((always_inline)) inline float cg_mul(float a,float b){volatile float x=a*b;return x;}
static __attribute__((always_inline)) inline float cg_add(float a,float b){volatile float x=a+b;return x;}
static __attribute__((always_inline)) inline long cg_round(float x){long y=(long)x;float r=x-(float)y;if(r>0.5f||(r==0.5f&&(y&1)))++y;if(r< -0.5f||(r== -0.5f&&(y&1)))--y;return y;}
static __attribute__((always_inline)) inline float cg_roundf(float x,int even){
 union{float f;uint32_t u;}bits={x};uint32_t magnitude=bits.u&0x7fffffffU;
 if(magnitude>=0x4b000000U||magnitude==0)return x;
 long y=(long)x;float r=x-(float)y;
 if(r>0.5f||(r==0.5f&&(!even||(y&1))))++y;
 if(r< -0.5f||(r== -0.5f&&(!even||(y&1))))--y;
 if(y==0){bits.u&=0x80000000U;return bits.f;}return (float)y;
}
static __attribute__((always_inline)) inline double cg_roundd(double x,int even){
 union{double f;uint64_t u;}bits={x};uint64_t magnitude=bits.u&0x7fffffffffffffffULL;
 if(magnitude>=0x4330000000000000ULL||magnitude==0)return x;
 long y=(long)x;double r=x-(double)y;
 if(r>0.5||(r==0.5&&(!even||(y&1))))++y;
 if(r< -0.5||(r== -0.5&&(!even||(y&1))))--y;
 if(y==0){bits.u&=0x8000000000000000ULL;return bits.f;}return (double)y;
}
static __attribute__((always_inline)) inline float cg_minmax(float a,float b,int minimum,int number){
 if(a!=a)return number?b:a;if(b!=b)return number?a:b;
 if(a==0.0f&&b==0.0f){union{float f;unsigned int u;}x={a},y={b},z;z.u=minimum?(x.u|y.u):(x.u&y.u);return z.f;}
 return minimum?(a<b?a:b):(a>b?a:b);
}
static __attribute__((always_inline)) inline float cg_sqrt(float x){float y;__asm__("fsqrt.s %0,%1":"=f"(y):"f"(x));return y;}
static __attribute__((always_inline)) inline float cg_exp(float x){
 if(x!=x)return x;if(x < -104.0f)return 0.0f;if(x>89.0f){union{unsigned int u;float f;}a={0x7f800000U};return a.f;}
 int k=(int)cg_round(cg_mul(x,1.4426950408889634f));float r=x-(float)k*0.6931471805599453f;
 float p=1.0f+r*(1.0f+r*(0.5f+r*(0.16666666666666666f+r*(0.041666666666666664f+r*(0.008333333333333333f+r*(0.001388888888888889f+r*(0.0001984126984126984f+r*0.0000248015873015873f)))))));
 if(k< -126){union{unsigned int u;float f;}a={(unsigned int)(k+151)<<23};return p*a.f*0.000000059604644775390625f;}
 if(k==128){union{unsigned int u;float f;}a={254U<<23};return (p*2.0f)*a.f;}union{unsigned int u;float f;}a={(unsigned int)(k+127)<<23};return p*a.f;
}
static __attribute__((always_inline)) inline float cg_sincos(float x,int cosine){
 union{float f;uint32_t u;}input={x};if((input.u&0x7f800000U)==0x7f800000U)return x-x;double z=(double)x*0.63661977236758134308;long q=(long)(z+(z<0?-0.5:0.5));double r=(double)x-(double)q*1.57079632679489661923,rr=r*r;
 double si=r*(1.0+rr*(-0.16666666666666666667+rr*(0.00833333333333333333+rr*(-0.00019841269841269841+rr*(0.00000275573192239859+rr*(-0.00000002505210838544+rr*0.00000000016059043837))))));
 double co=1.0+rr*(-0.5+rr*(0.04166666666666666667+rr*(-0.00138888888888888889+rr*(0.00002480158730158730+rr*(-0.00000027557319223986+rr*0.00000000208767569879)))));
 unsigned k=((unsigned)q+(unsigned)cosine)&3;return (float)(k==0?si:k==1?co:k==2?-si:-co);
}
static __attribute__((always_inline)) inline float cg_tanh(float x){if(x>10)return 1;if(x< -10)return -1;float a=x<0?-x:x;if(a<0.01f){float t=x*x;return x*(1.0f+t*(-0.3333333333333333f+t*(0.1333333333333333f-t*0.05396825396825397f)));}float e=cg_exp(-2*a),y=(1-e)/(1+e);return x<0?-y:y;}
__attribute__((noinline,used)) void kernel_complete(void){__asm__ volatile("fence" ::: "memory");}
)C";
 std::istringstream lines(body.str());std::ostringstream tagged;std::string line;while(std::getline(lines,line)){if(line.rfind("#line ",0)==0)tagged<<"\n"<<line<<"\n";else tagged<<line<<" ";}tagged<<"\n";
 return std::string("#include <stdint.h>\n")+deviceRuntime()+residualRuntime()+convRuntime()+poolRuntime()+"\n"+helpers+decl.str()+"\n"+signature+"{\n"+(arr(cb,"commands").empty()?"":"dg_flush();\n")+tagged.str()+"\n}\n";
}
}
