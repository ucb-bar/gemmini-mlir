#include "host.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <regex>
#include <set>
namespace mc { namespace {
std::string trim(std::string s){size_t a=s.find_first_not_of(" \t\n"),b=s.find_last_not_of(" \t\n");return a==std::string::npos?"":s.substr(a,b-a+1);}
std::vector<std::string> split(std::string s){std::vector<std::string>r;size_t p=0;int dep=0;for(size_t i=0;i<s.size();i++){if(s[i]=='('||s[i]=='['||s[i]=='<')dep++;if(s[i]==')'||s[i]==']'||s[i]=='>')dep--;if(s[i]==','&&dep==0){r.push_back(trim(s.substr(p,i-p)));p=i+1;}}if(p<s.size())r.push_back(trim(s.substr(p)));return r;}
const Obj& item(const Arr*a,size_t i){if(!a||i>=a->size()||!(*a)[i].getAsObject())throw std::runtime_error("missing source operand");return *(*a)[i].getAsObject();}
std::string literal(const Val&v){if(auto s=v.getAsString()){if(*s=="-inf"||*s=="-Infinity")return "(-__builtin_inff())";if(*s=="inf"||*s=="Infinity")return "__builtin_inff()";if(*s=="nan"||*s=="NaN")return "__builtin_nanf(\"\")";}if(auto n=v.getAsNumber())if(*n==0.0&&std::signbit(*n))return "(-0.0)";if(auto n=v.getAsInteger())return std::to_string(*n);if(auto n=v.getAsNumber()){if(std::isnan(*n))return "__builtin_nanf(\"\")";if(std::isinf(*n))return *n<0?"(-__builtin_inff())":"__builtin_inff()";std::ostringstream s;s<<std::setprecision(17)<<*n;return s.str();}throw std::runtime_error("non numeric literal");}
std::string constAttr(std::string s){s=trim(s);auto pos=s.find(" : ");if(pos!=std::string::npos)s=s.substr(0,pos);if(s=="true")return "1";if(s=="false")return "0";if(s.find("0x")==0){ // MLIR hexadecimal floating constants are bit patterns.
return "__host_f32bits("+s+"u)";
}if(s.find("inf")!=std::string::npos)return s[0]=='-'?"(-__builtin_inff())":"__builtin_inff()";if(s.find("nan")!=std::string::npos)return "__builtin_nanf(\"\")";return s;}
std::vector<int64_t> attrInts(const Obj&a,const std::string&k){std::string s=trim(str(a,k));std::vector<int64_t>r;size_t splatCount=1;if(s.rfind("dense<",0)==0){auto end=s.find('>');if(end==std::string::npos)throw std::runtime_error("malformed dense integer attribute");auto type=s.substr(end+1);auto tensor=type.find("tensor<");if(tensor!=std::string::npos){auto shape=type.substr(tensor+7);auto dimEnd=shape.find('x');if(dimEnd!=std::string::npos)splatCount=std::stoul(shape.substr(0,dimEnd));}s=s.substr(6,end-6);if(!s.empty()&&s[0]=='[')splatCount=1;}else if(s.rfind("array<",0)==0){auto colon=s.find(':');if(colon!=std::string::npos)s=s.substr(colon+1);}
std::regex re("(-?[0-9]+)([[:space:]]*:[[:space:]]*i[0-9]+)?");for(auto i=std::sregex_iterator(s.begin(),s.end(),re);i!=std::sregex_iterator();++i){if(i->position()>0&&s[i->position()-1]=='i')continue;r.push_back(std::stoll((*i)[1].str()));}if(splatCount>1&&r.size()==1)r.resize(splatCount,r[0]);return r;}
std::vector<std::string> mapResults(const std::string&s){auto p=s.find("->");if(p==std::string::npos)throw std::runtime_error("malformed affine map");auto a=s.find('(',p),b=s.rfind(')');if(a==std::string::npos||b<a)throw std::runtime_error("malformed affine results");return split(s.substr(a+1,b-a-1));}
std::string affine(std::string s,const std::vector<std::string>&dims){for(int i=(int)dims.size()-1;i>=0;i--)s=std::regex_replace(s,std::regex("\\bd"+std::to_string(i)+"\\b"),dims[i]);s=std::regex_replace(s,std::regex("\\bmod\\b"),"%");s=std::regex_replace(s,std::regex("\\bfloordiv\\b"),"/");if(s.find("ceildiv")!=std::string::npos)throw std::runtime_error("affine ceildiv unsupported");return s;}
std::string flat(const std::vector<std::string>&coords,const std::vector<int64_t>&shape){if(coords.size()!=shape.size())throw std::runtime_error("indexing map tensor rank mismatch");std::string x="0";for(size_t i=0;i<shape.size();i++)x="("+x+"*"+std::to_string(shape[i])+"+("+coords[i]+"))";return x;}
std::string load(Context&c,const std::string&name,const Obj&rec,const std::string&index="0"){if(auto v=rec.get("const_value"))return "(("+ctype(str(rec,"dtype"))+ ")("+literal(*v)+"))";if(!name.empty())return c.ptr(name)+"["+index+"]";auto src=rec.getArray("source");if(src&&src->size()>1&&(*src)[0].getAsString()&&*(*src)[0].getAsString()=="init")return "0";throw std::runtime_error("source scalar value unavailable");}
std::string scalarType(std::string t){if(t=="index")return "int64_t";return ctype(t);}
std::string unsignedType(std::string t){if(t=="i1"||t=="i8"||t=="si8"||t=="ui8")return "uint8_t";if(t=="i16"||t=="si16"||t=="ui16")return "uint16_t";if(t=="i32"||t=="si32"||t=="ui32")return "uint32_t";if(t=="i64"||t=="si64"||t=="ui64"||t=="index")return "uint64_t";throw std::runtime_error("expected integer type "+t);}
std::string scalarExpr(const Obj&op,const std::vector<std::string>&a,const std::vector<std::string>&types,const std::vector<std::string>&dims,const std::vector<std::string>&operandTypes){std::string k=str(op,"op");const Obj empty;auto ap=op.getObject("attributes");auto&at=ap?*ap:empty;auto x=[&](size_t i)->std::string {if(i>=a.size())throw std::runtime_error("scalar operand arity "+k);return "("+a[i]+")";};auto bin=[&](std::string o){return "("+x(0)+o+x(1)+")";};if(k=="arith.constant")return constAttr(str(at,"value"));if(k=="linalg.index"){int64_t d=0;auto s=str(at,"dim");if(!s.empty())d=std::stoll(s);if(d<0||size_t(d)>=dims.size())throw std::runtime_error("linalg.index dimension");return dims[d];}
if(k=="arith.addi"||k=="arith.subi"||k=="arith.muli"){
 // uint8/uint16 operands otherwise promote to signed int before multiplying.
 // Compute in an unsigned type with at least 32 bits, then truncate to the
 // declared width. An i1 result retains its low bit rather than a full byte.
 auto t=types.at(0),u=unsignedType(t);if(u=="uint8_t"||u=="uint16_t")u="uint32_t";
 auto expression="(("+u+")"+x(0)+(k=="arith.addi"?"+":k=="arith.subi"?"-":"*")+"("+u+")"+x(1)+")";
 if(t=="i1")expression="("+expression+"&1u)";
 return "(("+scalarType(t)+")"+expression+")";
}if(k=="arith.addf")return bin("+");if(k=="arith.subf")return bin("-");if(k=="arith.mulf")return bin("*");if(k=="arith.divf"||k=="arith.divsi")return bin("/");if(k=="arith.divui"||k=="arith.remui"||k=="arith.shrui"){auto u=unsignedType(operandTypes.at(0));return "(("+u+")"+x(0)+(k=="arith.divui"?"/":k=="arith.remui"?"%":">>")+"("+u+")"+x(1)+")";}if(k=="arith.remsi")return bin("%");if(k=="arith.andi")return bin("&");if(k=="arith.ori")return bin("|");if(k=="arith.xori")return bin("^");if(k=="arith.shli")return "(("+unsignedType(operandTypes.at(0))+")"+x(0)+"<<"+x(1)+")";if(k=="arith.shrsi")return bin(">>");
if(k=="arith.select")return "("+x(0)+"?"+x(1)+":"+x(2)+")";if(k=="arith.negf")return "(-"+x(0)+")";
if(k=="arith.cmpf"||k=="arith.cmpi"){std::string p=str(at,"predicate");auto colon=p.find(':');if(colon!=std::string::npos)p=trim(p.substr(0,colon));int n=std::stoi(p);static const char*ip[]={"==","!=","<","<=",">",">=","<","<=",">",">="};static const char*fp[]={"","==",">",">=","<","<=","!=","","==",">",">=","<","<=","!=","",""};if(k=="arith.cmpi"){if(n<0||n>9)throw std::runtime_error("cmpi predicate");if(n>=6){auto u=unsignedType(operandTypes.at(0));return "(("+u+")"+x(0)+ip[n]+"("+u+")"+x(1)+")";}return bin(ip[n]);}auto nan="(__builtin_isnan("+x(0)+")||__builtin_isnan("+x(1)+"))";if(n==0)return "0";if(n==15)return "1";if(n==7)return "(!"+nan+")";if(n==14)return nan;if(n<1||n>13)throw std::runtime_error("cmpf predicate");return "("+std::string(n>=8?nan+"||":"!"+nan+"&&")+bin(fp[n])+")";}
if(k=="arith.maximumf"||k=="arith.minimumf"||k=="arith.maxnumf"||k=="arith.minnumf"||k=="arith.maxsi"||k=="arith.minsi"){bool mn=k.find("min")!=std::string::npos;auto cmp="("+x(0)+(mn?"<":">")+x(1)+"?"+x(0)+":"+x(1)+")";if(k.find("f")!=std::string::npos)cmp="(("+x(0)+"==0.0f&&"+x(1)+"==0.0f)?((__builtin_signbit("+x(0)+")"+(mn?"||":"&&")+"__builtin_signbit("+x(1)+"))?-0.0f:0.0f):"+cmp+")";if(k=="arith.maximumf"||k=="arith.minimumf")return "(__builtin_isnan("+x(0)+")?"+x(0)+":(__builtin_isnan("+x(1)+")?"+x(1)+":"+cmp+"))";if(k=="arith.maxnumf"||k=="arith.minnumf")return "(__builtin_isnan("+x(0)+")?"+x(1)+":(__builtin_isnan("+x(1)+")?"+x(0)+":"+cmp+"))";return cmp;}
if(k=="arith.fptoui")return "(("+scalarType(types.at(0))+")("+unsignedType(types.at(0))+")"+x(0)+")";
if((k=="arith.trunci"||k=="arith.index_cast")&&types.at(0)=="i1")return "((uint8_t)((uint64_t)"+x(0)+"&1u))";
if((k=="arith.extsi"||k=="arith.sitofp"||k=="arith.index_cast")&&operandTypes.at(0)=="i1")return "(("+scalarType(types.at(0))+")(-((int)"+x(0)+"&1)))";
if(k=="arith.extsi"||k=="arith.trunci"||k=="arith.index_cast"||k=="arith.sitofp"||k=="arith.fptosi"||k=="arith.extf"||k=="arith.truncf")return "(("+scalarType(types.at(0))+")"+x(0)+")";
if(k=="arith.extui"||k=="arith.uitofp"){return "(("+scalarType(types.at(0))+")("+unsignedType(operandTypes.at(0))+")"+x(0)+")";}
if(k=="math.rsqrt")return "(1.0f/__builtin_sqrtf("+x(0)+"))";
static const std::map<std::string,std::string> fun={{"math.exp","expf"},{"math.exp2","exp2f"},{"math.log","logf"},{"math.log2","log2f"},{"math.sqrt","sqrtf"},{"math.tanh","tanhf"},{"math.sin","sinf"},{"math.cos","cosf"},{"math.absf","fabsf"},{"math.ceil","ceilf"},{"math.floor","floorf"},{"math.round","roundf"},{"math.roundeven","rintf"},{"math.erf","erff"},{"math.powf","powf"},{"math.fma","fmaf"}};auto it=fun.find(k);if(it!=fun.end()){std::string s="__builtin_"+it->second+"(";for(size_t i=0;i<a.size();i++){if(i)s+=",";s+=x(i);}return s+")";}throw std::runtime_error("unsupported scalar operation "+k);}
int storageBytes(const std::string&t){if(t=="i1"||t=="i8"||t=="si8"||t=="ui8")return 1;if(t=="f16"||t=="bf16"||t=="i16"||t=="si16"||t=="ui16")return 2;if(t=="f32"||t=="i32"||t=="si32"||t=="ui32")return 4;if(t=="f64"||t=="i64"||t=="si64"||t=="ui64"||t=="index")return 8;throw std::runtime_error("DMA storage type "+t);}
std::vector<int64_t> byteStrides(const Tensor&t){std::vector<int64_t>s(t.shape.size());int64_t v=storageBytes(t.dtype);for(size_t i=t.shape.size();i-->0;){s[i]=v;v*=t.shape[i];}return s;}
// Widen each stored signed byte into the accumulator, then narrow with scale
// one. Every i8 value is exact, preserving all bits including float encodings.
// Full-width copies instead bypass scaling on raw accumulator readout.
// Both configurations are explicit; contractions may change strides/scales.
void dmaRect(Context&c,const std::string&src,const std::string&dst,int64_t width,int64_t rows,int64_t ss,int64_t ds,const std::string&p,int unit=1){
 if(width<=0||rows<=0)return;
 c.usesDma=true;
 c.body<<"{gm_config_execute();gm_config_load("<<ss<<",1.0f,"<<(unit==4?0:1)<<");gm_config_store("<<ds<<",1.0f,0);\n";
 // Static shapes need only four possible tile bodies: full/full, full/tail,
 // tail/full and tail/tail. Loop bodies therefore have constant DMA geometry.
 auto tile=[&](const std::string&r,const std::string&col,int nr,int nc){
   c.body<<"gm_mvin((const unsigned char*)("<<src<<")+("<<r<<")*"<<ss<<"+("<<col<<")*"<<unit<<",0x80000000U,"<<nc<<","<<nr<<");\n"
   <<"gm_mvout((unsigned char*)("<<dst<<")+("<<r<<")*"<<ds<<"+("<<col<<")*"<<unit<<","<<(unit==4?"0xa0000000U":"0x80000000U")<<","<<nc<<","<<nr<<");\n";
 };
 auto columns=[&](const std::string&r,int nr){
   if(width>=16){auto col=p+"c";c.body<<"for(int64_t "<<col<<"=0;"<<col<<"<"<<width/16*16<<";"<<col<<"+=16){\n";tile(r,col,nr,16);c.body<<"}\n";}
   if(width%16)tile(r,std::to_string(width/16*16),nr,width%16);
 };
 if(rows>=16){auto r=p+"r";c.body<<"for(int64_t "<<r<<"=0;"<<r<<"<"<<rows/16*16<<";"<<r<<"+=16){\n";columns(r,16);c.body<<"}\n";}
 if(rows%16)columns(std::to_string(rows/16*16),rows%16);
 c.body<<"}\n";
}
void dmaLinear(Context&c,const std::string&src,const std::string&dst,int64_t bytes,const std::string&p,int unit=1){
 if(unit!=1&&unit!=4)throw std::runtime_error("invalid DMA copy word size");if(bytes%unit)throw std::runtime_error("unaligned DMA copy size");int64_t rowBytes=16*unit;
 c.body<<"gm_fence();\n";dmaRect(c,src,dst,16,bytes/rowBytes,rowBytes,rowBytes,p,unit);
 if(bytes%rowBytes)dmaRect(c,"((const unsigned char*)("+src+")+"+std::to_string(bytes/rowBytes*rowBytes)+")","((unsigned char*)("+dst+")+"+std::to_string(bytes/rowBytes*rowBytes)+")",bytes%rowBytes/unit,1,0,0,p+"tail",unit);
 c.body<<"gm_fence();\n";
}
// A rectangular affine copy with independent byte strides covers permutations,
// rank-reduced static slices, and concatenation without scalar tensor traffic.
void dmaLayout(Context&c,const std::string&src,const std::string&dst,std::vector<int64_t>shape,std::vector<int64_t>ss,std::vector<int64_t>ds,int bytes,const std::string&p){
 if(shape.size()!=ss.size()||shape.size()!=ds.size())throw std::runtime_error("DMA layout rank");
 for(auto n:shape)if(n==0)return;
 // All callers derive their base offsets from typed element indices. Preserve
 // byte address arithmetic while selecting full accumulator words for aligned
 // layouts; f64/i64 elements become two consecutive raw 32-bit words.
 int unit=bytes>=4&&bytes%4==0?4:1;
 for(auto stride:ss)if(stride%4)unit=1;
 for(auto stride:ds)if(stride%4)unit=1;
 int64_t width=bytes;size_t rank=shape.size();while(rank&&ss[rank-1]==width&&ds[rank-1]==width){width*=shape[rank-1];--rank;}
 if(width%unit)unit=1;
 if(!rank){dmaLinear(c,src,dst,width,p,unit);return;}
 c.body<<"gm_fence();{\n";std::string si="0",di="0";
 for(size_t d=0;d+1<rank;d++){auto v=p+"d"+std::to_string(d);c.body<<"for(int64_t "<<v<<"=0;"<<v<<"<"<<shape[d]<<";++"<<v<<"){\n";si+="+"+v+"*"+std::to_string(ss[d]);di+="+"+v+"*"+std::to_string(ds[d]);}
 dmaRect(c,"((const unsigned char*)("+src+")+"+si+")","((unsigned char*)("+dst+")+"+di+")",width/unit,shape[rank-1],ss[rank-1],ds[rank-1],p,unit);
 for(size_t d=0;d+1<rank;d++)c.body<<"}\n";c.body<<"}gm_fence();\n";
}
} // namespace
bool emitHost(Context&c,const Obj&rec,const std::vector<std::string>&ins,const std::vector<std::string>&outs,const std::vector<std::string>&res,const std::vector<std::string>&captures){
std::string kind=str(rec,"kind",str(rec,"operation"));auto inrec=rec.getArray("ins"),outrec=rec.getArray("outs");const Obj empty;auto attrs=rec.getObject("attributes");const Obj&at=attrs?*attrs:empty;int id=c.serial++;std::string pre="h"+std::to_string(id)+"_";
if(kind=="linalg.generic"||kind=="linalg.map"||kind=="linalg.reduce"){
 auto sb=rec.getObject("scalar_body");auto maps=rec.getArray("indexing_maps"),its=rec.getArray("iterator_types");if(!sb||!maps||!its||maps->size()!=ins.size()+outs.size()||outs.size()!=res.size())throw std::runtime_error("incomplete generic scalar/map contract");
 // Eligible int8 contractions must have been intercepted by accelerator lowering.
 if(str(rec,"family")=="contraction"&&ins.size()>=2&&!ins[0].empty()&&!ins[1].empty()&&c.tensor(ins[0]).dtype=="i8"&&c.tensor(ins[1]).dtype=="i8")throw std::runtime_error("eligible contraction reached host lowering");
 size_t rank=its->size();std::vector<int64_t> bounds(rank,-1);std::vector<std::vector<std::string>> coords;
 for(size_t n=0;n<maps->size();n++){auto s=(*maps)[n].getAsString();if(!s)throw std::runtime_error("invalid affine map");auto v=mapResults(s->str());coords.push_back(v);auto sh=ints(n<ins.size()?item(inrec,n):item(outrec,n-ins.size()),"shape");if(v.size()!=sh.size())throw std::runtime_error("affine rank mismatch");for(size_t d=0;d<v.size();d++)if(std::regex_match(v[d],std::regex("d[0-9]+"))){size_t k=std::stoul(v[d].substr(1));if(k>=rank)throw std::runtime_error("affine dim out of bounds");if(bounds[k]>=0&&bounds[k]!=sh[d])throw std::runtime_error("loop extent disagreement");bounds[k]=sh[d];}}
 for(auto b:bounds)if(b<0)throw std::runtime_error("cannot derive affine loop extent");
 // A scalar region that directly yields an input is a data movement. Derive
 // independent iterator byte strides from projected permutations and broadcasts.
 auto tryCopyRegion=[&](){
   auto operations=sb->getArray("operations"),yields=sb->getArray("yields");
   if(!operations||!operations->empty()||!yields||yields->size()!=res.size())return false;
   if(auto effects=sb->getArray("effects"))if(!effects->empty())return false;
   for(auto&iterator:*its)if(iterator.getAsString().value_or("")!="parallel")return false;
   struct CopyPlan{std::string src,dst;std::vector<int64_t>ss,ds;int bytes;};std::vector<CopyPlan>plans;
   auto strides=[&](size_t mapIndex,const Tensor&t,bool output,std::vector<int64_t>&result){
     if(coords[mapIndex].size()!=t.shape.size())return false;
     auto physical=byteStrides(t);result.assign(rank,0);std::set<size_t>seen;
     for(size_t d=0;d<t.shape.size();d++){
       auto expression=trim(coords[mapIndex][d]);
       if(expression=="0"){if(t.shape[d]!=1)return false;continue;}
       if(!std::regex_match(expression,std::regex("d[0-9]+")))return false;
       size_t k=std::stoul(expression.substr(1));if(k>=rank||!seen.insert(k).second||t.shape[d]!=bounds[k])return false;
       result[k]=physical[d];
     }
     if(output)for(size_t k=0;k<rank;k++)if(bounds[k]>1&&!seen.count(k))return false;
     return true;
   };
   for(size_t n=0;n<res.size();n++){
     auto ref=(*yields)[n].getAsString();if(!ref)return false;auto name=ref->str();
     if(!std::regex_match(name,std::regex("arg:[0-9]+")))return false;
     size_t i=std::stoul(name.substr(4));if(i>=ins.size()||ins[i].empty())return false;
     const auto&source=c.tensor(ins[i]);const auto&target=c.tensor(res[n]);if(source.dtype!=target.dtype)return false;
     // Do not reorder multi-result observations when storage aliases an input.
     for(auto&result:res)if(c.ptr(result)==source.var)return false;
     CopyPlan plan{source.var,target.var,{},{},storageBytes(target.dtype)};
     if(!strides(i,source,false,plan.ss)||!strides(ins.size()+n,target,true,plan.ds))return false;
     plans.push_back(std::move(plan));
   }
   for(size_t n=0;n<plans.size();n++){auto&p=plans[n];dmaLayout(c,p.src,p.dst,bounds,p.ss,p.ds,p.bytes,pre+"copy"+std::to_string(n));}
   return true;
 };
 if(tryCopyRegion())return true;
 // Keep reduction state in scalar locals when parallel coordinates identify
 // exactly one output and every reduction iterator is a suffix. The scalar DAG
 // and reduction iteration order remain unchanged.
 size_t parallelRank=0;while(parallelRank<rank&&(*its)[parallelRank].getAsString().value_or("")=="parallel")++parallelRank;
 bool registerReduction=parallelRank<rank;
 for(size_t k=parallelRank;k<rank;k++)if((*its)[k].getAsString().value_or("")!="reduction")registerReduction=false;
 for(size_t n=0;n<res.size()&&registerReduction;n++){
   auto shape=ints(item(outrec,n),"shape");std::set<size_t>seen;
   for(size_t d=0;d<shape.size();d++){
     const auto&expression=coords[ins.size()+n][d];
     if(expression=="0"&&shape[d]==1)continue;
     if(!std::regex_match(expression,std::regex("d[0-9]+"))){registerReduction=false;break;}
     size_t k=std::stoul(expression.substr(1));if(k>=parallelRank||!seen.insert(k).second||bounds[k]!=shape[d]){registerReduction=false;break;}
   }
   for(size_t k=0;k<parallelRank;k++)if(bounds[k]>1&&!seen.count(k))registerReduction=false;
 }
 // Destination initialization is observable only if the scalar DAG reads an
 // outs argument, or if its output map does not prove complete coverage.
 std::set<std::string> bodyReads;
 if(auto operations=sb->getArray("operations"))for(auto&v:*operations)if(auto operation=v.getAsObject())if(auto operands=operation->getArray("operands"))for(auto&operand:*operands)if(auto ref=operand.getAsString())bodyReads.insert(ref->str());
 if(auto yields=sb->getArray("yields"))for(auto&yield:*yields)if(auto ref=yield.getAsString())bodyReads.insert(ref->str());
 c.body<<"{\n";for(size_t n=0;n<res.size()&&!registerReduction;n++){auto&t=c.tensor(res[n]);bool full=true;std::set<size_t> usedDims;auto shape=ints(item(outrec,n),"shape");auto&map=coords[ins.size()+n];
 for(size_t d=0;d<map.size();d++){
   if(std::regex_match(map[d],std::regex("d[0-9]+"))){size_t k=std::stoul(map[d].substr(1));if(k>=rank||!usedDims.insert(k).second||bounds[k]!=shape[d])full=false;}
   else if(!(map[d]=="0"&&shape[d]==1))full=false;
 }
 if(full&&!bodyReads.count("arg:"+std::to_string(ins.size()+n)))continue;
 std::string init="0";if(n<outs.size()&&!outs[n].empty())init=c.ptr(outs[n])+"["+pre+"z]";else if(auto v=item(outrec,n).get("const_value"))init=literal(*v);c.body<<"for(int64_t "<<pre<<"z=0;"<<pre<<"z<"<<t.count()<<";++"<<pre<<"z)"<<t.var<<"["<<pre<<"z]="<<init<<";\n";}
 std::vector<std::string>dims;for(size_t n=0;n<rank;n++)dims.push_back(pre+"d"+std::to_string(n));
 auto index=[&](size_t n,const Obj&o){std::vector<std::string>v;for(auto&e:coords[n])v.push_back(affine(e,dims));return flat(v,ints(o,"shape"));};
 // Cache immutable SSA inputs at the shallowest loop scope that defines their
 // map dimensions. Broadcast scalars/vectors are then read once per shared run.
 std::vector<std::vector<size_t>>cacheAt(rank+1);std::vector<std::string>cached(ins.size());
 for(size_t n=0;n<ins.size();n++)if(!ins[n].empty()&&std::none_of(bounds.begin(),bounds.end(),[](int64_t n){return n==0;})){
   bool aliases=false;for(auto&result:res)if(c.ptr(ins[n])==c.ptr(result))aliases=true;if(aliases)continue;
   size_t scope=0;for(auto&expression:coords[n]){std::regex dim("\\bd([0-9]+)\\b");for(auto it=std::sregex_iterator(expression.begin(),expression.end(),dim);it!=std::sregex_iterator();++it)scope=std::max(scope,size_t(std::stoul((*it)[1].str())+1));}
   if(scope<rank){cacheAt[scope].push_back(n);cached[n]=pre+"arg"+std::to_string(n);}
 }
 auto emitCaches=[&](size_t scope){for(auto n:cacheAt[scope])c.body<<ctype(c.tensor(ins[n]).dtype)<<" "<<cached[n]<<"="<<load(c,ins[n],item(inrec,n),index(n,item(inrec,n)))<<";\n";};
 auto emitAccumulators=[&](){for(size_t n=0;n<res.size();n++){std::string init="0";if(!outs[n].empty())init=load(c,outs[n],item(outrec,n),index(ins.size()+n,item(outrec,n)));else if(auto v=item(outrec,n).get("const_value"))init=literal(*v);c.body<<ctype(c.tensor(res[n]).dtype)<<" "<<pre<<"acc"<<n<<"="<<init<<";\n";}};
 emitCaches(0);if(registerReduction&&parallelRank==0)emitAccumulators();
 for(size_t n=0;n<rank;n++){c.body<<"for(int64_t "<<dims[n]<<"=0;"<<dims[n]<<"<"<<bounds[n]<<";++"<<dims[n]<<"){\n";emitCaches(n+1);if(registerReduction&&n+1==parallelRank)emitAccumulators();}
 std::map<std::string,std::string> valueTypes;if(auto a=sb->getArray("arguments"))for(size_t i=0;i<a->size();i++)valueTypes["arg:"+std::to_string(i)]=(*a)[i].getAsString()->str();if(auto a=sb->getArray("captures"))for(size_t i=0;i<a->size();i++)valueTypes["capture:"+std::to_string(i)]=str(item(a,i),"dtype");std::map<std::string,std::string> values;for(size_t n=0;n<ins.size();n++)values["arg:"+std::to_string(n)]=cached[n].empty()?load(c,ins[n],item(inrec,n),index(n,item(inrec,n))):cached[n];for(size_t n=0;n<res.size();n++)values["arg:"+std::to_string(ins.size()+n)]=registerReduction?pre+"acc"+std::to_string(n):c.ptr(res[n])+"["+index(ins.size()+n,item(outrec,n))+"]";
 // A captured tensor is an SSA value with its full shape, not its first element.
 // Keep tensor.extract's source reference distinct from scalar captures, including rank zero.
 std::set<std::string> tensorReferences;
 if(auto operations=sb->getArray("operations"))for(auto&v:*operations)if(auto operation=v.getAsObject())if(str(*operation,"op")=="tensor.extract")if(auto operands=operation->getArray("operands"))if(!operands->empty())if(auto ref=(*operands)[0].getAsString())tensorReferences.insert(ref->str());
 std::map<std::string,std::vector<int64_t>> tensorShapes;
 if(auto cr=sb->getArray("captures"))for(size_t n=0;n<cr->size();n++){
   auto ref="capture:"+std::to_string(n);const auto&capture=item(cr,n);auto shape=ints(capture,"shape");auto name=n<captures.size()?captures[n]:"";
   if(!shape.empty()||tensorReferences.count(ref)){
     tensorShapes[ref]=shape;
     if(!name.empty())values[ref]=c.ptr(name);
     else if(auto scalar=capture.get("const_value")){
       // Constant splats can answer every extraction with the same scalar value.
       values[ref]="";
       values[ref+":splat"]="(("+ctype(str(capture,"dtype"))+")("+literal(*scalar)+"))";
     }else if(auto data=capture.getArray("const_values")){
       int64_t count=1;for(auto d:shape)count*=d;if(int64_t(data->size())!=count)throw std::runtime_error("captured dense constant shape mismatch");
       auto var=pre+"capture"+std::to_string(n);c.globals<<"static const "<<ctype(str(capture,"dtype"))<<" "<<var<<"["<<count<<"]={";
       for(size_t i=0;i<data->size();i++){if(i)c.globals<<",";c.globals<<literal((*data)[i]);}c.globals<<"};\n";values[ref]=var;
     }else throw std::runtime_error("captured tensor storage unavailable");
   }else values[ref]=load(c,name,capture);
 }
 auto ops=sb->getArray("operations");if(!ops)throw std::runtime_error("missing scalar operations");for(size_t n=0;n<ops->size();n++){const auto&o=item(ops,n);std::vector<std::string>a,types,operandTypes;if(auto os=o.getArray("operands"))for(auto&v:*os){auto s=v.getAsString();if(!s||!values.count(s->str()))throw std::runtime_error("unknown scalar SSA edge");a.push_back(values[s->str()]);operandTypes.push_back(valueTypes.at(s->str()));}if(auto ts=o.getArray("results"))for(auto&v:*ts){auto s=v.getAsString();if(!s)throw std::runtime_error("scalar result type");types.push_back(s->str());}if(types.size()!=1)throw std::runtime_error("unsupported scalar multi-result operation");auto var=pre+"s"+std::to_string(n);std::string expr;
 if(str(o,"op")=="tensor.extract"){
   auto operands=o.getArray("operands");if(!operands||operands->empty())throw std::runtime_error("tensor.extract source missing");
   auto ref=(*operands)[0].getAsString();if(!ref||!tensorShapes.count(ref->str()))throw std::runtime_error("tensor.extract source has no tensor shape");
   const auto&shape=tensorShapes.at(ref->str());if(a.size()!=shape.size()+1)throw std::runtime_error("tensor.extract index rank mismatch");
   if(values.count(ref->str()+":splat"))expr=values.at(ref->str()+":splat");
   else expr="("+a[0]+")["+flat(std::vector<std::string>(a.begin()+1,a.end()),shape)+"]";
 }else expr=scalarExpr(o,a,types,dims,operandTypes);
 c.body<<scalarType(types[0])<<" "<<var<<"="<<expr<<";\n";values["op:"+std::to_string(n)+":0"]=var;valueTypes["op:"+std::to_string(n)+":0"]=types[0];}
 auto yields=sb->getArray("yields");if(!yields||yields->size()!=res.size())throw std::runtime_error("scalar yield arity"); // Stage simultaneous yields to preserve multi-result reduction semantics.
 for(size_t n=0;n<res.size();n++){auto y=(*yields)[n].getAsString();if(!y||!values.count(y->str()))throw std::runtime_error("unknown yield edge");c.body<<ctype(c.tensor(res[n]).dtype)<<" "<<pre<<"y"<<n<<"="<<values[y->str()]<<";\n";}
 for(size_t n=0;n<res.size();n++)c.body<<(registerReduction?pre+"acc"+std::to_string(n):c.ptr(res[n])+"["+index(ins.size()+n,item(outrec,n))+"]")<<"="<<pre<<"y"<<n<<";\n";
 if(registerReduction){
   for(size_t n=parallelRank;n<rank;n++)c.body<<"}\n";
   for(size_t n=0;n<res.size();n++)c.body<<c.ptr(res[n])<<"["<<index(ins.size()+n,item(outrec,n))<<"]="<<pre<<"acc"<<n<<";\n";
   for(size_t n=0;n<parallelRank;n++)c.body<<"}\n";
 }else for(size_t n=0;n<rank;n++)c.body<<"}\n";
 c.body<<"}\n";return true;
}
if(res.empty())return false;
auto&dst=c.tensor(res[0]);
if(kind=="linalg.matmul"||kind=="linalg.batch_matmul"||kind=="linalg.matmul_transpose_b"){
 if(ins.size()!=2||ins[0].empty()||ins[1].empty()||res.size()!=1)throw std::runtime_error("named matmul operand arity");auto a=c.tensor(ins[0]),b=c.tensor(ins[1]);bool trans=kind=="linalg.matmul_transpose_b",batch=kind=="linalg.batch_matmul";size_t rank=batch?3:2;
 if(a.shape.size()!=rank||b.shape.size()!=rank||dst.shape.size()!=rank)throw std::runtime_error("named matmul rank");if(a.dtype=="i8"&&b.dtype=="i8")throw std::runtime_error("eligible named matmul reached host lowering");if(a.dtype!=dst.dtype||b.dtype!=dst.dtype||(dst.dtype!="f32"&&dst.dtype!="f64"&&dst.dtype!="f16"))throw std::runtime_error("named matmul host precision unsupported");int64_t B=batch?a.shape[0]:1,M=a.shape[rank-2],K=a.shape[rank-1],N=b.shape[rank-(trans?2:1)];if(b.shape[rank-(trans?1:2)]!=K||dst.shape[rank-2]!=M||dst.shape[rank-1]!=N||(batch&&(b.shape[0]!=B||dst.shape[0]!=B)))throw std::runtime_error("named matmul extent mismatch");
 std::string q=pre+"b",m=pre+"m",n=pre+"n",k=pre+"k",z="("+q+"*"+std::to_string(M*N)+"+"+m+"*"+std::to_string(N)+"+"+n+")",ai="("+q+"*"+std::to_string(M*K)+"+"+m+"*"+std::to_string(K)+"+"+k+")",bi="("+q+"*"+std::to_string(K*N)+"+"+(trans?n+"*"+std::to_string(K)+"+"+k:k+"*"+std::to_string(N)+"+"+n)+")";std::string init="0";if(!outs.empty())init=load(c,outs[0],item(outrec,0),z);
 c.body<<"for(int64_t "<<q<<"=0;"<<q<<"<"<<B<<";++"<<q<<")for(int64_t "<<m<<"=0;"<<m<<"<"<<M<<";++"<<m<<")for(int64_t "<<n<<"=0;"<<n<<"<"<<N<<";++"<<n<<"){\n"<<ctype(dst.dtype)<<" "<<pre<<"acc="<<init<<";\nfor(int64_t "<<k<<"=0;"<<k<<"<"<<K<<";++"<<k<<"){\n"<<ctype(dst.dtype)<<" "<<pre<<"prod="<<a.var<<"["<<ai<<"]*"<<b.var<<"["<<bi<<"];\n"<<pre<<"acc="<<pre<<"acc+"<<pre<<"prod;\n}\n"<<dst.var<<"["<<z<<"]="<<pre<<"acc;\n}\n";return true;
}
if(kind=="linalg.conv_2d_nhwc_hwcf"||kind=="linalg.conv_2d_nchw_fchw"){
 if(ins.size()!=2||ins[0].empty()||ins[1].empty()||res.size()!=1)throw std::runtime_error("named convolution operands");auto a=c.tensor(ins[0]),w=c.tensor(ins[1]);bool nhwc=kind=="linalg.conv_2d_nhwc_hwcf";if(a.shape.size()!=4||w.shape.size()!=4||dst.shape.size()!=4)throw std::runtime_error("named convolution rank");if(a.dtype=="i8"&&w.dtype=="i8")throw std::runtime_error("eligible named convolution reached host lowering");if(a.dtype!=dst.dtype||w.dtype!=dst.dtype||(dst.dtype!="f32"&&dst.dtype!="f64"&&dst.dtype!="f16"))throw std::runtime_error("named convolution host precision unsupported");auto strides=attrInts(at,"strides"),dilations=attrInts(at,"dilations");if(strides.empty())strides={1,1};if(dilations.empty())dilations={1,1};if(strides.size()!=2||dilations.size()!=2)throw std::runtime_error("convolution strides rank");int64_t B=a.shape[0],C=a.shape[nhwc?3:1],KH=w.shape[nhwc?0:2],KW=w.shape[nhwc?1:3],F=w.shape[nhwc?3:0],OH=dst.shape[nhwc?1:2],OW=dst.shape[nhwc?2:3];if(w.shape[nhwc?2:1]!=C||dst.shape[0]!=B||dst.shape[nhwc?3:1]!=F)throw std::runtime_error("convolution channels mismatch");
 std::vector<std::string>ds;std::vector<int64_t>bounds={B,OH,OW,F,KH,KW,C};for(int i=0;i<7;i++)ds.push_back(pre+"c"+std::to_string(i));auto n=ds[0],oh=ds[1],ow=ds[2],f=ds[3],kh=ds[4],kw=ds[5],ch=ds[6],ih="("+oh+"*"+std::to_string(strides[0])+"+"+kh+"*"+std::to_string(dilations[0])+")",iw="("+ow+"*"+std::to_string(strides[1])+"+"+kw+"*"+std::to_string(dilations[1])+")";auto oi=flat(nhwc?std::vector<std::string>{n,oh,ow,f}:std::vector<std::string>{n,f,oh,ow},dst.shape),ai=flat(nhwc?std::vector<std::string>{n,ih,iw,ch}:std::vector<std::string>{n,ch,ih,iw},a.shape),wi=flat(nhwc?std::vector<std::string>{kh,kw,ch,f}:std::vector<std::string>{f,ch,kh,kw},w.shape);std::string init="0";if(!outs.empty())init=load(c,outs[0],item(outrec,0),oi);
 for(int d=0;d<4;d++)c.body<<"for(int64_t "<<ds[d]<<"=0;"<<ds[d]<<"<"<<bounds[d]<<";++"<<ds[d]<<"){\n";c.body<<ctype(dst.dtype)<<" "<<pre<<"acc="<<init<<";\n";std::vector<int>order=nhwc?std::vector<int>{4,5,6}:std::vector<int>{6,4,5};for(int d:order)c.body<<"for(int64_t "<<ds[d]<<"=0;"<<ds[d]<<"<"<<bounds[d]<<";++"<<ds[d]<<"){\n";c.body<<ctype(dst.dtype)<<" "<<pre<<"prod="<<a.var<<"["<<ai<<"]*"<<w.var<<"["<<wi<<"];\n"<<pre<<"acc="<<pre<<"acc+"<<pre<<"prod;\n";for(int i=0;i<3;i++)c.body<<"}\n";c.body<<dst.var<<"["<<oi<<"]="<<pre<<"acc;\n";for(int i=0;i<4;i++)c.body<<"}\n";return true;
}
if(kind=="quant_ext.quantize_per_tensor"||kind=="quant_ext.dequantize_per_tensor"){
 if(ins.size()<3||ins[0].empty())throw std::runtime_error("per tensor quantization operands");if(dst.count()==0)return true;std::string scale=pre+"scale",zero=pre+"zero";c.body<<"{\n"<<ctype(str(item(inrec,1),"dtype"))<<" "<<scale<<"="<<load(c,ins[1],item(inrec,1))<<";\n"<<ctype(str(item(inrec,2),"dtype"))<<" "<<zero<<"="<<load(c,ins[2],item(inrec,2))<<";\n";std::string i=pre+"i",x=c.ptr(ins[0])+"["+i+"]";c.body<<"for(int64_t "<<i<<"=0;"<<i<<"<"<<dst.count()<<";++"<<i<<"){\n";
 if(kind=="quant_ext.dequantize_per_tensor")c.body<<dst.var<<"["<<i<<"]=((float)"<<x<<"-(float)("<<zero<<"))*(float)("<<scale<<");\n";
 else{auto mn=constAttr(str(at,"quant_min","-128")),mx=constAttr(str(at,"quant_max","127"));auto q=pre+"q";c.body<<"float "<<q<<"=__builtin_rintf("<<x<<"/(float)("<<scale<<"))+(float)("<<zero<<");\n"<<q<<"="<<q<<"<"<<mn<<"?"<<mn<<":"<<q<<";\n"<<q<<"="<<q<<">"<<mx<<"?"<<mx<<":"<<q<<";\n"<<dst.var<<"["<<i<<"]=("<<ctype(dst.dtype)<<")"<<q<<";\n";}c.body<<"}}\n";return true;
}
if(kind=="tensor.collapse_shape"||kind=="tensor.expand_shape"||kind=="tensor.cast"||kind=="tensor.reshape"){
 if(ins.empty()||ins[0].empty()||c.tensor(ins[0]).count()!=dst.count()||c.tensor(ins[0]).dtype!=dst.dtype)throw std::runtime_error("invalid tensor reshape");dmaLinear(c,c.ptr(ins[0]),dst.var,dst.count()*storageBytes(dst.dtype),pre,storageBytes(dst.dtype)>=4?4:1);return true;
}
if(kind=="tensor.empty"||kind=="tensor.splat"||kind=="linalg.fill"){
 std::string val="0";bool constant=kind=="tensor.empty";if(kind!="tensor.empty"){if(ins.empty())throw std::runtime_error("fill input missing");val=load(c,ins[0],item(inrec,0));constant=ins[0].empty()&&item(inrec,0).get("const_value");}
 if(constant){int bytes=storageBytes(dst.dtype);auto tile=pre+"fill";c.globals<<"static const "<<ctype(dst.dtype)<<" "<<tile<<"["<<16/bytes<<"]={";for(int i=0;i<16/bytes;i++){if(i)c.globals<<",";c.globals<<val;}c.globals<<"};\n";int64_t count=dst.count()*bytes;c.body<<"gm_fence();\n";dmaRect(c,tile,dst.var,bytes>=4?4:16,count/16,0,16,pre,bytes>=4?4:1);if(count%16)dmaRect(c,tile,"((unsigned char*)"+dst.var+"+"+std::to_string(count/16*16)+")",count%16/(bytes>=4?4:1),1,0,0,pre+"tail",bytes>=4?4:1);c.body<<"gm_fence();\n";}
 else c.body<<"for(int64_t "<<pre<<"i=0;"<<pre<<"i<"<<dst.count()<<";++"<<pre<<"i)"<<dst.var<<"["<<pre<<"i]="<<val<<";\n";return true;
}
if(kind=="arith.constant"){
 std::string s=str(at,"value");if(dst.shape.empty()&&s.find("dense<")==std::string::npos){c.body<<dst.var<<"[0]="<<constAttr(s)<<";\n";return true;}auto p=s.find("dense<"),end=s.rfind('>');if(p==std::string::npos)throw std::runtime_error("tensor constant lacks dense value");std::string v=trim(s.substr(p+6,end-p-6));auto close=v.find(">");if(close!=std::string::npos)v=v.substr(0,close);std::vector<std::string>vals;if(!v.empty()&&v[0]=='['){std::string atom;for(char ch:v){if(ch=='['||ch==']'||ch==','){if(!trim(atom).empty())vals.push_back(trim(atom));atom.clear();}else atom+=ch;}if(!trim(atom).empty())vals.push_back(trim(atom));}else vals.push_back(v);if(vals.size()!=1&&int64_t(vals.size())!=dst.count())throw std::runtime_error("dense constant shape mismatch");if(vals.size()==1)c.body<<"for(int64_t "<<pre<<"i=0;"<<pre<<"i<"<<dst.count()<<";++"<<pre<<"i)"<<dst.var<<"["<<pre<<"i]="<<constAttr(vals[0])<<";\n";else for(size_t i=0;i<vals.size();i++)c.body<<dst.var<<"["<<i<<"]="<<constAttr(vals[i])<<";\n";return true;
}
if(kind=="linalg.transpose"){
 if(ins.empty()||ins[0].empty())throw std::runtime_error("transpose input missing");auto perm=attrInts(at,"permutation");auto&src=c.tensor(ins[0]);if(perm.size()!=dst.shape.size()||perm.size()!=src.shape.size()||src.dtype!=dst.dtype)throw std::runtime_error("transpose rank/type");auto ss=byteStrides(src),ds=byteStrides(dst);std::vector<int64_t>ps;std::set<int64_t>seen;for(size_t d=0;d<perm.size();d++){if(perm[d]<0||size_t(perm[d])>=ss.size()||!seen.insert(perm[d]).second||src.shape[perm[d]]!=dst.shape[d])throw std::runtime_error("transpose permutation");ps.push_back(ss[perm[d]]);}dmaLayout(c,src.var,dst.var,dst.shape,ps,ds,storageBytes(dst.dtype),pre);return true;
}
if(kind=="tensor.extract_slice"||kind=="tensor.insert_slice"){
 bool insert=kind=="tensor.insert_slice";if(ins.empty()||ins[0].empty()||(insert&&(ins.size()<2||ins[1].empty())))throw std::runtime_error("slice operands");auto off=attrInts(at,"static_offsets"),sizes=attrInts(at,"static_sizes"),steps=attrInts(at,"static_strides");auto&src=c.tensor(ins[0]);auto&large=insert?dst:src;auto&small=insert?src:dst;if(off.size()!=large.shape.size()||steps.size()!=off.size()||sizes.size()!=off.size()||src.dtype!=dst.dtype)throw std::runtime_error("dynamic slice/type unsupported");auto ls=byteStrides(large),sm=byteStrides(small);std::vector<int64_t>smallStrides(sizes.size(),0);size_t sr=0;int64_t base=0;for(size_t d=0;d<sizes.size();d++){if(off[d]<0||sizes[d]<0||steps[d]<=0||(sizes[d]&&off[d]+(sizes[d]-1)*steps[d]>=large.shape[d]))throw std::runtime_error("invalid static slice");base+=off[d]*ls[d];ls[d]*=steps[d];if(sizes.size()==small.shape.size()||sizes[d]!=1||small.shape.size()-sr==sizes.size()-d){if(sr>=sm.size()||small.shape[sr]!=sizes[d])throw std::runtime_error("slice shape mismatch");smallStrides[d]=sm[sr++];}}if(sr!=small.shape.size())throw std::runtime_error("slice rank reduction mismatch");
 if(insert){if(c.tensor(ins[1]).dtype!=dst.dtype||c.tensor(ins[1]).count()!=dst.count())throw std::runtime_error("insert destination type");dmaLinear(c,c.ptr(ins[1]),dst.var,dst.count()*storageBytes(dst.dtype),pre+"init");dmaLayout(c,src.var,"((unsigned char*)"+dst.var+"+"+std::to_string(base)+")",sizes,smallStrides,ls,storageBytes(dst.dtype),pre);}
 else dmaLayout(c,"((const unsigned char*)"+src.var+"+"+std::to_string(base)+")",dst.var,sizes,ls,smallStrides,storageBytes(dst.dtype),pre);return true;
}
if(kind=="tensor.concat"){
 auto dim=attrInts(at,"dim");if(dim.empty())throw std::runtime_error("concat dimension absent");int64_t axis=dim[0],offset=0;if(axis<0||size_t(axis)>=dst.shape.size())throw std::runtime_error("concat axis");auto ds=byteStrides(dst);for(size_t n=0;n<ins.size();n++){if(ins[n].empty())throw std::runtime_error("concat input");auto&src=c.tensor(ins[n]);if(src.dtype!=dst.dtype||src.shape.size()!=dst.shape.size())throw std::runtime_error("concat type/rank");for(size_t d=0;d<src.shape.size();d++)if(d!=size_t(axis)&&src.shape[d]!=dst.shape[d])throw std::runtime_error("concat extent");dmaLayout(c,src.var,"((unsigned char*)"+dst.var+"+"+std::to_string(offset*ds[axis])+")",src.shape,byteStrides(src),ds,storageBytes(dst.dtype),pre+"part"+std::to_string(n));offset+=src.shape[axis];}if(offset!=dst.shape[axis])throw std::runtime_error("concat result extent");return true;
}
return false;
}
} // namespace mc
