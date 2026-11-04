#pragma once
#include "llvm/Support/JSON.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <stdexcept>
namespace mc {
using Obj=llvm::json::Object; using Arr=llvm::json::Array; using Val=llvm::json::Value;
inline std::string str(const Obj &o, llvm::StringRef k, std::string d=""){auto x=o.getString(k);return x?x->str():d;}
inline int64_t num(const Obj &o,llvm::StringRef k,int64_t d=0){auto x=o.getInteger(k);return x?*x:d;}
inline double real(const Obj &o,llvm::StringRef k,double d=0){auto x=o.getNumber(k);return x?*x:d;}
inline std::vector<int64_t> ints(const Obj&o,llvm::StringRef k){std::vector<int64_t> r;if(auto a=o.getArray(k))for(auto &v:*a)if(auto n=v.getAsInteger())r.push_back(*n);return r;}
inline bool hasStage(const Obj&o,const std::string&s){if(auto a=o.getArray("epilogue"))for(auto&v:*a)if(v.getAsString()&&*v.getAsString()==s)return true;return false;}
inline Val copy(const Val&v){return llvm::cantFail(llvm::json::parse(llvm::formatv("{0}",v).str()));}
struct Tensor{std::string name,dtype,role,var;std::vector<int64_t> shape;int64_t count()const{int64_t n=1;for(auto x:shape)n*=x;return n;}};
struct Context {
 Obj cb; Obj inventory; std::string source,sourceHash; std::vector<Tensor> tensors; std::map<std::string,size_t> names;
 std::ostringstream globals,body; int serial=0; bool usesDma=false;
 Tensor &tensor(const std::string&n){auto it=names.find(n);if(it==names.end())throw std::runtime_error("unknown tensor "+n);return tensors[it->second];}
 std::string ptr(const std::string&n){return tensor(n).var;}
};
// Frontend turns the supplied semantic reader's JSON into a command buffer.
void lowerInventory(Context& c);
// Common command-buffer kernel dispatch; helpers emit C, later translated to LLVM dialect.
void emitKernel(Context& c);
// All accelerator helpers below are force-inlined in generated C.
std::string deviceSupport();
// Emit tiled i8 matmul, with optional transposed RHS; epilogue attributes control readout.
// Expressions are C pointer expressions; batch calls can use offsets. Output is logical row-major.
void emitMatmul(Context&c,const std::string&A,const std::string&W,const std::string&Y,int64_t M,int64_t K,int64_t N,const Obj&attrs,const std::string&bias="",bool transpose=false);
void emitConv(Context&c,const Obj&command);
void emitPool(Context&c,const std::string&src,const std::string&dst,int64_t M,int64_t N,const Obj&attrs,const std::string&tag);
bool emitOther(Context&c,const Obj&command);
void emitReadout(Context&c,const std::string&src,const std::string&dst,int64_t rows,int64_t cols,const Obj&attrs);
std::string ctype(const std::string&dtype);
}
