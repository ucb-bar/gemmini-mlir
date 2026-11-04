#pragma once
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
namespace backend {
using Object=llvm::json::Object; using Array=llvm::json::Array; using Value=llvm::json::Value;
// Shared normalized frontend: numerical_program.v1 plus source_op_count,
// expressions[].source_op_indices and optional expressions[].region_id.
// JSON input expressions map 1:1 to source ops. Generic MLIR may group source
// operations but must cover all direct ops, including empty/constant/fill.
// Additional expression operators may be added with explicit interfaces.
Object parseProgram(const std::string &source); // frontend.cpp; detects JSON/MLIR
Object lowerCommands(const Object &program, const std::string &sourceSHA); // commands.cpp
// Device helpers authored from RTL; generated C is freestanding RV64GC.
std::string residualRuntime();
std::string deviceRuntime(); // device.cpp
std::string blockedRuntime();
std::string convRuntime();
std::string poolRuntime();
std::string poolHardwareRuntime();
std::string compilerSettingString(const std::string&name,const std::string&fallback);
bool compilerSettingBool(const std::string&name,bool fallback); // blocked.cpp
// Emit call computing dense row-major i8 A[M,K] @ i8 B[K,N] -> i32 C[M,N].
// Inputs C expressions are pointers; task/source tagging handled by parent.
std::string deviceMatmul(const std::string&A,const std::string&B,const std::string&C,
 int64_t M,int64_t K,int64_t N); // device.cpp
std::string deviceMovement(const std::string&A,const std::string&C,int64_t M,int64_t N,const std::string &outDtype); // device.cpp
std::string emitC(const Object &program, const Object &cb); // codegen.cpp
inline std::string str(const Object&o,llvm::StringRef k,std::string d=""){auto s=o.getString(k);return s?s->str():d;}
inline int64_t integer(const Object&o,llvm::StringRef k,int64_t d=0){auto v=o.getInteger(k);return v?*v:d;}
inline const Array& arr(const Object&o,llvm::StringRef k){auto*p=o.getArray(k);if(!p)throw std::runtime_error("missing array "+k.str());return *p;}
inline std::vector<int64_t> shape(const Object&o){std::vector<int64_t>s;for(auto&v:arr(o,"shape"))s.push_back(*v.getAsInteger());return s;}
inline int64_t elements(const Object&o){int64_t n=1;for(auto x:shape(o))n*=x;return n;}
inline Object clone(const Object&o){return Object(o);}
}
