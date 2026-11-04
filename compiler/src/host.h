#pragma once
#include "compiler.h"
namespace mc {
bool emitHost(Context&,const Obj&,const std::vector<std::string>& inputs,const std::vector<std::string>& outputs,const std::vector<std::string>& results,const std::vector<std::string>& captures={});
}
