#include "compiler.h"
#include <fstream>
#include <sstream>
namespace backend {
namespace {
const Object &settings() {
 static const Object loaded=[] {
  std::ifstream input(DEFAULT_COMPILER_SETTINGS);
  if(!input)return Object{};
  std::ostringstream bytes;bytes<<input.rdbuf();
  auto parsed=llvm::json::parse(bytes.str());
  if(!parsed)throw std::runtime_error("invalid compiler settings: "+llvm::toString(parsed.takeError()));
  auto *object=parsed->getAsObject();
  if(!object)throw std::runtime_error("compiler settings must be an object");
  return Object(*object);
 }();
 return loaded;
}
}
std::string compilerSettingString(const std::string&name,const std::string&fallback) {
 auto value=settings().getString(name);return value?value->str():fallback;
}
bool compilerSettingBool(const std::string&name,bool fallback) {
 auto value=settings().getBoolean(name);return value?*value:fallback;
}
}
