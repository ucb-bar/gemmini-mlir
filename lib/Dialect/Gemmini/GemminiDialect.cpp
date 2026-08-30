//===- GemminiDialect.cpp - gemmini target dialect registration -------------===//
#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/GemminiTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace gemmini;

#include "Gemmini/GemminiOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Gemmini/GemminiOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Gemmini/GemminiOps.cpp.inc"

void GemminiDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Gemmini/GemminiOps.cpp.inc"
      >();
  registerTypes();
}

void GemminiDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Gemmini/GemminiOpsTypes.cpp.inc"
      >();
}
