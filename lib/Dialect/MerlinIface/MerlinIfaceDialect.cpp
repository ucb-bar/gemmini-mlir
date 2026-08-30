//===- MerlinIfaceDialect.cpp - merlin_iface input dialect registration -----===//
#include "MerlinIface/MerlinIfaceDialect.h"
#include "MerlinIface/MerlinIfaceOps.h"
#include "MerlinIface/MerlinIfaceTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace merlin_iface;

#include "MerlinIface/MerlinIfaceOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "MerlinIface/MerlinIfaceOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "MerlinIface/MerlinIfaceOps.cpp.inc"

void MerlinIfaceDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MerlinIface/MerlinIfaceOps.cpp.inc"
      >();
  registerTypes();
}

void MerlinIfaceDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "MerlinIface/MerlinIfaceOpsTypes.cpp.inc"
      >();
}
