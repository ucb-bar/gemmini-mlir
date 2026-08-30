#ifndef GEMMINI_OPS_H
#define GEMMINI_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Gemmini/GemminiTypes.h"

#define GET_OP_CLASSES
#include "Gemmini/GemminiOps.h.inc"

#endif // GEMMINI_OPS_H
