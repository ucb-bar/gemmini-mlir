#ifndef GEMMINI_TYPES_H
#define GEMMINI_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "Gemmini/GemminiDialect.h"

#define GET_TYPEDEF_CLASSES
#include "Gemmini/GemminiOpsTypes.h.inc"

#endif // GEMMINI_TYPES_H
