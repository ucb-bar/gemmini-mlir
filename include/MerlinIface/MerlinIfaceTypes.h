#ifndef MERLIN_IFACE_TYPES_H
#define MERLIN_IFACE_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "MerlinIface/MerlinIfaceDialect.h"

#define GET_TYPEDEF_CLASSES
#include "MerlinIface/MerlinIfaceOpsTypes.h.inc"

#endif // MERLIN_IFACE_TYPES_H
