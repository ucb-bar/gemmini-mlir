#ifndef MERLIN_IFACE_OPS_H
#define MERLIN_IFACE_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "MerlinIface/MerlinIfaceTypes.h"

#define GET_OP_CLASSES
#include "MerlinIface/MerlinIfaceOps.h.inc"

#endif // MERLIN_IFACE_OPS_H
