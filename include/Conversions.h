#ifndef GEMMINI_OOT_CONVERSIONS_H
#define GEMMINI_OOT_CONVERSIONS_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace gemmini_oot {

// merlin_iface module -> gemmini-dialect module. Returns null on failure (diagnostic emitted).
mlir::OwningOpRef<mlir::ModuleOp>
convertInterfaceToGemmini(mlir::ModuleOp ifaceModule, mlir::MLIRContext &ctx);

// gemmini-dialect module -> command_buffer.json text. Sets `ok=false` on failure.
std::string emitCommandBuffer(mlir::ModuleOp gemminiModule, bool &ok);

// gemmini-dialect module -> llvm-dialect module defining @gemmini_kernel. Null on failure.
mlir::OwningOpRef<mlir::ModuleOp>
convertGemminiToLLVM(mlir::ModuleOp gemminiModule, mlir::MLIRContext &ctx);

} // namespace gemmini_oot

#endif // GEMMINI_OOT_CONVERSIONS_H
