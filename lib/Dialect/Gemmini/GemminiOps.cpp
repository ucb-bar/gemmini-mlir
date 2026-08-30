//===- GemminiOps.cpp - gemmini op verifiers --------------------------------===//
#include "Gemmini/GemminiOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace gemmini;

// gemmini.commit verifier (Correction C: real verifiers on the target dialect):
//   - every epilogue stage must be in {relu, acc_scale}
//   - the acc_scale attr is present iff "acc_scale" appears in the epilogue
//   - output_dtype must be i8 or i32
LogicalResult CommitOp::verify() {
  bool hasAccScaleStage = false;
  for (Attribute a : getEpilogue()) {
    auto s = dyn_cast<StringAttr>(a);
    if (!s)
      return emitOpError("epilogue entries must be string attributes");
    StringRef v = s.getValue();
    if (v != "relu" && v != "acc_scale")
      return emitOpError("unsupported epilogue stage '") << v << "' (have: relu, acc_scale)";
    if (v == "acc_scale")
      hasAccScaleStage = true;
  }
  if (hasAccScaleStage != getAccScale().has_value())
    return emitOpError("acc_scale attribute must be present iff 'acc_scale' is in the epilogue");
  StringRef dt = getOutputDtype();
  if (dt != "i8" && dt != "i32")
    return emitOpError("output_dtype must be i8 or i32, got '") << dt << "'";
  return success();
}
