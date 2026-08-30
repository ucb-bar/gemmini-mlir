#ifndef GEMMINI_TYPES
#define GEMMINI_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Gemmini/GemminiDialect.td"

class Gemmini_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Gemmini_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// !gemmini.resident_tensor<tensor<KxNxi8>> — packed / stationary weight handle.
def Gemmini_ResidentTensorType : Gemmini_Type<"ResidentTensor", "resident_tensor"> {
  let summary = "Packed/stationary resident weight handle (carries the underlying tensor type).";
  let parameters = (ins "mlir::Type":$elemType);
  let assemblyFormat = "`<` $elemType `>`";
}

// !gemmini.accumulator<tensor<MxNxi32>> — i32 accumulator, not yet committed.
def Gemmini_AccumulatorType : Gemmini_Type<"Accumulator", "accumulator"> {
  let summary = "Integer (i32) accumulator handle (carries the underlying tensor type).";
  let parameters = (ins "mlir::Type":$elemType);
  let assemblyFormat = "`<` $elemType `>`";
}

#endif // GEMMINI_TYPES
