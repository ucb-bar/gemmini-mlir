#ifndef GEMMINI_DIALECT
#define GEMMINI_DIALECT

include "mlir/IR/OpBase.td"

// The `gemmini` TARGET dialect: a target-level view of a single-tile Gemmini systolic backend.
// `merlin_iface` (input) lowers to this dialect (InterfaceToGemmini); this dialect lowers to a
// command buffer (GemminiToCommandBuffer) and to LLVM/RoCC (GemminiToLLVM).
def Gemmini_Dialect : Dialect {
    let name = "gemmini";
    let summary = "Single-tile Gemmini systolic target dialect (Experiment ABI v0.1).";
    let cppNamespace = "::gemmini";
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{ void registerTypes(); }];
}

class Gemmini_Op<string mnemonic, list<Trait> traits = []> :
        Op<Gemmini_Dialect, mnemonic, traits>;

#endif // GEMMINI_DIALECT
