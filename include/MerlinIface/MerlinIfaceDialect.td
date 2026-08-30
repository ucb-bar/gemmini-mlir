#ifndef MERLIN_IFACE_DIALECT
#define MERLIN_IFACE_DIALECT

include "mlir/IR/OpBase.td"

// The frozen experiment-ABI input grammar (merlin_iface v0.1). Re-authored from
// bench_contract/interface_grammar.md. This is the package's INPUT dialect (parsed from the
// *.interface.mlir the runner hands us); the package's TARGET dialect is `gemmini` (see Gemmini/).
def MerlinIface_Dialect : Dialect {
    let name = "merlin_iface";
    let summary = "Experiment-ABI v0.1 interface grammar (input to a target backend package).";
    let cppNamespace = "::merlin_iface";
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{ void registerTypes(); }];
}

class MerlinIface_Op<string mnemonic, list<Trait> traits = []> :
        Op<MerlinIface_Dialect, mnemonic, traits>;

#endif // MERLIN_IFACE_DIALECT
