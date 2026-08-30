#ifndef MERLIN_IFACE_TYPES
#define MERLIN_IFACE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "MerlinIface/MerlinIfaceDialect.td"

class MerlinIface_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MerlinIface_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MerlinIface_ResidentType : MerlinIface_Type<"Resident", "resident"> {
  let summary = "Opaque resident (packed, stationary) weight handle.";
}

def MerlinIface_AccType : MerlinIface_Type<"Acc", "acc"> {
  let summary = "Opaque integer accumulator handle.";
  let parameters = (ins "mlir::Type":$elemType);
  let assemblyFormat = "`<` $elemType `>`";
}

#endif // MERLIN_IFACE_TYPES
