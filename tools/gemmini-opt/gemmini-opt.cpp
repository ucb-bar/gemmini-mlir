//===- gemmini-opt.cpp - agent_spec_v0_mlir_oot driver ----------------------===//
// Four entrypoints (see ../../manifest.yaml):
//   --verify-diagnostics                 parse+verify the merlin_iface input (nonzero on error)
//   --convert-iface-to-gemmini           print the gemmini-dialect module (verified)
//   --emit-command-buffer=<file>         write command_buffer.json
//   --convert-gemmini-to-llvm-rocc       print the llvm-dialect @gemmini_kernel
// The gemmini module is always verify()'d before use (Correction C: a decorative/non-verifying
// target module is a hard failure, surfaced as a nonzero exit -> the runner routes target_dialect).
#include "Conversions.h"

#include "Gemmini/GemminiDialect.h"
#include "MerlinIface/MerlinIfaceDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <string>

using namespace mlir;

int main(int argc, char **argv) {
  std::string input, emitCb;
  bool toLLVM = false, toGemmini = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--convert-gemmini-to-llvm-rocc") toLLVM = true;
    else if (a == "--convert-iface-to-gemmini") toGemmini = true;
    else if (a.rfind("--emit-command-buffer=", 0) == 0) emitCb = a.substr(22);
    else if (a == "--verify-diagnostics") { /* parse-only */ }
    else if (!a.empty() && a[0] != '-') input = a;
  }
  if (input.empty()) { llvm::errs() << "gemmini-opt: no input file\n"; return 1; }

  MLIRContext ctx;
  ctx.getOrLoadDialect<merlin_iface::MerlinIfaceDialect>();
  ctx.getOrLoadDialect<gemmini::GemminiDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> ifaceModule = parseSourceFile<ModuleOp>(input, &ctx);
  if (!ifaceModule) return 1;  // parse / verify of the input failed

  // --verify-diagnostics: input parsed + verified OK.
  if (!toLLVM && !toGemmini && emitCb.empty()) return 0;

  // All remaining paths need the gemmini module.
  OwningOpRef<ModuleOp> gem = gemmini_oot::convertInterfaceToGemmini(*ifaceModule, ctx);
  if (!gem) { llvm::errs() << "gemmini-opt: interface->gemmini conversion failed\n"; return 1; }
  if (failed(verify(*gem))) {       // Correction C: target dialect must verify
    llvm::errs() << "gemmini-opt: gemmini-dialect module failed verify()\n";
    return 1;
  }

  if (!emitCb.empty()) {
    bool ok = false;
    std::string json = gemmini_oot::emitCommandBuffer(*gem, ok);
    if (!ok) { llvm::errs() << "gemmini-opt: command-buffer emission failed\n"; return 1; }
    std::ofstream out(emitCb);
    if (!out) { llvm::errs() << "gemmini-opt: cannot open " << emitCb << "\n"; return 1; }
    out << json;
    return 0;
  }

  if (toLLVM) {
    OwningOpRef<ModuleOp> low = gemmini_oot::convertGemminiToLLVM(*gem, ctx);
    if (!low) { llvm::errs() << "gemmini-opt: gemmini->llvm conversion failed\n"; return 1; }
    if (failed(verify(*low))) { llvm::errs() << "gemmini-opt: llvm module failed verify()\n"; return 1; }
    low->print(llvm::outs());
    return 0;
  }

  // toGemmini only
  gem->print(llvm::outs());
  return 0;
}
