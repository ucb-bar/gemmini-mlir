//===- GemminiToLLVM.cpp - gemmini dialect -> llvm.func @gemmini_kernel -----===//
// Builds an llvm-dialect kernel of RoCC `.insn` ops (custom-3 / 0x7b) for the single-/multi-tile
// weight-stationary Gemmini sequence. This is a self-contained C++ port of the CERTIFIED encoding
// in merlin/python/merlin/runtime/backends/gemmini_codegen_mlir.py (see inputs/runtime_plan.yaml;
// the reused constants are tagged as a Merlin tooling advantage in inputs/source_manifest.yaml).
//
// Arg order MUST match the runner-owned harness: @gemmini_kernel(weight*, lhs_0*.., out_0*..).
#include "Conversions.h"

#include "Gemmini/GemminiOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"

#include <cstdint>

using namespace mlir;

namespace gemmini_oot {
namespace {

constexpr uint64_t DIM = 16;
constexpr uint64_t F1 = 0x3F800000ULL;        // float 1.0 bits
constexpr uint64_t GARBAGE = 0xFFFFFFFFULL;
constexpr uint64_t C_ACC = 0xA0000000ULL;      // full-i32 readout
constexpr uint64_t ACC_I8 = 0x80000000ULL;     // scaled i8 readout (drops full_C)
constexpr uint64_t ACC_ACCUM = 0x40000000ULL;  // K-accumulate-onto bit
enum Funct { K_CONFIG = 0, K_MVIN = 2, K_MVOUT = 3, K_COMPUTE_PRELOADED = 4, K_PRELOAD = 6, K_FLUSH = 7 };
constexpr uint64_t CFG_EX_RS1 = (F1 << 32) | (1ULL << 16) | (1ULL << 2);
constexpr uint64_t CFG_EX_RS2 = (1ULL << 48);
constexpr uint64_t CFG_LD_RS1 = (F1 << 32) | (DIM << 16) | (1ULL << 8) | 1ULL;

static uint64_t pack(uint64_t addr) { return (DIM << 48) | (DIM << 32) | (addr & 0xFFFFFFFFULL); }
static uint64_t ceilDim(int64_t x) { return ((uint64_t(x) + DIM - 1) / DIM) * DIM; }
static uint64_t f32bits(double scale) { return llvm::bit_cast<uint32_t>((float)scale); }

struct Job {
  unsigned lhsArg, outArg;
  int64_t m;
  bool relu, i8out;
  double scale;
};

} // namespace

// Movement kernel: gemmini_kernel(src*, dst*) tiles [M,N] i8 by 16x16, MVIN each tile to a
// scratchpad slot then MVOUT it to DRAM. No EX-config, no preload, no compute (pure data movement).
static OwningOpRef<ModuleOp> buildMovementKernel(ModuleOp gemminiModule, MLIRContext &ctx,
                                                 llvm::ArrayRef<gemmini::MovementOp> moves) {
  auto mv = moves.front();
  auto ty = cast<RankedTensorType>(mv.getSrc().getType());
  int64_t M = ty.getShape()[0], N = ty.getShape()[1];
  uint64_t mp = ceilDim(M), np = ceilDim(N);
  uint64_t Mt = mp / DIM, Nt = np / DIM;

  OpBuilder b(&ctx);
  Location loc = gemminiModule.getLoc();
  auto module = ModuleOp::create(loc);
  b.setInsertionPointToEnd(module.getBody());
  auto i64 = b.getI64Type();
  auto ptr = LLVM::LLVMPointerType::get(&ctx);
  auto voidTy = LLVM::LLVMVoidType::get(&ctx);
  llvm::SmallVector<Type> argTys(2, ptr);          // (src, dst)
  auto kernel = b.create<LLVM::LLVMFuncOp>(
      loc, "gemmini_kernel", LLVM::LLVMFunctionType::get(voidTy, argTys));
  Block *entry = kernel.addEntryBlock(b);
  b.setInsertionPointToStart(entry);

  auto konst = [&](uint64_t v) -> Value {
    return b.create<LLVM::ConstantOp>(loc, i64, b.getI64IntegerAttr((int64_t)v));
  };
  auto inlineAsm = [&](StringRef s, StringRef cons, ValueRange ops) {
    b.create<LLVM::InlineAsmOp>(loc, TypeRange{}, ops, s, cons, true, false,
                                LLVM::TailCallKind::None, LLVM::AsmDialectAttr{}, ArrayAttr{});
  };
  auto rocc = [&](int funct, Value rs1, Value rs2) {
    inlineAsm(".insn r 0x7b, 0x3, " + std::to_string(funct) + ", x0, $0, $1", "r,r",
              ValueRange{rs1, rs2});
  };
  Value srcInt = b.create<LLVM::PtrToIntOp>(loc, i64, kernel.getArgument(0));
  Value dstInt = b.create<LLVM::PtrToIntOp>(loc, i64, kernel.getArgument(1));
  auto addr = [&](Value base, uint64_t off) -> Value {
    return off == 0 ? base : b.create<LLVM::AddOp>(loc, base, konst(off));
  };

  inlineAsm("fence", "", ValueRange{});
  rocc(K_FLUSH, konst(0), konst(0));
  // config_ld stride = np bytes (i8 mvin); config_st stride = np bytes, identity scale, i8 readout.
  rocc(K_CONFIG, konst(CFG_LD_RS1), konst(np));
  rocc(K_CONFIG, konst(2), konst((F1 << 32) | np));   // CONFIG_ST: (acc_act=0<<2)|2 ; scale=1.0, stride=np
  for (uint64_t mi = 0; mi < Mt; ++mi)
    for (uint64_t nj = 0; nj < Nt; ++nj) {
      uint64_t slot = (mi * Nt + nj) * DIM;
      uint64_t off = (mi * DIM) * np + nj * DIM;
      rocc(K_MVIN, addr(srcInt, off), konst(pack(slot)));     // DRAM src tile -> scratchpad slot
      rocc(K_MVOUT, addr(dstInt, off), konst(pack(slot)));    // scratchpad slot -> DRAM dst tile
    }
  inlineAsm("fence", "", ValueRange{});
  b.create<LLVM::ReturnOp>(loc, ValueRange{});
  return module;
}

OwningOpRef<ModuleOp> convertGemminiToLLVM(ModuleOp gemminiModule, MLIRContext &ctx) {
  ctx.getOrLoadDialect<LLVM::LLVMDialect>();
  func::FuncOp fn;
  for (auto f : gemminiModule.getBody()->getOps<func::FuncOp>()) { fn = f; break; }
  if (!fn) return nullptr;

  // Movement-only module (MVIN -> MVOUT, no compute): emit the movement kernel and return.
  llvm::SmallVector<gemmini::MovementOp> moves;
  for (Operation &op : fn.getBody().front().getOperations())
    if (auto mv = dyn_cast<gemmini::MovementOp>(op)) moves.push_back(mv);
  if (!moves.empty())
    return buildMovementKernel(gemminiModule, ctx, moves);

  // Reconstruct structural info (single resident weight; matmuls paired to commits by def-use).
  gemmini::ResPackOp pack0;
  llvm::SmallVector<gemmini::MatmulOp> mms;
  for (Operation &op : fn.getBody().front().getOperations()) {
    if (auto p = dyn_cast<gemmini::ResPackOp>(op)) { if (!pack0) pack0 = p; }
    else if (auto m = dyn_cast<gemmini::MatmulOp>(op)) mms.push_back(m);
  }
  if (!pack0 || mms.empty()) return nullptr;
  auto weightTy = cast<RankedTensorType>(pack0.getSrc().getType());
  int64_t K = weightTy.getShape()[0], N = weightTy.getShape()[1];

  unsigned R = mms.size();
  llvm::SmallVector<Job> jobs;
  for (unsigned j = 0; j < R; ++j) {
    gemmini::CommitOp commit;
    for (Operation *u : mms[j].getResult().getUsers())
      if (auto c = dyn_cast<gemmini::CommitOp>(u)) commit = c;
    if (!commit) return nullptr;
    bool relu = false;
    for (Attribute a : commit.getEpilogue())
      if (cast<StringAttr>(a).getValue() == "relu") relu = true;
    bool i8out = commit.getOutputDtype() == "i8";
    double scale = 1.0;
    if (auto sc = commit.getAccScale()) scale = sc->convertToDouble();
    auto lhsTy = cast<RankedTensorType>(mms[j].getLhs().getType());
    jobs.push_back({1 + j, 1 + R + j, lhsTy.getShape()[0], relu, i8out, scale});
  }

  OpBuilder b(&ctx);
  Location loc = gemminiModule.getLoc();
  auto module = ModuleOp::create(loc);
  b.setInsertionPointToEnd(module.getBody());

  auto i64 = b.getI64Type();
  auto ptr = LLVM::LLVMPointerType::get(&ctx);
  auto voidTy = LLVM::LLVMVoidType::get(&ctx);
  llvm::SmallVector<Type> argTys(1 + 2 * R, ptr);
  auto fnTy = LLVM::LLVMFunctionType::get(voidTy, argTys);
  auto kernel = b.create<LLVM::LLVMFuncOp>(loc, "gemmini_kernel", fnTy);
  Block *entry = kernel.addEntryBlock(b);
  b.setInsertionPointToStart(entry);

  auto konst = [&](uint64_t v) -> Value {
    return b.create<LLVM::ConstantOp>(loc, i64, b.getI64IntegerAttr((int64_t)v));
  };
  auto inlineAsm = [&](StringRef s, StringRef cons, ValueRange ops) {
    b.create<LLVM::InlineAsmOp>(loc, TypeRange{}, ops, s, cons,
                                /*has_side_effects=*/true, /*is_align_stack=*/false,
                                /*tail_call_kind=*/LLVM::TailCallKind::None,
                                /*asm_dialect=*/LLVM::AsmDialectAttr{}, /*operand_attrs=*/ArrayAttr{});
  };
  auto rocc = [&](int funct, Value rs1, Value rs2) {
    std::string s = ".insn r 0x7b, 0x3, " + std::to_string(funct) + ", x0, $0, $1";
    inlineAsm(s, "r,r", ValueRange{rs1, rs2});
  };

  llvm::SmallVector<Value> pint(1 + 2 * R);
  for (unsigned i = 0; i < pint.size(); ++i)
    pint[i] = b.create<LLVM::PtrToIntOp>(loc, i64, kernel.getArgument(i));
  auto addr = [&](unsigned argIdx, uint64_t off) -> Value {
    if (off == 0) return pint[argIdx];
    return b.create<LLVM::AddOp>(loc, pint[argIdx], konst(off));
  };

  uint64_t kp = ceilDim(K), np = ceilDim(N);
  uint64_t Kt = kp / DIM, Nt = np / DIM, a_slot = Kt * Nt * DIM;

  int64_t lastLd = -1;
  auto configLd = [&](uint64_t stride) {
    if ((int64_t)stride != lastLd) { rocc(K_CONFIG, konst(CFG_LD_RS1), konst(stride)); lastLd = stride; }
  };

  inlineAsm("fence", "", ValueRange{});
  rocc(K_FLUSH, konst(0), konst(0));
  rocc(K_CONFIG, konst(CFG_EX_RS1), konst(CFG_EX_RS2));

  // Resident weight: mvin all Kt x Nt tiles once (stride = np bytes, elem_t = 1B).
  configLd(np);
  for (uint64_t kt = 0; kt < Kt; ++kt)
    for (uint64_t nj = 0; nj < Nt; ++nj) {
      uint64_t wRow = (kt * Nt + nj) * DIM;
      uint64_t off = (kt * DIM) * np + nj * DIM;
      rocc(K_MVIN, addr(0, off), konst(pack(wRow)));
    }
  configLd(kp);  // activation stride

  for (const Job &job : jobs) {
    uint64_t mp = ceilDim(job.m), Mt = mp / DIM;
    uint64_t accAct = job.relu ? 1 : 0;
    uint64_t scaleBits = job.i8out ? f32bits(job.scale) : F1;
    uint64_t elt = job.i8out ? 1 : 4;
    uint64_t readBase = job.i8out ? ACC_I8 : C_ACC;
    rocc(K_CONFIG, konst((accAct << 2) | 2), konst((scaleBits << 32) | (np * elt)));
    for (uint64_t mi = 0; mi < Mt; ++mi)
      for (uint64_t nj = 0; nj < Nt; ++nj) {
        for (uint64_t kt = 0; kt < Kt; ++kt) {
          uint64_t aOff = (mi * DIM) * kp + kt * DIM;
          rocc(K_MVIN, addr(job.lhsArg, aOff), konst(pack(a_slot)));
          uint64_t cad = (kt == 0) ? C_ACC : (C_ACC | ACC_ACCUM);
          rocc(K_PRELOAD, konst(pack((kt * Nt + nj) * DIM)), konst(pack(cad)));
          rocc(K_COMPUTE_PRELOADED, konst(pack(a_slot)), konst(pack(GARBAGE)));
        }
        uint64_t cOff = ((mi * DIM) * np + nj * DIM) * elt;
        rocc(K_MVOUT, addr(job.outArg, cOff), konst(pack(readBase)));
      }
  }

  inlineAsm("fence", "", ValueRange{});
  b.create<LLVM::ReturnOp>(loc, ValueRange{});
  return module;
}

} // namespace gemmini_oot
