//===- GemminiToCommandBuffer.cpp - gemmini dialect -> command_buffer.json --===//
// Walks the gemmini func and serializes RES_PACK / MATMUL_RESIDENT / COMMIT / EVICT plus the
// tensors{} block (weight role from res_pack, input role from matmul). Mirrors the xDSL prototype
// (xdsl/dialect.py: emit_command_buffer), which was validated to deep-equal the golden g0 cb.
#include "Conversions.h"

#include "Gemmini/GemminiOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>

using namespace mlir;

namespace gemmini_oot {

static std::string dtypeOf(Type elem) {
  if (auto it = dyn_cast<IntegerType>(elem))
    return it.getWidth() == 8 ? "i8" : "i32";
  return "i32";
}

static std::string shapeDtypeRole(RankedTensorType t, StringRef role) {
  auto sh = t.getShape();
  std::string s = "{\"shape\": [" + std::to_string(sh[0]) + ", " + std::to_string(sh[1]) +
                  "], \"dtype\": \"" + dtypeOf(t.getElementType()) + "\", \"role\": \"" +
                  role.str() + "\"}";
  return s;
}

static std::string fmtScale(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.8g", v);
  return buf;
}

static int64_t iarr(ArrayAttr a, unsigned i) { return cast<IntegerAttr>(a[i]).getInt(); }

std::string emitCommandBuffer(ModuleOp gemminiModule, bool &ok) {
  ok = false;
  func::FuncOp fn;
  for (auto f : gemminiModule.getBody()->getOps<func::FuncOp>()) { fn = f; break; }
  if (!fn) return "";

  std::string tensors, commands, recipes;
  auto appendTensor = [&](StringRef name, const std::string &body) {
    if (!tensors.empty()) tensors += ", ";
    tensors += "\"" + name.str() + "\": " + body;
  };
  auto appendRecipe = [&](const std::string &r) {
    if (!recipes.empty()) recipes += ", ";
    recipes += r;
  };
  auto appendCmd = [&](const std::string &c) {
    if (!commands.empty()) commands += ", ";
    commands += c;
  };

  for (Operation &op : fn.getBody().front().getOperations()) {
    if (auto p = dyn_cast<gemmini::ResPackOp>(op)) {
      auto t = cast<RankedTensorType>(p.getSrc().getType());
      appendTensor(p.getSrcName(), shapeDtypeRole(t, "weight"));
      appendCmd("{\"opcode\": \"RES_PACK\", \"operands\": {\"src\": \"" + p.getSrcName().str() +
                "\", \"dst\": \"" + p.getDstName().str() + "\"}, \"attributes\": {\"layout\": \"" +
                p.getLayout().str() + "\"}}");
    } else if (auto m = dyn_cast<gemmini::MatmulOp>(op)) {
      auto t = cast<RankedTensorType>(m.getLhs().getType());
      appendTensor(m.getLhsName(), shapeDtypeRole(t, "input"));
      // conv2d lowering: also declare the 4D IFM leaf + an im2col recipe (runner materializes Acol).
      if (auto src = m.getIm2colSource()) {
        auto ifmShape = m.getIfmShape().value();
        auto p = m.getIm2colParams().value();   // [kh,kw,ci,sh,sw,pt,pl,pb,pr,dh,dw]
        std::string sh;
        for (unsigned i = 0; i < ifmShape.size(); ++i)
          sh += (i ? ", " : "") + std::to_string(iarr(ifmShape, i));
        appendTensor(*src, "{\"shape\": [" + sh + "], \"dtype\": \"i8\", \"role\": \"input\"}");
        std::string layout = m.getIm2colLayout() ? m.getIm2colLayout()->str() : "nhwc";
        appendRecipe(
            "{\"target\": \"" + m.getLhsName().str() + "\", \"source\": \"" + src->str() +
            "\", \"kh\": " + std::to_string(iarr(p, 0)) + ", \"kw\": " + std::to_string(iarr(p, 1)) +
            ", \"ci\": " + std::to_string(iarr(p, 2)) +
            ", \"stride\": [" + std::to_string(iarr(p, 3)) + ", " + std::to_string(iarr(p, 4)) +
            "], \"padding\": [" + std::to_string(iarr(p, 5)) + ", " + std::to_string(iarr(p, 6)) +
            ", " + std::to_string(iarr(p, 7)) + ", " + std::to_string(iarr(p, 8)) +
            "], \"dilation\": [" + std::to_string(iarr(p, 9)) + ", " + std::to_string(iarr(p, 10)) +
            "], \"layout\": \"" + layout + "\"}");
      }
      // rhs dst name is the res_pack's dst_name (the resident handle).
      auto pack = m.getRhs().getDefiningOp<gemmini::ResPackOp>();
      std::string rhsName = pack ? pack.getDstName().str() : "W_res";
      appendCmd("{\"opcode\": \"MATMUL_RESIDENT\", \"operands\": {\"lhs\": \"" + m.getLhsName().str() +
                "\", \"rhs\": \"" + rhsName + "\", \"dst\": \"" + m.getDstName().str() + "\"}}");
    } else if (auto c = dyn_cast<gemmini::CommitOp>(op)) {
      std::string epi;
      for (Attribute a : c.getEpilogue()) {
        if (!epi.empty()) epi += ", ";
        epi += "\"" + cast<StringAttr>(a).getValue().str() + "\"";
      }
      std::string attrs = "\"epilogue\": [" + epi + "], \"output_dtype\": \"" +
                          c.getOutputDtype().str() + "\"";
      if (auto sc = c.getAccScale())
        attrs += ", \"acc_scale\": " + fmtScale(sc->convertToDouble());
      appendCmd("{\"opcode\": \"COMMIT\", \"operands\": {\"src\": \"" + c.getSrcName().str() +
                "\", \"dst\": \"" + c.getDstName().str() + "\"}, \"attributes\": {" + attrs + "}}");
    } else if (auto mv = dyn_cast<gemmini::MovementOp>(op)) {
      auto t = cast<RankedTensorType>(mv.getSrc().getType());
      auto rt = cast<RankedTensorType>(mv.getResult().getType());
      appendTensor(mv.getSrcName(), shapeDtypeRole(t, "input"));
      appendTensor(mv.getDstName(), shapeDtypeRole(rt, "output"));
      appendCmd("{\"opcode\": \"VECTOR_MAP\", \"operands\": {\"lhs\": \"" + mv.getSrcName().str() +
                "\", \"dst\": \"" + mv.getDstName().str() +
                "\"}, \"attributes\": {\"combine\": \"identity\"}}");
    } else if (auto e = dyn_cast<gemmini::EvictOp>(op)) {
      appendCmd("{\"opcode\": \"EVICT\", \"operands\": {\"handle\": \"" + e.getHandleName().str() +
                "\"}}");
    }
  }

  std::string params;
  if (!recipes.empty())
    params = ",\n  \"params\": {\"im2col_recipes\": [" + recipes + "]}";
  std::string json = "{\n  \"abi_version\": \"0.1\",\n  \"target\": \"gemmini\",\n  \"tensors\": {" +
                     tensors + "},\n  \"commands\": [" + commands + "]" + params + "\n}\n";
  ok = true;
  return json;
}

} // namespace gemmini_oot
