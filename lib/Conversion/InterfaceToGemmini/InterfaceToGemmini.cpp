//===- InterfaceToGemmini.cpp - merlin_iface -> gemmini dialect -------------===//
// Walks a merlin_iface module and builds an equivalent gemmini-dialect module:
//   func.func @kernel(<leaf tensors as block args>) -> <committed outputs> {
//     gemmini.res_pack / gemmini.matmul / gemmini.commit / gemmini.evict ; func.return
//   }
// Names are carried on the gemmini ops (src_name/dst_name/lhs_name/handle_name) so the downstream
// command-buffer emission is self-sufficient. The caller (gemmini-opt) runs verify() on the result.
#include "Conversions.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/GemminiTypes.h"
#include "MerlinIface/MerlinIfaceDialect.h"
#include "MerlinIface/MerlinIfaceOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace gemmini_oot {

static RankedTensorType i32TensorLike(RankedTensorType lhs, RankedTensorType weight,
                                      MLIRContext &ctx) {
  // accumulator shape = [M, N] from lhs [M,K] and weight [K,N], element i32.
  int64_t M = lhs.getShape()[0];
  int64_t N = weight.getShape()[1];
  return RankedTensorType::get({M, N}, IntegerType::get(&ctx, 32));
}

static int64_t arrAt(ArrayAttr a, unsigned i) {
  return cast<IntegerAttr>(a[i]).getInt();
}

// im2col output dims [M, K] for a conv2d: M = N*Ho*Wo, K = kh*kw*ci, from the NHWC ifm + attrs.
struct ConvDims { int64_t M, K, Co, N, H, W, Ci, Ho, Wo; };
static ConvDims convDims(RankedTensorType ifm, Operation *conv) {
  auto kern = conv->getAttrOfType<ArrayAttr>("kernel");      // [kh,kw,ci,co]
  auto strd = conv->getAttrOfType<ArrayAttr>("stride");      // [sh,sw]
  auto pad = conv->getAttrOfType<ArrayAttr>("padding");      // [pt,pl,pb,pr]
  auto dil = conv->getAttrOfType<ArrayAttr>("dilation");     // [dh,dw]
  int64_t kh = arrAt(kern, 0), kw = arrAt(kern, 1), ci = arrAt(kern, 2), co = arrAt(kern, 3);
  int64_t sh = arrAt(strd, 0), sw = arrAt(strd, 1);
  int64_t pt = arrAt(pad, 0), pl = arrAt(pad, 1), pb = arrAt(pad, 2), pr = arrAt(pad, 3);
  int64_t dh = arrAt(dil, 0), dw = arrAt(dil, 1);
  auto sh4 = ifm.getShape();
  int64_t N = sh4[0], H = sh4[1], W = sh4[2], Ci = sh4[3];
  int64_t Ho = (H + pt + pb - (dh * (kh - 1) + 1)) / sh + 1;
  int64_t Wo = (W + pl + pr - (dw * (kw - 1) + 1)) / sw + 1;
  return {N * Ho * Wo, kh * kw * ci, co, N, H, W, Ci, Ho, Wo};
}

OwningOpRef<ModuleOp> convertInterfaceToGemmini(ModuleOp ifaceModule, MLIRContext &ctx) {
  ctx.getOrLoadDialect<gemmini::GemminiDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  OpBuilder b(&ctx);
  Location loc = ifaceModule.getLoc();

  // Conv2d IFMs are NOT kernel args; their derived im2col activation [M,K] is. Map ifm -> conv op.
  llvm::DenseMap<Value, Operation *> convForIfm;
  for (Operation &op : ifaceModule.getBody()->getOperations())
    if (op.getName().getStringRef() == "merlin_iface.conv2d")
      convForIfm[op.getOperand(0)] = &op;

  // Per-conv-IFM leaf, remember the im2col recipe so the matmul can carry it downstream.
  struct ConvLeaf { std::string ifmName, acolName; ConvDims d; SmallVector<int64_t> params, ifmShape;
                    std::string layout; };
  llvm::DenseMap<Value, ConvLeaf> convLeaf;  // iface tensor result -> recipe

  // Pass 1: collect leaf tensors (in declaration order) -> they become func args.
  SmallVector<StringRef> leafNames;
  SmallVector<std::string> leafNameStore;       // backing storage for synthesized names
  SmallVector<Type> leafTypes;
  llvm::DenseMap<Value, unsigned> leafIndex;     // iface tensor result -> func arg index
  for (Operation &op : ifaceModule.getBody()->getOperations()) {
    if (op.getName().getStringRef() != "merlin_iface.tensor")
      continue;
    Value res = op.getResult(0);
    auto name = op.getAttrOfType<StringAttr>("name");
    unsigned idx = leafTypes.size();
    leafIndex[res] = idx;
    if (auto it = convForIfm.find(res); it != convForIfm.end()) {
      // conv IFM: arg is the im2col activation [M,K] i8; the IFM itself is only a recipe source.
      Operation *conv = it->second;
      auto ifmTy = cast<RankedTensorType>(res.getType());
      ConvDims d = convDims(ifmTy, conv);
      auto kern = conv->getAttrOfType<ArrayAttr>("kernel");
      auto strd = conv->getAttrOfType<ArrayAttr>("stride");
      auto pad = conv->getAttrOfType<ArrayAttr>("padding");
      auto dil = conv->getAttrOfType<ArrayAttr>("dilation");
      ConvLeaf cl;
      cl.ifmName = name.getValue().str();
      cl.acolName = cl.ifmName + "_im2col";
      cl.d = d;
      // im2col_params = [kh,kw,ci,sh,sw,pt,pl,pb,pr,dh,dw]
      cl.params = {arrAt(kern, 0), arrAt(kern, 1), arrAt(kern, 2),
                   arrAt(strd, 0), arrAt(strd, 1), arrAt(pad, 0), arrAt(pad, 1),
                   arrAt(pad, 2), arrAt(pad, 3), arrAt(dil, 0), arrAt(dil, 1)};
      cl.ifmShape = {d.N, d.H, d.W, d.Ci};
      cl.layout = conv->getAttrOfType<StringAttr>("layout").getValue().str();
      convLeaf[res] = cl;
      leafNameStore.push_back(cl.acolName);
      leafTypes.push_back(RankedTensorType::get({d.M, d.K}, IntegerType::get(&ctx, 8)));
    } else {
      leafNameStore.push_back(name.getValue().str());
      leafTypes.push_back(res.getType());
    }
  }
  for (auto &s : leafNameStore) leafNames.push_back(s);

  auto module = ModuleOp::create(loc);
  b.setInsertionPointToEnd(module.getBody());

  // Build func @kernel; result types come from commits AND conv2d (which commits internally).
  SmallVector<Type> resultTypes;
  for (Operation &op : ifaceModule.getBody()->getOperations()) {
    StringRef nm = op.getName().getStringRef();
    if (nm == "merlin_iface.commit" || nm == "merlin_iface.conv2d" ||
        nm == "merlin_iface.movement")
      resultTypes.push_back(op.getResult(0).getType());
  }

  auto fnType = b.getFunctionType(leafTypes, resultTypes);
  auto fn = b.create<func::FuncOp>(loc, "kernel", fnType);
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);

  llvm::DenseMap<Value, Value> remap;            // iface value -> gemmini value
  llvm::DenseMap<Value, std::string> gname;      // gemmini value -> name
  for (auto &kv : leafIndex) {
    Value arg = entry->getArgument(kv.second);
    remap[kv.first] = arg;
    gname[arg] = leafNames[kv.second].str();
  }

  auto namedStr = [&](StringRef k, StringRef v) {
    return b.getNamedAttr(k, b.getStringAttr(v));
  };

  unsigned accCtr = 0;
  SmallVector<Value> outs;
  for (Operation &op : ifaceModule.getBody()->getOperations()) {
    StringRef nm = op.getName().getStringRef();
    if (nm == "merlin_iface.resident_pack") {
      Value src = remap[op.getOperand(0)];
      std::string srcName = gname[src];
      std::string dstName = srcName + "_res";
      auto resTy = gemmini::ResidentTensorType::get(&ctx, src.getType());
      SmallVector<NamedAttribute> attrs{
          namedStr("src_name", srcName), namedStr("dst_name", dstName),
          namedStr("layout", op.getAttrOfType<StringAttr>("layout").getValue())};
      auto packed = b.create<gemmini::ResPackOp>(loc, TypeRange{resTy}, ValueRange{src}, attrs);
      remap[op.getResult(0)] = packed.getResult();
      gname[packed.getResult()] = dstName;
    } else if (nm == "merlin_iface.matmul") {
      Value lhs = remap[op.getOperand(0)];
      Value rhs = remap[op.getOperand(1)];
      auto weightTy = cast<RankedTensorType>(
          cast<gemmini::ResidentTensorType>(rhs.getType()).getElemType());
      auto accTensor = i32TensorLike(cast<RankedTensorType>(lhs.getType()), weightTy, ctx);
      auto accTy = gemmini::AccumulatorType::get(&ctx, accTensor);
      std::string dstName = "acc" + std::to_string(accCtr++);
      SmallVector<NamedAttribute> attrs{
          namedStr("lhs_name", gname[lhs]), namedStr("dst_name", dstName)};
      auto mm = b.create<gemmini::MatmulOp>(loc, TypeRange{accTy}, ValueRange{lhs, rhs}, attrs);
      remap[op.getResult(0)] = mm.getResult();
      gname[mm.getResult()] = dstName;
    } else if (nm == "merlin_iface.conv2d") {
      // Lower conv2d -> matmul(im2col_activation, resident_weight) [+ recipe attrs] -> commit.
      Value lhs = remap[op.getOperand(0)];   // the synthesized im2col activation arg
      Value rhs = remap[op.getOperand(1)];   // resident weight
      auto weightTy = cast<RankedTensorType>(
          cast<gemmini::ResidentTensorType>(rhs.getType()).getElemType());
      auto accTensor = i32TensorLike(cast<RankedTensorType>(lhs.getType()), weightTy, ctx);
      auto accTy = gemmini::AccumulatorType::get(&ctx, accTensor);
      std::string dstName = "acc" + std::to_string(accCtr++);
      const ConvLeaf &cl = convLeaf[op.getOperand(0)];
      SmallVector<NamedAttribute> mmAttrs{
          namedStr("lhs_name", gname[lhs]), namedStr("dst_name", dstName),
          namedStr("im2col_source", cl.ifmName),
          b.getNamedAttr("im2col_params", b.getI64ArrayAttr(cl.params)),
          b.getNamedAttr("ifm_shape", b.getI64ArrayAttr(cl.ifmShape)),
          namedStr("im2col_layout", cl.layout)};
      auto mm = b.create<gemmini::MatmulOp>(loc, TypeRange{accTy}, ValueRange{lhs, rhs}, mmAttrs);
      gname[mm.getResult()] = dstName;
      // commit carrying the conv's epilogue/output_dtype/acc_scale/name
      SmallVector<NamedAttribute> cAttrs{
          namedStr("src_name", dstName),
          namedStr("dst_name", op.getAttrOfType<StringAttr>("name").getValue()),
          b.getNamedAttr("epilogue", op.getAttrOfType<ArrayAttr>("epilogue")),
          namedStr("output_dtype", op.getAttrOfType<StringAttr>("output_dtype").getValue())};
      if (auto sc = op.getAttrOfType<FloatAttr>("acc_scale"))
        cAttrs.push_back(b.getNamedAttr("acc_scale", sc));
      auto commit = b.create<gemmini::CommitOp>(
          loc, TypeRange{op.getResult(0).getType()}, ValueRange{mm.getResult()}, cAttrs);
      remap[op.getResult(0)] = commit.getResult();
      outs.push_back(commit.getResult());
    } else if (nm == "merlin_iface.commit") {
      Value src = remap[op.getOperand(0)];
      SmallVector<NamedAttribute> attrs{
          namedStr("src_name", gname[src]),
          namedStr("dst_name", op.getAttrOfType<StringAttr>("name").getValue()),
          b.getNamedAttr("epilogue", op.getAttrOfType<ArrayAttr>("epilogue")),
          namedStr("output_dtype", op.getAttrOfType<StringAttr>("output_dtype").getValue())};
      if (auto sc = op.getAttrOfType<FloatAttr>("acc_scale"))
        attrs.push_back(b.getNamedAttr("acc_scale", sc));
      auto commit = b.create<gemmini::CommitOp>(
          loc, TypeRange{op.getResult(0).getType()}, ValueRange{src}, attrs);
      remap[op.getResult(0)] = commit.getResult();
      outs.push_back(commit.getResult());
    } else if (nm == "merlin_iface.movement") {
      Value src = remap[op.getOperand(0)];
      std::string dstName = op.getAttrOfType<StringAttr>("name").getValue().str();
      SmallVector<NamedAttribute> attrs{namedStr("src_name", gname[src]),
                                        namedStr("dst_name", dstName)};
      auto mv = b.create<gemmini::MovementOp>(
          loc, TypeRange{op.getResult(0).getType()}, ValueRange{src}, attrs);
      remap[op.getResult(0)] = mv.getResult();
      gname[mv.getResult()] = dstName;
      outs.push_back(mv.getResult());
    } else if (nm == "merlin_iface.evict") {
      Value h = remap[op.getOperand(0)];
      SmallVector<NamedAttribute> attrs{namedStr("handle_name", gname[h])};
      b.create<gemmini::EvictOp>(loc, TypeRange{}, ValueRange{h}, attrs);
    }
  }
  b.create<func::ReturnOp>(loc, outs);
  return module;
}

} // namespace gemmini_oot
