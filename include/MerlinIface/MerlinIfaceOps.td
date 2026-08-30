#ifndef MERLIN_IFACE_OPS
#define MERLIN_IFACE_OPS

include "MerlinIface/MerlinIfaceTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>
def MerlinIface_TensorOp : MerlinIface_Op<"tensor"> {
  let summary = "Declare a leaf input/weight tensor (materialized deterministically by name).";
  let arguments = (ins StrAttr:$name, StrAttr:$role);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<..>) -> !merlin_iface.resident
def MerlinIface_ResidentPackOp : MerlinIface_Op<"resident_pack"> {
  let summary = "Make a weight resident (packed / stationary).";
  let arguments = (ins AnyRankedTensor:$src, StrAttr:$layout);
  let results = (outs MerlinIface_ResidentType:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

// %acc = merlin_iface.matmul %A, %W_res : (tensor<..>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
def MerlinIface_MatmulOp : MerlinIface_Op<"matmul"> {
  let summary = "Matmul against a resident weight.";
  let arguments = (ins AnyRankedTensor:$lhs, MerlinIface_ResidentType:$rhs);
  let results = (outs MerlinIface_AccType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

// %Y = merlin_iface.commit %acc {name=.., epilogue=[..], output_dtype=.., acc_scale=..} : (!acc) -> tensor<..>
def MerlinIface_CommitOp : MerlinIface_Op<"commit"> {
  let summary = "Apply epilogue, commit accumulator to an output tensor.";
  let arguments = (ins MerlinIface_AccType:$src, StrAttr:$name, StrArrayAttr:$epilogue,
                       StrAttr:$output_dtype, OptionalAttr<F32Attr>:$acc_scale);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

// %Y = merlin_iface.conv2d %IFM, %W_res {kernel=[kh,kw,ci,co], stride=[sh,sw], padding=[pt,pl,pb,pr],
//        dilation=[dh,dw], name=.., epilogue=[..], output_dtype=.., acc_scale=.., layout="nhwc"}
//        : (tensor<NxHxWxCxi8>, !merlin_iface.resident) -> tensor<MxCoxi32>
// The compiler lowers this to im2col + a weight-stationary matmul on the resident weight.
def MerlinIface_Conv2dOp : MerlinIface_Op<"conv2d"> {
  let summary = "2D convolution; compiler-lowered to im2col + Gemmini matmul.";
  let arguments = (ins AnyRankedTensor:$ifm, MerlinIface_ResidentType:$weight,
                       I64ArrayAttr:$kernel, I64ArrayAttr:$stride, I64ArrayAttr:$padding,
                       I64ArrayAttr:$dilation, StrAttr:$name, StrArrayAttr:$epilogue,
                       StrAttr:$output_dtype, OptionalAttr<F32Attr>:$acc_scale, StrAttr:$layout);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$ifm `,` $weight attr-dict `:` functional-type(operands, results)";
}

// %Y = merlin_iface.movement %X {name="Y"} : (tensor<MxNxi8>) -> tensor<MxNxi8>
// Pure data movement (MVIN -> MVOUT, no compute); output equals input after the layout move.
def MerlinIface_MovementOp : MerlinIface_Op<"movement"> {
  let summary = "Memory movement (MVIN/MVOUT only, no compute).";
  let arguments = (ins AnyRankedTensor:$src, StrAttr:$name);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

// merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
def MerlinIface_EvictOp : MerlinIface_Op<"evict"> {
  let summary = "Release a resident weight.";
  let arguments = (ins MerlinIface_ResidentType:$handle);
  let assemblyFormat = "$handle attr-dict `:` functional-type(operands, results)";
}

#endif // MERLIN_IFACE_OPS
