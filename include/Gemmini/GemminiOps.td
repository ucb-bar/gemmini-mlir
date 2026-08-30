#ifndef GEMMINI_OPS
#define GEMMINI_OPS

include "Gemmini/GemminiTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// %W_res = gemmini.res_pack %W {src_name=.., dst_name=.., layout=..} : (tensor<..>) -> !gemmini.resident_tensor<..>
def Gemmini_ResPackOp : Gemmini_Op<"res_pack"> {
  let summary = "Make a weight resident (packed / stationary).";
  let arguments = (ins AnyRankedTensor:$src, StrAttr:$src_name, StrAttr:$dst_name, StrAttr:$layout);
  let results = (outs Gemmini_ResidentTensorType:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

// %acc = gemmini.matmul %A, %W_res {lhs_name=.., dst_name=..} : (tensor<..>, !resident) -> !accumulator
def Gemmini_MatmulOp : Gemmini_Op<"matmul"> {
  let summary = "Matmul of lhs against a resident weight, producing an i32 accumulator.";
  // Optional im2col recipe (set when this matmul is the lowering of a conv2d): the lhs is a
  // derived im2col activation gathered (runner-side) from $im2col_source per these params.
  let arguments = (ins AnyRankedTensor:$lhs, Gemmini_ResidentTensorType:$rhs,
                       StrAttr:$lhs_name, StrAttr:$dst_name,
                       OptionalAttr<StrAttr>:$im2col_source,
                       OptionalAttr<I64ArrayAttr>:$im2col_params,
                       OptionalAttr<I64ArrayAttr>:$ifm_shape,
                       OptionalAttr<StrAttr>:$im2col_layout);
  let results = (outs Gemmini_AccumulatorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

// %Y = gemmini.commit %acc {src_name=.., dst_name=.., epilogue=[..], output_dtype=.., acc_scale=..}
def Gemmini_CommitOp : Gemmini_Op<"commit"> {
  let summary = "Apply epilogue (subset of {relu, acc_scale}) and commit the accumulator.";
  let arguments = (ins Gemmini_AccumulatorType:$src, StrAttr:$src_name, StrAttr:$dst_name,
                       StrArrayAttr:$epilogue, StrAttr:$output_dtype, OptionalAttr<F32Attr>:$acc_scale);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// %Y = gemmini.movement %X {src_name=.., dst_name=..} : (tensor<..>) -> tensor<..>
// Pure data movement (MVIN to scratchpad, MVOUT to DRAM); no compute.
def Gemmini_MovementOp : Gemmini_Op<"movement"> {
  let summary = "Memory movement (MVIN/MVOUT only).";
  let arguments = (ins AnyRankedTensor:$src, StrAttr:$src_name, StrAttr:$dst_name);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

// gemmini.evict %W_res {handle_name=..} : (!resident) -> ()
def Gemmini_EvictOp : Gemmini_Op<"evict"> {
  let summary = "Release a resident weight.";
  let arguments = (ins Gemmini_ResidentTensorType:$handle, StrAttr:$handle_name);
  let assemblyFormat = "$handle attr-dict `:` functional-type(operands, results)";
}

#endif // GEMMINI_OPS
