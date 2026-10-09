module {
func.func @forward(%a:tensor<1x2x1x2xi8>) -> tensor<1x2xi8> {
 %s = arith.constant 1.74337542 : f32
 %scale = tensor.splat %s : tensor<f32>
 %z = arith.constant 0 : i64
 %zero = tensor.splat %z : tensor<i64>
 %dq = "quant_ext.dequantize_per_tensor"(%a,%scale,%zero) <{quant_min=-128:i64,quant_max=127:i64,output_dtype="float32"}> : (tensor<1x2x1x2xi8>,tensor<f32>,tensor<i64>) -> tensor<1x2x1x2xf32>
 %fzero = arith.constant 0.0 : f32
 %init = tensor.splat %fzero : tensor<1x2xf32>
 %sum = linalg.reduce ins(%dq:tensor<1x2x1x2xf32>) outs(%init:tensor<1x2xf32>) dimensions=[2,3]
 (%x:f32,%acc:f32) {
  %v = arith.addf %x,%acc : f32
  linalg.yield %v : f32
 }
 %n = arith.constant 2.0 : f32
 %den = tensor.splat %n : tensor<1x2xf32>
 %empty = tensor.empty() : tensor<1x2xf32>
 %mean = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%sum,%den:tensor<1x2xf32>,tensor<1x2xf32>) outs(%empty:tensor<1x2xf32>) {
 ^bb0(%v:f32,%d:f32,%unused:f32):
  %r = arith.divf %v,%d : f32
  linalg.yield %r : f32
 } -> tensor<1x2xf32>
 %os = arith.constant 1.35259748 : f32
 %outscale = tensor.splat %os : tensor<f32>
 %q = "quant_ext.quantize_per_tensor"(%mean,%outscale,%zero) <{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}> : (tensor<1x2xf32>,tensor<f32>,tensor<i64>) -> tensor<1x2xi8>
 return %q : tensor<1x2xi8>
}
}
