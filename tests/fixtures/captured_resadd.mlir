builtin.module {
  func.func @residual(%a: tensor<16x64xi8>, %b: tensor<16x64xi8>) -> tensor<16x64xi8> {
    %one = arith.constant 1.0 : f32
    %scale = tensor.splat %one : tensor<f32>
    %zero = arith.constant 0 : i64
    %zp = tensor.splat %zero : tensor<i64>
    %da = "quant_ext.dequantize_per_tensor"(%a, %scale, %zp) <{quant_min = -128 : i64, quant_max = 127 : i64}> : (tensor<16x64xi8>, tensor<f32>, tensor<i64>) -> tensor<16x64xf32>
    %db = "quant_ext.dequantize_per_tensor"(%b, %scale, %zp) <{quant_min = -128 : i64, quant_max = 127 : i64}> : (tensor<16x64xi8>, tensor<f32>, tensor<i64>) -> tensor<16x64xf32>
    %empty = tensor.empty() : tensor<16x64xf32>
    %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%da, %db : tensor<16x64xf32>, tensor<16x64xf32>) outs(%empty : tensor<16x64xf32>) {
      ^bb0(%x: f32, %y: f32, %out: f32):
        %s = arith.addf %x, %y : f32
        linalg.yield %s : f32
    } -> tensor<16x64xf32>
    %q = "quant_ext.quantize_per_tensor"(%sum, %scale, %zp) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "residual"} : (tensor<16x64xf32>, tensor<f32>, tensor<i64>) -> tensor<16x64xi8>
    func.return %q : tensor<16x64xi8>
  }
}
