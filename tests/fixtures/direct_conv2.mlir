builtin.module {
  func.func @captured_conv(%0: tensor<64x576xi8>, %1: tensor<1x64x58x58xi8>) -> tensor<64x3136xi32> {
    %2 = tensor.empty() : tensor<64x3x3x1x56x56xi8>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x64x58x58xi8>) outs(%2 : tensor<64x3x3x1x56x56xi8>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2", prov.role = "gather"} {
    ^bb0(%4: i8, %5: i8):
      linalg.yield %4 : i8
    } -> tensor<64x3x3x1x56x56xi8>
    %6 = tensor.collapse_shape %3 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<64x3x3x1x56x56xi8> into tensor<1806336xi8>
    %7 = tensor.expand_shape %6 [[0 : i64, 1 : i64]] output_shape [576, 3136] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<1806336xi8> into tensor<576x3136xi8>
    %8 = arith.constant 0 : i32
    %9 = tensor.empty() : tensor<64x3136xi32>
    %10 = linalg.fill ins(%8 : i32) outs(%9 : tensor<64x3136xi32>) -> tensor<64x3136xi32>
    %11 = linalg.matmul {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2", prov.role = "contraction"} ins(%0, %7 : tensor<64x576xi8>, tensor<576x3136xi8>) outs(%10 : tensor<64x3136xi32>) -> tensor<64x3136xi32>
    func.return %11 : tensor<64x3136xi32>
  }
}