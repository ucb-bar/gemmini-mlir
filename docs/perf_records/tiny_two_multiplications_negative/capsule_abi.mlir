#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
module {
  func.func @forward(%arg0: tensor<1x8x2048xf32>, %arg1: tensor<1x8x2048xi32>, %arg2: tensor<2048xf32>) -> tensor<1x8x2048xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 6.85257779E-4 : f32
    %0 = tensor.empty() : tensor<1x8x2048xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<1x8x2048xf32>, tensor<1x8x2048xi32>, tensor<2048xf32>) outs(%0 : tensor<1x8x2048xf32>) {
    ^bb0(%in: f32, %in_0: i32, %in_1: f32, %out: f32):
      %2 = arith.sitofp %in_0 : i32 to f32
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %3, %in_1 : f32
      %5 = arith.addf %in, %4 {prov.origin_node_ids = ["g:original:root:n292", "g:quantized:root:n758"], prov.source_node_ids = ["g:prepared:root:n809"], prov.trace_role = "lowering"} : f32
      linalg.yield %5 : f32
    } -> tensor<1x8x2048xf32>
    return %1 : tensor<1x8x2048xf32>
  }
}

