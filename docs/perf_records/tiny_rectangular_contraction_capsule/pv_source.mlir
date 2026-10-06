#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @forward(%arg0: tensor<32x8x8xf32>, %arg1: tensor<32x8x64xf32>) -> tensor<32x8x64xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x8x64xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x8x8xf32>, tensor<32x8x64xf32>) outs(%cst : tensor<32x8x64xf32>) attrs =  {prov._pattern_hint = "batch_matmul", prov.aten = "aten.bmm.default", prov.family = "contraction", prov.fqn = "lm.model.layers.0.self_attn", prov.module = "lm", prov.op = "batch_matmul", prov.orig_dtype = "float32", prov.origin_node_ids = ["g:original:root:n288", "g:quantized:root:n752"], prov.region_id = "matmul_5", prov.source_node_ids = ["g:prepared:root:n798"], prov.trace_role = "lowering"} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 {prov.origin_node_ids = ["g:original:root:n288", "g:quantized:root:n752"], prov.source_node_ids = ["g:prepared:root:n798"], prov.trace_role = "lowering"} : f32
      %2 = arith.addf %out, %1 {prov.origin_node_ids = ["g:original:root:n288", "g:quantized:root:n752"], prov.source_node_ids = ["g:prepared:root:n798"], prov.trace_role = "lowering"} : f32
      linalg.yield %2 {prov.origin_node_ids = ["g:original:root:n288", "g:quantized:root:n752"], prov.source_node_ids = ["g:prepared:root:n798"], prov.trace_role = "lowering"} : f32
    } -> tensor<32x8x64xf32>
    return %0 : tensor<32x8x64xf32>
  }
}

