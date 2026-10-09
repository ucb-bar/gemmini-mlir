#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module {
  func.func @forward(%arg0: tensor<2048xf32>, %arg1: tensor<1x8x2048xf32>) -> tensor<1x8x2048xi8> attributes {llvm.emit_c_interface} {
    %expanded = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 8, 1, 2048] : tensor<1x8x2048xf32> into tensor<1x8x1x2048xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8x1xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expanded : tensor<1x8x1x2048xf32>) outs(%cst : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.mulf %in, %in {prov.origin_node_ids = ["g:original:root:n249", "g:quantized:root:n709"], prov.source_node_ids = ["g:prepared:root:n729"], prov.trace_role = "lowering"} : f32
      %4 = arith.addf %3, %out {prov.origin_node_ids = ["g:original:root:n250", "g:quantized:root:n710"], prov.source_node_ids = ["g:prepared:root:n730"], prov.trace_role = "lowering"} : f32
      linalg.yield %4 : f32
    } -> tensor<1x8x1xf32>
    %1 = tensor.empty() : tensor<1x8x2048xi8>
    %cst_0 = arith.constant 2.048000e+03 : f32
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %cst_2 = arith.constant 6.469040e+01 : f32
    %cst_3 = arith.constant -1.280000e+02 : f32
    %cst_4 = arith.constant 1.270000e+02 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %c1_i8 = arith.constant 1 : i8
    %c0_i8 = arith.constant 0 : i8
    %cst_6 = arith.constant {prov._pattern_hint = "batch_matmul", prov.aten = "aten.bmm.default", prov.family = "contraction", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "batch_matmul", prov.orig_dtype = "float32", prov.origin_node_ids = ["g:original:root:n244", "g:quantized:submod_1:n16"], prov.region_id = "matmul_0", prov.source_node_ids = ["g:prepared:root:n718"], prov.trace_role = "lowering"} 0.000000e+00 : f32
    %c-1_i8 = arith.constant -1 : i8
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %0 : tensor<2048xf32>, tensor<1x8x2048xf32>, tensor<1x8x1xf32>) outs(%1 : tensor<1x8x2048xi8>) {
    ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: i8):
      %3 = arith.divf %in_8, %cst_0 {prov.origin_node_ids = ["g:original:root:n250", "g:quantized:root:n710"], prov.source_node_ids = ["g:prepared:root:n730"], prov.trace_role = "lowering"} : f32
      %4 = arith.addf %3, %cst_1 {prov.origin_node_ids = ["g:original:root:n251", "g:quantized:root:n711"], prov.source_node_ids = ["g:prepared:root:n731"], prov.trace_role = "lowering"} : f32
      %5 = math.rsqrt %4 {prov.origin_node_ids = ["g:original:root:n252", "g:quantized:root:n712"], prov.source_node_ids = ["g:prepared:root:n732"], prov.trace_role = "lowering"} : f32
      %6 = arith.mulf %in_7, %5 {prov.origin_node_ids = ["g:original:root:n253", "g:quantized:root:n713"], prov.source_node_ids = ["g:prepared:root:n733"], prov.trace_role = "lowering"} : f32
      %7 = arith.mulf %in, %6 {prov.origin_node_ids = ["g:original:root:n256", "g:quantized:root:n716"], prov.source_node_ids = ["g:prepared:root:n735"], prov.trace_role = "lowering"} : f32
      %8 = arith.mulf %7, %cst_2 : f32
      %9 = arith.maximumf %8, %cst_3 : f32
      %10 = arith.minimumf %9, %cst_4 : f32
      %11 = arith.fptosi %10 : f32 to i8
      %12 = arith.sitofp %11 : i8 to f32
      %13 = arith.subf %10, %12 : f32
      %14 = arith.negf %13 : f32
      %15 = arith.maximumf %13, %14 : f32
      %16 = arith.cmpf ogt, %15, %cst_5 : f32
      %17 = arith.cmpf oeq, %15, %cst_5 : f32
      %18 = arith.andi %11, %c1_i8 : i8
      %19 = arith.cmpi ne, %18, %c0_i8 : i8
      %20 = arith.andi %17, %19 : i1
      %21 = arith.ori %16, %20 : i1
      %22 = arith.cmpf olt, %10, %cst_6 : f32
      %23 = arith.select %22, %c-1_i8, %c1_i8 : i8
      %24 = arith.select %21, %23, %c0_i8 : i8
      %25 = arith.addi %11, %24 {prov.transforms = "fuse_quantize_round_convert"} : i8
      linalg.yield %25 : i8
    } -> tensor<1x8x2048xi8>
    return %2 : tensor<1x8x2048xi8>
  }
}

