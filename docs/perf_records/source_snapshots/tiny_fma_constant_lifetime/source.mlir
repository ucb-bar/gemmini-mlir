#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
module {
  func.func @forward(%arg0: tensor<1x2x5632xi32>, %arg1: tensor<5632xf32>, %arg2: tensor<1x2x5632xi32>, %arg3: tensor<5632xf32>) -> tensor<1x2x5632xi8> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.018829124 : f32
    %cst_0 = arith.constant -8.700000e+01 : f32
    %cst_1 = arith.constant 8.800000e+01 : f32
    %cst_2 = arith.constant 1.44269502 : f32
    %cst_3 = arith.constant 0x4B400000 : f32
    %cst_4 = arith.constant -0.693147182 : f32
    %cst_5 = arith.constant 1.90465421E-9 : f32
    %cst_6 = arith.constant 0.00138888892 : f32
    %cst_7 = arith.constant 0.00833333377 : f32
    %cst_8 = arith.constant 0.0416666679 : f32
    %cst_9 = arith.constant 0.166666672 : f32
    %cst_10 = arith.constant 5.000000e-01 : f32
    %cst_11 = arith.constant {prov._pattern_hint = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "mul", prov.orig_dtype = "float32", prov.origin_node_ids = ["g:original:root:n244", "g:quantized:root:n704", "g:quantized:root:n705", "g:quantized:submod_1:n20"], prov.region_id = "mul_0", prov.source_node_ids = ["g:prepared:root:n723"], prov.trace_role = "lowering"} 1.000000e+00 : f32
    %c127_i32 = arith.constant 127 : i32
    %c23_i32 = arith.constant 23 : i32
    %cst_12 = arith.constant 293.566284 : f32
    %cst_13 = arith.constant -1.280000e+02 : f32
    %cst_14 = arith.constant 1.270000e+02 : f32
    %c1_i8 = arith.constant 1 : i8
    %c0_i8 = arith.constant 0 : i8
    %cst_15 = arith.constant {prov._pattern_hint = "batch_matmul", prov.aten = "aten.bmm.default", prov.family = "contraction", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "batch_matmul", prov.orig_dtype = "float32", prov.origin_node_ids = ["g:original:root:n244", "g:quantized:submod_1:n16"], prov.region_id = "matmul_0", prov.source_node_ids = ["g:prepared:root:n718"], prov.trace_role = "lowering"} 0.000000e+00 : f32
    %c-1_i8 = arith.constant -1 : i8
    %0 = tensor.empty() : tensor<1x2x5632xi8>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<1x2x5632xi32>, tensor<5632xf32>, tensor<1x2x5632xi32>, tensor<5632xf32>) outs(%0 : tensor<1x2x5632xi8>) {
    ^bb0(%in: i32, %in_16: f32, %in_17: i32, %in_18: f32, %out: i8):
      %2 = arith.sitofp %in_17 : i32 to f32
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %3, %in_18 : f32
      %5 = arith.sitofp %in : i32 to f32
      %6 = arith.mulf %5, %cst : f32
      %7 = arith.mulf %6, %in_16 : f32
      %8 = arith.negf %7 {prov.origin_node_ids = ["g:original:root:n304", "g:quantized:root:n773"], prov.source_node_ids = ["g:prepared:root:n825"], prov.trace_role = "lowering"} : f32
      %9 = arith.maximumf %8, %cst_0 : f32
      %10 = arith.minimumf %9, %cst_1 : f32
      %11 = arith.mulf %10, %cst_2 : f32
      %12 = arith.addf %11, %cst_3 : f32
      %13 = arith.subf %12, %cst_3 : f32
      %14 = math.fma %13, %cst_4, %10 : f32
      %15 = math.fma %13, %cst_5, %14 : f32
      %16 = math.fma %cst_6, %15, %cst_7 : f32
      %17 = math.fma %16, %15, %cst_8 : f32
      %18 = math.fma %17, %15, %cst_9 : f32
      %19 = math.fma %18, %15, %cst_10 : f32
      %20 = math.fma %19, %15, %cst_11 : f32
      %21 = math.fma %20, %15, %cst_11 : f32
      %22 = arith.fptosi %13 : f32 to i32
      %23 = arith.addi %22, %c127_i32 : i32
      %24 = arith.shli %23, %c23_i32 : i32
      %25 = arith.bitcast %24 : i32 to f32
      %26 = arith.mulf %21, %25 : f32
      %27 = arith.addf %26, %cst_11 {prov.origin_node_ids = ["g:original:root:n304", "g:quantized:root:n773"], prov.source_node_ids = ["g:prepared:root:n825"], prov.trace_role = "lowering"} : f32
      %28 = arith.divf %cst_11, %27 {prov.origin_node_ids = ["g:original:root:n304", "g:quantized:root:n773"], prov.source_node_ids = ["g:prepared:root:n825"], prov.trace_role = "lowering"} : f32
      %29 = arith.mulf %7, %28 {prov.origin_node_ids = ["g:original:root:n304", "g:quantized:root:n773"], prov.source_node_ids = ["g:prepared:root:n826"], prov.trace_role = "lowering"} : f32
      %30 = arith.mulf %29, %4 {prov.origin_node_ids = ["g:original:root:n306", "g:quantized:root:n775"], prov.source_node_ids = ["g:prepared:root:n831"], prov.trace_role = "lowering"} : f32
      %31 = arith.mulf %30, %cst_12 : f32
      %32 = arith.maximumf %31, %cst_13 : f32
      %33 = arith.minimumf %32, %cst_14 : f32
      %34 = arith.fptosi %33 : f32 to i8
      %35 = arith.sitofp %34 : i8 to f32
      %36 = arith.subf %33, %35 : f32
      %37 = arith.negf %36 : f32
      %38 = arith.maximumf %36, %37 : f32
      %39 = arith.cmpf ogt, %38, %cst_10 : f32
      %40 = arith.cmpf oeq, %38, %cst_10 : f32
      %41 = arith.andi %34, %c1_i8 : i8
      %42 = arith.cmpi ne, %41, %c0_i8 : i8
      %43 = arith.andi %40, %42 : i1
      %44 = arith.ori %39, %43 : i1
      %45 = arith.cmpf olt, %33, %cst_15 : f32
      %46 = arith.select %45, %c-1_i8, %c1_i8 : i8
      %47 = arith.select %44, %46, %c0_i8 : i8
      %48 = arith.addi %34, %47 {prov.transforms = "fuse_quantize_round_convert"} : i8
      linalg.yield %48 : i8
    } -> tensor<1x2x5632xi8>
    return %1 : tensor<1x2x5632xi8>
  }
}

