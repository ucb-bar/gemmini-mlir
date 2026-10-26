"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<289x17xi8>, tensor<289x17xi8>) -> tensor<1x8x8x17xi8>}> ({
  ^bb0(%0: tensor<289x17xi8>, %1: tensor<289x17xi8>):
    %operand_sum_empty = "tensor.empty"() : () -> tensor<289x17xi32>
    %operand_sum = "linalg.generic"(%0, %1, %operand_sum_empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb1(%x0: i8, %x1: i8, %dest: i32):
      %a = "arith.sitofp"(%x0) : (i8) -> f32
      %b = "arith.sitofp"(%x1) : (i8) -> f32
      %s0 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %s1 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %p0 = "arith.mulf"(%a, %s0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %p1 = "arith.mulf"(%b, %s1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %p = "arith.addf"(%p0, %p1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %r = "math.roundeven"(%p) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      %v = "arith.fptosi"(%r) : (f32) -> i32
      "linalg.yield"(%v) : (i32) -> ()
    }) {prov.trace_role = "node0_operand_sum"} : (tensor<289x17xi8>, tensor<289x17xi8>, tensor<289x17xi32>) -> tensor<289x17xi32>
    %convert_empty = "tensor.empty"() : () -> tensor<289x17xf32>
    %convert = "linalg.generic"(%operand_sum, %convert_empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb2(%x0_1: i32, %dest_1: f32):
      %f = "arith.sitofp"(%x0_1) : (i32) -> f32
      "linalg.yield"(%f) : (f32) -> ()
    }) {prov.trace_role = "node0_convert"} : (tensor<289x17xi32>, tensor<289x17xf32>) -> tensor<289x17xf32>
    %gains = "arith.constant"() <{value = dense<"0xABAA2A3E"> : tensor<17xf32>}> : () -> tensor<17xf32>
    %scale_empty = "tensor.empty"() : () -> tensor<289x17xf32>
    %scale = "linalg.generic"(%convert, %gains, %scale_empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb3(%x0_2: f32, %x1_1: f32, %dest_2: f32):
      %p_1 = "arith.mulf"(%x0_2, %x1_1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%p_1) : (f32) -> ()
    }) {prov.trace_role = "node0_scale"} : (tensor<289x17xf32>, tensor<17xf32>, tensor<289x17xf32>) -> tensor<289x17xf32>
    %image = "tensor.expand_shape"(%scale) <{reassociation = [[0 : i64, 1 : i64, 2 : i64], [3 : i64]], static_output_shape = array<i64: 1, 17, 17, 17>}> : (tensor<289x17xf32>) -> tensor<1x17x17x17xf32>
    %ninf = "arith.constant"() <{value = 0xff800000 : f32}> : () -> f32
    %pi = "tensor.splat"(%ninf) : (f32) -> tensor<1x8x8x17xf32>
    %win = "tensor.empty"() : () -> tensor<2x2xf32>
    %pooled = "linalg.generic"(%image, %win, %pi) <{indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, ((d1 * 2) + d4), ((d2 * 2) + d5), d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb4(%v_1: f32, %w: f32, %o: f32):
      %r_1 = "arith.maximumf"(%v_1, %o) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%r_1) : (f32) -> ()
    }) {prov.trace_role = "node0_pool"} : (tensor<1x17x17x17xf32>, tensor<2x2xf32>, tensor<1x8x8x17xf32>) -> tensor<1x8x8x17xf32>
    %qs = "arith.constant"() <{value = dense<"0x0000803F"> : tensor<f32>}> : () -> tensor<f32>
    %qz = "arith.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %readout = "quant_ext.quantize_per_tensor"(%pooled, %qs, %qz) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.trace_role = "node0_readout"} : (tensor<1x8x8x17xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x8x17xi8>
    "func.return"(%readout) : (tensor<1x8x8x17xi8>) -> ()
  }) : () -> ()
}) : () -> ()
