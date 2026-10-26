"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<289x16xi8>, tensor<289x16xi8>) -> tensor<1x9x9x16xi8>}> ({
  ^bb0(%0: tensor<289x16xi8>, %1: tensor<289x16xi8>):
    %left_requant_gains = "arith.constant"() <{value = dense<"0xABAA2A3E"> : tensor<16xf32>}> : () -> tensor<16xf32>
    %left_requant_empty = "tensor.empty"() : () -> tensor<289x16xi8>
    %left_requant = "linalg.generic"(%0, %left_requant_gains, %left_requant_empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb1(%x0: i8, %x1: f32, %dest: i8):
      %f = "arith.sitofp"(%x0) : (i8) -> f32
      %p = "arith.mulf"(%f, %x1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %r = "math.roundeven"(%p) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      %lo = "arith.constant"() <{value = -1.280000e+02 : f32}> : () -> f32
      %hi = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
      %l = "arith.maximumf"(%r, %lo) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %h = "arith.minimumf"(%l, %hi) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %v = "arith.fptosi"(%h) : (f32) -> i8
      "linalg.yield"(%v) : (i8) -> ()
    }) {prov.trace_role = "node0_left_requant"} : (tensor<289x16xi8>, tensor<16xf32>, tensor<289x16xi8>) -> tensor<289x16xi8>
    %right_requant_gains = "arith.constant"() <{value = dense<"0xCDCCCC3D"> : tensor<16xf32>}> : () -> tensor<16xf32>
    %right_requant_empty = "tensor.empty"() : () -> tensor<289x16xi8>
    %right_requant = "linalg.generic"(%1, %right_requant_gains, %right_requant_empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb2(%x0_1: i8, %x1_1: f32, %dest_1: i8):
      %f_1 = "arith.sitofp"(%x0_1) : (i8) -> f32
      %p_1 = "arith.mulf"(%f_1, %x1_1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %r_1 = "math.roundeven"(%p_1) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      %lo_1 = "arith.constant"() <{value = -1.280000e+02 : f32}> : () -> f32
      %hi_1 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
      %l_1 = "arith.maximumf"(%r_1, %lo_1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %h_1 = "arith.minimumf"(%l_1, %hi_1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %v_1 = "arith.fptosi"(%h_1) : (f32) -> i8
      "linalg.yield"(%v_1) : (i8) -> ()
    }) {prov.trace_role = "node0_right_requant"} : (tensor<289x16xi8>, tensor<16xf32>, tensor<289x16xi8>) -> tensor<289x16xi8>
    %operand_sum_empty = "tensor.empty"() : () -> tensor<289x16xi32>
    %operand_sum = "linalg.generic"(%left_requant, %right_requant, %operand_sum_empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb3(%x0_2: i8, %x1_2: i8, %dest_2: i32):
      %a = "arith.extsi"(%x0_2) : (i8) -> i32
      %b = "arith.extsi"(%x1_2) : (i8) -> i32
      %v_2 = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "linalg.yield"(%v_2) : (i32) -> ()
    }) {prov.trace_role = "node0_operand_sum"} : (tensor<289x16xi8>, tensor<289x16xi8>, tensor<289x16xi32>) -> tensor<289x16xi32>
    %convert_empty = "tensor.empty"() : () -> tensor<289x16xf32>
    %convert = "linalg.generic"(%operand_sum, %convert_empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb4(%x0_3: i32, %dest_3: f32):
      %f_2 = "arith.sitofp"(%x0_3) : (i32) -> f32
      "linalg.yield"(%f_2) : (f32) -> ()
    }) {prov.trace_role = "node0_convert"} : (tensor<289x16xi32>, tensor<289x16xf32>) -> tensor<289x16xf32>
    %gains = "arith.constant"() <{value = dense<"0xABAA2A3E"> : tensor<16xf32>}> : () -> tensor<16xf32>
    %scale_empty = "tensor.empty"() : () -> tensor<289x16xf32>
    %scale = "linalg.generic"(%convert, %gains, %scale_empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb5(%x0_4: f32, %x1_3: f32, %dest_4: f32):
      %p_2 = "arith.mulf"(%x0_4, %x1_3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%p_2) : (f32) -> ()
    }) {prov.trace_role = "node0_scale"} : (tensor<289x16xf32>, tensor<16xf32>, tensor<289x16xf32>) -> tensor<289x16xf32>
    %activation_empty = "tensor.empty"() : () -> tensor<289x16xf32>
    %activation = "linalg.generic"(%scale, %activation_empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb6(%x0_5: f32, %dest_5: f32):
      %z = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %v_3 = "arith.maximumf"(%x0_5, %z) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%v_3) : (f32) -> ()
    }) {prov.trace_role = "node0_activation"} : (tensor<289x16xf32>, tensor<289x16xf32>) -> tensor<289x16xf32>
    %image = "tensor.expand_shape"(%activation) <{reassociation = [[0 : i64, 1 : i64, 2 : i64], [3 : i64]], static_output_shape = array<i64: 1, 17, 17, 16>}> : (tensor<289x16xf32>) -> tensor<1x17x17x16xf32>
    %pz = "arith.constant"() <{value = 0xff800000 : f32}> : () -> f32
    %border = "tensor.splat"(%pz) : (f32) -> tensor<1x19x19x16xf32>
    %framed = "tensor.insert_slice"(%image, %border) <{static_offsets = array<i64: 0, 1, 1, 0>, static_sizes = array<i64: 1, 17, 17, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (tensor<1x17x17x16xf32>, tensor<1x19x19x16xf32>) -> tensor<1x19x19x16xf32>
    %ninf = "arith.constant"() <{value = 0xff800000 : f32}> : () -> f32
    %pi = "tensor.splat"(%ninf) : (f32) -> tensor<1x9x9x16xf32>
    %win = "tensor.empty"() : () -> tensor<2x2xf32>
    %pooled = "linalg.generic"(%framed, %win, %pi) <{indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, ((d1 * 2) + d4), ((d2 * 2) + d5), d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb7(%v_4: f32, %w: f32, %o: f32):
      %r_2 = "arith.maximumf"(%v_4, %o) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%r_2) : (f32) -> ()
    }) {prov.trace_role = "node0_pool"} : (tensor<1x19x19x16xf32>, tensor<2x2xf32>, tensor<1x9x9x16xf32>) -> tensor<1x9x9x16xf32>
    %qs = "arith.constant"() <{value = dense<"0x0000803F"> : tensor<f32>}> : () -> tensor<f32>
    %qz = "arith.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %readout = "quant_ext.quantize_per_tensor"(%pooled, %qs, %qz) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.trace_role = "node0_readout"} : (tensor<1x9x9x16xf32>, tensor<f32>, tensor<i64>) -> tensor<1x9x9x16xi8>
    "func.return"(%readout) : (tensor<1x9x9x16xi8>) -> ()
  }) : () -> ()
}) : () -> ()
