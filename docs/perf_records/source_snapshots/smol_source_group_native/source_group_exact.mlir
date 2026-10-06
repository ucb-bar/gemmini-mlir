builtin.module {
  func.func public @source_group_exact(%0: tensor<1x12x256x64xbf16>, %1: tensor<1x12x512x64xbf16>, %2: tensor<1x1x256x512xi1>, %3: tensor<1x12x192x64xbf16>, %4: tensor<1x12x192x64xbf16>, %5: tensor<1x12x128x64xbf16>, %6: tensor<1x12x512x64xbf16>, %7: tensor<1x1x256x512xi1>, %8: tensor<1x12x192x64xbf16>, %9: tensor<1x12x192x64xbf16>, %10: tensor<1x12x128x64xbf16>) -> tensor<1x12x256x64xbf16> attributes {llvm.emit_c_interface} {
    %11 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %12 = tensor.splat %11 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x512xf32>
    %13 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0xff800000 : f32
    %14 = tensor.splat %13 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256xf32>
    %15 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0xff800000 : f32
    %16 = tensor.splat %15 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256xf32>
    %17 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %18 = tensor.splat %17 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x8xf32>
    %19 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %20 = tensor.splat %19 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256xf32>
    %21 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %22 = tensor.splat %21 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x64xf32>
    %23 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %24 = tensor.splat %23 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x64xf32>
    %25 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %26 = tensor.splat %25 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x64xf32>
    %27 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %28 = tensor.splat %27 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x64xf32>
    %29 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %30 = tensor.splat %29 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x512xf32>
    %31 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0xff800000 : f32
    %32 = tensor.splat %31 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256xf32>
    %33 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %34 = tensor.splat %33 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x8xf32>
    %35 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %36 = tensor.splat %35 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x64xf32>
    %37 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %38 = tensor.splat %37 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x64xf32>
    %39 = arith.constant {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} 0.000000e+00 : f32
    %40 = tensor.splat %39 {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x64xf32>
    %41 = tensor.empty() : tensor<1x12x256x512xf32>
    %42 = tensor.empty() : tensor<1x12x256x512xf32>
    %43 = tensor.empty() : tensor<1x12x256xf32>
    %44 = tensor.empty() : tensor<1x12x256xf32>
    %45 = tensor.empty() : tensor<1x12x256x512xf32>
    %46 = tensor.empty() : tensor<1x12x256x512xf32>
    %47 = tensor.empty() : tensor<1x12x256xf32>
    %48 = tensor.empty() : tensor<1x12x256xf32>
    %49 = tensor.empty() : tensor<1x12x256xf32>
    %50 = tensor.empty() : tensor<1x12x256x512xbf16>
    %51 = tensor.empty() : tensor<1x12x256x64xf32>
    %52 = tensor.empty() : tensor<1x12x256x64xf32>
    %53 = tensor.empty() : tensor<1x12x256x64xf32>
    %54 = tensor.empty() : tensor<1x12x256x64xf32>
    %55 = tensor.empty() : tensor<1x12x256x512xf32>
    %56 = tensor.empty() : tensor<1x12x256x512xf32>
    %57 = tensor.empty() : tensor<1x12x256xf32>
    %58 = tensor.empty() : tensor<1x12x256xf32>
    %59 = tensor.empty() : tensor<1x12x256x512xf32>
    %60 = tensor.empty() : tensor<1x12x256x512xf32>
    %61 = tensor.empty() : tensor<1x12x256xf32>
    %62 = tensor.empty() : tensor<1x12x256xf32>
    %63 = tensor.empty() : tensor<1x12x256xf32>
    %64 = tensor.empty() : tensor<1x12x256x512xbf16>
    %65 = tensor.empty() : tensor<1x12x256x64xf32>
    %66 = tensor.empty() : tensor<1x12x256x64xf32>
    %67 = tensor.empty() : tensor<1x12x256x64xf32>
    %68 = tensor.empty() : tensor<1x12x256x64xf32>
    %69 = tensor.empty() : tensor<1x12x256xf32>
    %70 = tensor.empty() : tensor<1x12x256x64xbf16>
    %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<1x12x256x64xbf16>, tensor<1x12x512x64xbf16>) outs(%12 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb0(%72: bf16, %73: bf16, %74: f32):
      %75 = arith.extf %72 : bf16 to f32
      %76 = arith.extf %73 : bf16 to f32
      %77 = math.fma %75, %76, %74 : f32
      linalg.yield %77 : f32
    } -> tensor<1x12x256x512xf32>
    %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71 : tensor<1x12x256x512xf32>) outs(%41 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb1(%79: f32, %80: f32):
      %81 = arith.constant 1.250000e-01 : f32
      %82 = arith.mulf %79, %81 : f32
      linalg.yield %82 : f32
    } -> tensor<1x12x256x512xf32>
    %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%78, %2 : tensor<1x12x256x512xf32>, tensor<1x1x256x512xi1>) outs(%42 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb2(%84: f32, %85: i1, %86: f32):
      %87 = arith.constant 0xff800000 : f32
      %88 = arith.select %85, %84, %87 : f32
      linalg.yield %88 : f32
    } -> tensor<1x12x256x512xf32>
    %89 = "linalg.reduce"(%83, %14) <{dimensions = array<i64: 3>}> ({
    ^bb3(%90: f32, %91: f32):
      %92 = "arith.maximumf"(%90, %91) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%92) : (f32) -> ()
    }) {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x512xf32>, tensor<1x12x256xf32>) -> tensor<1x12x256xf32>
    %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %89 : tensor<1x12x256xf32>, tensor<1x12x256xf32>) outs(%43 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb4(%94: f32, %95: f32, %96: f32):
      %97 = arith.maximumf %94, %95 : f32
      linalg.yield %97 : f32
    } -> tensor<1x12x256xf32>
    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93 : tensor<1x12x256xf32>) outs(%44 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb5(%99: f32, %100: f32):
      %101 = arith.constant 0xff800000 : f32
      %102 = arith.constant 0.000000e+00 : f32
      %103 = arith.cmpf oeq, %99, %101 : f32
      %104 = arith.select %103, %102, %99 : f32
      linalg.yield %104 : f32
    } -> tensor<1x12x256xf32>
    %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%83, %98 : tensor<1x12x256x512xf32>, tensor<1x12x256xf32>) outs(%45 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb6(%106: f32, %107: f32, %108: f32):
      %109 = arith.subf %106, %107 : f32
      linalg.yield %109 : f32
    } -> tensor<1x12x256x512xf32>
    %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105 : tensor<1x12x256x512xf32>) outs(%46 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb7(%111: f32, %112: f32):
      %113 = arith.constant 0.000000e+00 : f32
      %114 = arith.constant -87.3365479 : f32
      %115 = arith.cmpf olt, %111, %114 : f32
      %116 = arith.select %115, %113, %111 : f32
      %117 = arith.constant 1.44269502 : f32
      %118 = arith.mulf %116, %117 : f32
      %119 = math.floor %118 : f32
      %120 = arith.subf %118, %119 : f32
      %121 = arith.constant -0.079204239 : f32
      %122 = arith.constant -0.224338368 : f32
      %123 = math.fma %120, %121, %122 : f32
      %124 = arith.constant 0.303542614 : f32
      %125 = math.fma %120, %123, %124 : f32
      %126 = arith.constant 0.00010703435 : f32
      %127 = math.fma %120, %125, %126 : f32
      %128 = arith.subf %118, %127 : f32
      %129 = arith.constant 0x4B000000 : f32
      %130 = arith.constant 1.06535322e+09 : f32
      %131 = math.fma %129, %128, %130 : f32
      %132 = arith.fptosi %131 : f32 to i32
      %133 = arith.bitcast %132 : i32 to f32
      %134 = arith.select %115, %113, %133 : f32
      linalg.yield %134 : f32
    } -> tensor<1x12x256x512xf32>
    %135 = tensor.expand_shape %110 [[0 : i64], [1 : i64], [2 : i64], [3 : i64, 4 : i64]] output_shape [1, 12, 256, 64, 8] {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x512xf32> into tensor<1x12x256x64x8xf32>
    %136 = "linalg.reduce"(%135, %18) <{dimensions = array<i64: 3>}> ({
    ^bb8(%137: f32, %138: f32):
      %139 = "arith.addf"(%137, %138) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%139) : (f32) -> ()
    }) {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x64x8xf32>, tensor<1x12x256x8xf32>) -> tensor<1x12x256x8xf32>
    %140 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 3)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 4)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 5)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 6)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 7)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%136, %136, %136, %136, %136, %136, %136, %136 : tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>) outs(%47 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb9(%141: f32, %142: f32, %143: f32, %144: f32, %145: f32, %146: f32, %147: f32, %148: f32, %149: f32):
      %150 = arith.addf %141, %145 : f32
      %151 = arith.addf %142, %146 : f32
      %152 = arith.addf %143, %147 : f32
      %153 = arith.addf %144, %148 : f32
      %154 = arith.addf %150, %152 : f32
      %155 = arith.addf %151, %153 : f32
      %156 = arith.addf %154, %155 : f32
      linalg.yield %156 : f32
    } -> tensor<1x12x256xf32>
    %157 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %93 : tensor<1x12x256xf32>, tensor<1x12x256xf32>) outs(%48 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb10(%158: f32, %159: f32, %160: f32):
      %161 = arith.constant 0xff800000 : f32
      %162 = arith.constant 0.000000e+00 : f32
      %163 = arith.cmpf oeq, %159, %161 : f32
      %164 = arith.select %163, %162, %158 : f32
      %165 = arith.select %163, %162, %159 : f32
      %166 = arith.subf %164, %165 : f32
      %167 = math.exp %166 : f32
      linalg.yield %167 : f32
    } -> tensor<1x12x256xf32>
    %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%157, %20, %140 : tensor<1x12x256xf32>, tensor<1x12x256xf32>, tensor<1x12x256xf32>) outs(%49 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb11(%169: f32, %170: f32, %171: f32, %172: f32):
      %173 = math.fma %169, %170, %171 : f32
      linalg.yield %173 : f32
    } -> tensor<1x12x256xf32>
    %174 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110 : tensor<1x12x256x512xf32>) outs(%50 : tensor<1x12x256x512xbf16>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb12(%175: f32, %176: bf16):
      %177 = arith.truncf %175 : f32 to bf16
      linalg.yield %177 : bf16
    } -> tensor<1x12x256x512xbf16>
    %178 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %157 : tensor<1x12x256x64xf32>, tensor<1x12x256xf32>) outs(%51 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb13(%179: f32, %180: f32, %181: f32):
      %182 = arith.mulf %179, %180 : f32
      linalg.yield %182 : f32
    } -> tensor<1x12x256x64xf32>
    %183 = "tensor.extract_slice"(%174) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 12, 256, 192>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x512xbf16>) -> tensor<1x12x256x192xbf16>
    %184 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%183, %3 : tensor<1x12x256x192xbf16>, tensor<1x12x192x64xbf16>) outs(%24 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb14(%185: bf16, %186: bf16, %187: f32):
      %188 = arith.extf %185 : bf16 to f32
      %189 = arith.extf %186 : bf16 to f32
      %190 = math.fma %188, %189, %187 : f32
      linalg.yield %190 : f32
    } -> tensor<1x12x256x64xf32>
    %191 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%184, %178 : tensor<1x12x256x64xf32>, tensor<1x12x256x64xf32>) outs(%52 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb15(%192: f32, %193: f32, %194: f32):
      %195 = arith.addf %192, %193 : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x256x64xf32>
    %196 = "tensor.extract_slice"(%174) <{static_offsets = array<i64: 0, 0, 0, 192>, static_sizes = array<i64: 1, 12, 256, 192>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x512xbf16>) -> tensor<1x12x256x192xbf16>
    %197 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%196, %4 : tensor<1x12x256x192xbf16>, tensor<1x12x192x64xbf16>) outs(%26 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb16(%198: bf16, %199: bf16, %200: f32):
      %201 = arith.extf %198 : bf16 to f32
      %202 = arith.extf %199 : bf16 to f32
      %203 = math.fma %201, %202, %200 : f32
      linalg.yield %203 : f32
    } -> tensor<1x12x256x64xf32>
    %204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%197, %191 : tensor<1x12x256x64xf32>, tensor<1x12x256x64xf32>) outs(%53 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb17(%205: f32, %206: f32, %207: f32):
      %208 = arith.addf %205, %206 : f32
      linalg.yield %208 : f32
    } -> tensor<1x12x256x64xf32>
    %209 = "tensor.extract_slice"(%174) <{static_offsets = array<i64: 0, 0, 0, 384>, static_sizes = array<i64: 1, 12, 256, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x512xbf16>) -> tensor<1x12x256x128xbf16>
    %210 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%209, %5 : tensor<1x12x256x128xbf16>, tensor<1x12x128x64xbf16>) outs(%28 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb18(%211: bf16, %212: bf16, %213: f32):
      %214 = arith.extf %211 : bf16 to f32
      %215 = arith.extf %212 : bf16 to f32
      %216 = math.fma %214, %215, %213 : f32
      linalg.yield %216 : f32
    } -> tensor<1x12x256x64xf32>
    %217 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%210, %204 : tensor<1x12x256x64xf32>, tensor<1x12x256x64xf32>) outs(%54 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb19(%218: f32, %219: f32, %220: f32):
      %221 = arith.addf %218, %219 : f32
      linalg.yield %221 : f32
    } -> tensor<1x12x256x64xf32>
    %222 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%0, %6 : tensor<1x12x256x64xbf16>, tensor<1x12x512x64xbf16>) outs(%30 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb20(%223: bf16, %224: bf16, %225: f32):
      %226 = arith.extf %223 : bf16 to f32
      %227 = arith.extf %224 : bf16 to f32
      %228 = math.fma %226, %227, %225 : f32
      linalg.yield %228 : f32
    } -> tensor<1x12x256x512xf32>
    %229 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%222 : tensor<1x12x256x512xf32>) outs(%55 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb21(%230: f32, %231: f32):
      %232 = arith.constant 1.250000e-01 : f32
      %233 = arith.mulf %230, %232 : f32
      linalg.yield %233 : f32
    } -> tensor<1x12x256x512xf32>
    %234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%229, %7 : tensor<1x12x256x512xf32>, tensor<1x1x256x512xi1>) outs(%56 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb22(%235: f32, %236: i1, %237: f32):
      %238 = arith.constant 0xff800000 : f32
      %239 = arith.select %236, %235, %238 : f32
      linalg.yield %239 : f32
    } -> tensor<1x12x256x512xf32>
    %240 = "linalg.reduce"(%234, %32) <{dimensions = array<i64: 3>}> ({
    ^bb23(%241: f32, %242: f32):
      %243 = "arith.maximumf"(%241, %242) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%243) : (f32) -> ()
    }) {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x512xf32>, tensor<1x12x256xf32>) -> tensor<1x12x256xf32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93, %240 : tensor<1x12x256xf32>, tensor<1x12x256xf32>) outs(%57 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb24(%245: f32, %246: f32, %247: f32):
      %248 = arith.maximumf %245, %246 : f32
      linalg.yield %248 : f32
    } -> tensor<1x12x256xf32>
    %249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%244 : tensor<1x12x256xf32>) outs(%58 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb25(%250: f32, %251: f32):
      %252 = arith.constant 0xff800000 : f32
      %253 = arith.constant 0.000000e+00 : f32
      %254 = arith.cmpf oeq, %250, %252 : f32
      %255 = arith.select %254, %253, %250 : f32
      linalg.yield %255 : f32
    } -> tensor<1x12x256xf32>
    %256 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%234, %249 : tensor<1x12x256x512xf32>, tensor<1x12x256xf32>) outs(%59 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb26(%257: f32, %258: f32, %259: f32):
      %260 = arith.subf %257, %258 : f32
      linalg.yield %260 : f32
    } -> tensor<1x12x256x512xf32>
    %261 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%256 : tensor<1x12x256x512xf32>) outs(%60 : tensor<1x12x256x512xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb27(%262: f32, %263: f32):
      %264 = arith.constant 0.000000e+00 : f32
      %265 = arith.constant -87.3365479 : f32
      %266 = arith.cmpf olt, %262, %265 : f32
      %267 = arith.select %266, %264, %262 : f32
      %268 = arith.constant 1.44269502 : f32
      %269 = arith.mulf %267, %268 : f32
      %270 = math.floor %269 : f32
      %271 = arith.subf %269, %270 : f32
      %272 = arith.constant -0.079204239 : f32
      %273 = arith.constant -0.224338368 : f32
      %274 = math.fma %271, %272, %273 : f32
      %275 = arith.constant 0.303542614 : f32
      %276 = math.fma %271, %274, %275 : f32
      %277 = arith.constant 0.00010703435 : f32
      %278 = math.fma %271, %276, %277 : f32
      %279 = arith.subf %269, %278 : f32
      %280 = arith.constant 0x4B000000 : f32
      %281 = arith.constant 1.06535322e+09 : f32
      %282 = math.fma %280, %279, %281 : f32
      %283 = arith.fptosi %282 : f32 to i32
      %284 = arith.bitcast %283 : i32 to f32
      %285 = arith.select %266, %264, %284 : f32
      linalg.yield %285 : f32
    } -> tensor<1x12x256x512xf32>
    %286 = tensor.expand_shape %261 [[0 : i64], [1 : i64], [2 : i64], [3 : i64, 4 : i64]] output_shape [1, 12, 256, 64, 8] {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : tensor<1x12x256x512xf32> into tensor<1x12x256x64x8xf32>
    %287 = "linalg.reduce"(%286, %34) <{dimensions = array<i64: 3>}> ({
    ^bb28(%288: f32, %289: f32):
      %290 = "arith.addf"(%288, %289) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%290) : (f32) -> ()
    }) {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x64x8xf32>, tensor<1x12x256x8xf32>) -> tensor<1x12x256x8xf32>
    %291 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 3)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 4)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 5)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 6)>, affine_map<(d0, d1, d2) -> (d0, d1, d2, 7)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%287, %287, %287, %287, %287, %287, %287, %287 : tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>, tensor<1x12x256x8xf32>) outs(%61 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb29(%292: f32, %293: f32, %294: f32, %295: f32, %296: f32, %297: f32, %298: f32, %299: f32, %300: f32):
      %301 = arith.addf %292, %296 : f32
      %302 = arith.addf %293, %297 : f32
      %303 = arith.addf %294, %298 : f32
      %304 = arith.addf %295, %299 : f32
      %305 = arith.addf %301, %303 : f32
      %306 = arith.addf %302, %304 : f32
      %307 = arith.addf %305, %306 : f32
      linalg.yield %307 : f32
    } -> tensor<1x12x256xf32>
    %308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93, %244 : tensor<1x12x256xf32>, tensor<1x12x256xf32>) outs(%62 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb30(%309: f32, %310: f32, %311: f32):
      %312 = arith.constant 0xff800000 : f32
      %313 = arith.constant 0.000000e+00 : f32
      %314 = arith.cmpf oeq, %310, %312 : f32
      %315 = arith.select %314, %313, %309 : f32
      %316 = arith.select %314, %313, %310 : f32
      %317 = arith.subf %315, %316 : f32
      %318 = math.exp %317 : f32
      linalg.yield %318 : f32
    } -> tensor<1x12x256xf32>
    %319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%308, %168, %291 : tensor<1x12x256xf32>, tensor<1x12x256xf32>, tensor<1x12x256xf32>) outs(%63 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb31(%320: f32, %321: f32, %322: f32, %323: f32):
      %324 = math.fma %320, %321, %322 : f32
      linalg.yield %324 : f32
    } -> tensor<1x12x256xf32>
    %325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%261 : tensor<1x12x256x512xf32>) outs(%64 : tensor<1x12x256x512xbf16>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb32(%326: f32, %327: bf16):
      %328 = arith.truncf %326 : f32 to bf16
      linalg.yield %328 : bf16
    } -> tensor<1x12x256x512xbf16>
    %329 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%217, %308 : tensor<1x12x256x64xf32>, tensor<1x12x256xf32>) outs(%65 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb33(%330: f32, %331: f32, %332: f32):
      %333 = arith.mulf %330, %331 : f32
      linalg.yield %333 : f32
    } -> tensor<1x12x256x64xf32>
    %334 = "tensor.extract_slice"(%325) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 12, 256, 192>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x512xbf16>) -> tensor<1x12x256x192xbf16>
    %335 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%334, %8 : tensor<1x12x256x192xbf16>, tensor<1x12x192x64xbf16>) outs(%36 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb34(%336: bf16, %337: bf16, %338: f32):
      %339 = arith.extf %336 : bf16 to f32
      %340 = arith.extf %337 : bf16 to f32
      %341 = math.fma %339, %340, %338 : f32
      linalg.yield %341 : f32
    } -> tensor<1x12x256x64xf32>
    %342 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%335, %329 : tensor<1x12x256x64xf32>, tensor<1x12x256x64xf32>) outs(%66 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb35(%343: f32, %344: f32, %345: f32):
      %346 = arith.addf %343, %344 : f32
      linalg.yield %346 : f32
    } -> tensor<1x12x256x64xf32>
    %347 = "tensor.extract_slice"(%325) <{static_offsets = array<i64: 0, 0, 0, 192>, static_sizes = array<i64: 1, 12, 256, 192>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x512xbf16>) -> tensor<1x12x256x192xbf16>
    %348 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%347, %9 : tensor<1x12x256x192xbf16>, tensor<1x12x192x64xbf16>) outs(%38 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb36(%349: bf16, %350: bf16, %351: f32):
      %352 = arith.extf %349 : bf16 to f32
      %353 = arith.extf %350 : bf16 to f32
      %354 = math.fma %352, %353, %351 : f32
      linalg.yield %354 : f32
    } -> tensor<1x12x256x64xf32>
    %355 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%348, %342 : tensor<1x12x256x64xf32>, tensor<1x12x256x64xf32>) outs(%67 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb37(%356: f32, %357: f32, %358: f32):
      %359 = arith.addf %356, %357 : f32
      linalg.yield %359 : f32
    } -> tensor<1x12x256x64xf32>
    %360 = "tensor.extract_slice"(%325) <{static_offsets = array<i64: 0, 0, 0, 384>, static_sizes = array<i64: 1, 12, 256, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} : (tensor<1x12x256x512xbf16>) -> tensor<1x12x256x128xbf16>
    %361 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%360, %10 : tensor<1x12x256x128xbf16>, tensor<1x12x128x64xbf16>) outs(%40 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb38(%362: bf16, %363: bf16, %364: f32):
      %365 = arith.extf %362 : bf16 to f32
      %366 = arith.extf %363 : bf16 to f32
      %367 = math.fma %365, %366, %364 : f32
      linalg.yield %367 : f32
    } -> tensor<1x12x256x64xf32>
    %368 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%361, %355 : tensor<1x12x256x64xf32>, tensor<1x12x256x64xf32>) outs(%68 : tensor<1x12x256x64xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb39(%369: f32, %370: f32, %371: f32):
      %372 = arith.addf %369, %370 : f32
      linalg.yield %372 : f32
    } -> tensor<1x12x256x64xf32>
    %373 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%319 : tensor<1x12x256xf32>) outs(%69 : tensor<1x12x256xf32>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb40(%374: f32, %375: f32):
      %376 = arith.constant 1.000000e+00 : f32
      %377 = arith.constant 0.000000e+00 : f32
      %378 = arith.cmpf oeq, %374, %377 : f32
      %379 = arith.select %378, %376, %374 : f32
      %380 = arith.divf %376, %379 : f32
      linalg.yield %380 : f32
    } -> tensor<1x12x256xf32>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%368, %373 : tensor<1x12x256x64xf32>, tensor<1x12x256xf32>) outs(%70 : tensor<1x12x256x64xbf16>) attrs =  {merlin.sdpa_backend_policy = "torch_cpu_flash_bf16_avx2", merlin.sdpa_pv_backend_policy = "mkl_2024_0_u2_def_zen_k192", prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.source_node_ids = ["g:prepared:root:n1036"], prov.origin_node_ids = ["g:quantized:root:n1872"], prov.trace_role = "lowering", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn"} {
    ^bb41(%382: f32, %383: f32, %384: bf16):
      %385 = arith.mulf %382, %383 : f32
      %386 = arith.truncf %385 : f32 to bf16
      linalg.yield %386 : bf16
    } -> tensor<1x12x256x64xbf16>
    func.return %381 : tensor<1x12x256x64xbf16>
  }
}