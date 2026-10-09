module {func.func @forward(%a:tensor<12x7xf32>,%seed:tensor<12xf32>) -> tensor<12xf32> attributes {llvm.emit_c_interface} {
 %r=linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0)>],iterator_types=["parallel","reduction"]}
 ins(%a:tensor<12x7xf32>) outs(%seed:tensor<12xf32>) {^bb0(%x:f32,%z:f32):
 %p=arith.mulf %x,%x:f32 %v=arith.addf %p,%z:f32 linalg.yield %v:f32} -> tensor<12xf32>
 return %r:tensor<12xf32>
}}