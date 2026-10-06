module { func.func @forward(%a: tensor<2x2x7xf32>, %b: tensor<2x8x7xf32>, %c: tensor<2x2x8xf32>) -> tensor<2x2x8xf32> attributes {llvm.emit_c_interface} {
      %r = linalg.generic {indexing_maps=[affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>,affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>,affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>], iterator_types=["parallel","parallel","parallel","reduction"]}
      ins(%a,%b:tensor<2x2x7xf32>,tensor<2x8x7xf32>) outs(%c:tensor<2x2x8xf32>) {
      ^bb0(%x:f32,%y:f32,%z:f32):
        %p = arith.mulf %x,%y:f32
        %s = arith.addf %z,%p:f32
        linalg.yield %s:f32
      } -> tensor<2x2x8xf32>
      return %r:tensor<2x2x8xf32>
    } }