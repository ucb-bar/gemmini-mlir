module { func.func @forward(%a: tensor<2x2x3xf32>, %b: tensor<2x3x0xf32>, %c: tensor<2x2x0xf32>) -> tensor<2x2x0xf32> attributes {llvm.emit_c_interface} {
      %r = linalg.generic {indexing_maps=[affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>,affine_map<(d0,d1,d2,d3)->(d0,d3,d2)>,affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>], iterator_types=["parallel","parallel","parallel","reduction"]}
      ins(%a,%b:tensor<2x2x3xf32>,tensor<2x3x0xf32>) outs(%c:tensor<2x2x0xf32>) {
      ^bb0(%x:f32,%y:f32,%z:f32):
        %p = arith.mulf %x,%y:f32
        %s = arith.addf %p,%z:f32
        linalg.yield %s:f32
      } -> tensor<2x2x0xf32>
      return %r:tensor<2x2x0xf32>
    } }