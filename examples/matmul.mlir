module {
  func.func @forward(%a: tensor<16x16xi8>, %b: tensor<16x16xi8>) -> tensor<16x16xi32> {
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<16x16xi32>
    %initial = linalg.fill ins(%zero : i32) outs(%empty : tensor<16x16xi32>) -> tensor<16x16xi32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<16x16xi8>, tensor<16x16xi8>) outs(%initial : tensor<16x16xi32>) {
    ^bb0(%left: i8, %right: i8, %acc: i32):
      %l = arith.extsi %left : i8 to i32
      %r = arith.extsi %right : i8 to i32
      %product = arith.muli %l, %r : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
    } -> tensor<16x16xi32>
    return %result : tensor<16x16xi32>
  }
}
