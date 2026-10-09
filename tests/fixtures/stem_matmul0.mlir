builtin.module {
  func.func @captured_stem(%0: tensor<1x3x230x230xi8>, %1: tensor<147x64xi8>) -> tensor<12544x64xi32> {
    %2 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %3 = "tensor.extract_slice"(%2) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %4 = tensor.empty() : tensor<1x112x112x3xi8>
    %5 = linalg.transpose ins(%3:tensor<1x3x112x112xi8>) outs(%4:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %6 = tensor.collapse_shape %5 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %7 = tensor.expand_shape %6 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %8 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %9 = "tensor.extract_slice"(%8) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %10 = tensor.empty() : tensor<1x112x112x3xi8>
    %11 = linalg.transpose ins(%9:tensor<1x3x112x112xi8>) outs(%10:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %12 = tensor.collapse_shape %11 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %13 = tensor.expand_shape %12 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %14 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %15 = "tensor.extract_slice"(%14) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %16 = tensor.empty() : tensor<1x112x112x3xi8>
    %17 = linalg.transpose ins(%15:tensor<1x3x112x112xi8>) outs(%16:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %18 = tensor.collapse_shape %17 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %19 = tensor.expand_shape %18 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %20 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %21 = "tensor.extract_slice"(%20) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %22 = tensor.empty() : tensor<1x112x112x3xi8>
    %23 = linalg.transpose ins(%21:tensor<1x3x112x112xi8>) outs(%22:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %24 = tensor.collapse_shape %23 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %25 = tensor.expand_shape %24 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %26 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %27 = "tensor.extract_slice"(%26) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %28 = tensor.empty() : tensor<1x112x112x3xi8>
    %29 = linalg.transpose ins(%27:tensor<1x3x112x112xi8>) outs(%28:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %30 = tensor.collapse_shape %29 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %31 = tensor.expand_shape %30 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %32 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %33 = "tensor.extract_slice"(%32) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %34 = tensor.empty() : tensor<1x112x112x3xi8>
    %35 = linalg.transpose ins(%33:tensor<1x3x112x112xi8>) outs(%34:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %36 = tensor.collapse_shape %35 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %37 = tensor.expand_shape %36 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %38 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %39 = "tensor.extract_slice"(%38) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %40 = tensor.empty() : tensor<1x112x112x3xi8>
    %41 = linalg.transpose ins(%39:tensor<1x3x112x112xi8>) outs(%40:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %42 = tensor.collapse_shape %41 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %43 = tensor.expand_shape %42 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %44 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %45 = "tensor.extract_slice"(%44) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %46 = tensor.empty() : tensor<1x112x112x3xi8>
    %47 = linalg.transpose ins(%45:tensor<1x3x112x112xi8>) outs(%46:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %48 = tensor.collapse_shape %47 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %49 = tensor.expand_shape %48 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %50 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %51 = "tensor.extract_slice"(%50) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %52 = tensor.empty() : tensor<1x112x112x3xi8>
    %53 = linalg.transpose ins(%51:tensor<1x3x112x112xi8>) outs(%52:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %54 = tensor.collapse_shape %53 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %55 = tensor.expand_shape %54 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %56 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %57 = "tensor.extract_slice"(%56) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %58 = tensor.empty() : tensor<1x112x112x3xi8>
    %59 = linalg.transpose ins(%57:tensor<1x3x112x112xi8>) outs(%58:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %60 = tensor.collapse_shape %59 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %61 = tensor.expand_shape %60 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %62 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %63 = "tensor.extract_slice"(%62) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %64 = tensor.empty() : tensor<1x112x112x3xi8>
    %65 = linalg.transpose ins(%63:tensor<1x3x112x112xi8>) outs(%64:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %66 = tensor.collapse_shape %65 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %67 = tensor.expand_shape %66 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %68 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %69 = "tensor.extract_slice"(%68) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %70 = tensor.empty() : tensor<1x112x112x3xi8>
    %71 = linalg.transpose ins(%69:tensor<1x3x112x112xi8>) outs(%70:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %72 = tensor.collapse_shape %71 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %73 = tensor.expand_shape %72 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %74 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %75 = "tensor.extract_slice"(%74) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %76 = tensor.empty() : tensor<1x112x112x3xi8>
    %77 = linalg.transpose ins(%75:tensor<1x3x112x112xi8>) outs(%76:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %78 = tensor.collapse_shape %77 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %79 = tensor.expand_shape %78 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %80 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %81 = "tensor.extract_slice"(%80) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %82 = tensor.empty() : tensor<1x112x112x3xi8>
    %83 = linalg.transpose ins(%81:tensor<1x3x112x112xi8>) outs(%82:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %84 = tensor.collapse_shape %83 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %85 = tensor.expand_shape %84 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %86 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %87 = "tensor.extract_slice"(%86) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %88 = tensor.empty() : tensor<1x112x112x3xi8>
    %89 = linalg.transpose ins(%87:tensor<1x3x112x112xi8>) outs(%88:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %90 = tensor.collapse_shape %89 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %91 = tensor.expand_shape %90 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %92 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %93 = "tensor.extract_slice"(%92) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %94 = tensor.empty() : tensor<1x112x112x3xi8>
    %95 = linalg.transpose ins(%93:tensor<1x3x112x112xi8>) outs(%94:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %96 = tensor.collapse_shape %95 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %97 = tensor.expand_shape %96 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %98 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %99 = "tensor.extract_slice"(%98) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %100 = tensor.empty() : tensor<1x112x112x3xi8>
    %101 = linalg.transpose ins(%99:tensor<1x3x112x112xi8>) outs(%100:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %102 = tensor.collapse_shape %101 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %103 = tensor.expand_shape %102 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %104 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %105 = "tensor.extract_slice"(%104) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %106 = tensor.empty() : tensor<1x112x112x3xi8>
    %107 = linalg.transpose ins(%105:tensor<1x3x112x112xi8>) outs(%106:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %108 = tensor.collapse_shape %107 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %109 = tensor.expand_shape %108 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %110 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %111 = "tensor.extract_slice"(%110) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %112 = tensor.empty() : tensor<1x112x112x3xi8>
    %113 = linalg.transpose ins(%111:tensor<1x3x112x112xi8>) outs(%112:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %114 = tensor.collapse_shape %113 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %115 = tensor.expand_shape %114 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %116 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %117 = "tensor.extract_slice"(%116) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %118 = tensor.empty() : tensor<1x112x112x3xi8>
    %119 = linalg.transpose ins(%117:tensor<1x3x112x112xi8>) outs(%118:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %120 = tensor.collapse_shape %119 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %121 = tensor.expand_shape %120 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %122 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %123 = "tensor.extract_slice"(%122) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %124 = tensor.empty() : tensor<1x112x112x3xi8>
    %125 = linalg.transpose ins(%123:tensor<1x3x112x112xi8>) outs(%124:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %126 = tensor.collapse_shape %125 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %127 = tensor.expand_shape %126 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %128 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %129 = "tensor.extract_slice"(%128) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %130 = tensor.empty() : tensor<1x112x112x3xi8>
    %131 = linalg.transpose ins(%129:tensor<1x3x112x112xi8>) outs(%130:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %132 = tensor.collapse_shape %131 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %133 = tensor.expand_shape %132 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %134 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %135 = "tensor.extract_slice"(%134) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %136 = tensor.empty() : tensor<1x112x112x3xi8>
    %137 = linalg.transpose ins(%135:tensor<1x3x112x112xi8>) outs(%136:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %138 = tensor.collapse_shape %137 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %139 = tensor.expand_shape %138 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %140 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %141 = "tensor.extract_slice"(%140) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %142 = tensor.empty() : tensor<1x112x112x3xi8>
    %143 = linalg.transpose ins(%141:tensor<1x3x112x112xi8>) outs(%142:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %144 = tensor.collapse_shape %143 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %145 = tensor.expand_shape %144 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %146 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %147 = "tensor.extract_slice"(%146) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %148 = tensor.empty() : tensor<1x112x112x3xi8>
    %149 = linalg.transpose ins(%147:tensor<1x3x112x112xi8>) outs(%148:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %150 = tensor.collapse_shape %149 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %151 = tensor.expand_shape %150 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %152 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %153 = "tensor.extract_slice"(%152) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %154 = tensor.empty() : tensor<1x112x112x3xi8>
    %155 = linalg.transpose ins(%153:tensor<1x3x112x112xi8>) outs(%154:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %156 = tensor.collapse_shape %155 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %157 = tensor.expand_shape %156 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %158 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %159 = "tensor.extract_slice"(%158) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %160 = tensor.empty() : tensor<1x112x112x3xi8>
    %161 = linalg.transpose ins(%159:tensor<1x3x112x112xi8>) outs(%160:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %162 = tensor.collapse_shape %161 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %163 = tensor.expand_shape %162 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %164 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %165 = "tensor.extract_slice"(%164) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %166 = tensor.empty() : tensor<1x112x112x3xi8>
    %167 = linalg.transpose ins(%165:tensor<1x3x112x112xi8>) outs(%166:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %168 = tensor.collapse_shape %167 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %169 = tensor.expand_shape %168 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %170 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %171 = "tensor.extract_slice"(%170) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %172 = tensor.empty() : tensor<1x112x112x3xi8>
    %173 = linalg.transpose ins(%171:tensor<1x3x112x112xi8>) outs(%172:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %174 = tensor.collapse_shape %173 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %175 = tensor.expand_shape %174 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %176 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %177 = "tensor.extract_slice"(%176) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %178 = tensor.empty() : tensor<1x112x112x3xi8>
    %179 = linalg.transpose ins(%177:tensor<1x3x112x112xi8>) outs(%178:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %180 = tensor.collapse_shape %179 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %181 = tensor.expand_shape %180 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %182 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %183 = "tensor.extract_slice"(%182) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %184 = tensor.empty() : tensor<1x112x112x3xi8>
    %185 = linalg.transpose ins(%183:tensor<1x3x112x112xi8>) outs(%184:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %186 = tensor.collapse_shape %185 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %187 = tensor.expand_shape %186 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %188 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %189 = "tensor.extract_slice"(%188) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %190 = tensor.empty() : tensor<1x112x112x3xi8>
    %191 = linalg.transpose ins(%189:tensor<1x3x112x112xi8>) outs(%190:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %192 = tensor.collapse_shape %191 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %193 = tensor.expand_shape %192 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %194 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %195 = "tensor.extract_slice"(%194) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %196 = tensor.empty() : tensor<1x112x112x3xi8>
    %197 = linalg.transpose ins(%195:tensor<1x3x112x112xi8>) outs(%196:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %198 = tensor.collapse_shape %197 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %199 = tensor.expand_shape %198 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %200 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %201 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %202 = tensor.empty() : tensor<1x112x112x3xi8>
    %203 = linalg.transpose ins(%201:tensor<1x3x112x112xi8>) outs(%202:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %204 = tensor.collapse_shape %203 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %205 = tensor.expand_shape %204 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %206 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %207 = "tensor.extract_slice"(%206) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %208 = tensor.empty() : tensor<1x112x112x3xi8>
    %209 = linalg.transpose ins(%207:tensor<1x3x112x112xi8>) outs(%208:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %210 = tensor.collapse_shape %209 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %211 = tensor.expand_shape %210 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %212 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %213 = "tensor.extract_slice"(%212) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %214 = tensor.empty() : tensor<1x112x112x3xi8>
    %215 = linalg.transpose ins(%213:tensor<1x3x112x112xi8>) outs(%214:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %216 = tensor.collapse_shape %215 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %217 = tensor.expand_shape %216 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %218 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %219 = "tensor.extract_slice"(%218) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %220 = tensor.empty() : tensor<1x112x112x3xi8>
    %221 = linalg.transpose ins(%219:tensor<1x3x112x112xi8>) outs(%220:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %222 = tensor.collapse_shape %221 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %223 = tensor.expand_shape %222 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %224 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %225 = "tensor.extract_slice"(%224) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %226 = tensor.empty() : tensor<1x112x112x3xi8>
    %227 = linalg.transpose ins(%225:tensor<1x3x112x112xi8>) outs(%226:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %228 = tensor.collapse_shape %227 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %229 = tensor.expand_shape %228 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %230 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %231 = "tensor.extract_slice"(%230) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %232 = tensor.empty() : tensor<1x112x112x3xi8>
    %233 = linalg.transpose ins(%231:tensor<1x3x112x112xi8>) outs(%232:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %234 = tensor.collapse_shape %233 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %235 = tensor.expand_shape %234 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %236 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %237 = "tensor.extract_slice"(%236) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %238 = tensor.empty() : tensor<1x112x112x3xi8>
    %239 = linalg.transpose ins(%237:tensor<1x3x112x112xi8>) outs(%238:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %240 = tensor.collapse_shape %239 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %241 = tensor.expand_shape %240 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %242 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %243 = "tensor.extract_slice"(%242) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %244 = tensor.empty() : tensor<1x112x112x3xi8>
    %245 = linalg.transpose ins(%243:tensor<1x3x112x112xi8>) outs(%244:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %246 = tensor.collapse_shape %245 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %247 = tensor.expand_shape %246 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %248 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %249 = "tensor.extract_slice"(%248) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %250 = tensor.empty() : tensor<1x112x112x3xi8>
    %251 = linalg.transpose ins(%249:tensor<1x3x112x112xi8>) outs(%250:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %252 = tensor.collapse_shape %251 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %253 = tensor.expand_shape %252 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %254 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %255 = "tensor.extract_slice"(%254) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %256 = tensor.empty() : tensor<1x112x112x3xi8>
    %257 = linalg.transpose ins(%255:tensor<1x3x112x112xi8>) outs(%256:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %258 = tensor.collapse_shape %257 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %259 = tensor.expand_shape %258 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %260 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %261 = "tensor.extract_slice"(%260) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %262 = tensor.empty() : tensor<1x112x112x3xi8>
    %263 = linalg.transpose ins(%261:tensor<1x3x112x112xi8>) outs(%262:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %264 = tensor.collapse_shape %263 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %265 = tensor.expand_shape %264 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %266 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %267 = "tensor.extract_slice"(%266) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %268 = tensor.empty() : tensor<1x112x112x3xi8>
    %269 = linalg.transpose ins(%267:tensor<1x3x112x112xi8>) outs(%268:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %270 = tensor.collapse_shape %269 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %271 = tensor.expand_shape %270 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %272 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %273 = "tensor.extract_slice"(%272) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %274 = tensor.empty() : tensor<1x112x112x3xi8>
    %275 = linalg.transpose ins(%273:tensor<1x3x112x112xi8>) outs(%274:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %276 = tensor.collapse_shape %275 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %277 = tensor.expand_shape %276 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %278 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %279 = "tensor.extract_slice"(%278) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %280 = tensor.empty() : tensor<1x112x112x3xi8>
    %281 = linalg.transpose ins(%279:tensor<1x3x112x112xi8>) outs(%280:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %282 = tensor.collapse_shape %281 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %283 = tensor.expand_shape %282 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %284 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %285 = "tensor.extract_slice"(%284) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %286 = tensor.empty() : tensor<1x112x112x3xi8>
    %287 = linalg.transpose ins(%285:tensor<1x3x112x112xi8>) outs(%286:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %288 = tensor.collapse_shape %287 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %289 = tensor.expand_shape %288 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %290 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %291 = "tensor.extract_slice"(%290) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %292 = tensor.empty() : tensor<1x112x112x3xi8>
    %293 = linalg.transpose ins(%291:tensor<1x3x112x112xi8>) outs(%292:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %294 = tensor.collapse_shape %293 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %295 = tensor.expand_shape %294 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %296 = tensor.concat dim(1) %7, %13, %19, %25, %31, %37, %43, %49, %55, %61, %67, %73, %79, %85, %91, %97, %103, %109, %115, %121, %127, %133, %139, %145, %151, %157, %163, %169, %175, %181, %187, %193, %199, %205, %211, %217, %223, %229, %235, %241, %247, %253, %259, %265, %271, %277, %283, %289, %295 : (tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>) -> tensor<12544x147xi8>
    %297 = arith.constant 0 : i32
    %298 = tensor.splat %297 : tensor<12544x64xi32>
    %299 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%296, %1 : tensor<12544x147xi8>, tensor<147x64xi8>) outs(%298 : tensor<12544x64xi32>) {
    ^bb0(%300: i8, %301: i8, %302: i32):
      %303 = arith.extsi %300 : i8 to i32
      %304 = arith.extsi %301 : i8 to i32
      %305 = arith.muli %303, %304 : i32
      %306 = arith.addi %302, %305 : i32
      linalg.yield %306 : i32
    } -> tensor<12544x64xi32>
    func.return %299 : tensor<12544x64xi32>
  }
}