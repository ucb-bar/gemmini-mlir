builtin.module {
  func.func @captured_stem_pool(%0: tensor<1x3x230x230xi8>, %1: tensor<147x64xi8>, %2: tensor<64xf32>) -> tensor<1x64x56x56xi8> {
    %3 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %4 = "tensor.extract_slice"(%3) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %5 = tensor.empty() : tensor<1x112x112x3xi8>
    %6 = linalg.transpose ins(%4:tensor<1x3x112x112xi8>) outs(%5:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %7 = tensor.collapse_shape %6 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %8 = tensor.expand_shape %7 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %9 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %10 = "tensor.extract_slice"(%9) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %11 = tensor.empty() : tensor<1x112x112x3xi8>
    %12 = linalg.transpose ins(%10:tensor<1x3x112x112xi8>) outs(%11:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %13 = tensor.collapse_shape %12 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %14 = tensor.expand_shape %13 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %15 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %16 = "tensor.extract_slice"(%15) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %17 = tensor.empty() : tensor<1x112x112x3xi8>
    %18 = linalg.transpose ins(%16:tensor<1x3x112x112xi8>) outs(%17:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %19 = tensor.collapse_shape %18 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %20 = tensor.expand_shape %19 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %21 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %22 = "tensor.extract_slice"(%21) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %23 = tensor.empty() : tensor<1x112x112x3xi8>
    %24 = linalg.transpose ins(%22:tensor<1x3x112x112xi8>) outs(%23:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %25 = tensor.collapse_shape %24 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %26 = tensor.expand_shape %25 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %27 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %28 = "tensor.extract_slice"(%27) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %29 = tensor.empty() : tensor<1x112x112x3xi8>
    %30 = linalg.transpose ins(%28:tensor<1x3x112x112xi8>) outs(%29:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %31 = tensor.collapse_shape %30 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %32 = tensor.expand_shape %31 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %33 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %34 = "tensor.extract_slice"(%33) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %35 = tensor.empty() : tensor<1x112x112x3xi8>
    %36 = linalg.transpose ins(%34:tensor<1x3x112x112xi8>) outs(%35:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %37 = tensor.collapse_shape %36 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %38 = tensor.expand_shape %37 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %39 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %40 = "tensor.extract_slice"(%39) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %41 = tensor.empty() : tensor<1x112x112x3xi8>
    %42 = linalg.transpose ins(%40:tensor<1x3x112x112xi8>) outs(%41:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %43 = tensor.collapse_shape %42 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %44 = tensor.expand_shape %43 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %45 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %46 = "tensor.extract_slice"(%45) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %47 = tensor.empty() : tensor<1x112x112x3xi8>
    %48 = linalg.transpose ins(%46:tensor<1x3x112x112xi8>) outs(%47:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %49 = tensor.collapse_shape %48 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %50 = tensor.expand_shape %49 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %51 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %52 = "tensor.extract_slice"(%51) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %53 = tensor.empty() : tensor<1x112x112x3xi8>
    %54 = linalg.transpose ins(%52:tensor<1x3x112x112xi8>) outs(%53:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %55 = tensor.collapse_shape %54 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %56 = tensor.expand_shape %55 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %57 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %58 = "tensor.extract_slice"(%57) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %59 = tensor.empty() : tensor<1x112x112x3xi8>
    %60 = linalg.transpose ins(%58:tensor<1x3x112x112xi8>) outs(%59:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %61 = tensor.collapse_shape %60 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %62 = tensor.expand_shape %61 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %63 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %64 = "tensor.extract_slice"(%63) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %65 = tensor.empty() : tensor<1x112x112x3xi8>
    %66 = linalg.transpose ins(%64:tensor<1x3x112x112xi8>) outs(%65:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %67 = tensor.collapse_shape %66 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %68 = tensor.expand_shape %67 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %69 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %70 = "tensor.extract_slice"(%69) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %71 = tensor.empty() : tensor<1x112x112x3xi8>
    %72 = linalg.transpose ins(%70:tensor<1x3x112x112xi8>) outs(%71:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %73 = tensor.collapse_shape %72 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %74 = tensor.expand_shape %73 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %75 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %76 = "tensor.extract_slice"(%75) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %77 = tensor.empty() : tensor<1x112x112x3xi8>
    %78 = linalg.transpose ins(%76:tensor<1x3x112x112xi8>) outs(%77:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %79 = tensor.collapse_shape %78 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %80 = tensor.expand_shape %79 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %81 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 1, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %82 = "tensor.extract_slice"(%81) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %83 = tensor.empty() : tensor<1x112x112x3xi8>
    %84 = linalg.transpose ins(%82:tensor<1x3x112x112xi8>) outs(%83:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %85 = tensor.collapse_shape %84 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %86 = tensor.expand_shape %85 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %87 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %88 = "tensor.extract_slice"(%87) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %89 = tensor.empty() : tensor<1x112x112x3xi8>
    %90 = linalg.transpose ins(%88:tensor<1x3x112x112xi8>) outs(%89:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %91 = tensor.collapse_shape %90 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %92 = tensor.expand_shape %91 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %93 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %94 = "tensor.extract_slice"(%93) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %95 = tensor.empty() : tensor<1x112x112x3xi8>
    %96 = linalg.transpose ins(%94:tensor<1x3x112x112xi8>) outs(%95:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %97 = tensor.collapse_shape %96 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %98 = tensor.expand_shape %97 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %99 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %100 = "tensor.extract_slice"(%99) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %101 = tensor.empty() : tensor<1x112x112x3xi8>
    %102 = linalg.transpose ins(%100:tensor<1x3x112x112xi8>) outs(%101:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %103 = tensor.collapse_shape %102 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %104 = tensor.expand_shape %103 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %105 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %106 = "tensor.extract_slice"(%105) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %107 = tensor.empty() : tensor<1x112x112x3xi8>
    %108 = linalg.transpose ins(%106:tensor<1x3x112x112xi8>) outs(%107:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %109 = tensor.collapse_shape %108 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %110 = tensor.expand_shape %109 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %111 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %112 = "tensor.extract_slice"(%111) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %113 = tensor.empty() : tensor<1x112x112x3xi8>
    %114 = linalg.transpose ins(%112:tensor<1x3x112x112xi8>) outs(%113:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %115 = tensor.collapse_shape %114 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %116 = tensor.expand_shape %115 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %117 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %118 = "tensor.extract_slice"(%117) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %119 = tensor.empty() : tensor<1x112x112x3xi8>
    %120 = linalg.transpose ins(%118:tensor<1x3x112x112xi8>) outs(%119:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %121 = tensor.collapse_shape %120 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %122 = tensor.expand_shape %121 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %123 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 2, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %124 = "tensor.extract_slice"(%123) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %125 = tensor.empty() : tensor<1x112x112x3xi8>
    %126 = linalg.transpose ins(%124:tensor<1x3x112x112xi8>) outs(%125:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %127 = tensor.collapse_shape %126 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %128 = tensor.expand_shape %127 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %129 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %130 = "tensor.extract_slice"(%129) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %131 = tensor.empty() : tensor<1x112x112x3xi8>
    %132 = linalg.transpose ins(%130:tensor<1x3x112x112xi8>) outs(%131:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %133 = tensor.collapse_shape %132 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %134 = tensor.expand_shape %133 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %135 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %136 = "tensor.extract_slice"(%135) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %137 = tensor.empty() : tensor<1x112x112x3xi8>
    %138 = linalg.transpose ins(%136:tensor<1x3x112x112xi8>) outs(%137:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %139 = tensor.collapse_shape %138 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %140 = tensor.expand_shape %139 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %141 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %142 = "tensor.extract_slice"(%141) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %143 = tensor.empty() : tensor<1x112x112x3xi8>
    %144 = linalg.transpose ins(%142:tensor<1x3x112x112xi8>) outs(%143:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %145 = tensor.collapse_shape %144 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %146 = tensor.expand_shape %145 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %147 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %148 = "tensor.extract_slice"(%147) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %149 = tensor.empty() : tensor<1x112x112x3xi8>
    %150 = linalg.transpose ins(%148:tensor<1x3x112x112xi8>) outs(%149:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %151 = tensor.collapse_shape %150 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %152 = tensor.expand_shape %151 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %153 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %154 = "tensor.extract_slice"(%153) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %155 = tensor.empty() : tensor<1x112x112x3xi8>
    %156 = linalg.transpose ins(%154:tensor<1x3x112x112xi8>) outs(%155:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %157 = tensor.collapse_shape %156 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %158 = tensor.expand_shape %157 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %159 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %160 = "tensor.extract_slice"(%159) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %161 = tensor.empty() : tensor<1x112x112x3xi8>
    %162 = linalg.transpose ins(%160:tensor<1x3x112x112xi8>) outs(%161:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %163 = tensor.collapse_shape %162 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %164 = tensor.expand_shape %163 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %165 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 3, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %166 = "tensor.extract_slice"(%165) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %167 = tensor.empty() : tensor<1x112x112x3xi8>
    %168 = linalg.transpose ins(%166:tensor<1x3x112x112xi8>) outs(%167:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %169 = tensor.collapse_shape %168 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %170 = tensor.expand_shape %169 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %171 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %172 = "tensor.extract_slice"(%171) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %173 = tensor.empty() : tensor<1x112x112x3xi8>
    %174 = linalg.transpose ins(%172:tensor<1x3x112x112xi8>) outs(%173:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %175 = tensor.collapse_shape %174 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %176 = tensor.expand_shape %175 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %177 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %178 = "tensor.extract_slice"(%177) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %179 = tensor.empty() : tensor<1x112x112x3xi8>
    %180 = linalg.transpose ins(%178:tensor<1x3x112x112xi8>) outs(%179:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %181 = tensor.collapse_shape %180 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %182 = tensor.expand_shape %181 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %183 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %184 = "tensor.extract_slice"(%183) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %185 = tensor.empty() : tensor<1x112x112x3xi8>
    %186 = linalg.transpose ins(%184:tensor<1x3x112x112xi8>) outs(%185:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %187 = tensor.collapse_shape %186 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %188 = tensor.expand_shape %187 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %189 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %190 = "tensor.extract_slice"(%189) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %191 = tensor.empty() : tensor<1x112x112x3xi8>
    %192 = linalg.transpose ins(%190:tensor<1x3x112x112xi8>) outs(%191:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %193 = tensor.collapse_shape %192 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %194 = tensor.expand_shape %193 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %195 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %196 = "tensor.extract_slice"(%195) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %197 = tensor.empty() : tensor<1x112x112x3xi8>
    %198 = linalg.transpose ins(%196:tensor<1x3x112x112xi8>) outs(%197:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %199 = tensor.collapse_shape %198 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %200 = tensor.expand_shape %199 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %201 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %202 = "tensor.extract_slice"(%201) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %203 = tensor.empty() : tensor<1x112x112x3xi8>
    %204 = linalg.transpose ins(%202:tensor<1x3x112x112xi8>) outs(%203:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %205 = tensor.collapse_shape %204 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %206 = tensor.expand_shape %205 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %207 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 4, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %208 = "tensor.extract_slice"(%207) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %209 = tensor.empty() : tensor<1x112x112x3xi8>
    %210 = linalg.transpose ins(%208:tensor<1x3x112x112xi8>) outs(%209:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %211 = tensor.collapse_shape %210 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %212 = tensor.expand_shape %211 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %213 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %214 = "tensor.extract_slice"(%213) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %215 = tensor.empty() : tensor<1x112x112x3xi8>
    %216 = linalg.transpose ins(%214:tensor<1x3x112x112xi8>) outs(%215:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %217 = tensor.collapse_shape %216 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %218 = tensor.expand_shape %217 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %219 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %220 = "tensor.extract_slice"(%219) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %221 = tensor.empty() : tensor<1x112x112x3xi8>
    %222 = linalg.transpose ins(%220:tensor<1x3x112x112xi8>) outs(%221:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %223 = tensor.collapse_shape %222 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %224 = tensor.expand_shape %223 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %225 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %226 = "tensor.extract_slice"(%225) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %227 = tensor.empty() : tensor<1x112x112x3xi8>
    %228 = linalg.transpose ins(%226:tensor<1x3x112x112xi8>) outs(%227:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %229 = tensor.collapse_shape %228 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %230 = tensor.expand_shape %229 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %231 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %232 = "tensor.extract_slice"(%231) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %233 = tensor.empty() : tensor<1x112x112x3xi8>
    %234 = linalg.transpose ins(%232:tensor<1x3x112x112xi8>) outs(%233:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %235 = tensor.collapse_shape %234 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %236 = tensor.expand_shape %235 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %237 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %238 = "tensor.extract_slice"(%237) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %239 = tensor.empty() : tensor<1x112x112x3xi8>
    %240 = linalg.transpose ins(%238:tensor<1x3x112x112xi8>) outs(%239:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %241 = tensor.collapse_shape %240 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %242 = tensor.expand_shape %241 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %243 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %244 = "tensor.extract_slice"(%243) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %245 = tensor.empty() : tensor<1x112x112x3xi8>
    %246 = linalg.transpose ins(%244:tensor<1x3x112x112xi8>) outs(%245:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %247 = tensor.collapse_shape %246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %248 = tensor.expand_shape %247 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %249 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 5, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %250 = "tensor.extract_slice"(%249) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %251 = tensor.empty() : tensor<1x112x112x3xi8>
    %252 = linalg.transpose ins(%250:tensor<1x3x112x112xi8>) outs(%251:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %253 = tensor.collapse_shape %252 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %254 = tensor.expand_shape %253 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %255 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %256 = "tensor.extract_slice"(%255) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %257 = tensor.empty() : tensor<1x112x112x3xi8>
    %258 = linalg.transpose ins(%256:tensor<1x3x112x112xi8>) outs(%257:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %259 = tensor.collapse_shape %258 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %260 = tensor.expand_shape %259 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %261 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %262 = "tensor.extract_slice"(%261) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %263 = tensor.empty() : tensor<1x112x112x3xi8>
    %264 = linalg.transpose ins(%262:tensor<1x3x112x112xi8>) outs(%263:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %265 = tensor.collapse_shape %264 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %266 = tensor.expand_shape %265 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %267 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %268 = "tensor.extract_slice"(%267) <{static_offsets = array<i64: 0, 0, 0, 2>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %269 = tensor.empty() : tensor<1x112x112x3xi8>
    %270 = linalg.transpose ins(%268:tensor<1x3x112x112xi8>) outs(%269:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %271 = tensor.collapse_shape %270 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %272 = tensor.expand_shape %271 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %273 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %274 = "tensor.extract_slice"(%273) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %275 = tensor.empty() : tensor<1x112x112x3xi8>
    %276 = linalg.transpose ins(%274:tensor<1x3x112x112xi8>) outs(%275:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %277 = tensor.collapse_shape %276 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %278 = tensor.expand_shape %277 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %279 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %280 = "tensor.extract_slice"(%279) <{static_offsets = array<i64: 0, 0, 0, 4>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %281 = tensor.empty() : tensor<1x112x112x3xi8>
    %282 = linalg.transpose ins(%280:tensor<1x3x112x112xi8>) outs(%281:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %283 = tensor.collapse_shape %282 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %284 = tensor.expand_shape %283 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %285 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %286 = "tensor.extract_slice"(%285) <{static_offsets = array<i64: 0, 0, 0, 5>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %287 = tensor.empty() : tensor<1x112x112x3xi8>
    %288 = linalg.transpose ins(%286:tensor<1x3x112x112xi8>) outs(%287:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %289 = tensor.collapse_shape %288 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %290 = tensor.expand_shape %289 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %291 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 6, 0>, static_sizes = array<i64: 1, 3, 112, 230>, static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x230x230xi8>) -> tensor<1x3x112x230xi8>
    %292 = "tensor.extract_slice"(%291) <{static_offsets = array<i64: 0, 0, 0, 6>, static_sizes = array<i64: 1, 3, 112, 112>, static_strides = array<i64: 1, 1, 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x3x112x230xi8>) -> tensor<1x3x112x112xi8>
    %293 = tensor.empty() : tensor<1x112x112x3xi8>
    %294 = linalg.transpose ins(%292:tensor<1x3x112x112xi8>) outs(%293:tensor<1x112x112x3xi8>) permutation = [0, 2, 3, 1]
    %295 = tensor.collapse_shape %294 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<1x112x112x3xi8> into tensor<37632xi8>
    %296 = tensor.expand_shape %295 [[0 : i64, 1 : i64]] output_shape [12544, 3] : tensor<37632xi8> into tensor<12544x3xi8>
    %297 = tensor.concat dim(1) %8, %14, %20, %26, %32, %38, %44, %50, %56, %62, %68, %74, %80, %86, %92, %98, %104, %110, %116, %122, %128, %134, %140, %146, %152, %158, %164, %170, %176, %182, %188, %194, %200, %206, %212, %218, %224, %230, %236, %242, %248, %254, %260, %266, %272, %278, %284, %290, %296 : (tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>, tensor<12544x3xi8>) -> tensor<12544x147xi8>
    %298 = arith.constant 0 : i32
    %299 = tensor.splat %298 : tensor<12544x64xi32>
    %300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%297, %1 : tensor<12544x147xi8>, tensor<147x64xi8>) outs(%299 : tensor<12544x64xi32>) {
    ^bb0(%301: i8, %302: i8, %303: i32):
      %304 = arith.extsi %301 : i8 to i32
      %305 = arith.extsi %302 : i8 to i32
      %306 = arith.muli %304, %305 : i32
      %307 = arith.addi %303, %306 : i32
      linalg.yield %307 : i32
    } -> tensor<12544x64xi32>
    %308 = tensor.empty() : tensor<12544x64xf32>
    %309 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%300 : tensor<12544x64xi32>) outs(%308 : tensor<12544x64xf32>) {
    ^bb1(%310: i32, %311: f32):
      %312 = arith.sitofp %310 : i32 to f32
      linalg.yield %312 : f32
    } -> tensor<12544x64xf32>
    %313 = arith.constant 0.0313860513 : f32
    %314 = tensor.splat %313 : tensor<12544x64xf32>
    %315 = tensor.empty() : tensor<12544x64xf32>
    %316 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%309, %314 : tensor<12544x64xf32>, tensor<12544x64xf32>) outs(%315 : tensor<12544x64xf32>) {
    ^bb2(%317: f32, %318: f32, %319: f32):
      %320 = arith.mulf %317, %318 : f32
      linalg.yield %320 : f32
    } -> tensor<12544x64xf32>
    %321 = arith.constant 0.000717962219 : f32
    %322 = tensor.splat %321 : tensor<12544x64xf32>
    %323 = tensor.empty() : tensor<12544x64xf32>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%316, %322 : tensor<12544x64xf32>, tensor<12544x64xf32>) outs(%323 : tensor<12544x64xf32>) {
    ^bb3(%325: f32, %326: f32, %327: f32):
      %328 = arith.mulf %325, %326 : f32
      linalg.yield %328 : f32
    } -> tensor<12544x64xf32>
    %329 = tensor.collapse_shape %324 [[0 : i64, 1 : i64]] : tensor<12544x64xf32> into tensor<802816xf32>
    %330 = tensor.expand_shape %329 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 112, 112, 64] : tensor<802816xf32> into tensor<1x112x112x64xf32>
    %331 = tensor.empty() : tensor<1x64x112x112xf32>
    %332 = linalg.transpose ins(%330:tensor<1x112x112x64xf32>) outs(%331:tensor<1x64x112x112xf32>) permutation = [0, 3, 1, 2]
    %333 = tensor.expand_shape %2 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %334 = tensor.empty() : tensor<1x64x112x112xf32>
    %335 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%332, %333 : tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) outs(%334 : tensor<1x64x112x112xf32>) {
    ^bb4(%336: f32, %337: f32, %338: f32):
      %339 = arith.addf %336, %337 : f32
      linalg.yield %339 : f32
    } -> tensor<1x64x112x112xf32>
    %340 = tensor.empty() : tensor<1x64x112x112xf32>
    %341 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%335 : tensor<1x64x112x112xf32>) outs(%340 : tensor<1x64x112x112xf32>) {
    ^bb5(%342: f32, %343: f32):
      %344 = arith.constant 0.000000e+00 : f32
      %345 = arith.maximumf %342, %344 : f32
      linalg.yield %345 : f32
    } -> tensor<1x64x112x112xf32>
    %346 = arith.constant 0xff800000 : f32
    %347 = tensor.splat %346 : tensor<1x64x114x114xf32>
    %348 = "tensor.insert_slice"(%341, %347) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 64, 112, 112>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (tensor<1x64x112x112xf32>, tensor<1x64x114x114xf32>) -> tensor<1x64x114x114xf32>
    %349 = tensor.empty() : tensor<3x3xf32>
    %350 = tensor.splat %346 : tensor<1x64x56x56xf32>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, ((d2 * 2) + d4), ((d3 * 2) + d5))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%348, %349 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%350 : tensor<1x64x56x56xf32>) {
    ^bb6(%352: f32, %353: f32, %354: f32):
      %355 = arith.maximumf %352, %354 : f32
      linalg.yield %355 : f32
    } -> tensor<1x64x56x56xf32>
    %356 = arith.constant 0.0108702034 : f32
    %357 = tensor.splat %356 : tensor<f32>
    %358 = arith.constant 0 : i64
    %359 = tensor.splat %358 : tensor<i64>
    %360 = "quant_ext.quantize_per_tensor"(%351, %357, %359) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> : (tensor<1x64x56x56xf32>, tensor<f32>, tensor<i64>) -> tensor<1x64x56x56xi8>
    func.return %360 : tensor<1x64x56x56xi8>
  }
}