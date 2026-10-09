; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @packet(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31) {
  br label %33

33:                                               ; preds = %95, %32
  %34 = phi i64 [ %96, %95 ], [ 0, %32 ]
  %35 = icmp slt i64 %34, 8
  br i1 %35, label %36, label %97

36:                                               ; preds = %33
  br label %37

37:                                               ; preds = %40, %36
  %38 = phi i64 [ %94, %40 ], [ 0, %36 ]
  %39 = icmp slt i64 %38, 2048
  br i1 %39, label %40, label %95

40:                                               ; preds = %37
  %41 = add i64 %38, 1
  %42 = add i64 %38, 2
  %43 = add i64 %38, 3
  %44 = mul nuw nsw i64 %34, 2048
  %45 = add nuw nsw i64 0, %44
  %46 = add nuw nsw i64 %45, %38
  %47 = getelementptr inbounds nuw float, ptr %1, i64 %46
  %48 = load float, ptr %47, align 4
  %49 = getelementptr inbounds nuw i32, ptr %10, i64 %46
  %50 = load i32, ptr %49, align 4
  %51 = getelementptr inbounds nuw float, ptr %19, i64 %38
  %52 = load float, ptr %51, align 4
  %53 = add nuw nsw i64 %45, %41
  %54 = getelementptr inbounds nuw float, ptr %1, i64 %53
  %55 = load float, ptr %54, align 4
  %56 = getelementptr inbounds nuw i32, ptr %10, i64 %53
  %57 = load i32, ptr %56, align 4
  %58 = getelementptr inbounds nuw float, ptr %19, i64 %41
  %59 = load float, ptr %58, align 4
  %60 = add nuw nsw i64 %45, %42
  %61 = getelementptr inbounds nuw float, ptr %1, i64 %60
  %62 = load float, ptr %61, align 4
  %63 = getelementptr inbounds nuw i32, ptr %10, i64 %60
  %64 = load i32, ptr %63, align 4
  %65 = getelementptr inbounds nuw float, ptr %19, i64 %42
  %66 = load float, ptr %65, align 4
  %67 = add nuw nsw i64 %45, %43
  %68 = getelementptr inbounds nuw float, ptr %1, i64 %67
  %69 = load float, ptr %68, align 4
  %70 = getelementptr inbounds nuw i32, ptr %10, i64 %67
  %71 = load i32, ptr %70, align 4
  %72 = getelementptr inbounds nuw float, ptr %19, i64 %43
  %73 = load float, ptr %72, align 4
  %74 = sitofp i32 %50 to float
  %75 = sitofp i32 %57 to float
  %76 = sitofp i32 %64 to float
  %77 = sitofp i32 %71 to float
  %78 = fmul float %74, 0x3F46745BE0000000
  %79 = fmul float %75, 0x3F46745BE0000000
  %80 = fmul float %76, 0x3F46745BE0000000
  %81 = fmul float %77, 0x3F46745BE0000000
  %82 = fmul float %78, %52
  %83 = fmul float %79, %59
  %84 = fmul float %80, %66
  %85 = fmul float %81, %73
  %86 = fadd float %48, %82
  %87 = fadd float %55, %83
  %88 = fadd float %62, %84
  %89 = fadd float %69, %85
  %90 = getelementptr inbounds nuw float, ptr %24, i64 %46
  store float %86, ptr %90, align 4
  %91 = getelementptr inbounds nuw float, ptr %24, i64 %53
  store float %87, ptr %91, align 4
  %92 = getelementptr inbounds nuw float, ptr %24, i64 %60
  store float %88, ptr %92, align 4
  %93 = getelementptr inbounds nuw float, ptr %24, i64 %67
  store float %89, ptr %93, align 4
  %94 = add i64 %38, 4
  br label %37

95:                                               ; preds = %37
  %96 = add i64 %34, 1
  br label %33

97:                                               ; preds = %33
  ret void
}

define void @_mlir_ciface_packet(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %0, align 8
  %6 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 0
  %7 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 1
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 2
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 3, 0
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 3, 1
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 3, 2
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 4, 0
  %13 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 4, 1
  %14 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 4, 2
  %15 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8
  %16 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 0
  %17 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 1
  %18 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 2
  %19 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 3, 0
  %20 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 3, 1
  %21 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 3, 2
  %22 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 4, 0
  %23 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 4, 1
  %24 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 4, 2
  %25 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %26 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 0
  %27 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 1
  %28 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 2
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 3, 0
  %30 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 4, 0
  %31 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %3, align 8
  %32 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, 0
  %33 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, 1
  %34 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, 2
  %35 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, 3, 0
  %36 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, 3, 1
  %37 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, 3, 2
  %38 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, 4, 0
  %39 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, 4, 1
  %40 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, 4, 2
  call void @packet(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, ptr %26, ptr %27, i64 %28, i64 %29, i64 %30, ptr %32, ptr %33, i64 %34, i64 %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
