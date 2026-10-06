; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @control(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31) {
  br label %33

33:                                               ; preds = %63, %32
  %34 = phi i64 [ %64, %63 ], [ 0, %32 ]
  %35 = icmp slt i64 %34, 1
  br i1 %35, label %36, label %65

36:                                               ; preds = %33
  br label %37

37:                                               ; preds = %61, %36
  %38 = phi i64 [ %62, %61 ], [ 0, %36 ]
  %39 = icmp slt i64 %38, 8
  br i1 %39, label %40, label %63

40:                                               ; preds = %37
  br label %41

41:                                               ; preds = %44, %40
  %42 = phi i64 [ %60, %44 ], [ 0, %40 ]
  %43 = icmp slt i64 %42, 2048
  br i1 %43, label %44, label %61

44:                                               ; preds = %41
  %45 = mul nuw nsw i64 %34, 16384
  %46 = mul nuw nsw i64 %38, 2048
  %47 = add nuw nsw i64 %45, %46
  %48 = add nuw nsw i64 %47, %42
  %49 = getelementptr inbounds nuw float, ptr %1, i64 %48
  %50 = load float, ptr %49, align 4
  %51 = getelementptr inbounds nuw i32, ptr %10, i64 %48
  %52 = load i32, ptr %51, align 4
  %53 = getelementptr inbounds nuw float, ptr %19, i64 %42
  %54 = load float, ptr %53, align 4
  %55 = sitofp i32 %52 to float
  %56 = fmul float %55, 0x3F46745BE0000000
  %57 = fmul float %56, %54
  %58 = fadd float %50, %57
  %59 = getelementptr inbounds nuw float, ptr %24, i64 %48
  store float %58, ptr %59, align 4
  %60 = add i64 %42, 1
  br label %41

61:                                               ; preds = %41
  %62 = add i64 %38, 1
  br label %37

63:                                               ; preds = %37
  %64 = add i64 %34, 1
  br label %33

65:                                               ; preds = %33
  ret void
}

define void @_mlir_ciface_control(ptr %0, ptr %1, ptr %2, ptr %3) {
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
  call void @control(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, ptr %26, ptr %27, i64 %28, i64 %29, i64 %30, ptr %32, ptr %33, i64 %34, i64 %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
