; ModuleID = '/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny-pointwise-outline-20261006/outlined.native.premerge.ll'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare ptr @malloc(i64)

define void @outlined(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, ptr %28, ptr %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36) {
  %.loc1 = alloca i1, align 1
  %.loc = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, align 8
  %38 = call ptr @malloc(i64 11328)
  %39 = ptrtoint ptr %38 to i64
  %40 = add i64 %39, 63
  %41 = urem i64 %40, 64
  %42 = sub i64 %40, %41
  %43 = inttoptr i64 %42 to ptr
  %44 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %38, 0
  %45 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %44, ptr %43, 1
  %46 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %45, i64 0, 2
  %47 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %46, i64 1, 3, 0
  %48 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %47, i64 2, 3, 1
  %49 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %48, i64 5632, 3, 2
  %50 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, i64 11264, 4, 0
  %51 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %50, i64 5632, 4, 1
  %52 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %51, i64 1, 4, 2
  br label %codeRepl

codeRepl:                                         ; preds = %37
  call void @llvm.lifetime.start.p0(ptr %.loc)
  call void @llvm.lifetime.start.p0(ptr %.loc1)
  call void @outlined.extracted({ ptr, ptr, i64, [3 x i64], [3 x i64] } %52, ptr %1, ptr %10, ptr %15, ptr %24, ptr %.loc, ptr %.loc1)
  %.reload = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %.loc, align 8
  %.reload2 = load i1, ptr %.loc1, align 1
  call void @llvm.lifetime.end.p0(ptr %.loc)
  call void @llvm.lifetime.end.p0(ptr %.loc1)
  br label %53

53:                                               ; preds = %codeRepl
  %54 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.reload, 3, 0
  %55 = mul i64 %54, 1
  %56 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.reload, 3, 1
  %57 = mul i64 %55, %56
  %58 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.reload, 3, 2
  %59 = mul i64 %57, %58
  %60 = mul i64 %59, 1
  %61 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.reload, 1
  %62 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.reload, 2
  %63 = getelementptr i8, ptr %61, i64 %62
  %64 = getelementptr i8, ptr %29, i64 %30
  call void @llvm.memcpy.p0.p0.i64(ptr %64, ptr %63, i64 %60, i1 false)
  %65 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.reload, 0
  %66 = call ptr @malloc(i64 16)
  %67 = call ptr @malloc(i64 2)
  %68 = call ptr @malloc(i64 0)
  %69 = ptrtoint ptr %43 to i64
  store i64 %69, ptr %66, align 4
  %70 = ptrtoint ptr %61 to i64
  %71 = getelementptr inbounds nuw i64, ptr %66, i32 1
  store i64 %70, ptr %71, align 4
  store i1 true, ptr %67, align 1
  %72 = getelementptr inbounds nuw i1, ptr %67, i32 1
  store i1 %.reload2, ptr %72, align 1
  %73 = call ptr @malloc(i64 2)
  %74 = call ptr @malloc(i64 0)
  call void @outlined_dealloc_helper(ptr %66, ptr %66, i64 0, i64 2, i64 1, ptr %68, ptr %68, i64 0, i64 0, i64 1, ptr %67, ptr %67, i64 0, i64 2, i64 1, ptr %73, ptr %73, i64 0, i64 2, i64 1, ptr %74, ptr %74, i64 0, i64 0, i64 1)
  %75 = load i1, ptr %73, align 1
  br i1 %75, label %76, label %77

76:                                               ; preds = %53
  call void @free(ptr %38)
  br label %77

77:                                               ; preds = %76, %53
  %78 = getelementptr inbounds nuw i1, ptr %73, i32 1
  %79 = load i1, ptr %78, align 1
  br i1 %79, label %80, label %81

80:                                               ; preds = %77
  call void @free(ptr %65)
  br label %81

81:                                               ; preds = %80, %77
  call void @free(ptr %66)
  call void @free(ptr %68)
  call void @free(ptr %67)
  call void @free(ptr %73)
  call void @free(ptr %74)
  ret void
}

define void @_mlir_ciface_outlined(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) {
  %6 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %0, align 8
  %7 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, 0
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, 1
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, 2
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, 3, 0
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, 3, 1
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, 3, 2
  %13 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, 4, 0
  %14 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, 4, 1
  %15 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, 4, 2
  %16 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 0
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 1
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 2
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 3, 0
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 4, 0
  %22 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %2, align 8
  %23 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 0
  %24 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 1
  %25 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 2
  %26 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 3, 0
  %27 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 3, 1
  %28 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 3, 2
  %29 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 4, 0
  %30 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 4, 1
  %31 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 4, 2
  %32 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %33 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 0
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 1
  %35 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 2
  %36 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 3, 0
  %37 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 4, 0
  %38 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %4, align 8
  %39 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, 0
  %40 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, 1
  %41 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, 2
  %42 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, 3, 0
  %43 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, 3, 1
  %44 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, 3, 2
  %45 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, 4, 0
  %46 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, 4, 1
  %47 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, 4, 2
  call void @outlined(ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, ptr %33, ptr %34, i64 %35, i64 %36, i64 %37, ptr %39, ptr %40, i64 %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46, i64 %47)
  ret void
}

define void @outlined_dealloc_helper(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24) {
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, ptr %6, 1
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %7, 2
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 %8, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, ptr %1, 1
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, i64 %2, 2
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %3, 3, 0
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3
  %35 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %34, ptr %35, align 4
  %36 = getelementptr [1 x i64], ptr %35, i32 0, i32 0
  %37 = load i64, ptr %36, align 4
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 3
  %39 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %38, ptr %39, align 4
  %40 = getelementptr [1 x i64], ptr %39, i32 0, i32 0
  %41 = load i64, ptr %40, align 4
  br label %codeRepl1

codeRepl1:                                        ; preds = %25
  call void @outlined_dealloc_helper.extracted.1(i64 %41, ptr %21)
  br label %42

42:                                               ; preds = %codeRepl1
  br label %codeRepl

codeRepl:                                         ; preds = %42
  call void @outlined_dealloc_helper.extracted(i64 %37, ptr %1, ptr %11, i64 %41, ptr %16, ptr %6, ptr %21)
  br label %43

43:                                               ; preds = %codeRepl
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimum.f32(float, float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.roundeven.f32(float) #1

; Function Attrs: noinline
define internal void @outlined.extracted({ ptr, ptr, i64, [3 x i64], [3 x i64] } %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %.out, ptr %.out1) #2 {
newFuncRoot:
  br label %5

5:                                                ; preds = %16, %newFuncRoot
  %6 = phi i64 [ %18, %16 ], [ 0, %newFuncRoot ]
  %7 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %13, %16 ], [ %0, %newFuncRoot ]
  %8 = phi i1 [ %17, %16 ], [ false, %newFuncRoot ]
  store i1 %8, ptr %.out1, align 1
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, ptr %.out, align 8
  %9 = icmp slt i64 %6, 1
  br i1 %9, label %10, label %.exitStub

10:                                               ; preds = %5
  br label %11

11:                                               ; preds = %25, %10
  %12 = phi i64 [ %27, %25 ], [ 0, %10 ]
  %13 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %22, %25 ], [ %7, %10 ]
  %14 = phi i1 [ %26, %25 ], [ false, %10 ]
  %15 = icmp slt i64 %12, 2
  br i1 %15, label %19, label %16

16:                                               ; preds = %11
  %17 = or i1 %14, %8
  %18 = add i64 %6, 1
  br label %5

19:                                               ; preds = %11
  br label %20

20:                                               ; preds = %28, %19
  %21 = phi i64 [ %148, %28 ], [ 0, %19 ]
  %22 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %22, %28 ], [ %13, %19 ]
  %23 = phi i1 [ %23, %28 ], [ false, %19 ]
  %24 = icmp slt i64 %21, 5632
  br i1 %24, label %28, label %25

25:                                               ; preds = %20
  %26 = or i1 %23, %14
  %27 = add i64 %12, 1
  br label %11

28:                                               ; preds = %20
  %29 = add i64 %21, 1
  %30 = mul nuw nsw i64 %6, 11264
  %31 = mul nuw nsw i64 %12, 5632
  %32 = add nuw nsw i64 %30, %31
  %33 = add nuw nsw i64 %32, %21
  %34 = getelementptr inbounds nuw i32, ptr %1, i64 %33
  %35 = load i32, ptr %34, align 4
  %36 = getelementptr inbounds nuw float, ptr %2, i64 %21
  %37 = load float, ptr %36, align 4
  %38 = getelementptr inbounds nuw i32, ptr %3, i64 %33
  %39 = load i32, ptr %38, align 4
  %40 = getelementptr inbounds nuw float, ptr %4, i64 %21
  %41 = load float, ptr %40, align 4
  %42 = add nuw nsw i64 %32, %29
  %43 = getelementptr inbounds nuw i32, ptr %1, i64 %42
  %44 = load i32, ptr %43, align 4
  %45 = getelementptr inbounds nuw float, ptr %2, i64 %29
  %46 = load float, ptr %45, align 4
  %47 = getelementptr inbounds nuw i32, ptr %3, i64 %42
  %48 = load i32, ptr %47, align 4
  %49 = getelementptr inbounds nuw float, ptr %4, i64 %29
  %50 = load float, ptr %49, align 4
  %51 = sitofp i32 %39 to float
  %52 = sitofp i32 %48 to float
  %53 = fmul float %51, f0x3C9A3F89
  %54 = fmul float %52, f0x3C9A3F89
  %55 = fmul float %53, %41
  %56 = fmul float %54, %50
  %57 = sitofp i32 %35 to float
  %58 = sitofp i32 %44 to float
  %59 = fmul float %57, f0x3C9A3F89
  %60 = fmul float %58, f0x3C9A3F89
  %61 = fmul float %59, %37
  %62 = fmul float %60, %46
  %63 = fneg float %61
  %64 = fneg float %62
  %65 = call float @llvm.maximum.f32(float %63, float -8.700000e+01)
  %66 = call float @llvm.maximum.f32(float %64, float -8.700000e+01)
  %67 = call float @llvm.minimum.f32(float %65, float 8.800000e+01)
  %68 = call float @llvm.minimum.f32(float %66, float 8.800000e+01)
  %69 = fmul float %67, f0x3FB8AA3B
  %70 = fmul float %68, f0x3FB8AA3B
  %71 = fadd float %69, f0x4B400000
  %72 = fadd float %70, f0x4B400000
  %73 = fsub float %71, f0x4B400000
  %74 = fsub float %72, f0x4B400000
  %75 = call float @llvm.fma.f32(float %73, float f0xBF317218, float %67)
  %76 = call float @llvm.fma.f32(float %74, float f0xBF317218, float %68)
  %77 = call float @llvm.fma.f32(float %73, float f0x3102E308, float %75)
  %78 = call float @llvm.fma.f32(float %74, float f0x3102E308, float %76)
  %79 = call float @llvm.fma.f32(float f0x3AB60B61, float %77, float f0x3C088889)
  %80 = call float @llvm.fma.f32(float f0x3AB60B61, float %78, float f0x3C088889)
  %81 = call float @llvm.fma.f32(float %79, float %77, float f0x3D2AAAAB)
  %82 = call float @llvm.fma.f32(float %80, float %78, float f0x3D2AAAAB)
  %83 = call float @llvm.fma.f32(float %81, float %77, float f0x3E2AAAAB)
  %84 = call float @llvm.fma.f32(float %82, float %78, float f0x3E2AAAAB)
  %85 = call float @llvm.fma.f32(float %83, float %77, float 5.000000e-01)
  %86 = call float @llvm.fma.f32(float %84, float %78, float 5.000000e-01)
  %87 = call float @llvm.fma.f32(float %85, float %77, float 1.000000e+00)
  %88 = call float @llvm.fma.f32(float %86, float %78, float 1.000000e+00)
  %89 = call float @llvm.fma.f32(float %87, float %77, float 1.000000e+00)
  %90 = call float @llvm.fma.f32(float %88, float %78, float 1.000000e+00)
  %91 = fptosi float %73 to i32
  %92 = fptosi float %74 to i32
  %93 = add i32 %91, 127
  %94 = add i32 %92, 127
  %95 = shl i32 %93, 23
  %96 = shl i32 %94, 23
  %97 = bitcast i32 %95 to float
  %98 = bitcast i32 %96 to float
  %99 = fmul float %89, %97
  %100 = fmul float %90, %98
  %101 = fadd float %99, 1.000000e+00
  %102 = fadd float %100, 1.000000e+00
  %103 = fdiv float 1.000000e+00, %101
  %104 = fdiv float 1.000000e+00, %102
  %105 = fmul float %61, %103
  %106 = fmul float %62, %104
  %107 = fmul float %105, %55
  %108 = fmul float %106, %56
  %109 = fmul float %107, f0x4392C87C
  %110 = fmul float %108, f0x4392C87C
  %111 = call float @llvm.maximum.f32(float %109, float -1.280000e+02)
  %112 = call float @llvm.maximum.f32(float %110, float -1.280000e+02)
  %113 = call float @llvm.minimum.f32(float %111, float 1.270000e+02)
  %114 = call float @llvm.minimum.f32(float %112, float 1.270000e+02)
  %115 = fptosi float %113 to i8
  %116 = fptosi float %114 to i8
  %117 = sitofp i8 %115 to float
  %118 = sitofp i8 %116 to float
  %119 = fsub float %113, %117
  %120 = fsub float %114, %118
  %121 = fneg float %119
  %122 = fneg float %120
  %123 = call float @llvm.maximum.f32(float %119, float %121)
  %124 = call float @llvm.maximum.f32(float %120, float %122)
  %125 = fcmp ogt float %123, 5.000000e-01
  %126 = fcmp ogt float %124, 5.000000e-01
  %127 = fcmp oeq float %123, 5.000000e-01
  %128 = fcmp oeq float %124, 5.000000e-01
  %129 = and i8 %115, 1
  %130 = and i8 %116, 1
  %131 = icmp ne i8 %129, 0
  %132 = icmp ne i8 %130, 0
  %133 = and i1 %127, %131
  %134 = and i1 %128, %132
  %135 = or i1 %125, %133
  %136 = or i1 %126, %134
  %137 = fcmp olt float %113, 0.000000e+00
  %138 = fcmp olt float %114, 0.000000e+00
  %139 = select i1 %137, i8 -1, i8 1
  %140 = select i1 %138, i8 -1, i8 1
  %141 = select i1 %135, i8 %139, i8 0
  %142 = select i1 %136, i8 %140, i8 0
  %merlin.rne.0 = call float @llvm.roundeven.f32(float %113)
  %143 = fptosi float %merlin.rne.0 to i8
  %merlin.rne.1 = call float @llvm.roundeven.f32(float %114)
  %144 = fptosi float %merlin.rne.1 to i8
  %145 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, 1
  %146 = getelementptr inbounds nuw i8, ptr %145, i64 %33
  store i8 %143, ptr %146, align 1
  %147 = getelementptr inbounds nuw i8, ptr %145, i64 %42
  store i8 %144, ptr %147, align 1
  %148 = add i64 %21, 2
  br label %20

.exitStub:                                        ; preds = %5
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #0

; Function Attrs: noinline
define internal void @outlined_dealloc_helper.extracted(i64 %0, ptr %1, ptr %2, i64 %3, ptr %4, ptr %5, ptr %6) #2 {
newFuncRoot:
  br label %7

7:                                                ; preds = %24, %newFuncRoot
  %8 = phi i64 [ %27, %24 ], [ 0, %newFuncRoot ]
  %9 = icmp slt i64 %8, %0
  br i1 %9, label %10, label %.exitStub

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw i64, ptr %1, i64 %8
  %12 = load i64, ptr %11, align 4
  %13 = getelementptr inbounds nuw i1, ptr %2, i64 %8
  %14 = load i1, ptr %13, align 1
  br label %15

15:                                               ; preds = %42, %10
  %16 = phi i64 [ %45, %42 ], [ 0, %10 ]
  %17 = phi i1 [ %44, %42 ], [ true, %10 ]
  %18 = icmp slt i64 %16, %3
  br i1 %18, label %34, label %19

19:                                               ; preds = %15
  br label %20

20:                                               ; preds = %28, %19
  %21 = phi i64 [ %33, %28 ], [ 0, %19 ]
  %22 = phi i1 [ %32, %28 ], [ %17, %19 ]
  %23 = icmp slt i64 %21, %8
  br i1 %23, label %28, label %24

24:                                               ; preds = %20
  %25 = and i1 %22, %14
  %26 = getelementptr inbounds nuw i1, ptr %4, i64 %8
  store i1 %25, ptr %26, align 1
  %27 = add i64 %8, 1
  br label %7

28:                                               ; preds = %20
  %29 = getelementptr inbounds nuw i64, ptr %1, i64 %21
  %30 = load i64, ptr %29, align 4
  %31 = icmp ne i64 %30, %12
  %32 = and i1 %22, %31
  %33 = add i64 %21, 1
  br label %20

34:                                               ; preds = %15
  %35 = getelementptr inbounds nuw i64, ptr %5, i64 %16
  %36 = load i64, ptr %35, align 4
  %37 = icmp eq i64 %36, %12
  br i1 %37, label %38, label %42

38:                                               ; preds = %34
  %39 = getelementptr inbounds nuw i1, ptr %6, i64 %16
  %40 = load i1, ptr %39, align 1
  %41 = or i1 %40, %14
  store i1 %41, ptr %39, align 1
  br label %42

42:                                               ; preds = %38, %34
  %43 = icmp ne i64 %36, %12
  %44 = and i1 %17, %43
  %45 = add i64 %16, 1
  br label %15

.exitStub:                                        ; preds = %7
  ret void
}

; Function Attrs: noinline
define internal void @outlined_dealloc_helper.extracted.1(i64 %0, ptr %1) #2 {
newFuncRoot:
  br label %2

2:                                                ; preds = %5, %newFuncRoot
  %3 = phi i64 [ %7, %5 ], [ 0, %newFuncRoot ]
  %4 = icmp slt i64 %3, %0
  br i1 %4, label %5, label %.exitStub

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i1, ptr %1, i64 %3
  store i1 false, ptr %6, align 1
  %7 = add i64 %3, 1
  br label %2

.exitStub:                                        ; preds = %2
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { noinline }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
