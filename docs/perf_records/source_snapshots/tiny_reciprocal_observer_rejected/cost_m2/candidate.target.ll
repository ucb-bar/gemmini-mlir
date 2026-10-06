; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define void @candidate(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, ptr %28, ptr %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36) {
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
  br label %53

53:                                               ; preds = %194, %37
  %54 = phi i64 [ %196, %194 ], [ 0, %37 ]
  %55 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %61, %194 ], [ %52, %37 ]
  %56 = phi i1 [ %195, %194 ], [ false, %37 ]
  %57 = icmp slt i64 %54, 1
  br i1 %57, label %58, label %197

58:                                               ; preds = %53
  br label %59

59:                                               ; preds = %191, %58
  %60 = phi i64 [ %193, %191 ], [ 0, %58 ]
  %61 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %67, %191 ], [ %55, %58 ]
  %62 = phi i1 [ %192, %191 ], [ false, %58 ]
  %63 = icmp slt i64 %60, 2
  br i1 %63, label %64, label %194

64:                                               ; preds = %59
  br label %65

65:                                               ; preds = %70, %64
  %66 = phi i64 [ %190, %70 ], [ 0, %64 ]
  %67 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %67, %70 ], [ %61, %64 ]
  %68 = phi i1 [ %68, %70 ], [ false, %64 ]
  %69 = icmp slt i64 %66, 5632
  br i1 %69, label %70, label %191

70:                                               ; preds = %65
  %71 = add i64 %66, 1
  %72 = mul nuw nsw i64 %54, 11264
  %73 = mul nuw nsw i64 %60, 5632
  %74 = add nuw nsw i64 %72, %73
  %75 = add nuw nsw i64 %74, %66
  %76 = getelementptr inbounds nuw i32, ptr %1, i64 %75
  %77 = load i32, ptr %76, align 4
  %78 = getelementptr inbounds nuw float, ptr %10, i64 %66
  %79 = load float, ptr %78, align 4
  %80 = getelementptr inbounds nuw i32, ptr %15, i64 %75
  %81 = load i32, ptr %80, align 4
  %82 = getelementptr inbounds nuw float, ptr %24, i64 %66
  %83 = load float, ptr %82, align 4
  %84 = add nuw nsw i64 %74, %71
  %85 = getelementptr inbounds nuw i32, ptr %1, i64 %84
  %86 = load i32, ptr %85, align 4
  %87 = getelementptr inbounds nuw float, ptr %10, i64 %71
  %88 = load float, ptr %87, align 4
  %89 = getelementptr inbounds nuw i32, ptr %15, i64 %84
  %90 = load i32, ptr %89, align 4
  %91 = getelementptr inbounds nuw float, ptr %24, i64 %71
  %92 = load float, ptr %91, align 4
  %93 = sitofp i32 %81 to float
  %94 = sitofp i32 %90 to float
  %95 = fmul float %93, f0x3C9A3F89
  %96 = fmul float %94, f0x3C9A3F89
  %97 = fmul float %95, %83
  %98 = fmul float %96, %92
  %99 = sitofp i32 %77 to float
  %100 = sitofp i32 %86 to float
  %101 = fmul float %99, f0x3C9A3F89
  %102 = fmul float %100, f0x3C9A3F89
  %103 = fmul float %101, %79
  %104 = fmul float %102, %88
  %105 = fneg float %103
  %106 = fneg float %104
  %107 = call float @llvm.maximum.f32(float %105, float -8.700000e+01)
  %108 = call float @llvm.maximum.f32(float %106, float -8.700000e+01)
  %109 = call float @llvm.minimum.f32(float %107, float 8.800000e+01)
  %110 = call float @llvm.minimum.f32(float %108, float 8.800000e+01)
  %111 = fmul float %109, f0x3FB8AA3B
  %112 = fmul float %110, f0x3FB8AA3B
  %113 = fadd float %111, f0x4B400000
  %114 = fadd float %112, f0x4B400000
  %115 = fsub float %113, f0x4B400000
  %116 = fsub float %114, f0x4B400000
  %117 = call float @llvm.fma.f32(float %115, float f0xBF317218, float %109)
  %118 = call float @llvm.fma.f32(float %116, float f0xBF317218, float %110)
  %119 = call float @llvm.fma.f32(float %115, float f0x3102E308, float %117)
  %120 = call float @llvm.fma.f32(float %116, float f0x3102E308, float %118)
  %121 = call float @llvm.fma.f32(float f0x3AB60B61, float %119, float f0x3C088889)
  %122 = call float @llvm.fma.f32(float f0x3AB60B61, float %120, float f0x3C088889)
  %123 = call float @llvm.fma.f32(float %121, float %119, float f0x3D2AAAAB)
  %124 = call float @llvm.fma.f32(float %122, float %120, float f0x3D2AAAAB)
  %125 = call float @llvm.fma.f32(float %123, float %119, float f0x3E2AAAAB)
  %126 = call float @llvm.fma.f32(float %124, float %120, float f0x3E2AAAAB)
  %127 = call float @llvm.fma.f32(float %125, float %119, float 5.000000e-01)
  %128 = call float @llvm.fma.f32(float %126, float %120, float 5.000000e-01)
  %129 = call float @llvm.fma.f32(float %127, float %119, float 1.000000e+00)
  %130 = call float @llvm.fma.f32(float %128, float %120, float 1.000000e+00)
  %131 = call float @llvm.fma.f32(float %129, float %119, float 1.000000e+00)
  %132 = call float @llvm.fma.f32(float %130, float %120, float 1.000000e+00)
  %133 = fptosi float %115 to i32
  %134 = fptosi float %116 to i32
  %135 = add i32 %133, 127
  %136 = add i32 %134, 127
  %137 = shl i32 %135, 23
  %138 = shl i32 %136, 23
  %139 = bitcast i32 %137 to float
  %140 = bitcast i32 %138 to float
  %141 = fmul float %131, %139
  %142 = fmul float %132, %140
  %143 = fadd float %141, 1.000000e+00
  %144 = fadd float %142, 1.000000e+00
  %145 = fdiv float 1.000000e+00, %143
  %146 = fdiv float 1.000000e+00, %144
  %147 = fmul float %103, %145
  %148 = fmul float %104, %146
  %149 = fmul float %147, %97
  %150 = fmul float %148, %98
  %151 = call float @exported_observer(float %143, float %103, float %97, float f0x4392C87C)
  %152 = call float @exported_observer(float %144, float %104, float %98, float f0x4392C87C)
  %153 = call float @llvm.maximum.f32(float %151, float -1.280000e+02)
  %154 = call float @llvm.maximum.f32(float %152, float -1.280000e+02)
  %155 = call float @llvm.minimum.f32(float %153, float 1.270000e+02)
  %156 = call float @llvm.minimum.f32(float %154, float 1.270000e+02)
  %157 = fptosi float %155 to i8
  %158 = fptosi float %156 to i8
  %159 = sitofp i8 %157 to float
  %160 = sitofp i8 %158 to float
  %161 = fsub float %155, %159
  %162 = fsub float %156, %160
  %163 = fneg float %161
  %164 = fneg float %162
  %165 = call float @llvm.maximum.f32(float %161, float %163)
  %166 = call float @llvm.maximum.f32(float %162, float %164)
  %167 = fcmp ogt float %165, 5.000000e-01
  %168 = fcmp ogt float %166, 5.000000e-01
  %169 = fcmp oeq float %165, 5.000000e-01
  %170 = fcmp oeq float %166, 5.000000e-01
  %171 = and i8 %157, 1
  %172 = and i8 %158, 1
  %173 = icmp ne i8 %171, 0
  %174 = icmp ne i8 %172, 0
  %175 = and i1 %169, %173
  %176 = and i1 %170, %174
  %177 = or i1 %167, %175
  %178 = or i1 %168, %176
  %179 = fcmp olt float %155, 0.000000e+00
  %180 = fcmp olt float %156, 0.000000e+00
  %181 = select i1 %179, i8 -1, i8 1
  %182 = select i1 %180, i8 -1, i8 1
  %183 = select i1 %177, i8 %181, i8 0
  %184 = select i1 %178, i8 %182, i8 0
  %merlin.rne.0 = call i32 asm "fmax.s ft0, $1, $2\0Afmin.s ft0, ft0, $3\0Afcvt.w.s $0, ft0, rne", "=r,f,f,f,~{ft0}"(float %151, float -1.280000e+02, float 1.270000e+02)
  %185 = trunc i32 %merlin.rne.0 to i8
  %merlin.rne.1 = call i32 asm "fmax.s ft0, $1, $2\0Afmin.s ft0, ft0, $3\0Afcvt.w.s $0, ft0, rne", "=r,f,f,f,~{ft0}"(float %152, float -1.280000e+02, float 1.270000e+02)
  %186 = trunc i32 %merlin.rne.1 to i8
  %187 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %67, 1
  %188 = getelementptr inbounds nuw i8, ptr %187, i64 %75
  store i8 %185, ptr %188, align 1
  %189 = getelementptr inbounds nuw i8, ptr %187, i64 %84
  store i8 %186, ptr %189, align 1
  %190 = add i64 %66, 2
  br label %65

191:                                              ; preds = %65
  %192 = or i1 %68, %62
  %193 = add i64 %60, 1
  br label %59

194:                                              ; preds = %59
  %195 = or i1 %62, %56
  %196 = add i64 %54, 1
  br label %53

197:                                              ; preds = %53
  %198 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 3, 0
  %199 = mul i64 %198, 1
  %200 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 3, 1
  %201 = mul i64 %199, %200
  %202 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 3, 2
  %203 = mul i64 %201, %202
  %204 = mul i64 %203, 1
  %205 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 1
  %206 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 2
  %207 = getelementptr i8, ptr %205, i64 %206
  %208 = getelementptr i8, ptr %29, i64 %30
  call void @llvm.memcpy.p0.p0.i64(ptr %208, ptr %207, i64 %204, i1 false)
  %209 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 0
  %210 = call ptr @malloc(i64 16)
  %211 = call ptr @malloc(i64 2)
  %212 = call ptr @malloc(i64 0)
  %213 = ptrtoint ptr %43 to i64
  store i64 %213, ptr %210, align 4
  %214 = ptrtoint ptr %205 to i64
  %215 = getelementptr inbounds nuw i64, ptr %210, i32 1
  store i64 %214, ptr %215, align 4
  store i1 true, ptr %211, align 1
  %216 = getelementptr inbounds nuw i1, ptr %211, i32 1
  store i1 %56, ptr %216, align 1
  %217 = call ptr @malloc(i64 2)
  %218 = call ptr @malloc(i64 0)
  call void @candidate_dealloc_helper(ptr %210, ptr %210, i64 0, i64 2, i64 1, ptr %212, ptr %212, i64 0, i64 0, i64 1, ptr %211, ptr %211, i64 0, i64 2, i64 1, ptr %217, ptr %217, i64 0, i64 2, i64 1, ptr %218, ptr %218, i64 0, i64 0, i64 1)
  %219 = load i1, ptr %217, align 1
  br i1 %219, label %220, label %221

220:                                              ; preds = %197
  call void @free(ptr %38)
  br label %221

221:                                              ; preds = %220, %197
  %222 = getelementptr inbounds nuw i1, ptr %217, i32 1
  %223 = load i1, ptr %222, align 1
  br i1 %223, label %224, label %225

224:                                              ; preds = %221
  call void @free(ptr %209)
  br label %225

225:                                              ; preds = %224, %221
  call void @free(ptr %210)
  call void @free(ptr %212)
  call void @free(ptr %211)
  call void @free(ptr %217)
  call void @free(ptr %218)
  ret void
}

declare ptr @malloc(i64)

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimum.f32(float, float) #0

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

define void @candidate_dealloc_helper(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24) {
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
  br label %42

42:                                               ; preds = %45, %25
  %43 = phi i64 [ %47, %45 ], [ 0, %25 ]
  %44 = icmp slt i64 %43, %41
  br i1 %44, label %45, label %48

45:                                               ; preds = %42
  %46 = getelementptr inbounds nuw i1, ptr %21, i64 %43
  store i1 false, ptr %46, align 1
  %47 = add i64 %43, 1
  br label %42

48:                                               ; preds = %42
  br label %49

49:                                               ; preds = %84, %48
  %50 = phi i64 [ %87, %84 ], [ 0, %48 ]
  %51 = icmp slt i64 %50, %37
  br i1 %51, label %52, label %88

52:                                               ; preds = %49
  %53 = getelementptr inbounds nuw i64, ptr %1, i64 %50
  %54 = load i64, ptr %53, align 4
  %55 = getelementptr inbounds nuw i1, ptr %11, i64 %50
  %56 = load i1, ptr %55, align 1
  br label %57

57:                                               ; preds = %69, %52
  %58 = phi i64 [ %72, %69 ], [ 0, %52 ]
  %59 = phi i1 [ %71, %69 ], [ true, %52 ]
  %60 = icmp slt i64 %58, %41
  br i1 %60, label %61, label %73

61:                                               ; preds = %57
  %62 = getelementptr inbounds nuw i64, ptr %6, i64 %58
  %63 = load i64, ptr %62, align 4
  %64 = icmp eq i64 %63, %54
  br i1 %64, label %65, label %69

65:                                               ; preds = %61
  %66 = getelementptr inbounds nuw i1, ptr %21, i64 %58
  %67 = load i1, ptr %66, align 1
  %68 = or i1 %67, %56
  store i1 %68, ptr %66, align 1
  br label %69

69:                                               ; preds = %65, %61
  %70 = icmp ne i64 %63, %54
  %71 = and i1 %59, %70
  %72 = add i64 %58, 1
  br label %57

73:                                               ; preds = %57
  br label %74

74:                                               ; preds = %78, %73
  %75 = phi i64 [ %83, %78 ], [ 0, %73 ]
  %76 = phi i1 [ %82, %78 ], [ %59, %73 ]
  %77 = icmp slt i64 %75, %50
  br i1 %77, label %78, label %84

78:                                               ; preds = %74
  %79 = getelementptr inbounds nuw i64, ptr %1, i64 %75
  %80 = load i64, ptr %79, align 4
  %81 = icmp ne i64 %80, %54
  %82 = and i1 %76, %81
  %83 = add i64 %75, 1
  br label %74

84:                                               ; preds = %74
  %85 = and i1 %76, %56
  %86 = getelementptr inbounds nuw i1, ptr %16, i64 %50
  store i1 %85, ptr %86, align 1
  %87 = add i64 %50, 1
  br label %49

88:                                               ; preds = %49
  ret void
}

declare void @free(ptr)

define void @_mlir_ciface_candidate(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) {
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
  call void @candidate(ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, ptr %33, ptr %34, i64 %35, i64 %36, i64 %37, ptr %39, ptr %40, i64 %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46, i64 %47)
  ret void
}

; Function Attrs: alwaysinline nounwind
define dso_local float @exported_observer(float noundef %d, float noundef %a, float noundef %b, float noundef %c) #2 {
entry:
  %0 = bitcast float %d to i32
  %1 = add i32 %0, -2097152000
  %or.cond.i = icmp ult i32 %1, -2071986176
  br i1 %or.cond.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %entry
  %div.i = fdiv float 1.000000e+00, %d
  %mul.i1 = fmul float %a, %div.i
  %mul1.i = fmul float %mul.i1, %b
  %mul2.i = fmul float %mul1.i, %c
  br label %reciprocal_observer.exit

if.end.i:                                         ; preds = %entry
  %sub.i = sub nuw nsw i32 2129859011, %0
  %2 = bitcast i32 %sub.i to float
  %fneg.i = fneg float %d
  %call4.i = tail call float @fmaf(float noundef %fneg.i, float noundef %2, float noundef 2.000000e+00) #4
  %mul.i = fmul float %call4.i, %2
  %call6.i = tail call float @fmaf(float noundef %fneg.i, float noundef %mul.i, float noundef 2.000000e+00) #4
  %mul7.i = fmul float %mul.i, %call6.i
  %mul8.i = fmul float %a, %mul7.i
  %mul9.i = fmul float %b, %mul8.i
  %mul10.i = fmul float %c, %mul9.i
  %3 = bitcast float %mul8.i to i32
  %shr.i = lshr i32 %3, 23
  %and.i2 = and i32 %shr.i, 255
  %4 = add nsw i32 %and.i2, -253
  %5 = icmp ult i32 %4, -251
  br i1 %5, label %if.then18.i, label %lor.lhs.false12.i

lor.lhs.false12.i:                                ; preds = %if.end.i
  %6 = bitcast float %mul9.i to i32
  %shr.i3 = lshr i32 %6, 23
  %and.i4 = and i32 %shr.i3, 255
  %7 = add nsw i32 %and.i4, -253
  %8 = icmp ult i32 %7, -251
  br i1 %8, label %if.then18.i, label %lor.lhs.false15.i

lor.lhs.false15.i:                                ; preds = %lor.lhs.false12.i
  %9 = bitcast float %mul10.i to i32
  %shr.i6 = lshr i32 %9, 23
  %and.i7 = and i32 %shr.i6, 255
  %10 = add nsw i32 %and.i7, -253
  %11 = icmp ult i32 %10, -251
  br i1 %11, label %if.then18.i, label %if.end20.i

if.then18.i:                                      ; preds = %lor.lhs.false15.i, %lor.lhs.false12.i, %if.end.i
  %div.i9 = fdiv float 1.000000e+00, %d
  %mul.i10 = fmul float %a, %div.i9
  %mul1.i11 = fmul float %mul.i10, %b
  %mul2.i12 = fmul float %mul1.i11, %c
  br label %reciprocal_observer.exit

if.end20.i:                                       ; preds = %lor.lhs.false15.i
  %12 = tail call float @llvm.fabs.f32(float %mul10.i)
  %cmp23.i = fcmp ult float %12, 2.560000e+02
  br i1 %cmp23.i, label %if.end25.i, label %reciprocal_observer.exit

if.end25.i:                                       ; preds = %if.end20.i
  %add.i = fadd float %12, f0x4B000000
  %sub26.i = fadd float %add.i, f0xCB000000
  %sub27.i = fsub float %12, %sub26.i
  %13 = tail call float @llvm.fabs.f32(float %sub27.i)
  %cmp28.i = fcmp olt float %13, f0x3EFC0000
  br i1 %cmp28.i, label %reciprocal_observer.exit, label %if.end30.i

if.end30.i:                                       ; preds = %if.end25.i
  %div.i13 = fdiv float 1.000000e+00, %d
  %mul.i14 = fmul float %a, %div.i13
  %mul1.i15 = fmul float %mul.i14, %b
  %mul2.i16 = fmul float %mul1.i15, %c
  br label %reciprocal_observer.exit

reciprocal_observer.exit:                         ; preds = %if.end30.i, %if.end25.i, %if.end20.i, %if.then18.i, %if.then.i
  %retval.3.i = phi float [ %mul2.i, %if.then.i ], [ %mul2.i12, %if.then18.i ], [ %mul10.i, %if.end20.i ], [ %mul2.i16, %if.end30.i ], [ %mul10.i, %if.end25.i ]
  ret float %retval.3.i
}

; Function Attrs: nounwind
declare dso_local float @fmaf(float noundef, float noundef, float noundef) local_unnamed_addr #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #0

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfwbdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
attributes #3 = { nounwind "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfwbdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
attributes #4 = { nobuiltin nounwind "no-builtins" }

!llvm.module.flags = !{!0, !1, !2, !4, !5}
!llvm.ident = !{!6}
!llvm.errno.tbaa = !{!7}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"target-abi", !"lp64d"}
!2 = !{i32 6, !"riscv-isa", !3}
!3 = !{!"rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0"}
!4 = !{i32 1, !"Code Model", i32 3}
!5 = !{i32 8, !"SmallDataLimit", i32 0}
!6 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git a47bddccec30255619bb8c37fa59700e661d4e66)"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
