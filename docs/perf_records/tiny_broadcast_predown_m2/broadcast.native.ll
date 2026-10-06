; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare ptr @malloc(i64)

define void @broadcast(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, ptr %28, ptr %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36) {
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

53:                                               ; preds = %192, %37
  %54 = phi i64 [ %194, %192 ], [ 0, %37 ]
  %55 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %61, %192 ], [ %52, %37 ]
  %56 = phi i1 [ %193, %192 ], [ false, %37 ]
  %57 = icmp slt i64 %54, 2
  br i1 %57, label %58, label %195

58:                                               ; preds = %53
  br label %59

59:                                               ; preds = %189, %58
  %60 = phi i64 [ %191, %189 ], [ 0, %58 ]
  %61 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %67, %189 ], [ %55, %58 ]
  %62 = phi i1 [ %190, %189 ], [ false, %58 ]
  %63 = icmp slt i64 %60, 1
  br i1 %63, label %64, label %192

64:                                               ; preds = %59
  br label %65

65:                                               ; preds = %70, %64
  %66 = phi i64 [ %188, %70 ], [ 0, %64 ]
  %67 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %67, %70 ], [ %61, %64 ]
  %68 = phi i1 [ %68, %70 ], [ false, %64 ]
  %69 = icmp slt i64 %66, 5632
  br i1 %69, label %70, label %189

70:                                               ; preds = %65
  %71 = add i64 %54, 1
  %72 = mul nuw nsw i64 %60, 11264
  %73 = mul nuw nsw i64 %54, 5632
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
  %84 = mul nuw nsw i64 %71, 5632
  %85 = add nuw nsw i64 %72, %84
  %86 = add nuw nsw i64 %85, %66
  %87 = getelementptr inbounds nuw i32, ptr %1, i64 %86
  %88 = load i32, ptr %87, align 4
  %89 = getelementptr inbounds nuw i32, ptr %15, i64 %86
  %90 = load i32, ptr %89, align 4
  %91 = sitofp i32 %81 to float
  %92 = sitofp i32 %90 to float
  %93 = fmul float %91, 0x3F9347F120000000
  %94 = fmul float %92, 0x3F9347F120000000
  %95 = fmul float %93, %83
  %96 = fmul float %94, %83
  %97 = sitofp i32 %77 to float
  %98 = sitofp i32 %88 to float
  %99 = fmul float %97, 0x3F9347F120000000
  %100 = fmul float %98, 0x3F9347F120000000
  %101 = fmul float %99, %79
  %102 = fmul float %100, %79
  %103 = fneg float %101
  %104 = fneg float %102
  %105 = call float @llvm.maximum.f32(float %103, float -8.700000e+01)
  %106 = call float @llvm.maximum.f32(float %104, float -8.700000e+01)
  %107 = call float @llvm.minimum.f32(float %105, float 8.800000e+01)
  %108 = call float @llvm.minimum.f32(float %106, float 8.800000e+01)
  %109 = fmul float %107, 0x3FF7154760000000
  %110 = fmul float %108, 0x3FF7154760000000
  %111 = fadd float %109, 0x4168000000000000
  %112 = fadd float %110, 0x4168000000000000
  %113 = fsub float %111, 0x4168000000000000
  %114 = fsub float %112, 0x4168000000000000
  %115 = call float @llvm.fma.f32(float %113, float 0xBFE62E4300000000, float %107)
  %116 = call float @llvm.fma.f32(float %114, float 0xBFE62E4300000000, float %108)
  %117 = call float @llvm.fma.f32(float %113, float 0x3E205C6100000000, float %115)
  %118 = call float @llvm.fma.f32(float %114, float 0x3E205C6100000000, float %116)
  %119 = call float @llvm.fma.f32(float 0x3F56C16C20000000, float %117, float 0x3F81111120000000)
  %120 = call float @llvm.fma.f32(float 0x3F56C16C20000000, float %118, float 0x3F81111120000000)
  %121 = call float @llvm.fma.f32(float %119, float %117, float 0x3FA5555560000000)
  %122 = call float @llvm.fma.f32(float %120, float %118, float 0x3FA5555560000000)
  %123 = call float @llvm.fma.f32(float %121, float %117, float 0x3FC5555560000000)
  %124 = call float @llvm.fma.f32(float %122, float %118, float 0x3FC5555560000000)
  %125 = call float @llvm.fma.f32(float %123, float %117, float 5.000000e-01)
  %126 = call float @llvm.fma.f32(float %124, float %118, float 5.000000e-01)
  %127 = call float @llvm.fma.f32(float %125, float %117, float 1.000000e+00)
  %128 = call float @llvm.fma.f32(float %126, float %118, float 1.000000e+00)
  %129 = call float @llvm.fma.f32(float %127, float %117, float 1.000000e+00)
  %130 = call float @llvm.fma.f32(float %128, float %118, float 1.000000e+00)
  %131 = fptosi float %113 to i32
  %132 = fptosi float %114 to i32
  %133 = add i32 %131, 127
  %134 = add i32 %132, 127
  %135 = shl i32 %133, 23
  %136 = shl i32 %134, 23
  %137 = bitcast i32 %135 to float
  %138 = bitcast i32 %136 to float
  %139 = fmul float %129, %137
  %140 = fmul float %130, %138
  %141 = fadd float %139, 1.000000e+00
  %142 = fadd float %140, 1.000000e+00
  %143 = fdiv float 1.000000e+00, %141
  %144 = fdiv float 1.000000e+00, %142
  %145 = fmul float %101, %143
  %146 = fmul float %102, %144
  %147 = fmul float %145, %95
  %148 = fmul float %146, %96
  %149 = fmul float %147, 0x4072590F80000000
  %150 = fmul float %148, 0x4072590F80000000
  %151 = call float @llvm.maximum.f32(float %149, float -1.280000e+02)
  %152 = call float @llvm.maximum.f32(float %150, float -1.280000e+02)
  %153 = call float @llvm.minimum.f32(float %151, float 1.270000e+02)
  %154 = call float @llvm.minimum.f32(float %152, float 1.270000e+02)
  %155 = fptosi float %153 to i8
  %156 = fptosi float %154 to i8
  %157 = sitofp i8 %155 to float
  %158 = sitofp i8 %156 to float
  %159 = fsub float %153, %157
  %160 = fsub float %154, %158
  %161 = fneg float %159
  %162 = fneg float %160
  %163 = call float @llvm.maximum.f32(float %159, float %161)
  %164 = call float @llvm.maximum.f32(float %160, float %162)
  %165 = fcmp ogt float %163, 5.000000e-01
  %166 = fcmp ogt float %164, 5.000000e-01
  %167 = fcmp oeq float %163, 5.000000e-01
  %168 = fcmp oeq float %164, 5.000000e-01
  %169 = and i8 %155, 1
  %170 = and i8 %156, 1
  %171 = icmp ne i8 %169, 0
  %172 = icmp ne i8 %170, 0
  %173 = and i1 %167, %171
  %174 = and i1 %168, %172
  %175 = or i1 %165, %173
  %176 = or i1 %166, %174
  %177 = fcmp olt float %153, 0.000000e+00
  %178 = fcmp olt float %154, 0.000000e+00
  %179 = select i1 %177, i8 -1, i8 1
  %180 = select i1 %178, i8 -1, i8 1
  %181 = select i1 %175, i8 %179, i8 0
  %182 = select i1 %176, i8 %180, i8 0
  %merlin.rne.0 = call float @llvm.roundeven.f32(float %153)
  %183 = fptosi float %merlin.rne.0 to i8
  %merlin.rne.1 = call float @llvm.roundeven.f32(float %154)
  %184 = fptosi float %merlin.rne.1 to i8
  %185 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %67, 1
  %186 = getelementptr inbounds nuw i8, ptr %185, i64 %75
  store i8 %183, ptr %186, align 1
  %187 = getelementptr inbounds nuw i8, ptr %185, i64 %86
  store i8 %184, ptr %187, align 1
  %188 = add i64 %66, 1
  br label %65

189:                                              ; preds = %65
  %190 = or i1 %68, %62
  %191 = add i64 %60, 1
  br label %59

192:                                              ; preds = %59
  %193 = or i1 %62, %56
  %194 = add i64 %54, 2
  br label %53

195:                                              ; preds = %53
  %196 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 3, 0
  %197 = mul i64 %196, 1
  %198 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 3, 1
  %199 = mul i64 %197, %198
  %200 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 3, 2
  %201 = mul i64 %199, %200
  %202 = mul i64 %201, 1
  %203 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 1
  %204 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 2
  %205 = getelementptr i8, ptr %203, i64 %204
  %206 = getelementptr i8, ptr %29, i64 %30
  call void @llvm.memcpy.p0.p0.i64(ptr %206, ptr %205, i64 %202, i1 false)
  %207 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 0
  %208 = call ptr @malloc(i64 16)
  %209 = call ptr @malloc(i64 2)
  %210 = call ptr @malloc(i64 0)
  %211 = ptrtoint ptr %43 to i64
  store i64 %211, ptr %208, align 4
  %212 = ptrtoint ptr %203 to i64
  %213 = getelementptr inbounds nuw i64, ptr %208, i32 1
  store i64 %212, ptr %213, align 4
  store i1 true, ptr %209, align 1
  %214 = getelementptr inbounds nuw i1, ptr %209, i32 1
  store i1 %56, ptr %214, align 1
  %215 = call ptr @malloc(i64 2)
  %216 = call ptr @malloc(i64 0)
  call void @broadcast_dealloc_helper(ptr %208, ptr %208, i64 0, i64 2, i64 1, ptr %210, ptr %210, i64 0, i64 0, i64 1, ptr %209, ptr %209, i64 0, i64 2, i64 1, ptr %215, ptr %215, i64 0, i64 2, i64 1, ptr %216, ptr %216, i64 0, i64 0, i64 1)
  %217 = load i1, ptr %215, align 1
  br i1 %217, label %218, label %219

218:                                              ; preds = %195
  call void @free(ptr %38)
  br label %219

219:                                              ; preds = %218, %195
  %220 = getelementptr inbounds nuw i1, ptr %215, i32 1
  %221 = load i1, ptr %220, align 1
  br i1 %221, label %222, label %223

222:                                              ; preds = %219
  call void @free(ptr %207)
  br label %223

223:                                              ; preds = %222, %219
  call void @free(ptr %208)
  call void @free(ptr %210)
  call void @free(ptr %209)
  call void @free(ptr %215)
  call void @free(ptr %216)
  ret void
}

define void @_mlir_ciface_broadcast(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) {
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
  call void @broadcast(ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, ptr %33, ptr %34, i64 %35, i64 %36, i64 %37, ptr %39, ptr %40, i64 %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46, i64 %47)
  ret void
}

define void @broadcast_dealloc_helper(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24) {
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

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimum.f32(float, float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

declare float @llvm.roundeven.f32(float)
