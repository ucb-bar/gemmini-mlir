; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare ptr @malloc(i64)

define void @materialized(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, ptr %28, ptr %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36) {
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

53:                                               ; preds = %310, %37
  %54 = phi i64 [ %312, %310 ], [ 0, %37 ]
  %55 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %61, %310 ], [ %52, %37 ]
  %56 = phi i1 [ %311, %310 ], [ false, %37 ]
  %57 = icmp slt i64 %54, 1
  br i1 %57, label %58, label %313

58:                                               ; preds = %53
  br label %59

59:                                               ; preds = %307, %58
  %60 = phi i64 [ %309, %307 ], [ 0, %58 ]
  %61 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %67, %307 ], [ %55, %58 ]
  %62 = phi i1 [ %308, %307 ], [ false, %58 ]
  %63 = icmp slt i64 %60, 2
  br i1 %63, label %64, label %310

64:                                               ; preds = %59
  br label %65

65:                                               ; preds = %70, %64
  %66 = phi i64 [ %306, %70 ], [ 0, %64 ]
  %67 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %67, %70 ], [ %61, %64 ]
  %68 = phi i1 [ %68, %70 ], [ false, %64 ]
  %69 = icmp slt i64 %66, 5632
  br i1 %69, label %70, label %307

70:                                               ; preds = %65
  %71 = add i64 %66, 1
  %72 = add i64 %66, 2
  %73 = add i64 %66, 3
  %74 = mul nuw nsw i64 %54, 11264
  %75 = mul nuw nsw i64 %60, 5632
  %76 = add nuw nsw i64 %74, %75
  %77 = add nuw nsw i64 %76, %66
  %78 = getelementptr inbounds nuw i32, ptr %1, i64 %77
  %79 = load i32, ptr %78, align 4
  %80 = getelementptr inbounds nuw float, ptr %10, i64 %66
  %81 = load float, ptr %80, align 4
  %82 = getelementptr inbounds nuw i32, ptr %15, i64 %77
  %83 = load i32, ptr %82, align 4
  %84 = getelementptr inbounds nuw float, ptr %24, i64 %66
  %85 = load float, ptr %84, align 4
  %86 = add nuw nsw i64 %76, %71
  %87 = getelementptr inbounds nuw i32, ptr %1, i64 %86
  %88 = load i32, ptr %87, align 4
  %89 = getelementptr inbounds nuw float, ptr %10, i64 %71
  %90 = load float, ptr %89, align 4
  %91 = getelementptr inbounds nuw i32, ptr %15, i64 %86
  %92 = load i32, ptr %91, align 4
  %93 = getelementptr inbounds nuw float, ptr %24, i64 %71
  %94 = load float, ptr %93, align 4
  %95 = add nuw nsw i64 %76, %72
  %96 = getelementptr inbounds nuw i32, ptr %1, i64 %95
  %97 = load i32, ptr %96, align 4
  %98 = getelementptr inbounds nuw float, ptr %10, i64 %72
  %99 = load float, ptr %98, align 4
  %100 = getelementptr inbounds nuw i32, ptr %15, i64 %95
  %101 = load i32, ptr %100, align 4
  %102 = getelementptr inbounds nuw float, ptr %24, i64 %72
  %103 = load float, ptr %102, align 4
  %104 = add nuw nsw i64 %76, %73
  %105 = getelementptr inbounds nuw i32, ptr %1, i64 %104
  %106 = load i32, ptr %105, align 4
  %107 = getelementptr inbounds nuw float, ptr %10, i64 %73
  %108 = load float, ptr %107, align 4
  %109 = getelementptr inbounds nuw i32, ptr %15, i64 %104
  %110 = load i32, ptr %109, align 4
  %111 = getelementptr inbounds nuw float, ptr %24, i64 %73
  %112 = load float, ptr %111, align 4
  %113 = sitofp i32 %83 to float
  %114 = sitofp i32 %92 to float
  %115 = sitofp i32 %101 to float
  %116 = sitofp i32 %110 to float
  %117 = fmul float %113, 0x3F9347F120000000
  %118 = fmul float %114, 0x3F9347F120000000
  %119 = fmul float %115, 0x3F9347F120000000
  %120 = fmul float %116, 0x3F9347F120000000
  %121 = fmul float %117, %85
  %122 = fmul float %118, %94
  %123 = fmul float %119, %103
  %124 = fmul float %120, %112
  %125 = sitofp i32 %79 to float
  %126 = sitofp i32 %88 to float
  %127 = sitofp i32 %97 to float
  %128 = sitofp i32 %106 to float
  %129 = fmul float %125, 0x3F9347F120000000
  %130 = fmul float %126, 0x3F9347F120000000
  %131 = fmul float %127, 0x3F9347F120000000
  %132 = fmul float %128, 0x3F9347F120000000
  %133 = fmul float %129, %81
  %134 = fmul float %130, %90
  %135 = fmul float %131, %99
  %136 = fmul float %132, %108
  %137 = fneg float %133
  %138 = fneg float %134
  %139 = fneg float %135
  %140 = fneg float %136
  %141 = call float @llvm.maximum.f32(float %137, float -8.700000e+01)
  %142 = call float @llvm.maximum.f32(float %138, float -8.700000e+01)
  %143 = call float @llvm.maximum.f32(float %139, float -8.700000e+01)
  %144 = call float @llvm.maximum.f32(float %140, float -8.700000e+01)
  %145 = call float @llvm.minimum.f32(float %141, float 8.800000e+01)
  %146 = call float @llvm.minimum.f32(float %142, float 8.800000e+01)
  %147 = call float @llvm.minimum.f32(float %143, float 8.800000e+01)
  %148 = call float @llvm.minimum.f32(float %144, float 8.800000e+01)
  %149 = fmul float %145, 0x3FF7154760000000
  %150 = fmul float %146, 0x3FF7154760000000
  %151 = fmul float %147, 0x3FF7154760000000
  %152 = fmul float %148, 0x3FF7154760000000
  %153 = fadd float %149, 0x4168000000000000
  %154 = fadd float %150, 0x4168000000000000
  %155 = fadd float %151, 0x4168000000000000
  %156 = fadd float %152, 0x4168000000000000
  %157 = fsub float %153, 0x4168000000000000
  %158 = fsub float %154, 0x4168000000000000
  %159 = fsub float %155, 0x4168000000000000
  %160 = fsub float %156, 0x4168000000000000
  %161 = call float @llvm.fma.f32(float %157, float 0xBFE62E4300000000, float %145)
  %162 = call float @llvm.fma.f32(float %158, float 0xBFE62E4300000000, float %146)
  %163 = call float @llvm.fma.f32(float %159, float 0xBFE62E4300000000, float %147)
  %164 = call float @llvm.fma.f32(float %160, float 0xBFE62E4300000000, float %148)
  %165 = call float @llvm.fma.f32(float %157, float 0x3E205C6100000000, float %161)
  %166 = call float @llvm.fma.f32(float %158, float 0x3E205C6100000000, float %162)
  %167 = call float @llvm.fma.f32(float %159, float 0x3E205C6100000000, float %163)
  %168 = call float @llvm.fma.f32(float %160, float 0x3E205C6100000000, float %164)
  %169 = call float @llvm.fma.f32(float 0x3F56C16C20000000, float %165, float 0x3F81111120000000)
  %170 = call float @llvm.fma.f32(float 0x3F56C16C20000000, float %166, float 0x3F81111120000000)
  %171 = call float @llvm.fma.f32(float 0x3F56C16C20000000, float %167, float 0x3F81111120000000)
  %172 = call float @llvm.fma.f32(float 0x3F56C16C20000000, float %168, float 0x3F81111120000000)
  %constant.packet.0 = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %165, float %166, float %167, float %168, i32 1026206379, float %169, float %170, float %171, float %172)
  %173 = extractvalue { float, float, float, float } %constant.packet.0, 0
  %174 = extractvalue { float, float, float, float } %constant.packet.0, 1
  %175 = extractvalue { float, float, float, float } %constant.packet.0, 2
  %176 = extractvalue { float, float, float, float } %constant.packet.0, 3
  %constant.packet.1 = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %165, float %166, float %167, float %168, i32 1042983595, float %173, float %174, float %175, float %176)
  %177 = extractvalue { float, float, float, float } %constant.packet.1, 0
  %178 = extractvalue { float, float, float, float } %constant.packet.1, 1
  %179 = extractvalue { float, float, float, float } %constant.packet.1, 2
  %180 = extractvalue { float, float, float, float } %constant.packet.1, 3
  %constant.packet.2 = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %165, float %166, float %167, float %168, i32 1056964608, float %177, float %178, float %179, float %180)
  %181 = extractvalue { float, float, float, float } %constant.packet.2, 0
  %182 = extractvalue { float, float, float, float } %constant.packet.2, 1
  %183 = extractvalue { float, float, float, float } %constant.packet.2, 2
  %184 = extractvalue { float, float, float, float } %constant.packet.2, 3
  %constant.packet.3 = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %165, float %166, float %167, float %168, i32 1065353216, float %181, float %182, float %183, float %184)
  %185 = extractvalue { float, float, float, float } %constant.packet.3, 0
  %186 = extractvalue { float, float, float, float } %constant.packet.3, 1
  %187 = extractvalue { float, float, float, float } %constant.packet.3, 2
  %188 = extractvalue { float, float, float, float } %constant.packet.3, 3
  %constant.packet.4 = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %165, float %166, float %167, float %168, i32 1065353216, float %185, float %186, float %187, float %188)
  %189 = extractvalue { float, float, float, float } %constant.packet.4, 0
  %190 = extractvalue { float, float, float, float } %constant.packet.4, 1
  %191 = extractvalue { float, float, float, float } %constant.packet.4, 2
  %192 = extractvalue { float, float, float, float } %constant.packet.4, 3
  %193 = fptosi float %157 to i32
  %194 = fptosi float %158 to i32
  %195 = fptosi float %159 to i32
  %196 = fptosi float %160 to i32
  %197 = add i32 %193, 127
  %198 = add i32 %194, 127
  %199 = add i32 %195, 127
  %200 = add i32 %196, 127
  %201 = shl i32 %197, 23
  %202 = shl i32 %198, 23
  %203 = shl i32 %199, 23
  %204 = shl i32 %200, 23
  %205 = bitcast i32 %201 to float
  %206 = bitcast i32 %202 to float
  %207 = bitcast i32 %203 to float
  %208 = bitcast i32 %204 to float
  %209 = fmul float %189, %205
  %210 = fmul float %190, %206
  %211 = fmul float %191, %207
  %212 = fmul float %192, %208
  %213 = fadd float %209, 1.000000e+00
  %214 = fadd float %210, 1.000000e+00
  %215 = fadd float %211, 1.000000e+00
  %216 = fadd float %212, 1.000000e+00
  %217 = fdiv float 1.000000e+00, %213
  %218 = fdiv float 1.000000e+00, %214
  %219 = fdiv float 1.000000e+00, %215
  %220 = fdiv float 1.000000e+00, %216
  %221 = fmul float %133, %217
  %222 = fmul float %134, %218
  %223 = fmul float %135, %219
  %224 = fmul float %136, %220
  %225 = fmul float %221, %121
  %226 = fmul float %222, %122
  %227 = fmul float %223, %123
  %228 = fmul float %224, %124
  %229 = fmul float %225, 0x4072590F80000000
  %230 = fmul float %226, 0x4072590F80000000
  %231 = fmul float %227, 0x4072590F80000000
  %232 = fmul float %228, 0x4072590F80000000
  %233 = call float @llvm.maximum.f32(float %229, float -1.280000e+02)
  %234 = call float @llvm.maximum.f32(float %230, float -1.280000e+02)
  %235 = call float @llvm.maximum.f32(float %231, float -1.280000e+02)
  %236 = call float @llvm.maximum.f32(float %232, float -1.280000e+02)
  %237 = call float @llvm.minimum.f32(float %233, float 1.270000e+02)
  %238 = call float @llvm.minimum.f32(float %234, float 1.270000e+02)
  %239 = call float @llvm.minimum.f32(float %235, float 1.270000e+02)
  %240 = call float @llvm.minimum.f32(float %236, float 1.270000e+02)
  %241 = fptosi float %237 to i8
  %242 = fptosi float %238 to i8
  %243 = fptosi float %239 to i8
  %244 = fptosi float %240 to i8
  %245 = sitofp i8 %241 to float
  %246 = sitofp i8 %242 to float
  %247 = sitofp i8 %243 to float
  %248 = sitofp i8 %244 to float
  %249 = fsub float %237, %245
  %250 = fsub float %238, %246
  %251 = fsub float %239, %247
  %252 = fsub float %240, %248
  %253 = fneg float %249
  %254 = fneg float %250
  %255 = fneg float %251
  %256 = fneg float %252
  %257 = call float @llvm.maximum.f32(float %249, float %253)
  %258 = call float @llvm.maximum.f32(float %250, float %254)
  %259 = call float @llvm.maximum.f32(float %251, float %255)
  %260 = call float @llvm.maximum.f32(float %252, float %256)
  %261 = fcmp ogt float %257, 5.000000e-01
  %262 = fcmp ogt float %258, 5.000000e-01
  %263 = fcmp ogt float %259, 5.000000e-01
  %264 = fcmp ogt float %260, 5.000000e-01
  %265 = fcmp oeq float %257, 5.000000e-01
  %266 = fcmp oeq float %258, 5.000000e-01
  %267 = fcmp oeq float %259, 5.000000e-01
  %268 = fcmp oeq float %260, 5.000000e-01
  %269 = and i8 %241, 1
  %270 = and i8 %242, 1
  %271 = and i8 %243, 1
  %272 = and i8 %244, 1
  %273 = icmp ne i8 %269, 0
  %274 = icmp ne i8 %270, 0
  %275 = icmp ne i8 %271, 0
  %276 = icmp ne i8 %272, 0
  %277 = and i1 %265, %273
  %278 = and i1 %266, %274
  %279 = and i1 %267, %275
  %280 = and i1 %268, %276
  %281 = or i1 %261, %277
  %282 = or i1 %262, %278
  %283 = or i1 %263, %279
  %284 = or i1 %264, %280
  %285 = fcmp olt float %237, 0.000000e+00
  %286 = fcmp olt float %238, 0.000000e+00
  %287 = fcmp olt float %239, 0.000000e+00
  %288 = fcmp olt float %240, 0.000000e+00
  %289 = select i1 %285, i8 -1, i8 1
  %290 = select i1 %286, i8 -1, i8 1
  %291 = select i1 %287, i8 -1, i8 1
  %292 = select i1 %288, i8 -1, i8 1
  %293 = select i1 %281, i8 %289, i8 0
  %294 = select i1 %282, i8 %290, i8 0
  %295 = select i1 %283, i8 %291, i8 0
  %296 = select i1 %284, i8 %292, i8 0
  %merlin.rne.0 = call i32 asm "fmax.s ft0, $1, $2\0Afmin.s ft0, ft0, $3\0Afcvt.w.s $0, ft0, rne", "=r,f,f,f,~{ft0}"(float %229, float -1.280000e+02, float 1.270000e+02)
  %297 = trunc i32 %merlin.rne.0 to i8
  %merlin.rne.1 = call i32 asm "fmax.s ft0, $1, $2\0Afmin.s ft0, ft0, $3\0Afcvt.w.s $0, ft0, rne", "=r,f,f,f,~{ft0}"(float %230, float -1.280000e+02, float 1.270000e+02)
  %298 = trunc i32 %merlin.rne.1 to i8
  %merlin.rne.2 = call i32 asm "fmax.s ft0, $1, $2\0Afmin.s ft0, ft0, $3\0Afcvt.w.s $0, ft0, rne", "=r,f,f,f,~{ft0}"(float %231, float -1.280000e+02, float 1.270000e+02)
  %299 = trunc i32 %merlin.rne.2 to i8
  %merlin.rne.3 = call i32 asm "fmax.s ft0, $1, $2\0Afmin.s ft0, ft0, $3\0Afcvt.w.s $0, ft0, rne", "=r,f,f,f,~{ft0}"(float %232, float -1.280000e+02, float 1.270000e+02)
  %300 = trunc i32 %merlin.rne.3 to i8
  %301 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %67, 1
  %302 = getelementptr inbounds nuw i8, ptr %301, i64 %77
  store i8 %297, ptr %302, align 1
  %303 = getelementptr inbounds nuw i8, ptr %301, i64 %86
  store i8 %298, ptr %303, align 1
  %304 = getelementptr inbounds nuw i8, ptr %301, i64 %95
  store i8 %299, ptr %304, align 1
  %305 = getelementptr inbounds nuw i8, ptr %301, i64 %104
  store i8 %300, ptr %305, align 1
  %306 = add i64 %66, 4
  br label %65

307:                                              ; preds = %65
  %308 = or i1 %68, %62
  %309 = add i64 %60, 1
  br label %59

310:                                              ; preds = %59
  %311 = or i1 %62, %56
  %312 = add i64 %54, 1
  br label %53

313:                                              ; preds = %53
  %314 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 3, 0
  %315 = mul i64 %314, 1
  %316 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 3, 1
  %317 = mul i64 %315, %316
  %318 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 3, 2
  %319 = mul i64 %317, %318
  %320 = mul i64 %319, 1
  %321 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 1
  %322 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 2
  %323 = getelementptr i8, ptr %321, i64 %322
  %324 = getelementptr i8, ptr %29, i64 %30
  call void @llvm.memcpy.p0.p0.i64(ptr %324, ptr %323, i64 %320, i1 false)
  %325 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, 0
  %326 = call ptr @malloc(i64 16)
  %327 = call ptr @malloc(i64 2)
  %328 = call ptr @malloc(i64 0)
  %329 = ptrtoint ptr %43 to i64
  store i64 %329, ptr %326, align 4
  %330 = ptrtoint ptr %321 to i64
  %331 = getelementptr inbounds nuw i64, ptr %326, i32 1
  store i64 %330, ptr %331, align 4
  store i1 true, ptr %327, align 1
  %332 = getelementptr inbounds nuw i1, ptr %327, i32 1
  store i1 %56, ptr %332, align 1
  %333 = call ptr @malloc(i64 2)
  %334 = call ptr @malloc(i64 0)
  call void @materialized_dealloc_helper(ptr %326, ptr %326, i64 0, i64 2, i64 1, ptr %328, ptr %328, i64 0, i64 0, i64 1, ptr %327, ptr %327, i64 0, i64 2, i64 1, ptr %333, ptr %333, i64 0, i64 2, i64 1, ptr %334, ptr %334, i64 0, i64 0, i64 1)
  %335 = load i1, ptr %333, align 1
  br i1 %335, label %336, label %337

336:                                              ; preds = %313
  call void @free(ptr %38)
  br label %337

337:                                              ; preds = %336, %313
  %338 = getelementptr inbounds nuw i1, ptr %333, i32 1
  %339 = load i1, ptr %338, align 1
  br i1 %339, label %340, label %341

340:                                              ; preds = %337
  call void @free(ptr %325)
  br label %341

341:                                              ; preds = %340, %337
  call void @free(ptr %326)
  call void @free(ptr %328)
  call void @free(ptr %327)
  call void @free(ptr %333)
  call void @free(ptr %334)
  ret void
}

define void @_mlir_ciface_materialized(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) {
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
  call void @materialized(ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, ptr %33, ptr %34, i64 %35, i64 %36, i64 %37, ptr %39, ptr %40, i64 %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46, i64 %47)
  ret void
}

define void @materialized_dealloc_helper(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24) {
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
