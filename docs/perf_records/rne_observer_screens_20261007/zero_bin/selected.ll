define void @cells_M8_rne(ptr %s.0, ptr %s.1, ptr %s.2, ptr %s.3, ptr %s.4) #4 {
newFuncRoot:
  br label %s.5

s.5:                                                ; preds = %12, %newFuncRoot
  %s.6 = phi i64 [ %s.13, %s.12 ], [ 0, %newFuncRoot ]
  %s.7 = icmp slt i64 %s.6, 8
  br i1 %s.7, label %s.8, label %.exitStub

s.8:                                                ; preds = %5
  br label %s.9

s.9:                                                ; preds = %14, %8
  %s.10 = phi i64 [ %s.132, %s.14 ], [ 0, %s.8 ]
  %s.11 = icmp slt i64 %s.10, 5632
  br i1 %s.11, label %s.14, label %s.12

s.12:                                               ; preds = %9
  %s.13 = add i64 %s.6, 1
  br label %s.5

s.14:                                               ; preds = %9
  %s.15 = add i64 %s.10, 1
  %s.16 = mul nuw nsw i64 %s.6, 5632
  %s.17 = add nuw nsw i64 0, %s.16
  %s.18 = add nuw nsw i64 %s.17, %s.10
  %s.19 = getelementptr inbounds nuw i32, ptr %s.0, i64 %s.18
  %s.20 = load i32, ptr %s.19, align 4
  %s.21 = getelementptr inbounds nuw float, ptr %s.1, i64 %s.10
  %s.22 = load float, ptr %s.21, align 4
  %s.23 = getelementptr inbounds nuw i32, ptr %s.2, i64 %s.18
  %s.24 = load i32, ptr %s.23, align 4
  %s.25 = getelementptr inbounds nuw float, ptr %s.3, i64 %s.10
  %s.26 = load float, ptr %s.25, align 4
  %s.27 = add nuw nsw i64 %s.17, %s.15
  %s.28 = getelementptr inbounds nuw i32, ptr %s.0, i64 %s.27
  %s.29 = load i32, ptr %s.28, align 4
  %s.30 = getelementptr inbounds nuw float, ptr %s.1, i64 %s.15
  %s.31 = load float, ptr %s.30, align 4
  %s.32 = getelementptr inbounds nuw i32, ptr %s.2, i64 %s.27
  %s.33 = load i32, ptr %s.32, align 4
  %s.34 = getelementptr inbounds nuw float, ptr %s.3, i64 %s.15
  %s.35 = load float, ptr %s.34, align 4
  %s.36 = sitofp i32 %s.24 to float
  %s.37 = sitofp i32 %s.33 to float
  %s.38 = fmul float %s.36, f0x3C9A3F89
  %s.39 = fmul float %s.37, f0x3C9A3F89
  %s.40 = fmul float %s.38, %s.26
  %s.41 = fmul float %s.39, %s.35
  %s.42 = sitofp i32 %s.20 to float
  %s.43 = sitofp i32 %s.29 to float
  %s.44 = fmul float %s.42, f0x3C9A3F89
  %s.45 = fmul float %s.43, f0x3C9A3F89
  %s.46 = fmul float %s.44, %s.22
  %s.47 = fmul float %s.45, %s.31
  %s.48 = fneg float %s.46
  %s.49 = fneg float %s.47
  %s.50 = call float @llvm.maximum.f32(float %s.48, float -8.700000e+01)
  %s.51 = call float @llvm.maximum.f32(float %s.49, float -8.700000e+01)
  %s.52 = call float @llvm.minimum.f32(float %s.50, float 8.800000e+01)
  %s.53 = call float @llvm.minimum.f32(float %s.51, float 8.800000e+01)
  %s.54 = fmul float %s.52, f0x3FB8AA3B
  %s.55 = fmul float %s.53, f0x3FB8AA3B
  %s.56 = fadd float %s.54, f0x4B400000
  %s.57 = fadd float %s.55, f0x4B400000
  %s.58 = fsub float %s.56, f0x4B400000
  %s.59 = fsub float %s.57, f0x4B400000
  %s.60 = call float @llvm.fma.f32(float %s.58, float f0xBF317218, float %s.52)
  %s.61 = call float @llvm.fma.f32(float %s.59, float f0xBF317218, float %s.53)
  %s.62 = call float @llvm.fma.f32(float %s.58, float f0x3102E308, float %s.60)
  %s.63 = call float @llvm.fma.f32(float %s.59, float f0x3102E308, float %s.61)
  %s.64 = call float @llvm.fma.f32(float f0x3AB60B61, float %s.62, float f0x3C088889)
  %s.65 = call float @llvm.fma.f32(float f0x3AB60B61, float %s.63, float f0x3C088889)
  %s.66 = call float @llvm.fma.f32(float %s.64, float %s.62, float f0x3D2AAAAB)
  %s.67 = call float @llvm.fma.f32(float %s.65, float %s.63, float f0x3D2AAAAB)
  %s.68 = call float @llvm.fma.f32(float %s.66, float %s.62, float f0x3E2AAAAB)
  %s.69 = call float @llvm.fma.f32(float %s.67, float %s.63, float f0x3E2AAAAB)
  %s.70 = call float @llvm.fma.f32(float %s.68, float %s.62, float 5.000000e-01)
  %s.71 = call float @llvm.fma.f32(float %s.69, float %s.63, float 5.000000e-01)
  %s.72 = call float @llvm.fma.f32(float %s.70, float %s.62, float 1.000000e+00)
  %s.73 = call float @llvm.fma.f32(float %s.71, float %s.63, float 1.000000e+00)
  %s.74 = call float @llvm.fma.f32(float %s.72, float %s.62, float 1.000000e+00)
  %s.75 = call float @llvm.fma.f32(float %s.73, float %s.63, float 1.000000e+00)
  %s.76 = fptosi float %s.58 to i32
  %s.77 = fptosi float %s.59 to i32
  %s.78 = add i32 %s.76, 127
  %s.79 = add i32 %s.77, 127
  %s.80 = shl i32 %s.78, 23
  %s.81 = shl i32 %s.79, 23
  %s.82 = bitcast i32 %s.80 to float
  %s.83 = bitcast i32 %s.81 to float
  %s.84 = fmul float %s.74, %s.82
  %s.85 = fmul float %s.75, %s.83
  %s.86 = fadd float %s.84, 1.000000e+00
  %s.87 = fadd float %s.85, 1.000000e+00
  %s.88 = fdiv float 1.000000e+00, %s.86
  %s.89 = fdiv float 1.000000e+00, %s.87
  %s.90 = fmul float %s.46, %s.88
  %s.91 = fmul float %s.47, %s.89
  %s.92 = fmul float %s.90, %s.40
  %s.93 = fmul float %s.91, %s.41
  %s.94 = fmul float %s.92, f0x4392C87C
  %s.95 = fmul float %s.93, f0x4392C87C
  %s.96 = call float @llvm.maximum.f32(float %s.94, float -1.280000e+02)
  %s.97 = call float @llvm.maximum.f32(float %s.95, float -1.280000e+02)
  %s.98 = call float @llvm.minimum.f32(float %s.96, float 1.270000e+02)
  %s.99 = call float @llvm.minimum.f32(float %s.97, float 1.270000e+02)
  %s.100 = fptosi float %s.98 to i8
  %s.101 = fptosi float %s.99 to i8
  %s.102 = sitofp i8 %s.100 to float
  %s.103 = sitofp i8 %s.101 to float
  %s.104 = fsub float %s.98, %s.102
  %s.105 = fsub float %s.99, %s.103
  %s.106 = fneg float %s.104
  %s.107 = fneg float %s.105
  %s.108 = call float @llvm.maximum.f32(float %s.104, float %s.106)
  %s.109 = call float @llvm.maximum.f32(float %s.105, float %s.107)
  %s.110 = fcmp ogt float %s.108, 5.000000e-01
  %s.111 = fcmp ogt float %s.109, 5.000000e-01
  %s.112 = fcmp oeq float %s.108, 5.000000e-01
  %s.113 = fcmp oeq float %s.109, 5.000000e-01
  %s.114 = and i8 %s.100, 1
  %s.115 = and i8 %s.101, 1
  %s.116 = icmp ne i8 %s.114, 0
  %s.117 = icmp ne i8 %s.115, 0
  %s.118 = and i1 %s.112, %s.116
  %s.119 = and i1 %s.113, %s.117
  %s.120 = or i1 %s.110, %s.118
  %s.121 = or i1 %s.111, %s.119
  %s.122 = fcmp olt float %s.98, 0.000000e+00
  %s.123 = fcmp olt float %s.99, 0.000000e+00
  %s.124 = select i1 %s.122, i8 -1, i8 1
  %s.125 = select i1 %s.123, i8 -1, i8 1
  %s.126 = select i1 %s.120, i8 %s.124, i8 0
  %s.127 = select i1 %s.121, i8 %s.125, i8 0
  %s.128 = call i8 @cells_lookup_activation(float %s.46, float %s.40, float f0x4392C87C)
  %s.129 = call i8 @cells_lookup_activation(float %s.47, float %s.41, float f0x4392C87C)
  %s.130 = getelementptr inbounds nuw i8, ptr %s.4, i64 %s.18
  store i8 %s.128, ptr %s.130, align 1
  %s.131 = getelementptr inbounds nuw i8, ptr %s.4, i64 %s.27
  store i8 %s.129, ptr %s.131, align 1
  %s.132 = add i64 %s.10, 2
  br label %s.9

.exitStub:                                        ; preds = %5
  ret void
}
declare void @free(ptr)
declare void @memrefCopy(i64, ptr, ptr)
declare ptr @malloc(i64)
declare float @expf(float) #0
declare float @rsqrtf(float) #0
declare float @sinf(float) #0
declare float @cosf(float) #0
declare void @_mlir_ciface_merlin_dev_gemmini_0__borrowed_write(ptr, ptr, ptr)
declare void @_mlir_ciface_merlin_dev_gemmini_1__borrowed_write(ptr, ptr, ptr)
declare void @_mlir_ciface_merlin_dev_gemmini_2__borrowed_write(ptr, ptr, ptr)
declare void @_mlir_ciface_merlin_dev_gemmini_3__borrowed_write(ptr, ptr, ptr)
declare void @_mlir_ciface_merlin_dev_gemmini_4__borrowed_write(ptr, ptr, ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
declare ptr @llvm.stacksave.p0() #2
declare void @llvm.stackrestore.p0(ptr) #2
declare float @llvm.maximum.f32(float, float) #3
declare float @llvm.minimum.f32(float, float) #3
declare float @llvm.fma.f32(float, float, float) #3
attributes #0 = { memory(none) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { noinline }
declare float @lookup_b16(float,float,float)
declare float @lookup_b18(float,float,float)
declare float @lookup_b20(float,float,float)

declare i8 @cells_lookup_activation(float,float,float)
