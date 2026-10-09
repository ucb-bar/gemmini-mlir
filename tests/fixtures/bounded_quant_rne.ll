define i8 @quant(float %x) {
  %v1 = call float @llvm.maximum.f32(float %x, float -1.280000e+02)
  %v2 = call float @llvm.minimum.f32(float %v1, float 1.270000e+02)
  %v3 = fptosi float %v2 to i8
  %v4 = sitofp i8 %v3 to float
  %v5 = fsub float %v2, %v4
  %v6 = fneg float %v5
  %v7 = call float @llvm.maximum.f32(float %v5, float %v6)
  %v8 = fcmp ogt float %v7, 5.000000e-01
  %v9 = fcmp oeq float %v7, 5.000000e-01
  %v10 = and i8 %v3, 1
  %v11 = icmp ne i8 %v10, 0
  %v12 = and i1 %v9, %v11
  %v13 = or i1 %v8, %v12
  %v14 = fcmp olt float %v2, 0.000000e+00
  %v15 = select i1 %v14, i8 -1, i8 1
  %v16 = select i1 %v13, i8 %v15, i8 0
  %v17 = add i8 %v3, %v16
  ret i8 %v17
}
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
