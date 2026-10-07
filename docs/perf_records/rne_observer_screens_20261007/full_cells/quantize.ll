; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i8 @cells_quantize(float %0) alwaysinline {
  %2 = call float @llvm.maximum.f32(float %0, float -1.280000e+02)
  %3 = call float @llvm.minimum.f32(float %2, float 1.270000e+02)
  %4 = fptosi float %3 to i8
  %5 = sitofp i8 %4 to float
  %6 = fsub float %3, %5
  %7 = fneg float %6
  %8 = call float @llvm.maximum.f32(float %6, float %7)
  %9 = fcmp ogt float %8, 5.000000e-01
  %10 = fcmp oeq float %8, 5.000000e-01
  %11 = and i8 %4, 1
  %12 = icmp ne i8 %11, 0
  %13 = and i1 %10, %12
  %14 = or i1 %9, %13
  %15 = fcmp olt float %3, 0.000000e+00
  %16 = select i1 %15, i8 -1, i8 1
  %17 = select i1 %14, i8 %16, i8 0
  %18 = add i8 %4, %17
  ret i8 %18
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimum.f32(float, float) #0

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
