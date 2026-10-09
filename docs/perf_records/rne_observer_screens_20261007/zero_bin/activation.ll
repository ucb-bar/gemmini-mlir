; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define float @cells_source_activation(float %0) noinline cold {
  %2 = fneg float %0
  %3 = call float @llvm.maximum.f32(float %2, float -8.700000e+01)
  %4 = call float @llvm.minimum.f32(float %3, float 8.800000e+01)
  %5 = fmul float %4, 0x3FF7154760000000
  %6 = fadd float %5, 0x4168000000000000
  %7 = fsub float %6, 0x4168000000000000
  %8 = call float @llvm.fma.f32(float %7, float 0xBFE62E4300000000, float %4)
  %9 = call float @llvm.fma.f32(float %7, float 0x3E205C6100000000, float %8)
  %10 = call float @llvm.fma.f32(float 0x3F56C16C20000000, float %9, float 0x3F81111120000000)
  %11 = call float @llvm.fma.f32(float %10, float %9, float 0x3FA5555560000000)
  %12 = call float @llvm.fma.f32(float %11, float %9, float 0x3FC5555560000000)
  %13 = call float @llvm.fma.f32(float %12, float %9, float 5.000000e-01)
  %14 = call float @llvm.fma.f32(float %13, float %9, float 1.000000e+00)
  %15 = call float @llvm.fma.f32(float %14, float %9, float 1.000000e+00)
  %16 = fptosi float %7 to i32
  %17 = add i32 %16, 127
  %18 = shl i32 %17, 23
  %19 = bitcast i32 %18 to float
  %20 = fmul float %15, %19
  %21 = fadd float %20, 1.000000e+00
  %22 = fdiv float 1.000000e+00, %21
  %23 = fmul float %0, %22
  ret float %23
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimum.f32(float, float) #0

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #0

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
