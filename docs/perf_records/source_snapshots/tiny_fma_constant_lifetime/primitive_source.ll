define void @source_probe0(ptr %a,ptr %b,ptr %o) {
  %ap0 = getelementptr float, ptr %a, i64 0
  %bp0 = getelementptr float, ptr %b, i64 0
  %a0 = load float, ptr %ap0
  %b0 = load float, ptr %bp0
  %ap1 = getelementptr float, ptr %a, i64 1
  %bp1 = getelementptr float, ptr %b, i64 1
  %a1 = load float, ptr %ap1
  %b1 = load float, ptr %bp1
  %ap2 = getelementptr float, ptr %a, i64 2
  %bp2 = getelementptr float, ptr %b, i64 2
  %a2 = load float, ptr %ap2
  %b2 = load float, ptr %bp2
  %ap3 = getelementptr float, ptr %a, i64 3
  %bp3 = getelementptr float, ptr %b, i64 3
  %a3 = load float, ptr %ap3
  %b3 = load float, ptr %bp3
  %c0 = bitcast i32 3207688728 to float
  %r0 = call float @llvm.fma.f32(float %a0, float %c0, float %b0)
  %r1 = call float @llvm.fma.f32(float %a1, float %c0, float %b1)
  %r2 = call float @llvm.fma.f32(float %a2, float %c0, float %b2)
  %r3 = call float @llvm.fma.f32(float %a3, float %c0, float %b3)
  %op0 = getelementptr float, ptr %o, i64 0
  store float %r0, ptr %op0
  %op1 = getelementptr float, ptr %o, i64 1
  store float %r1, ptr %op1
  %op2 = getelementptr float, ptr %o, i64 2
  store float %r2, ptr %op2
  %op3 = getelementptr float, ptr %o, i64 3
  store float %r3, ptr %op3
  ret void
}

define void @source_probe1(ptr %a,ptr %b,ptr %o) {
  %ap0 = getelementptr float, ptr %a, i64 0
  %bp0 = getelementptr float, ptr %b, i64 0
  %a0 = load float, ptr %ap0
  %b0 = load float, ptr %bp0
  %ap1 = getelementptr float, ptr %a, i64 1
  %bp1 = getelementptr float, ptr %b, i64 1
  %a1 = load float, ptr %ap1
  %b1 = load float, ptr %bp1
  %ap2 = getelementptr float, ptr %a, i64 2
  %bp2 = getelementptr float, ptr %b, i64 2
  %a2 = load float, ptr %ap2
  %b2 = load float, ptr %bp2
  %ap3 = getelementptr float, ptr %a, i64 3
  %bp3 = getelementptr float, ptr %b, i64 3
  %a3 = load float, ptr %ap3
  %b3 = load float, ptr %bp3
  %c0 = bitcast i32 822272776 to float
  %r0 = call float @llvm.fma.f32(float %a0, float %c0, float %b0)
  %r1 = call float @llvm.fma.f32(float %a1, float %c0, float %b1)
  %r2 = call float @llvm.fma.f32(float %a2, float %c0, float %b2)
  %r3 = call float @llvm.fma.f32(float %a3, float %c0, float %b3)
  %op0 = getelementptr float, ptr %o, i64 0
  store float %r0, ptr %op0
  %op1 = getelementptr float, ptr %o, i64 1
  store float %r1, ptr %op1
  %op2 = getelementptr float, ptr %o, i64 2
  store float %r2, ptr %op2
  %op3 = getelementptr float, ptr %o, i64 3
  store float %r3, ptr %op3
  ret void
}

define void @source_probe2(ptr %a,ptr %b,ptr %o) {
  %ap0 = getelementptr float, ptr %a, i64 0
  %bp0 = getelementptr float, ptr %b, i64 0
  %a0 = load float, ptr %ap0
  %b0 = load float, ptr %bp0
  %ap1 = getelementptr float, ptr %a, i64 1
  %bp1 = getelementptr float, ptr %b, i64 1
  %a1 = load float, ptr %ap1
  %b1 = load float, ptr %bp1
  %ap2 = getelementptr float, ptr %a, i64 2
  %bp2 = getelementptr float, ptr %b, i64 2
  %a2 = load float, ptr %ap2
  %b2 = load float, ptr %bp2
  %ap3 = getelementptr float, ptr %a, i64 3
  %bp3 = getelementptr float, ptr %b, i64 3
  %a3 = load float, ptr %ap3
  %b3 = load float, ptr %bp3
  %c0 = bitcast i32 985008993 to float
  %c1 = bitcast i32 1007192201 to float
  %r0 = call float @llvm.fma.f32(float %c0, float %a0, float %c1)
  %r1 = call float @llvm.fma.f32(float %c0, float %a1, float %c1)
  %r2 = call float @llvm.fma.f32(float %c0, float %a2, float %c1)
  %r3 = call float @llvm.fma.f32(float %c0, float %a3, float %c1)
  %op0 = getelementptr float, ptr %o, i64 0
  store float %r0, ptr %op0
  %op1 = getelementptr float, ptr %o, i64 1
  store float %r1, ptr %op1
  %op2 = getelementptr float, ptr %o, i64 2
  store float %r2, ptr %op2
  %op3 = getelementptr float, ptr %o, i64 3
  store float %r3, ptr %op3
  ret void
}

define void @source_probe3(ptr %a,ptr %b,ptr %o) {
  %ap0 = getelementptr float, ptr %a, i64 0
  %bp0 = getelementptr float, ptr %b, i64 0
  %a0 = load float, ptr %ap0
  %b0 = load float, ptr %bp0
  %ap1 = getelementptr float, ptr %a, i64 1
  %bp1 = getelementptr float, ptr %b, i64 1
  %a1 = load float, ptr %ap1
  %b1 = load float, ptr %bp1
  %ap2 = getelementptr float, ptr %a, i64 2
  %bp2 = getelementptr float, ptr %b, i64 2
  %a2 = load float, ptr %ap2
  %b2 = load float, ptr %bp2
  %ap3 = getelementptr float, ptr %a, i64 3
  %bp3 = getelementptr float, ptr %b, i64 3
  %a3 = load float, ptr %ap3
  %b3 = load float, ptr %bp3
  %c0 = bitcast i32 1026206379 to float
  %r0 = call float @llvm.fma.f32(float %a0, float %b0, float %c0)
  %r1 = call float @llvm.fma.f32(float %a1, float %b1, float %c0)
  %r2 = call float @llvm.fma.f32(float %a2, float %b2, float %c0)
  %r3 = call float @llvm.fma.f32(float %a3, float %b3, float %c0)
  %op0 = getelementptr float, ptr %o, i64 0
  store float %r0, ptr %op0
  %op1 = getelementptr float, ptr %o, i64 1
  store float %r1, ptr %op1
  %op2 = getelementptr float, ptr %o, i64 2
  store float %r2, ptr %op2
  %op3 = getelementptr float, ptr %o, i64 3
  store float %r3, ptr %op3
  ret void
}

define void @source_probe4(ptr %a,ptr %b,ptr %o) {
  %ap0 = getelementptr float, ptr %a, i64 0
  %bp0 = getelementptr float, ptr %b, i64 0
  %a0 = load float, ptr %ap0
  %b0 = load float, ptr %bp0
  %ap1 = getelementptr float, ptr %a, i64 1
  %bp1 = getelementptr float, ptr %b, i64 1
  %a1 = load float, ptr %ap1
  %b1 = load float, ptr %bp1
  %ap2 = getelementptr float, ptr %a, i64 2
  %bp2 = getelementptr float, ptr %b, i64 2
  %a2 = load float, ptr %ap2
  %b2 = load float, ptr %bp2
  %ap3 = getelementptr float, ptr %a, i64 3
  %bp3 = getelementptr float, ptr %b, i64 3
  %a3 = load float, ptr %ap3
  %b3 = load float, ptr %bp3
  %c0 = bitcast i32 1042983595 to float
  %r0 = call float @llvm.fma.f32(float %a0, float %b0, float %c0)
  %r1 = call float @llvm.fma.f32(float %a1, float %b1, float %c0)
  %r2 = call float @llvm.fma.f32(float %a2, float %b2, float %c0)
  %r3 = call float @llvm.fma.f32(float %a3, float %b3, float %c0)
  %op0 = getelementptr float, ptr %o, i64 0
  store float %r0, ptr %op0
  %op1 = getelementptr float, ptr %o, i64 1
  store float %r1, ptr %op1
  %op2 = getelementptr float, ptr %o, i64 2
  store float %r2, ptr %op2
  %op3 = getelementptr float, ptr %o, i64 3
  store float %r3, ptr %op3
  ret void
}

define void @source_probe5(ptr %a,ptr %b,ptr %o) {
  %ap0 = getelementptr float, ptr %a, i64 0
  %bp0 = getelementptr float, ptr %b, i64 0
  %a0 = load float, ptr %ap0
  %b0 = load float, ptr %bp0
  %ap1 = getelementptr float, ptr %a, i64 1
  %bp1 = getelementptr float, ptr %b, i64 1
  %a1 = load float, ptr %ap1
  %b1 = load float, ptr %bp1
  %ap2 = getelementptr float, ptr %a, i64 2
  %bp2 = getelementptr float, ptr %b, i64 2
  %a2 = load float, ptr %ap2
  %b2 = load float, ptr %bp2
  %ap3 = getelementptr float, ptr %a, i64 3
  %bp3 = getelementptr float, ptr %b, i64 3
  %a3 = load float, ptr %ap3
  %b3 = load float, ptr %bp3
  %c0 = bitcast i32 1056964608 to float
  %r0 = call float @llvm.fma.f32(float %a0, float %b0, float %c0)
  %r1 = call float @llvm.fma.f32(float %a1, float %b1, float %c0)
  %r2 = call float @llvm.fma.f32(float %a2, float %b2, float %c0)
  %r3 = call float @llvm.fma.f32(float %a3, float %b3, float %c0)
  %op0 = getelementptr float, ptr %o, i64 0
  store float %r0, ptr %op0
  %op1 = getelementptr float, ptr %o, i64 1
  store float %r1, ptr %op1
  %op2 = getelementptr float, ptr %o, i64 2
  store float %r2, ptr %op2
  %op3 = getelementptr float, ptr %o, i64 3
  store float %r3, ptr %op3
  ret void
}

define void @source_probe6(ptr %a,ptr %b,ptr %o) {
  %ap0 = getelementptr float, ptr %a, i64 0
  %bp0 = getelementptr float, ptr %b, i64 0
  %a0 = load float, ptr %ap0
  %b0 = load float, ptr %bp0
  %ap1 = getelementptr float, ptr %a, i64 1
  %bp1 = getelementptr float, ptr %b, i64 1
  %a1 = load float, ptr %ap1
  %b1 = load float, ptr %bp1
  %ap2 = getelementptr float, ptr %a, i64 2
  %bp2 = getelementptr float, ptr %b, i64 2
  %a2 = load float, ptr %ap2
  %b2 = load float, ptr %bp2
  %ap3 = getelementptr float, ptr %a, i64 3
  %bp3 = getelementptr float, ptr %b, i64 3
  %a3 = load float, ptr %ap3
  %b3 = load float, ptr %bp3
  %c0 = bitcast i32 1065353216 to float
  %r0 = call float @llvm.fma.f32(float %a0, float %b0, float %c0)
  %r1 = call float @llvm.fma.f32(float %a1, float %b1, float %c0)
  %r2 = call float @llvm.fma.f32(float %a2, float %b2, float %c0)
  %r3 = call float @llvm.fma.f32(float %a3, float %b3, float %c0)
  %op0 = getelementptr float, ptr %o, i64 0
  store float %r0, ptr %op0
  %op1 = getelementptr float, ptr %o, i64 1
  store float %r1, ptr %op1
  %op2 = getelementptr float, ptr %o, i64 2
  store float %r2, ptr %op2
  %op3 = getelementptr float, ptr %o, i64 3
  store float %r3, ptr %op3
  ret void
}

define void @source_probe7(ptr %a,ptr %b,ptr %o) {
  %ap0 = getelementptr float, ptr %a, i64 0
  %bp0 = getelementptr float, ptr %b, i64 0
  %a0 = load float, ptr %ap0
  %b0 = load float, ptr %bp0
  %ap1 = getelementptr float, ptr %a, i64 1
  %bp1 = getelementptr float, ptr %b, i64 1
  %a1 = load float, ptr %ap1
  %b1 = load float, ptr %bp1
  %ap2 = getelementptr float, ptr %a, i64 2
  %bp2 = getelementptr float, ptr %b, i64 2
  %a2 = load float, ptr %ap2
  %b2 = load float, ptr %bp2
  %ap3 = getelementptr float, ptr %a, i64 3
  %bp3 = getelementptr float, ptr %b, i64 3
  %a3 = load float, ptr %ap3
  %b3 = load float, ptr %bp3
  %c0 = bitcast i32 1065353216 to float
  %r0 = call float @llvm.fma.f32(float %a0, float %b0, float %c0)
  %r1 = call float @llvm.fma.f32(float %a1, float %b1, float %c0)
  %r2 = call float @llvm.fma.f32(float %a2, float %b2, float %c0)
  %r3 = call float @llvm.fma.f32(float %a3, float %b3, float %c0)
  %op0 = getelementptr float, ptr %o, i64 0
  store float %r0, ptr %op0
  %op1 = getelementptr float, ptr %o, i64 1
  store float %r1, ptr %op1
  %op2 = getelementptr float, ptr %o, i64 2
  store float %r2, ptr %op2
  %op3 = getelementptr float, ptr %o, i64 3
  store float %r3, ptr %op3
  ret void
}

declare float @llvm.fma.f32(float,float,float)
