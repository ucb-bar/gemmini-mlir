define void @source_case0(ptr %a,ptr %o) {
entry:
  %p0 = getelementptr float,ptr %a,i64 0
  %a0 = load float,ptr %p0
  %p1 = getelementptr float,ptr %a,i64 1
  %a1 = load float,ptr %p1
  %p2 = getelementptr float,ptr %a,i64 2
  %a2 = load float,ptr %p2
  %r0 = call float @llvm.fma.f32(float %a0, float 0xBFF0000000000000, float %a1)
  %r1 = call float @llvm.fma.f32(float %a1, float 0xBFF0000000000000, float %a2)
  %r2 = call float @llvm.fma.f32(float %a2, float 0xBFF0000000000000, float %a0)
  %q0 = getelementptr float,ptr %o,i64 0
  store float %r0,ptr %q0
  %q1 = getelementptr float,ptr %o,i64 1
  store float %r1,ptr %q1
  %q2 = getelementptr float,ptr %o,i64 2
  store float %r2,ptr %q2
  ret void
}

define void @source_case1(ptr %a,ptr %o) {
entry:
  %p0 = getelementptr float,ptr %a,i64 0
  %a0 = load float,ptr %p0
  %p1 = getelementptr float,ptr %a,i64 1
  %a1 = load float,ptr %p1
  %p2 = getelementptr float,ptr %a,i64 2
  %a2 = load float,ptr %p2
  %p3 = getelementptr float,ptr %a,i64 3
  %a3 = load float,ptr %p3
  %p4 = getelementptr float,ptr %a,i64 4
  %a4 = load float,ptr %p4
  %r0 = call float @llvm.fma.f32(float %a0, float %a1, float 0x400921FB60000000)
  %r1 = call float @llvm.fma.f32(float %a1, float %a2, float 0x400921FB60000000)
  %r2 = call float @llvm.fma.f32(float %a2, float %a3, float 0x400921FB60000000)
  %r3 = call float @llvm.fma.f32(float %a3, float %a4, float 0x400921FB60000000)
  %r4 = call float @llvm.fma.f32(float %a4, float %a0, float 0x400921FB60000000)
  %q0 = getelementptr float,ptr %o,i64 0
  store float %r0,ptr %q0
  %q1 = getelementptr float,ptr %o,i64 1
  store float %r1,ptr %q1
  %q2 = getelementptr float,ptr %o,i64 2
  store float %r2,ptr %q2
  %q3 = getelementptr float,ptr %o,i64 3
  store float %r3,ptr %q3
  %q4 = getelementptr float,ptr %o,i64 4
  store float %r4,ptr %q4
  ret void
}

define void @source_case2(ptr %a,ptr %o) {
entry:
  %p0 = getelementptr float,ptr %a,i64 0
  %a0 = load float,ptr %p0
  %p1 = getelementptr float,ptr %a,i64 1
  %a1 = load float,ptr %p1
  %p2 = getelementptr float,ptr %a,i64 2
  %a2 = load float,ptr %p2
  %p3 = getelementptr float,ptr %a,i64 3
  %a3 = load float,ptr %p3
  %p4 = getelementptr float,ptr %a,i64 4
  %a4 = load float,ptr %p4
  %p5 = getelementptr float,ptr %a,i64 5
  %a5 = load float,ptr %p5
  %p6 = getelementptr float,ptr %a,i64 6
  %a6 = load float,ptr %p6
  %r0 = call float @llvm.fma.f32(float 0x3FE8000000000000, float %a0, float 0x3FD0000000000000)
  %r1 = call float @llvm.fma.f32(float 0x3FE8000000000000, float %a1, float 0x3FD0000000000000)
  %r2 = call float @llvm.fma.f32(float 0x3FE8000000000000, float %a2, float 0x3FD0000000000000)
  %r3 = call float @llvm.fma.f32(float 0x3FE8000000000000, float %a3, float 0x3FD0000000000000)
  %r4 = call float @llvm.fma.f32(float 0x3FE8000000000000, float %a4, float 0x3FD0000000000000)
  %r5 = call float @llvm.fma.f32(float 0x3FE8000000000000, float %a5, float 0x3FD0000000000000)
  %r6 = call float @llvm.fma.f32(float 0x3FE8000000000000, float %a6, float 0x3FD0000000000000)
  %q0 = getelementptr float,ptr %o,i64 0
  store float %r0,ptr %q0
  %q1 = getelementptr float,ptr %o,i64 1
  store float %r1,ptr %q1
  %q2 = getelementptr float,ptr %o,i64 2
  store float %r2,ptr %q2
  %q3 = getelementptr float,ptr %o,i64 3
  store float %r3,ptr %q3
  %q4 = getelementptr float,ptr %o,i64 4
  store float %r4,ptr %q4
  %q5 = getelementptr float,ptr %o,i64 5
  store float %r5,ptr %q5
  %q6 = getelementptr float,ptr %o,i64 6
  store float %r6,ptr %q6
  ret void
}

define void @source_case3(ptr %a,ptr %o) {
entry:
  %p0 = getelementptr float,ptr %a,i64 0
  %a0 = load float,ptr %p0
  %p1 = getelementptr float,ptr %a,i64 1
  %a1 = load float,ptr %p1
  %p2 = getelementptr float,ptr %a,i64 2
  %a2 = load float,ptr %p2
  %p3 = getelementptr float,ptr %a,i64 3
  %a3 = load float,ptr %p3
  %p4 = getelementptr float,ptr %a,i64 4
  %a4 = load float,ptr %p4
  %p5 = getelementptr float,ptr %a,i64 5
  %a5 = load float,ptr %p5
  %r0 = call float @llvm.fma.f32(float %a0, float %a1, float 0x3810000000000000)
  %r1 = call float @llvm.fma.f32(float %a1, float %a2, float 0x3810000000000000)
  %r2 = call float @llvm.fma.f32(float %a2, float %a3, float 0x3810000000000000)
  %r3 = call float @llvm.fma.f32(float %a3, float %a4, float 0x3810000000000000)
  %r4 = call float @llvm.fma.f32(float %a4, float %a5, float 0x3810000000000000)
  %r5 = call float @llvm.fma.f32(float %a5, float %a0, float 0x3810000000000000)
  %q0 = getelementptr float,ptr %o,i64 0
  store float %r0,ptr %q0
  %q1 = getelementptr float,ptr %o,i64 1
  store float %r1,ptr %q1
  %q2 = getelementptr float,ptr %o,i64 2
  store float %r2,ptr %q2
  %q3 = getelementptr float,ptr %o,i64 3
  store float %r3,ptr %q3
  %q4 = getelementptr float,ptr %o,i64 4
  store float %r4,ptr %q4
  %q5 = getelementptr float,ptr %o,i64 5
  store float %r5,ptr %q5
  ret void
}

define void @source_case4(ptr %a,ptr %o) {
entry:
  %p0 = getelementptr float,ptr %a,i64 0
  %a0 = load float,ptr %p0
  %p1 = getelementptr float,ptr %a,i64 1
  %a1 = load float,ptr %p1
  %p2 = getelementptr float,ptr %a,i64 2
  %a2 = load float,ptr %p2
  %p3 = getelementptr float,ptr %a,i64 3
  %a3 = load float,ptr %p3
  %p4 = getelementptr float,ptr %a,i64 4
  %a4 = load float,ptr %p4
  %r0 = call float @llvm.fma.f32(float %a0, float 0x36A0000000000000, float %a1)
  %r1 = call float @llvm.fma.f32(float %a1, float 0x36A0000000000000, float %a2)
  %r2 = call float @llvm.fma.f32(float %a2, float 0x36A0000000000000, float %a3)
  %r3 = call float @llvm.fma.f32(float %a3, float 0x36A0000000000000, float %a4)
  %r4 = call float @llvm.fma.f32(float %a4, float 0x36A0000000000000, float %a0)
  %q0 = getelementptr float,ptr %o,i64 0
  store float %r0,ptr %q0
  %q1 = getelementptr float,ptr %o,i64 1
  store float %r1,ptr %q1
  %q2 = getelementptr float,ptr %o,i64 2
  store float %r2,ptr %q2
  %q3 = getelementptr float,ptr %o,i64 3
  store float %r3,ptr %q3
  %q4 = getelementptr float,ptr %o,i64 4
  store float %r4,ptr %q4
  ret void
}

define void @source_case5(ptr %a,ptr %o) {
entry:
  %p0 = getelementptr float,ptr %a,i64 0
  %a0 = load float,ptr %p0
  %p1 = getelementptr float,ptr %a,i64 1
  %a1 = load float,ptr %p1
  %p2 = getelementptr float,ptr %a,i64 2
  %a2 = load float,ptr %p2
  %p3 = getelementptr float,ptr %a,i64 3
  %a3 = load float,ptr %p3
  %p4 = getelementptr float,ptr %a,i64 4
  %a4 = load float,ptr %p4
  %p5 = getelementptr float,ptr %a,i64 5
  %a5 = load float,ptr %p5
  %p6 = getelementptr float,ptr %a,i64 6
  %a6 = load float,ptr %p6
  %r0 = call float @llvm.fma.f32(float 0x47EFFFFFE0000000, float %a0, float 0x3FD0000000000000)
  %r1 = call float @llvm.fma.f32(float 0x47EFFFFFE0000000, float %a1, float 0x3FD0000000000000)
  %r2 = call float @llvm.fma.f32(float 0x47EFFFFFE0000000, float %a2, float 0x3FD0000000000000)
  %r3 = call float @llvm.fma.f32(float 0x47EFFFFFE0000000, float %a3, float 0x3FD0000000000000)
  %r4 = call float @llvm.fma.f32(float 0x47EFFFFFE0000000, float %a4, float 0x3FD0000000000000)
  %r5 = call float @llvm.fma.f32(float 0x47EFFFFFE0000000, float %a5, float 0x3FD0000000000000)
  %r6 = call float @llvm.fma.f32(float 0x47EFFFFFE0000000, float %a6, float 0x3FD0000000000000)
  %q0 = getelementptr float,ptr %o,i64 0
  store float %r0,ptr %q0
  %q1 = getelementptr float,ptr %o,i64 1
  store float %r1,ptr %q1
  %q2 = getelementptr float,ptr %o,i64 2
  store float %r2,ptr %q2
  %q3 = getelementptr float,ptr %o,i64 3
  store float %r3,ptr %q3
  %q4 = getelementptr float,ptr %o,i64 4
  store float %r4,ptr %q4
  %q5 = getelementptr float,ptr %o,i64 5
  store float %r5,ptr %q5
  %q6 = getelementptr float,ptr %o,i64 6
  store float %r6,ptr %q6
  ret void
}

declare float @llvm.fma.f32(float,float,float)
