define void @selected_case0(ptr %a,ptr %o) {
entry:
  %p0 = getelementptr float,ptr %a,i64 0
  %a0 = load float,ptr %p0
  %p1 = getelementptr float,ptr %a,i64 1
  %a1 = load float,ptr %p1
  %p2 = getelementptr float,ptr %a,i64 2
  %a2 = load float,ptr %p2
  %constant.fma.packet.0 = call { float, float } asm "fmv.w.x ft0,$4\0Afmadd.s $0,$2,ft0,$0\0Afmadd.s $1,$3,ft0,$1", "=&f,=&f,f,f,r,0,1,~{ft0}"(float %a0, float %a1, i32 3212836864, float %a1, float %a2)
  %r0 = extractvalue { float, float } %constant.fma.packet.0, 0
  %r1 = extractvalue { float, float } %constant.fma.packet.0, 1
  %r2 = call float @llvm.fma.f32(float %a2, float 0xBFF0000000000000, float %a0)
  %q0 = getelementptr float,ptr %o,i64 0
  store float %r0,ptr %q0
  %q1 = getelementptr float,ptr %o,i64 1
  store float %r1,ptr %q1
  %q2 = getelementptr float,ptr %o,i64 2
  store float %r2,ptr %q2
  ret void
}

define void @selected_case1(ptr %a,ptr %o) {
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
  %constant.fma.packet.0 = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=&f,=&f,=&f,=&f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %a1, float %a2, float %a3, float %a4, i32 1078530011, float %a0, float %a1, float %a2, float %a3)
  %r0 = extractvalue { float, float, float, float } %constant.fma.packet.0, 0
  %r1 = extractvalue { float, float, float, float } %constant.fma.packet.0, 1
  %r2 = extractvalue { float, float, float, float } %constant.fma.packet.0, 2
  %r3 = extractvalue { float, float, float, float } %constant.fma.packet.0, 3
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

define void @selected_case2(ptr %a,ptr %o) {
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
  %constant.fma.packet.0 = call { float, float, float, float } asm "fmv.w.x ft0,$4\0Afmv.w.x ft1,$5\0Afmadd.s $0,ft0,$0,ft1\0Afmadd.s $1,ft0,$1,ft1\0Afmadd.s $2,ft0,$2,ft1\0Afmadd.s $3,ft0,$3,ft1", "=&f,=&f,=&f,=&f,r,r,0,1,2,3,~{ft0},~{ft1}"(i32 1061158912, i32 1048576000, float %a0, float %a1, float %a2, float %a3)
  %r0 = extractvalue { float, float, float, float } %constant.fma.packet.0, 0
  %r1 = extractvalue { float, float, float, float } %constant.fma.packet.0, 1
  %r2 = extractvalue { float, float, float, float } %constant.fma.packet.0, 2
  %r3 = extractvalue { float, float, float, float } %constant.fma.packet.0, 3
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

define void @selected_case3(ptr %a,ptr %o) {
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
  %constant.fma.packet.0 = call { float, float } asm "fmv.w.x ft0,$4\0Afmadd.s $0,$0,$2,ft0\0Afmadd.s $1,$1,$3,ft0", "=&f,=&f,f,f,r,0,1,~{ft0}"(float %a1, float %a2, i32 8388608, float %a0, float %a1)
  %r0 = extractvalue { float, float } %constant.fma.packet.0, 0
  %r1 = extractvalue { float, float } %constant.fma.packet.0, 1
  %constant.fma.packet.1 = call { float, float } asm "fmv.w.x ft0,$4\0Afmadd.s $0,$0,$2,ft0\0Afmadd.s $1,$1,$3,ft0", "=&f,=&f,f,f,r,0,1,~{ft0}"(float %a3, float %a4, i32 8388608, float %a2, float %a3)
  %r2 = extractvalue { float, float } %constant.fma.packet.1, 0
  %r3 = extractvalue { float, float } %constant.fma.packet.1, 1
  %constant.fma.packet.2 = call { float, float } asm "fmv.w.x ft0,$4\0Afmadd.s $0,$0,$2,ft0\0Afmadd.s $1,$1,$3,ft0", "=&f,=&f,f,f,r,0,1,~{ft0}"(float %a5, float %a0, i32 8388608, float %a4, float %a5)
  %r4 = extractvalue { float, float } %constant.fma.packet.2, 0
  %r5 = extractvalue { float, float } %constant.fma.packet.2, 1
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

define void @selected_case4(ptr %a,ptr %o) {
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
  %constant.fma.packet.0 = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$4,ft0,$0\0Afmadd.s $1,$5,ft0,$1\0Afmadd.s $2,$6,ft0,$2\0Afmadd.s $3,$7,ft0,$3", "=&f,=&f,=&f,=&f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %a0, float %a1, float %a2, float %a3, i32 1, float %a1, float %a2, float %a3, float %a4)
  %r0 = extractvalue { float, float, float, float } %constant.fma.packet.0, 0
  %r1 = extractvalue { float, float, float, float } %constant.fma.packet.0, 1
  %r2 = extractvalue { float, float, float, float } %constant.fma.packet.0, 2
  %r3 = extractvalue { float, float, float, float } %constant.fma.packet.0, 3
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

define void @selected_case5(ptr %a,ptr %o) {
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
  %constant.fma.packet.0 = call { float, float } asm "fmv.w.x ft0,$2\0Afmv.w.x ft1,$3\0Afmadd.s $0,ft0,$0,ft1\0Afmadd.s $1,ft0,$1,ft1", "=&f,=&f,r,r,0,1,~{ft0},~{ft1}"(i32 2139095039, i32 1048576000, float %a0, float %a1)
  %r0 = extractvalue { float, float } %constant.fma.packet.0, 0
  %r1 = extractvalue { float, float } %constant.fma.packet.0, 1
  %constant.fma.packet.1 = call { float, float } asm "fmv.w.x ft0,$2\0Afmv.w.x ft1,$3\0Afmadd.s $0,ft0,$0,ft1\0Afmadd.s $1,ft0,$1,ft1", "=&f,=&f,r,r,0,1,~{ft0},~{ft1}"(i32 2139095039, i32 1048576000, float %a2, float %a3)
  %r2 = extractvalue { float, float } %constant.fma.packet.1, 0
  %r3 = extractvalue { float, float } %constant.fma.packet.1, 1
  %constant.fma.packet.2 = call { float, float } asm "fmv.w.x ft0,$2\0Afmv.w.x ft1,$3\0Afmadd.s $0,ft0,$0,ft1\0Afmadd.s $1,ft0,$1,ft1", "=&f,=&f,r,r,0,1,~{ft0},~{ft1}"(i32 2139095039, i32 1048576000, float %a4, float %a5)
  %r4 = extractvalue { float, float } %constant.fma.packet.2, 0
  %r5 = extractvalue { float, float } %constant.fma.packet.2, 1
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
