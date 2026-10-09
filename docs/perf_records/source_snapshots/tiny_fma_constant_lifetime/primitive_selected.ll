define void @selected_probe0(ptr %a,ptr %b,ptr %o) {
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
  %pack = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$4,ft0,$0\0Afmadd.s $1,$5,ft0,$1\0Afmadd.s $2,$6,ft0,$2\0Afmadd.s $3,$7,ft0,$3", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %a0, float %a1, float %a2, float %a3, i32 3207688728, float %b0, float %b1, float %b2, float %b3)
  %r0 = extractvalue { float, float, float, float } %pack, 0
  %r1 = extractvalue { float, float, float, float } %pack, 1
  %r2 = extractvalue { float, float, float, float } %pack, 2
  %r3 = extractvalue { float, float, float, float } %pack, 3
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

define void @selected_probe1(ptr %a,ptr %b,ptr %o) {
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
  %pack = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$4,ft0,$0\0Afmadd.s $1,$5,ft0,$1\0Afmadd.s $2,$6,ft0,$2\0Afmadd.s $3,$7,ft0,$3", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %a0, float %a1, float %a2, float %a3, i32 822272776, float %b0, float %b1, float %b2, float %b3)
  %r0 = extractvalue { float, float, float, float } %pack, 0
  %r1 = extractvalue { float, float, float, float } %pack, 1
  %r2 = extractvalue { float, float, float, float } %pack, 2
  %r3 = extractvalue { float, float, float, float } %pack, 3
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

define void @selected_probe2(ptr %a,ptr %b,ptr %o) {
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
  %pack = call { float, float, float, float } asm "fmv.w.x ft0,$4\0Afmv.w.x ft1,$5\0Afmadd.s $0,ft0,$0,ft1\0Afmadd.s $1,ft0,$1,ft1\0Afmadd.s $2,ft0,$2,ft1\0Afmadd.s $3,ft0,$3,ft1", "=f,=f,=f,=f,r,r,0,1,2,3,~{ft0},~{ft1}"(i32 985008993, i32 1007192201, float %a0, float %a1, float %a2, float %a3)
  %r0 = extractvalue { float, float, float, float } %pack, 0
  %r1 = extractvalue { float, float, float, float } %pack, 1
  %r2 = extractvalue { float, float, float, float } %pack, 2
  %r3 = extractvalue { float, float, float, float } %pack, 3
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

define void @selected_probe3(ptr %a,ptr %b,ptr %o) {
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
  %pack = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %b0, float %b1, float %b2, float %b3, i32 1026206379, float %a0, float %a1, float %a2, float %a3)
  %r0 = extractvalue { float, float, float, float } %pack, 0
  %r1 = extractvalue { float, float, float, float } %pack, 1
  %r2 = extractvalue { float, float, float, float } %pack, 2
  %r3 = extractvalue { float, float, float, float } %pack, 3
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

define void @selected_probe4(ptr %a,ptr %b,ptr %o) {
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
  %pack = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %b0, float %b1, float %b2, float %b3, i32 1042983595, float %a0, float %a1, float %a2, float %a3)
  %r0 = extractvalue { float, float, float, float } %pack, 0
  %r1 = extractvalue { float, float, float, float } %pack, 1
  %r2 = extractvalue { float, float, float, float } %pack, 2
  %r3 = extractvalue { float, float, float, float } %pack, 3
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

define void @selected_probe5(ptr %a,ptr %b,ptr %o) {
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
  %pack = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %b0, float %b1, float %b2, float %b3, i32 1056964608, float %a0, float %a1, float %a2, float %a3)
  %r0 = extractvalue { float, float, float, float } %pack, 0
  %r1 = extractvalue { float, float, float, float } %pack, 1
  %r2 = extractvalue { float, float, float, float } %pack, 2
  %r3 = extractvalue { float, float, float, float } %pack, 3
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

define void @selected_probe6(ptr %a,ptr %b,ptr %o) {
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
  %pack = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %b0, float %b1, float %b2, float %b3, i32 1065353216, float %a0, float %a1, float %a2, float %a3)
  %r0 = extractvalue { float, float, float, float } %pack, 0
  %r1 = extractvalue { float, float, float, float } %pack, 1
  %r2 = extractvalue { float, float, float, float } %pack, 2
  %r3 = extractvalue { float, float, float, float } %pack, 3
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

define void @selected_probe7(ptr %a,ptr %b,ptr %o) {
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
  %pack = call { float, float, float, float } asm "fmv.w.x ft0,$8\0Afmadd.s $0,$0,$4,ft0\0Afmadd.s $1,$1,$5,ft0\0Afmadd.s $2,$2,$6,ft0\0Afmadd.s $3,$3,$7,ft0", "=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}"(float %b0, float %b1, float %b2, float %b3, i32 1065353216, float %a0, float %a1, float %a2, float %a3)
  %r0 = extractvalue { float, float, float, float } %pack, 0
  %r1 = extractvalue { float, float, float, float } %pack, 1
  %r2 = extractvalue { float, float, float, float } %pack, 2
  %r3 = extractvalue { float, float, float, float } %pack, 3
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
