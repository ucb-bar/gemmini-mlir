"builtin.module"() ({
  "llvm.mlir.global"() <{global_type = !llvm.array<247 x i8>, sym_name = "kernel_entry_scratch_0__attention_kt_3", linkage = #llvm.linkage<"internal">, addr_space = 0 : i32, alignment = 64 : i64}> ({
    %0 = "llvm.mlir.zero"() : () -> !llvm.array<247 x i8>
    "llvm.return"(%0) : (!llvm.array<247 x i8>) -> ()
  }) : () -> ()
  "llvm.func"() <{sym_name = "kernel_entry", function_type = !llvm.func<void (!llvm.ptr, !llvm.ptr, !llvm.ptr)>, CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<"external">, visibility_ = 0 : i64}> ({
  ^bb0(%1: !llvm.ptr, %2: !llvm.ptr, %3: !llvm.ptr):
    %4 = "llvm.mlir.addressof"() <{global_name = @kernel_entry_scratch_0__attention_kt_3}> : () -> !llvm.ptr
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    %5 = "llvm.mlir.constant"() <{value = 0 : i64}> : () -> i64
    "llvm.inline_asm"(%5, %5) <{asm_string = ".insn r 123, 3, 7, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    %6 = "llvm.mlir.constant"() <{value = 4575657221409472773 : i64}> : () -> i64
    %7 = "llvm.mlir.constant"() <{value = 19 : i64}> : () -> i64
    "llvm.inline_asm"(%6, %7) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %8 = "llvm.mlir.constant"() <{value = 2 : i64}> : () -> i64
    %9 = "llvm.mlir.constant"() <{value = 4575657221408423937 : i64}> : () -> i64
    "llvm.inline_asm"(%8, %9) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %10 = "llvm.mlir.constant"() <{value = 13 : i64}> : () -> i64
    %11 = "llvm.mlir.constant"() <{value = 2147483648 : i64}> : () -> i64
    %12 = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
    %13 = "llvm.mlir.constant"() <{value = 4294967295 : i64}> : () -> i64
    %14 = "llvm.mlir.constant"() <{value = 65535 : i64}> : () -> i64
    %15 = "llvm.mlir.constant"() <{value = 32 : i64}> : () -> i64
    %16 = "llvm.mlir.constant"() <{value = 48 : i64}> : () -> i64
    %17 = "llvm.mlir.constant"() <{value = 323 : i64}> : () -> i64
    %18 = "llvm.mlir.constant"() <{value = 247 : i64}> : () -> i64
    %19 = "llvm.mlir.constant"() <{value = 221 : i64}> : () -> i64
    %20 = "llvm.mlir.constant"() <{value = 4575657221408489476 : i64}> : () -> i64
    %21 = "llvm.mlir.constant"() <{value = 281474976710656 : i64}> : () -> i64
    %22 = "llvm.mlir.constant"() <{value = 4575657221409472769 : i64}> : () -> i64
    %23 = "llvm.mlir.constant"() <{value = 4575657221409472777 : i64}> : () -> i64
    %24 = "llvm.mlir.constant"() <{value = 16 : i64}> : () -> i64
    %25 = "llvm.mlir.constant"() <{value = 3 : i64}> : () -> i64
    %26 = "llvm.mlir.constant"() <{value = 8192 : i64}> : () -> i64
    %27 = "llvm.mlir.constant"() <{value = 30 : i64}> : () -> i64
    %28 = "llvm.mlir.constant"() <{value = 4575657221408423988 : i64}> : () -> i64
    %29 = "llvm.mlir.constant"() <{value = 2684354560 : i64}> : () -> i64
    %30 = "llvm.mlir.constant"() <{value = 536870911 : i64}> : () -> i64
    "llvm.br"(%5) [^bb1] : (i64) -> ()
  ^bb1(%31: i64):
    %32 = "llvm.icmp"(%31, %7) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%32) [^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb2:
    %33 = "llvm.mul"(%31, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %34 = "llvm.add"(%5, %33) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %35 = "llvm.add"(%34, %11) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %36 = "llvm.and"(%35, %13) : (i64, i64) -> i64
    %37 = "llvm.or"(%5, %36) : (i64, i64) -> i64
    %38 = "llvm.and"(%12, %14) : (i64, i64) -> i64
    %39 = "llvm.shl"(%38, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %40 = "llvm.or"(%37, %39) : (i64, i64) -> i64
    %41 = "llvm.and"(%10, %14) : (i64, i64) -> i64
    %42 = "llvm.shl"(%41, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %43 = "llvm.or"(%40, %42) : (i64, i64) -> i64
    %44 = "llvm.getelementptr"(%2, %31) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %45 = "llvm.ptrtoint"(%44) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%45, %43) <{asm_string = ".insn r 123, 3, 2, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %46 = "llvm.add"(%31, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%46) [^bb1] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb3:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.br"(%5) [^bb4] : (i64) -> ()
  ^bb4(%47: i64):
    %48 = "llvm.icmp"(%47, %7) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%48) [^bb5, ^bb6] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb5:
    %49 = "llvm.mul"(%47, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %50 = "llvm.mul"(%47, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %51 = "llvm.add"(%5, %50) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %52 = "llvm.add"(%51, %11) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %53 = "llvm.and"(%52, %13) : (i64, i64) -> i64
    %54 = "llvm.or"(%5, %53) : (i64, i64) -> i64
    %55 = "llvm.and"(%12, %14) : (i64, i64) -> i64
    %56 = "llvm.shl"(%55, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %57 = "llvm.or"(%54, %56) : (i64, i64) -> i64
    %58 = "llvm.and"(%10, %14) : (i64, i64) -> i64
    %59 = "llvm.shl"(%58, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %60 = "llvm.or"(%57, %59) : (i64, i64) -> i64
    %61 = "llvm.getelementptr"(%4, %49) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %62 = "llvm.ptrtoint"(%61) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%62, %60) <{asm_string = ".insn r 123, 3, 3, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %63 = "llvm.add"(%47, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%63) [^bb4] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb6:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.br"(%5) [^bb7] : (i64) -> ()
  ^bb7(%64: i64):
    %65 = "llvm.icmp"(%64, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%65) [^bb8, ^bb9] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb8:
    %66 = "llvm.mul"(%64, %17) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %67 = "llvm.getelementptr"(%1, %66) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %68 = "llvm.mul"(%64, %18) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %69 = "llvm.getelementptr"(%4, %68) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %70 = "llvm.mul"(%64, %19) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %71 = "llvm.getelementptr"(%3, %70) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"(%20, %21) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%22, %7) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%23, %10) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.br"(%5) [^bb10] : (i64) -> ()
  ^bb9:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.return"() : () -> ()
  ^bb10(%72: i64):
    %73 = "llvm.icmp"(%72, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%73) [^bb11, ^bb12] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb11:
    "llvm.br"(%5) [^bb13] : (i64) -> ()
  ^bb12:
    %74 = "llvm.add"(%64, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%74) [^bb7] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb13(%75: i64):
    %76 = "llvm.icmp"(%75, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%76) [^bb14, ^bb15] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb14:
    "llvm.br"(%5) [^bb16] : (i64) -> ()
  ^bb15:
    "llvm.br"(%5) [^bb22] : (i64) -> ()
  ^bb16(%77: i64):
    %78 = "llvm.icmp"(%77, %8) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%78) [^bb17, ^bb18] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb17:
    "llvm.br"(%5) [^bb19] : (i64) -> ()
  ^bb18:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    %79 = "llvm.add"(%75, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%79) [^bb13] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb19(%80: i64):
    %81 = "llvm.icmp"(%80, %8) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%81) [^bb20, ^bb21] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb20:
    %82 = "llvm.mul"(%72, %8) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %83 = "llvm.add"(%5, %82) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %84 = "llvm.add"(%83, %77) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %85 = "llvm.mul"(%72, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %86 = "llvm.add"(%5, %85) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %87 = "llvm.mul"(%77, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %88 = "llvm.add"(%86, %87) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %89 = "llvm.mul"(%80, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %90 = "llvm.icmp"(%72, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %91 = "llvm.icmp"(%77, %12) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %92 = "llvm.select"(%91, %12, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %93 = "llvm.select"(%90, %92, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %94 = "llvm.icmp"(%80, %12) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %95 = "llvm.select"(%94, %25, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %96 = "llvm.mul"(%77, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %97 = "llvm.add"(%5, %96) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %98 = "llvm.mul"(%80, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %99 = "llvm.add"(%97, %98) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %100 = "llvm.mul"(%88, %7) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %101 = "llvm.add"(%100, %89) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %102 = "llvm.getelementptr"(%67, %101) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %103 = "llvm.shl"(%95, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %104 = "llvm.or"(%99, %103) : (i64, i64) -> i64
    %105 = "llvm.shl"(%93, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %106 = "llvm.or"(%104, %105) : (i64, i64) -> i64
    %107 = "llvm.ptrtoint"(%102) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%107, %106) <{asm_string = ".insn r 123, 3, 2, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %108 = "llvm.add"(%80, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%108) [^bb19] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb21:
    %109 = "llvm.add"(%77, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%109) [^bb16] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb22(%110: i64):
    %111 = "llvm.icmp"(%110, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%111) [^bb23, ^bb24] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb23:
    "llvm.br"(%5) [^bb25] : (i64) -> ()
  ^bb24:
    %112 = "llvm.add"(%72, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%112) [^bb10] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb25(%113: i64):
    %114 = "llvm.icmp"(%113, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%114) [^bb26, ^bb27] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb26:
    "llvm.br"(%5) [^bb28] : (i64) -> ()
  ^bb27:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"(%8, %28) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.br"(%5) [^bb49] : (i64) -> ()
  ^bb28(%115: i64):
    %116 = "llvm.icmp"(%115, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%116) [^bb29, ^bb30] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb29:
    "llvm.br"(%5) [^bb31] : (i64) -> ()
  ^bb30:
    "llvm.br"(%5) [^bb37] : (i64) -> ()
  ^bb31(%117: i64):
    %118 = "llvm.icmp"(%117, %8) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%118) [^bb32, ^bb33] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb32:
    "llvm.br"(%5) [^bb34] : (i64) -> ()
  ^bb33:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    %119 = "llvm.add"(%115, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%119) [^bb28] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb34(%120: i64):
    %121 = "llvm.icmp"(%120, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%121) [^bb35, ^bb36] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb35:
    %122 = "llvm.mul"(%113, %8) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %123 = "llvm.add"(%5, %122) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %124 = "llvm.add"(%123, %117) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %125 = "llvm.add"(%5, %110) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %126 = "llvm.add"(%125, %120) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %127 = "llvm.mul"(%113, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %128 = "llvm.add"(%5, %127) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %129 = "llvm.mul"(%117, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %130 = "llvm.add"(%128, %129) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %131 = "llvm.mul"(%110, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %132 = "llvm.add"(%5, %131) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %133 = "llvm.mul"(%120, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %134 = "llvm.add"(%132, %133) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %135 = "llvm.icmp"(%113, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %136 = "llvm.icmp"(%117, %12) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %137 = "llvm.select"(%136, %25, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %138 = "llvm.select"(%135, %137, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %139 = "llvm.icmp"(%110, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %140 = "llvm.icmp"(%120, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %141 = "llvm.select"(%140, %10, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %142 = "llvm.select"(%139, %141, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %143 = "llvm.mul"(%117, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %144 = "llvm.add"(%5, %143) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %145 = "llvm.mul"(%120, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %146 = "llvm.add"(%144, %145) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %147 = "llvm.add"(%146, %26) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %148 = "llvm.mul"(%130, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %149 = "llvm.add"(%148, %134) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %150 = "llvm.getelementptr"(%69, %149) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %151 = "llvm.shl"(%142, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %152 = "llvm.or"(%147, %151) : (i64, i64) -> i64
    %153 = "llvm.shl"(%138, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %154 = "llvm.or"(%152, %153) : (i64, i64) -> i64
    %155 = "llvm.ptrtoint"(%150) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%155, %154) <{asm_string = ".insn r 123, 3, 1, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %156 = "llvm.add"(%120, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%156) [^bb34] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb36:
    %157 = "llvm.add"(%117, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%157) [^bb31] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb37(%158: i64):
    %159 = "llvm.icmp"(%158, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%159) [^bb38, ^bb39] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb38:
    "llvm.br"(%5) [^bb40] : (i64) -> ()
  ^bb39:
    %160 = "llvm.add"(%113, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%160) [^bb25] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb40(%161: i64):
    %162 = "llvm.icmp"(%161, %8) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%162) [^bb41, ^bb42] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb41:
    "llvm.br"(%5) [^bb43] : (i64) -> ()
  ^bb42:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    %163 = "llvm.add"(%158, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%163) [^bb37] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb43(%164: i64):
    %165 = "llvm.icmp"(%164, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%165) [^bb44, ^bb45] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb44:
    "llvm.br"(%5) [^bb46] : (i64) -> ()
  ^bb45:
    %166 = "llvm.add"(%161, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%166) [^bb40] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb46(%167: i64):
    %168 = "llvm.icmp"(%167, %8) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%168) [^bb47, ^bb48] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb47:
    %169 = "llvm.mul"(%161, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %170 = "llvm.add"(%5, %169) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %171 = "llvm.mul"(%164, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %172 = "llvm.add"(%170, %171) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %173 = "llvm.add"(%172, %26) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %174 = "llvm.mul"(%113, %8) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %175 = "llvm.add"(%5, %174) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %176 = "llvm.add"(%175, %161) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %177 = "llvm.add"(%5, %110) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %178 = "llvm.add"(%177, %164) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %179 = "llvm.mul"(%167, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %180 = "llvm.add"(%5, %179) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %181 = "llvm.mul"(%164, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %182 = "llvm.add"(%180, %181) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %183 = "llvm.icmp"(%113, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %184 = "llvm.icmp"(%161, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %185 = "llvm.select"(%184, %5, %12) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %186 = "llvm.select"(%183, %185, %12) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %187 = "llvm.icmp"(%72, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %188 = "llvm.icmp"(%167, %12) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %189 = "llvm.select"(%188, %12, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %190 = "llvm.select"(%187, %189, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %191 = "llvm.icmp"(%110, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %192 = "llvm.icmp"(%164, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %193 = "llvm.select"(%192, %10, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %194 = "llvm.select"(%191, %193, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %195 = "llvm.icmp"(%113, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %196 = "llvm.icmp"(%161, %12) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %197 = "llvm.select"(%196, %25, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %198 = "llvm.select"(%195, %197, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %199 = "llvm.or"(%182, %11) : (i64, i64) -> i64
    %200 = "llvm.shl"(%186, %27) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %201 = "llvm.or"(%199, %200) : (i64, i64) -> i64
    %202 = "llvm.shl"(%194, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %203 = "llvm.or"(%173, %202) : (i64, i64) -> i64
    %204 = "llvm.shl"(%198, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %205 = "llvm.or"(%203, %204) : (i64, i64) -> i64
    %206 = "llvm.shl"(%194, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %207 = "llvm.or"(%201, %206) : (i64, i64) -> i64
    %208 = "llvm.shl"(%190, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %209 = "llvm.or"(%207, %208) : (i64, i64) -> i64
    "llvm.inline_asm"(%205, %209) <{asm_string = ".insn r 123, 3, 6, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %210 = "llvm.mul"(%167, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %211 = "llvm.add"(%5, %210) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %212 = "llvm.mul"(%161, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %213 = "llvm.add"(%211, %212) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %214 = "llvm.mul"(%72, %8) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %215 = "llvm.add"(%5, %214) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %216 = "llvm.add"(%215, %167) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %217 = "llvm.mul"(%113, %8) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %218 = "llvm.add"(%5, %217) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %219 = "llvm.add"(%218, %161) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %220 = "llvm.icmp"(%72, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %221 = "llvm.icmp"(%167, %12) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %222 = "llvm.select"(%221, %12, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %223 = "llvm.select"(%220, %222, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %224 = "llvm.icmp"(%113, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %225 = "llvm.icmp"(%161, %12) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %226 = "llvm.select"(%225, %25, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %227 = "llvm.select"(%224, %226, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %228 = "llvm.shl"(%227, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %229 = "llvm.or"(%213, %228) : (i64, i64) -> i64
    %230 = "llvm.shl"(%223, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %231 = "llvm.or"(%229, %230) : (i64, i64) -> i64
    %232 = "llvm.shl"(%194, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %233 = "llvm.or"(%13, %232) : (i64, i64) -> i64
    %234 = "llvm.shl"(%227, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %235 = "llvm.or"(%233, %234) : (i64, i64) -> i64
    "llvm.inline_asm"(%231, %235) <{asm_string = ".insn r 123, 3, 4, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %236 = "llvm.add"(%167, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%236) [^bb46] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb48:
    %237 = "llvm.add"(%164, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%237) [^bb43] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb49(%238: i64):
    %239 = "llvm.icmp"(%238, %8) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%239) [^bb50, ^bb51] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb50:
    "llvm.br"(%5) [^bb52] : (i64) -> ()
  ^bb51:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    %240 = "llvm.add"(%110, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%240) [^bb22] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb52(%241: i64):
    %242 = "llvm.icmp"(%241, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%242) [^bb53, ^bb54] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb53:
    "llvm.br"(%5) [^bb55] : (i64) -> ()
  ^bb54:
    %243 = "llvm.add"(%238, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%243) [^bb49] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb55(%244: i64):
    %245 = "llvm.icmp"(%244, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%245) [^bb56, ^bb57] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb56:
    %246 = "llvm.mul"(%72, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %247 = "llvm.add"(%5, %246) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %248 = "llvm.mul"(%238, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %249 = "llvm.add"(%247, %248) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %250 = "llvm.mul"(%110, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %251 = "llvm.add"(%5, %250) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %252 = "llvm.mul"(%241, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %253 = "llvm.add"(%251, %252) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %254 = "llvm.icmp"(%72, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %255 = "llvm.icmp"(%238, %12) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %256 = "llvm.select"(%255, %12, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %257 = "llvm.select"(%254, %256, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %258 = "llvm.icmp"(%110, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %259 = "llvm.icmp"(%241, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %260 = "llvm.select"(%259, %10, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %261 = "llvm.select"(%258, %260, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %262 = "llvm.mul"(%238, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %263 = "llvm.add"(%5, %262) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %264 = "llvm.mul"(%241, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %265 = "llvm.add"(%263, %264) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %266 = "llvm.mul"(%249, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %267 = "llvm.add"(%266, %253) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %268 = "llvm.getelementptr"(%71, %267) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %269 = "llvm.and"(%265, %30) : (i64, i64) -> i64
    %270 = "llvm.or"(%29, %269) : (i64, i64) -> i64
    %271 = "llvm.and"(%270, %13) : (i64, i64) -> i64
    %272 = "llvm.or"(%5, %271) : (i64, i64) -> i64
    %273 = "llvm.and"(%261, %14) : (i64, i64) -> i64
    %274 = "llvm.shl"(%273, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %275 = "llvm.or"(%272, %274) : (i64, i64) -> i64
    %276 = "llvm.and"(%257, %14) : (i64, i64) -> i64
    %277 = "llvm.shl"(%276, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %278 = "llvm.or"(%275, %277) : (i64, i64) -> i64
    %279 = "llvm.ptrtoint"(%268) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%279, %278) <{asm_string = ".insn r 123, 3, 3, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %280 = "llvm.add"(%244, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%280) [^bb55] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb57:
    %281 = "llvm.add"(%241, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%281) [^bb52] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  }) : () -> ()
  "llvm.func"() <{sym_name = "kernel_complete", function_type = !llvm.func<void ()>, CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<"external">, visibility_ = 0 : i64}> ({
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
