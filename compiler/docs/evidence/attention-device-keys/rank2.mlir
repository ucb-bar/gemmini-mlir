"builtin.module"() ({
  "llvm.mlir.global"() <{global_type = !llvm.array<40 x i8>, sym_name = "kernel_entry_scratch_0__attention_kt_3", linkage = #llvm.linkage<"internal">, addr_space = 0 : i32, alignment = 64 : i64}> ({
    %0 = "llvm.mlir.zero"() : () -> !llvm.array<40 x i8>
    "llvm.return"(%0) : (!llvm.array<40 x i8>) -> ()
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
    %7 = "llvm.mlir.constant"() <{value = 8 : i64}> : () -> i64
    "llvm.inline_asm"(%6, %7) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %8 = "llvm.mlir.constant"() <{value = 2 : i64}> : () -> i64
    %9 = "llvm.mlir.constant"() <{value = 4575657221408423937 : i64}> : () -> i64
    "llvm.inline_asm"(%8, %9) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %10 = "llvm.mlir.constant"() <{value = 5 : i64}> : () -> i64
    %11 = "llvm.mlir.constant"() <{value = 2147483648 : i64}> : () -> i64
    %12 = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
    %13 = "llvm.mlir.constant"() <{value = 4294967295 : i64}> : () -> i64
    %14 = "llvm.mlir.constant"() <{value = 65535 : i64}> : () -> i64
    %15 = "llvm.mlir.constant"() <{value = 32 : i64}> : () -> i64
    %16 = "llvm.mlir.constant"() <{value = 48 : i64}> : () -> i64
    %17 = "llvm.mlir.constant"() <{value = 24 : i64}> : () -> i64
    %18 = "llvm.mlir.constant"() <{value = 40 : i64}> : () -> i64
    %19 = "llvm.mlir.constant"() <{value = 15 : i64}> : () -> i64
    %20 = "llvm.mlir.constant"() <{value = 4575657221408489476 : i64}> : () -> i64
    %21 = "llvm.mlir.constant"() <{value = 281474976710656 : i64}> : () -> i64
    %22 = "llvm.mlir.constant"() <{value = 4575657221409472769 : i64}> : () -> i64
    %23 = "llvm.mlir.constant"() <{value = 4575657221409472777 : i64}> : () -> i64
    %24 = "llvm.mlir.constant"() <{value = 16 : i64}> : () -> i64
    %25 = "llvm.mlir.constant"() <{value = 3 : i64}> : () -> i64
    %26 = "llvm.mlir.constant"() <{value = 8192 : i64}> : () -> i64
    %27 = "llvm.mlir.constant"() <{value = 30 : i64}> : () -> i64
    %28 = "llvm.mlir.constant"() <{value = 4575657221408423956 : i64}> : () -> i64
    %29 = "llvm.mlir.constant"() <{value = 2684354560 : i64}> : () -> i64
    "llvm.br"(%5) [^bb1] : (i64) -> ()
  ^bb1(%30: i64):
    %31 = "llvm.icmp"(%30, %7) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%31) [^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb2:
    %32 = "llvm.mul"(%30, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %33 = "llvm.add"(%5, %32) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %34 = "llvm.add"(%33, %11) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %35 = "llvm.and"(%34, %13) : (i64, i64) -> i64
    %36 = "llvm.or"(%5, %35) : (i64, i64) -> i64
    %37 = "llvm.and"(%12, %14) : (i64, i64) -> i64
    %38 = "llvm.shl"(%37, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %39 = "llvm.or"(%36, %38) : (i64, i64) -> i64
    %40 = "llvm.and"(%10, %14) : (i64, i64) -> i64
    %41 = "llvm.shl"(%40, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %42 = "llvm.or"(%39, %41) : (i64, i64) -> i64
    %43 = "llvm.getelementptr"(%2, %30) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %44 = "llvm.ptrtoint"(%43) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%44, %42) <{asm_string = ".insn r 123, 3, 2, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %45 = "llvm.add"(%30, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%45) [^bb1] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb3:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.br"(%5) [^bb4] : (i64) -> ()
  ^bb4(%46: i64):
    %47 = "llvm.icmp"(%46, %7) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%47) [^bb5, ^bb6] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb5:
    %48 = "llvm.mul"(%46, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %49 = "llvm.mul"(%46, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %50 = "llvm.add"(%5, %49) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %51 = "llvm.add"(%50, %11) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %52 = "llvm.and"(%51, %13) : (i64, i64) -> i64
    %53 = "llvm.or"(%5, %52) : (i64, i64) -> i64
    %54 = "llvm.and"(%12, %14) : (i64, i64) -> i64
    %55 = "llvm.shl"(%54, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %56 = "llvm.or"(%53, %55) : (i64, i64) -> i64
    %57 = "llvm.and"(%10, %14) : (i64, i64) -> i64
    %58 = "llvm.shl"(%57, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %59 = "llvm.or"(%56, %58) : (i64, i64) -> i64
    %60 = "llvm.getelementptr"(%4, %48) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %61 = "llvm.ptrtoint"(%60) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%61, %59) <{asm_string = ".insn r 123, 3, 3, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %62 = "llvm.add"(%46, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%62) [^bb4] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb6:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.br"(%5) [^bb7] : (i64) -> ()
  ^bb7(%63: i64):
    %64 = "llvm.icmp"(%63, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%64) [^bb8, ^bb9] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb8:
    %65 = "llvm.mul"(%63, %17) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %66 = "llvm.getelementptr"(%1, %65) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %67 = "llvm.mul"(%63, %18) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %68 = "llvm.getelementptr"(%4, %67) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %69 = "llvm.mul"(%63, %19) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %70 = "llvm.getelementptr"(%3, %69) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"(%20, %21) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%22, %7) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%23, %10) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.br"(%5) [^bb10] : (i64) -> ()
  ^bb9:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.return"() : () -> ()
  ^bb10(%71: i64):
    %72 = "llvm.icmp"(%71, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%72) [^bb11, ^bb12] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb11:
    "llvm.br"(%5) [^bb13] : (i64) -> ()
  ^bb12:
    %73 = "llvm.add"(%63, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%73) [^bb7] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb13(%74: i64):
    %75 = "llvm.icmp"(%74, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%75) [^bb14, ^bb15] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb14:
    "llvm.br"(%5) [^bb16] : (i64) -> ()
  ^bb15:
    %76 = "llvm.add"(%71, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%76) [^bb10] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb16(%77: i64):
    %78 = "llvm.icmp"(%77, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%78) [^bb17, ^bb18] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb17:
    "llvm.br"(%5) [^bb19] : (i64) -> ()
  ^bb18:
    "llvm.br"(%5) [^bb40] : (i64) -> ()
  ^bb19(%79: i64):
    %80 = "llvm.icmp"(%79, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%80) [^bb20, ^bb21] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb20:
    "llvm.br"(%5) [^bb22] : (i64) -> ()
  ^bb21:
    "llvm.br"(%5) [^bb25] : (i64) -> ()
  ^bb22(%81: i64):
    %82 = "llvm.icmp"(%81, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%82) [^bb23, ^bb24] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb23:
    %83 = "llvm.add"(%5, %71) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %84 = "llvm.add"(%83, %79) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %85 = "llvm.add"(%5, %77) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %86 = "llvm.add"(%85, %81) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %87 = "llvm.mul"(%71, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %88 = "llvm.add"(%5, %87) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %89 = "llvm.mul"(%79, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %90 = "llvm.add"(%88, %89) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %91 = "llvm.mul"(%77, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %92 = "llvm.add"(%5, %91) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %93 = "llvm.mul"(%81, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %94 = "llvm.add"(%92, %93) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %95 = "llvm.icmp"(%71, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %96 = "llvm.icmp"(%79, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %97 = "llvm.select"(%96, %25, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %98 = "llvm.select"(%95, %97, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %99 = "llvm.icmp"(%77, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %100 = "llvm.icmp"(%81, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %101 = "llvm.select"(%100, %7, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %102 = "llvm.select"(%99, %101, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %103 = "llvm.mul"(%79, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %104 = "llvm.add"(%5, %103) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %105 = "llvm.mul"(%81, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %106 = "llvm.add"(%104, %105) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %107 = "llvm.mul"(%90, %7) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %108 = "llvm.add"(%107, %94) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %109 = "llvm.getelementptr"(%66, %108) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %110 = "llvm.shl"(%102, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %111 = "llvm.or"(%106, %110) : (i64, i64) -> i64
    %112 = "llvm.shl"(%98, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %113 = "llvm.or"(%111, %112) : (i64, i64) -> i64
    %114 = "llvm.ptrtoint"(%109) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%114, %113) <{asm_string = ".insn r 123, 3, 2, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %115 = "llvm.add"(%81, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%115) [^bb22] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb24:
    %116 = "llvm.add"(%79, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%116) [^bb19] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb25(%117: i64):
    %118 = "llvm.icmp"(%117, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%118) [^bb26, ^bb27] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb26:
    "llvm.br"(%5) [^bb28] : (i64) -> ()
  ^bb27:
    "llvm.br"(%5) [^bb31] : (i64) -> ()
  ^bb28(%119: i64):
    %120 = "llvm.icmp"(%119, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%120) [^bb29, ^bb30] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb29:
    %121 = "llvm.add"(%5, %77) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %122 = "llvm.add"(%121, %117) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %123 = "llvm.add"(%5, %74) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %124 = "llvm.add"(%123, %119) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %125 = "llvm.mul"(%77, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %126 = "llvm.add"(%5, %125) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %127 = "llvm.mul"(%117, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %128 = "llvm.add"(%126, %127) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %129 = "llvm.mul"(%74, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %130 = "llvm.add"(%5, %129) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %131 = "llvm.mul"(%119, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %132 = "llvm.add"(%130, %131) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %133 = "llvm.icmp"(%77, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %134 = "llvm.icmp"(%117, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %135 = "llvm.select"(%134, %7, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %136 = "llvm.select"(%133, %135, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %137 = "llvm.icmp"(%74, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %138 = "llvm.icmp"(%119, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %139 = "llvm.select"(%138, %10, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %140 = "llvm.select"(%137, %139, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %141 = "llvm.mul"(%117, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %142 = "llvm.add"(%5, %141) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %143 = "llvm.mul"(%119, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %144 = "llvm.add"(%142, %143) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %145 = "llvm.add"(%144, %26) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %146 = "llvm.mul"(%128, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %147 = "llvm.add"(%146, %132) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %148 = "llvm.getelementptr"(%68, %147) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i8, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %149 = "llvm.shl"(%140, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %150 = "llvm.or"(%145, %149) : (i64, i64) -> i64
    %151 = "llvm.shl"(%136, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %152 = "llvm.or"(%150, %151) : (i64, i64) -> i64
    %153 = "llvm.ptrtoint"(%148) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%153, %152) <{asm_string = ".insn r 123, 3, 1, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %154 = "llvm.add"(%119, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%154) [^bb28] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb30:
    %155 = "llvm.add"(%117, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%155) [^bb25] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb31(%156: i64):
    %157 = "llvm.icmp"(%156, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%157) [^bb32, ^bb33] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb32:
    "llvm.br"(%5) [^bb34] : (i64) -> ()
  ^bb33:
    %158 = "llvm.add"(%77, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%158) [^bb16] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb34(%159: i64):
    %160 = "llvm.icmp"(%159, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%160) [^bb35, ^bb36] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb35:
    "llvm.br"(%5) [^bb37] : (i64) -> ()
  ^bb36:
    %161 = "llvm.add"(%156, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%161) [^bb31] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb37(%162: i64):
    %163 = "llvm.icmp"(%162, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%163) [^bb38, ^bb39] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb38:
    %164 = "llvm.mul"(%156, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %165 = "llvm.add"(%5, %164) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %166 = "llvm.mul"(%162, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %167 = "llvm.add"(%165, %166) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %168 = "llvm.add"(%167, %26) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %169 = "llvm.add"(%5, %77) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %170 = "llvm.add"(%169, %156) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %171 = "llvm.add"(%5, %74) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %172 = "llvm.add"(%171, %162) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %173 = "llvm.mul"(%159, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %174 = "llvm.add"(%5, %173) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %175 = "llvm.mul"(%162, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %176 = "llvm.add"(%174, %175) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %177 = "llvm.icmp"(%77, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %178 = "llvm.icmp"(%156, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %179 = "llvm.select"(%178, %5, %12) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %180 = "llvm.select"(%177, %179, %12) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %181 = "llvm.icmp"(%71, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %182 = "llvm.icmp"(%159, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %183 = "llvm.select"(%182, %25, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %184 = "llvm.select"(%181, %183, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %185 = "llvm.icmp"(%74, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %186 = "llvm.icmp"(%162, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %187 = "llvm.select"(%186, %10, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %188 = "llvm.select"(%185, %187, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %189 = "llvm.icmp"(%77, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %190 = "llvm.icmp"(%156, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %191 = "llvm.select"(%190, %7, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %192 = "llvm.select"(%189, %191, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %193 = "llvm.or"(%176, %11) : (i64, i64) -> i64
    %194 = "llvm.shl"(%180, %27) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %195 = "llvm.or"(%193, %194) : (i64, i64) -> i64
    %196 = "llvm.shl"(%188, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %197 = "llvm.or"(%168, %196) : (i64, i64) -> i64
    %198 = "llvm.shl"(%192, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %199 = "llvm.or"(%197, %198) : (i64, i64) -> i64
    %200 = "llvm.shl"(%188, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %201 = "llvm.or"(%195, %200) : (i64, i64) -> i64
    %202 = "llvm.shl"(%184, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %203 = "llvm.or"(%201, %202) : (i64, i64) -> i64
    "llvm.inline_asm"(%199, %203) <{asm_string = ".insn r 123, 3, 6, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %204 = "llvm.mul"(%159, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %205 = "llvm.add"(%5, %204) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %206 = "llvm.mul"(%156, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %207 = "llvm.add"(%205, %206) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %208 = "llvm.add"(%5, %71) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %209 = "llvm.add"(%208, %159) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %210 = "llvm.add"(%5, %77) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %211 = "llvm.add"(%210, %156) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %212 = "llvm.icmp"(%71, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %213 = "llvm.icmp"(%159, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %214 = "llvm.select"(%213, %25, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %215 = "llvm.select"(%212, %214, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %216 = "llvm.icmp"(%77, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %217 = "llvm.icmp"(%156, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %218 = "llvm.select"(%217, %7, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %219 = "llvm.select"(%216, %218, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %220 = "llvm.shl"(%219, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %221 = "llvm.or"(%207, %220) : (i64, i64) -> i64
    %222 = "llvm.shl"(%215, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %223 = "llvm.or"(%221, %222) : (i64, i64) -> i64
    %224 = "llvm.shl"(%188, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %225 = "llvm.or"(%13, %224) : (i64, i64) -> i64
    %226 = "llvm.shl"(%219, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %227 = "llvm.or"(%225, %226) : (i64, i64) -> i64
    "llvm.inline_asm"(%223, %227) <{asm_string = ".insn r 123, 3, 4, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    %228 = "llvm.add"(%162, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%228) [^bb37] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb39:
    %229 = "llvm.add"(%159, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%229) [^bb34] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb40(%230: i64):
    %231 = "llvm.icmp"(%230, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%231) [^bb41, ^bb42] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb41:
    "llvm.br"(%5) [^bb43] : (i64) -> ()
  ^bb42:
    %232 = "llvm.add"(%74, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%232) [^bb13] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb43(%233: i64):
    %234 = "llvm.icmp"(%233, %12) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%234) [^bb44, ^bb45] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb44:
    %235 = "llvm.mul"(%71, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %236 = "llvm.add"(%5, %235) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %237 = "llvm.mul"(%230, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %238 = "llvm.add"(%236, %237) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %239 = "llvm.mul"(%74, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %240 = "llvm.add"(%5, %239) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %241 = "llvm.mul"(%233, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %242 = "llvm.add"(%240, %241) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %243 = "llvm.icmp"(%71, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %244 = "llvm.icmp"(%230, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %245 = "llvm.select"(%244, %25, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %246 = "llvm.select"(%243, %245, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %247 = "llvm.icmp"(%74, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %248 = "llvm.icmp"(%233, %5) <{predicate = 0 : i64}> : (i64, i64) -> i1
    %249 = "llvm.select"(%248, %10, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %250 = "llvm.select"(%247, %249, %24) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %251 = "llvm.mul"(%230, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %252 = "llvm.add"(%5, %251) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %253 = "llvm.mul"(%233, %24) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %254 = "llvm.add"(%252, %253) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"(%8, %28) <{asm_string = ".insn r 123, 3, 0, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    %255 = "llvm.add"(%242, %5) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %256 = "llvm.mul"(%238, %10) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %257 = "llvm.add"(%256, %255) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %258 = "llvm.getelementptr"(%70, %257) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %259 = "llvm.or"(%254, %29) : (i64, i64) -> i64
    %260 = "llvm.shl"(%250, %15) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %261 = "llvm.or"(%259, %260) : (i64, i64) -> i64
    %262 = "llvm.shl"(%246, %16) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %263 = "llvm.or"(%261, %262) : (i64, i64) -> i64
    %264 = "llvm.ptrtoint"(%258) : (!llvm.ptr) -> i64
    "llvm.inline_asm"(%264, %263) <{asm_string = ".insn r 123, 3, 3, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    %265 = "llvm.add"(%233, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%265) [^bb43] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  ^bb45:
    %266 = "llvm.add"(%230, %12) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%266) [^bb40] {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>} : (i64) -> ()
  }) : () -> ()
  "llvm.func"() <{sym_name = "kernel_complete", function_type = !llvm.func<void ()>, CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<"external">, visibility_ = 0 : i64}> ({
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
