/* Exact binary32 round-to-nearest-even for the stock RV64GC host.
 * Binary32 magnitudes >= 2^23 are already integral. Smaller finite values
 * convert exactly back from i32 after RNE conversion; restore the zero sign.
 * Preserve existing fflags (roundeven must not raise inexact). sNaNs are
 * quieted through fadd, which retains normal invalid-exception behavior.
 */
.text
.align 2
.globl gemmini_golden_roundevenf
.type gemmini_golden_roundevenf,@function
gemmini_golden_roundevenf:
    fmv.x.w t0, fa0
    srli t1, t0, 23
    andi t1, t1, 255
    li t2, 150
    bgeu t1, t2, .Lround_large
    frflags t2
    fcvt.w.s t1, fa0, rne
    fcvt.s.w fa0, t1, rne
    fsflags t2
    fmv.x.w t1, fa0
    srli t0, t0, 31
    andi t0, t0, 1
    slli t0, t0, 31
    or t1, t1, t0
    fmv.w.x fa0, t1
    ret
.Lround_large:
    li t2, 255
    bne t1, t2, .Lround_return
    fadd.s fa0, fa0, fa0
.Lround_return:
    ret
.size gemmini_golden_roundevenf, .-gemmini_golden_roundevenf
.section .note.GNU-stack,"",@progbits
