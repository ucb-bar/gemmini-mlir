.section .rodata
.balign 64
.global input_a
input_a:
.incbin "/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny-packed-rhs-current-20261006/capture/a.bin"
.balign 64
.global input_b
input_b:
.incbin "/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny-packed-rhs-current-20261006/capture/b.bin"
.balign 64
.global expected
expected:
.incbin "/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny-packed-rhs-current-20261006/capture/expected.bin"
.balign 8
.global chosen_arm
chosen_arm: .quad 0
.global expected_b_xor
expected_b_xor: .quad 5720580802183809710
.global expected_a_xor
expected_a_xor: .quad 9877008592677756155
