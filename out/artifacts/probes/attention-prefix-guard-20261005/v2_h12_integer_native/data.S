.section .data
.balign 64
.global source_q
source_q:
.incbin "/scratch/agustin/tmp/gemmini-attention-prefix-guard-20261005/out/artifacts/probes/attention-prefix-guard-20261005/v2_h12_integer_native/source_q.bin"
.section .data
.balign 64
.global source_k
source_k:
.incbin "/scratch/agustin/tmp/gemmini-attention-prefix-guard-20261005/out/artifacts/probes/attention-prefix-guard-20261005/v2_h12_integer_native/source_k.bin"
.section .data
.balign 64
.global source_v
source_v:
.incbin "/scratch/agustin/tmp/gemmini-attention-prefix-guard-20261005/out/artifacts/probes/attention-prefix-guard-20261005/v2_h12_integer_native/source_v.bin"
.section .data
.balign 64
.global source_gold
source_gold:
.incbin "/scratch/agustin/tmp/gemmini-attention-prefix-guard-20261005/out/artifacts/probes/attention-prefix-guard-20261005/v2_h12_integer_native/source_gold.bin"
