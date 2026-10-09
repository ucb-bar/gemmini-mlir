.section .data
.balign 64
.global source_q
source_q:
.incbin "/scratch/agustin/tmp/gemmini-attention-prefix-guard-20261005/out/artifacts/probes/attention-prefix-guard-20261005/v4_h1_prepared/source_q.bin"
.section .data
.balign 64
.global source_k
source_k:
.incbin "/scratch/agustin/tmp/gemmini-attention-prefix-guard-20261005/out/artifacts/probes/attention-prefix-guard-20261005/v4_h1_prepared/source_k.bin"
.section .data
.balign 64
.global source_v
source_v:
.incbin "/scratch/agustin/tmp/gemmini-attention-prefix-guard-20261005/out/artifacts/probes/attention-prefix-guard-20261005/v4_h1_prepared/source_v.bin"
.section .data
.balign 64
.global source_gold
source_gold:
.incbin "/scratch/agustin/tmp/gemmini-attention-prefix-guard-20261005/out/artifacts/probes/attention-prefix-guard-20261005/v4_h1_prepared/source_gold.bin"
