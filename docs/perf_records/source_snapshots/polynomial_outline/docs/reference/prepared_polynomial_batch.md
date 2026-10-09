---
title: "Four-cell prepared polynomial scheduling"
kind: reference
status: current
owner: core
last_verified: 2026-10-06
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs: [src/merlin/llvmlower, merlin/runtime/c]
---

# Four-cell prepared polynomial scheduling

`emit_source_attention_frontier(..., polynomial_batch_four=True)` is an explicit,
default-off scheduling option for the prepared word-space softmax enclosure.
It requires the existing immutable source polynomial plan, source RNE,
nontrapping arithmetic and unobserved exception flags. Unsupported intervals or
plans use the existing scalar checked helper.

The portable helper stages four independent intervals and eight endpoints. Each
endpoint retains its source multiply, floor, subtract, three rounded Horner
FMAs, subtract, encoded FMA and integer conversion. Point/cutoff cases retain
scalar results; speculative discarded arithmetic is limited to the already
proved polynomial domain. It does not reassociate arithmetic or widen bounds.

The consumer computes these independent intervals after maximum refinement,
then processes cells in their original order. Exact replay, BF16 bin decisions,
per-lane denominator subsequences and the final tree remain unchanged. Tails
use only their active cells; dummy private lanes do not escape. Private scratch
is automatic storage, with no change to the caller workspace ABI.

Portable C expresses independence but does not promise target issue order.
Inspect actual compiler output and price the entire group, including scratch
and replay, before selecting this policy. The initial RV64GC compilation still
serializes each endpoint Horner chain; no eight-cell extension is justified by
that evidence.

## Optional helper outlining

`outline_polynomial_batch=True` requires `polynomial_batch_four=True` and a
compiler supporting the explicit no-inline attribute. Both options default to
false. The outlined function keeps the same private endpoint inputs, results,
source operation order and checked fallback. It introduces a synchronous call
boundary without changing the public writer or workspace ABI.

This option can reduce live caller state and stack loads, while adding calls
and callee-save traffic. Qualification must include those costs in the complete
caller and retain the original consumer observations. Instruction reductions do
not establish a cycle improvement. Default generated source remains unchanged.
