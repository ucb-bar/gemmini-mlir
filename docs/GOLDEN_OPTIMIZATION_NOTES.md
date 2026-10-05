
## Source-proven residual domains (2026-10-05; experiment only)

`golden_residual_domains` traces verified static reshape/transpose operations and
recognizes finite source DQ/add/ReLU/positive symmetric quantization. Unknown
producers retain [-128,127]. No sample values or region labels establish a range.
The closed recipe has 12 skips proven [0,127]; four projection residuals retain
both full signed domains. Every admissible pair is checked by the C++ interval
search and independently by NumPy with original float32 ordering. The source
MLIR, proof module, search source, and admissible output bytes are hashed.

The first exact representatives in the explicitly limited ratio neighborhood
reduce coefficient chunks in quantize_14 (9 to 3), quantize_27 (17 to 16), and
quantize_48 (8 to 7). This is not a globally optimal coefficient search. Residual
array issue floor falls from 5,193,216 to 4,854,528 cycles, saving 338,688 (6.52%).
The selected whole-model floor would remain 22,466,944 before DMA and host work.
The first residual still requires its full signed domain and 39 chunks.
No graph rewrite, current proof replacement, or hardware promotion is made.
Three tests cover finite source proof, refusal of unknown/no-ReLU/nonzero-threshold/
overflow cases, and propagation through verified reshape/permutation.

## Stock FireSim whole ResNet 1775

Exact virtual-padding candidate completed in 62,441,162 forward cycles, all 1,000
outputs bit-exact to the unchanged closed-recipe golden. ELF, staged ELF,
bitstream, and job-owned UART hashes are pinned in the receipt. This is 8.69%
below the corresponding padded candidate1774 (68,385,997); the unprofiled result
is retained independently of the separate boundary profiling job1777.

## Stock FireSim TinyLlama 1776

The large-N GEMM plus optimized-runtime candidate completed in 1,402,210,517
forward cycles versus 1,800,267,524 for baseline1747: 22.11% fewer cycles.
All 256,000 float32 outputs (1,024,000 little-endian bytes) match the validated
native/Spike reference SHA256 and the older baseline output bytes. Original
Torch tolerances and reference-quality checks remain unchanged. The full output
hash is outside the timed interval. ELF/staged ELF, bitstream, job-owned UART,
and strict native/Spike reference receipt hashes are in the hardware receipt.
This is a combined schedule/runtime result; it does not isolate either change.

## TinyLlama final-link device profile (queued1782)

Profile ELF0af1b3851cc69332af54a074625aeb4b3096c9a2cf8da4a0d1d2ef50729b33c5
reuses every optimized model, dense kernel, and runtime object from1776. Five
public three-pointer dense symbols are wrapped, preserving their whole-batch ABI
and private cores. The model linker constants are retained. No tensor-operation
instrumentation is inserted before compiler fusion.

Actual Gemmini Spike passes the complete256,000-value digest, zero-FSM audit,
155 calls, five symbols, exact per-symbol multiplicities22/1/44/44/44, and interval
conservation. Proxy forward678,967,350 = device18,254,522 + host660,712,828;
tail4,496,084. These are retired-instruction counters, not hardware cycles. Every
observed symbol also matches source catalog binding order, permitting an ordinal
to source-region map in the separate Spike attribution artifact. Repeated largest
host gaps are approximately17.98M instructions before M8N2048K2048 calls.
Hardware attribution remains pending;1776 remains the controlled unprofiled time.

## ResNet exact primitive-boundary hardware attribution1777

Stock FireSim completed with staged ELF/bitstream identity, all1,000 original
outputs exact, zero-FSM,70 calls and conserved intervals. Harness forward metric
61,467,502; interior profiled forward61,466,934 = device37,187,546 + host24,279,388.
The separate unprofiled1775 result62,441,162 remains the controlled time: profile
code placement and instrumentation can change timing, not merely add overhead.

Device categories: dense16,635,093; direct convolution12,330,645; wide residual
6,779,734; pooled stem1,442,074 cycles. Largest host gaps: prestem16,825,464;
preclassifier3,210,641; after integer readout matmul25 before26=1,990,199;
after48 before49=962,268; beforematmul14=614,548. The exact CPU readout work is
intentionally charged to host gaps because wrappers surround primitive device
calls. Per-call shapes, source regions, categories and intervals are retained in
the receipt. This provides measured priorities for host cleanup and device
schedule work; the analytical selected issue floor is not a measured runtime.
