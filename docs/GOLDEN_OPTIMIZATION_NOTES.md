
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

## ResNet exact uniform quantization + roundeven hardware1781

The controlled host-improvement candidate completed in55,239,221 stock FireSim
forward cycles versus62,441,162 for1775 (11.53% fewer). All1,000 original capture
outputs are bit-exact; staged ELF, bitstream and UART identities are pinned.
The no-FSM hardware schedules match1775. This combines two source-proven uniform
quantization rewrites and the target round-to-nearest-even helper; their separate
contributions are not isolated by this measurement.

## Exact reduction layout transfer: no baseline speedup claim

An isolated Merlin rewrite carries a common preferred input permutation through
pointwise producers into reductions only if parallel-axis and reduction-axis
relative orders are each preserved. Scalar bodies, initial accumulator values,
output layout, and H,W floating-point accumulation order remain unchanged.
Eight tests cover scalar-chain preservation and refusal of unknown/effectful or
order-changing layouts. Both whole-model experiments pass native and actual
Gemmini Spike against all1,000 original output bits, plus final zero-FSM audit.

The parallel-first generic version generates a model.o byte-identical to1775:
1fda829162ba289fff62efc7a70d60858671fa5128c5ab4a11797155f03d9e76,
and exactly the same19,286,601 Spike retired instructions. Existing downstream
fusion already eliminated the apparent final transpose and DQ materialization.
The physical-order named variant changes loops to H,W,C and uses19,552,971
instructions. Its cache behavior is unmeasured; it is not promoted. A source-level
copy census must not be described as remaining runtime copy traffic. The3.21M
hardware classifier gap motivates a genuinely different blocked reduction
schedule, not another equivalent transpose elimination.

## Tiny hardware device profile1782

Complete digest,155 calls with exact multiplicities, source symbol order and
interval conservation pass on stock FireSim. Harness1,399,827,329 cycles;
interior1,399,826,923 = device219,195,608 + host1,180,631,315, tail9,503,370.
Largest repeated pre-o_proj host gaps are26.74M cycles, covering attention and
its surrounding transforms. These intervals are not isolated attention timings.
The unprofiled1776 result remains1,402,210,517 cycles. Full per-call source-region
attribution is preserved in the hardware receipt.

## Explicit channel-block reduction experiment1789

Core opt-in `reduction_channel_block=64` splits a divisible static trailing
channel axis, then runs channel-block/H/W/channel-inner. The default is0.
H,W order for each output and all original DQ scalar arithmetic stay exact.
Ten focused tests pass. Real lowered LLVM retains fused i8-load/DQ-multiply/add;
there is no materialized f32 DQ tensor. Whole native and actual Spike match all
1,000 original outputs, final zero-FSM audit passes.

The candidate composes current uniform quantization and combined exact clamp/RNE
host optimizations. Runtime/weight objects and all device executable bytes match
control1786; only host scheduling differs. Spike11,629,325 versus11,355,701 is
2.41% more instructions. Hardware1789 measures the cache tradeoff; no speedup or
promotion is claimed before that A/B. Generic layout transfer without blocking
remains byte-identical to baseline, as recorded above.

Historical late LLVM relinks inherit the base harness build_hash. Their final
ELF SHA256, complete object/transform receipts and staged hardware identity are
the variant identity; the inherited build_hash alone is insufficient. The new
backend host-LLVM hook being developed by the parent will establish transformed
LLVM identity before model.o, harness hash and final linking for future builds.

## ResNet combined exact clamp/RNE hardware1786

Verified stock FireSim forward49,673,153 cycles, all1,000 unchanged original
outputs exact. Final/staged ELF and bitstream identities are pinned. This is
10.08% fewer cycles than1781 (55,239,221),20.45% fewer than1775 (62,441,162).
This becomes the controlled unblocked reference for queued blocked-reduction1789.
Its inherited base harness build_hash is not unique to the late transform;
final ELF/object/transform hashes supply that identity. No22M claim is made.

### Reduction rewrite SSA-use hygiene

Follow-up core f943a6664 constructs only the selected replacement, avoiding
orphan use-list entries from an uninserted unblocked alternative. A regression
test fails before and passes after;11 focused tests pass. On the actual full
pre-layout model, before/after emitted MLIR is byte-identical, preserving queued
1789's compiled correctness/performance identity. The earlier-stage layout
artifact file is later rewritten in-place by device binding; compare stage
snapshots or canonical stage emission, not that final file against an earlier
report hash. Immutable per-stage IR artifacts are an infrastructure improvement.

## Tiny scalar host scheduling hardware1788

Verified1,030,207,906 stock FireSim forward cycles with the complete256,000-value,
1,024,000-byte output SHA unchanged from original baseline. Staged ELF/bitstream,
UART and native/Spike/Torch reference receipts are pinned. Compared with1776's
1,402,210,517 cycles this is26.53% fewer; compared with1747's1,800,267,524 it is
42.77% fewer. Device arithmetic remains unchanged; the explicit scalar RV64GC
host schedule avoids target-inappropriate vectorized host IR. Stronger exact
scalar+quantization/RNE candidate1792 is queued separately, not yet measured.

## Exact blocked reduction hardware1789

Verified stock forward48,780,534 cycles, all1,000 original output bits exact,
final/staged ELF and bitstream pinned. Channel block64 saves892,619 cycles
(1.80%) versus strongest unblocked1786 at49,673,153, despite2.41% more Spike
instructions. Both retain the same source H,W arithmetic order and strongest
combined clamp/RNE host path; device executable bytes and runtime objects match.
This is one hardware observation per variant; blocking remains explicit/default
off. The historical late-relink harness hash alone is insufficient identity.

## Identified separate-B, block64 and exact host composition

New build selects the fully proved separate-B dense bundle plus verified block64
reduction, uniform quantization and exact combined clamp/RNE. The backend now
hashes selected host LLVM before compiling model.o and emitting harness identity
(f71783670baa). Explicit host_vectorize=True preserves the historical ResNet
policy. Full native and actual Spike outputs match all original1,000 float bits;
final ELF has zero FSM. Spike11,497,420 is an instruction proxy. Stock job1795
will compare against1789; no hardware result yet. Full4,000-byte SHA256 is emitted
after timing with one-value prefix, shortening serial output without weakening
full-output coverage. ELF4ef297f990815f309f80d6c2916af8e772b83e28d0b8cad686c5f5909b35092e.

### Scalar-host ResNet control is identical

Explicit host_vectorize=False versus1795's True completed full native/Spike
original-golden gates. Both emitted byte-identical final ELF4ef297f9..., selected
host LLVM and model.o; both retire11,497,420 Spike instructions. The backend
receipts confirm the distinct requested booleans. No duplicate hardware job is
submitted because this ResNet lowering contains no effective vectorization
change under these features. Tiny's scalar-host improvement does not transfer
to this already transformed ResNet instance.
