# Exact mean reduction on device

The permitted reference ZIP executes global average pooling through Gemmini
(`glue_gap` → `fx_gap`). Its 12,066-cycle “Other” timer includes that device
operation and glue. This motivates the same general algorithmic split for our
existing source-certified mean, without adopting its different numeric values.

Merlin's new `emit_integer_sum_finish` consumes exact externally produced i32
sums and the immutable original signed-i8 input. It reuses the complete rational
sum certificate and original ordered binary32 replay for ambiguous sums. The
caller must prove a zero-initialized, nonoverflowing integer producer and
disjoint input/sum/output buffers. The runtime guard checks the sum-table bounds;
it cannot establish that an arbitrary caller's supplied sum is exact.

The OOT provider recognizes the canonical static scalar-QDQ mean, zero init,
original reduction axes/order, no fastmath, and exact NHWC transpose. It emits
the integer product `ones[1,count] × input[count,channels] → sums[1,channels]`
through the ordinary xDSL Gemmini primitive compiler. Resource selection uses
actual scratchpad/accumulator bounds. Batch and channel tails are supported.
The producer completes its final fence before any CPU sum read or replay.

The normal tensor call has explicit read/write/write effects. Fresh i32 sums
and i8 output are distinct sole-use `tensor.empty` writers. The result is the
i8 destination. The normal writer conversion establishes aligned private
allocations and keeps original input live through finishing. There is no
streamed CPU/device access or additional coherence-granule assumption.

## Evidence

- The original physical input is captured at the existing mean boundary in a
  whole native run that preserves all 1,000 original binary32 output words.
- Complete common-address ranked GSIM pair: original CPU component 327,962
  cycles; new device producer plus unchanged source finishing 53,744 cycles,
  a section reduction of 274,218 cycles (83.613%). Both timed components are
  byte-identical to their actual current1992 or fresh normal model components.
- Both arms verify all 2,048 output bytes, all 2,048 exact i32 sums where
  produced, 8,192 guard bytes, all 100,352 immutable input bytes, and the returned
  descriptor. Strict RV64GC Spike and final no-FSM audits pass.
- Independent batch3/channel17/count7 capsule also passes complete target
  correctness and guards, measuring 6,726 → 6,527 GSIM cycles. This control is
  an independent scalar producer and does not price the original model.
- Eleven shared native tests include every signed-byte pair in both layouts,
  sizes 1/7/49/128, batch/channel tails, extremes, and input/sum immutability.
  Twenty OOT source/effect/type/resource/refusal and prior binding checks pass.
- Fresh normal control, fresh normal candidate, and frozen1992 controlled
  candidate all pass the unchanged full 1,000-word native and strict target
  gate at `atol=rtol=0` with no FSM instruction anywhere in the final ELF.

The exact current1992 baseline ELF is reproduced byte for byte. Its first four
device partial-link stages, all 52 paired/packet leaves, original39-chunk
residual, stem, runtime, and weights remain unchanged. The last partial link
substitutes only the mean component and matches the fresh normal catalog.
The controlled candidate ELF is
`dc693df17efe42b08e7ec137efdbc90ccc09aa9a8caba8e40eaa766c27c21e6f`.
Segmented projection inputs are a separate qualified candidate and are not
composed into this arm.

The GSIM ROI includes the full ranked producer, DMA, stores, final fence, and
source finishing, with preallocated private scratch. Whole allocation/boundary
costs and stock FireSim performance are unknown. No section savings are added
to another candidate's hardware result. The new option remains default off.

Initial native test failures came from dlopen retaining a library after pytest
reused its deleted temporary path; code-hash library names fixed that harness.
An independent fixture's overlapping string replacement and `IRUses` handling
were corrected. A missing historical benchmark-header path is preserved in an
incomplete capsule directory before the successful fresh run. Source arithmetic
and the original whole gate were unchanged.

Core topic commits: `3d9b14648`, `cc35923b3`. OOT implementation commits:
`ce105cd`, `456983e`. The final receipt and standalone recloser are
`resnet_integer_mean_current1992_qualified.json` and
`tests/device_mean_current1992_qualification_probe.py`. Immutable source
snapshots and all artifact hashes are included. Child token counters are
unavailable. A future conserved profile must add this new raw integer-sum
kernel explicitly; the existing 70-primitive profile does not include it.
