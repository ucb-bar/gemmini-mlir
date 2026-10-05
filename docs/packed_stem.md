# Packed 7x7 RGB stem

## Schedule and structural proof

`golden_stem.py` lowers the 7x7, stride-2, explicit-pad-3 RGB stem through typed
xDSL and the ordinary primitive Gemmini LLVM lowering. Input is NHWC including
its existing six halo rows/columns; all halo values are preserved. Output is
spatial-major. The source-bound integration returns i32 and preserves the
original floating epilogue and maxpool exactly.

The seven horizontal taps are contiguous: `kw * Cin = 7 * 3 = 21` bytes.
Overlapping DMA rows use stride six bytes. A wide MVIN with 21 columns produces
two scratchpad tiles, reduced as K=16 and K=5. The original HWIO-linear
147-by-Cout weight matrix is retained. There is no host weight repack, no
materialized im2col, and no 32-byte overread past a 21-byte source window.
For every output pixel, these two reductions visit exactly the same `(kh,kw,ci)`
products as the source's 49 tap panels, in the same integer accumulation order.

For H=W=224 and Cout=64:

| Command | Count |
|---|---:|
| Compute / preload | 43,904 each |
| A MVIN | 5,488 |
| B MVIN | 1,568 |
| i32 MVOUT | 3,136 |
| Padded array issue floor | 702,464 cycles |

The compute count matches the 14-K-block stem in the supplied q1013 analysis.
The analysis's estimated stem timing is not a measured per-layer reference.

`stem_binding.py` proves 49 ordered concat panels, NCHW-to-NHWC transpose,
complete flattening, common source, exact slice composition, RGB channels,
stride two, padded geometry, and signed from-zero i8 GEMM. It refuses reordered
taps, alternative strides, shape disagreements, and other panel structures.
`stem_bundle.py` emits the rewritten source, device object, checked descriptor
adapter, native oracle, hashes, and command census.

## Validation and whole-model integration

- GSIM H5/W35/Cout19: **9,380 kernel cycles**, all outputs and output guard pass.
  This exercises odd extents, partial spatial/channel tiles, and nonzero halos.
- Same small GSIM geometry with i8/ReLU and captured stem scale: **9,179 kernel
  cycles**, output and guard pass. This validates store quantization, not pooling.
- Full H224/W224/Cout64 primitive Spike: **802,816 outputs pass**, guard passes,
  175,988 kernel retired instructions. `--isa=rv64gc_zicntr` enables the probe's
  `rdcycle` without enabling vector instructions.
- Whole model, stem1/fused27/direct4/dense22: native and actual Gemmini Spike
  match all **1,000 outputs exactly**. Final ELF no-FSM audit passes.
- Whole-model Spike: **908,532,797 retired instructions**, versus 1,018,739,739
  for the same fused model with materialized stem im2col. This is a functional
  simulator proxy, not a FireSim cycle measurement.

Executable: `out/whole_requant_stem/build_direct/model.elf`, SHA256
`f3a432e5c037cf2704e3331c214a4acc38b2348043189c8d0d6dc6135e9176d8`.
Build hash: `b7077b7cbfa9`. Both host fusion environment flags are enabled;
experimental layout propagation is disabled.

Use `tests/fused_whole_model_probe.py --packed-stem` to add the stem before the
existing fused/direct/dense catalog. `stem_mixed_catalog.py` composes objects,
source identities, route coverage, and native oracle sources. The ordinary
whole-model native and Spike gates remain mandatory.

## Hardware maxpool follow-up

The current capture's 64 stem biases are all zero. The complete scalar threshold
proof accepts all 64 channels with store scale **0.0020730062387883663** and ReLU.
Positive scaling, ReLU, RNE, and saturation are monotone, so exchanging their
pointwise application with max can preserve values. The source uses 3x3 stride-2
maxpool with negative-infinity padding after ReLU. A future fused binder must
prove that structure and each valid window before replacing padding by the
hardware's zero contribution.

Hardware pooling is **not enabled** here. Stock StoreController pooling indexes
accumulator rows as `base + row * ocols + col` and forbids block MVOUTs while
pooling. The current accumulator tile order interleaves channel tiles and only
retains one output row, so its layout cannot be passed directly to pool-on-store.
An explicit spatial accumulator-layout abstraction, with checked row/channel
strides and lifetime/bank bounds, is needed before pooling is safely composable.
The scalar threshold census alone is not a hardware pool correctness proof.
