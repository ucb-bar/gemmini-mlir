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

The i32 packed-stem path above leaves hardware pooling disabled. Stock StoreController pooling indexes
accumulator rows as `base + row * ocols + col` and forbids block MVOUTs while
pooling. The current accumulator tile order interleaves channel tiles and only
retains one output row, so its layout cannot be passed directly to pool-on-store.
An explicit spatial accumulator-layout abstraction, with checked row/channel
strides and lifetime/bank bounds, is needed before pooling is safely composable.
The scalar threshold census alone is not a hardware pool correctness proof.

## Optional fused hardware-pooling implementation

The new `golden_stem_pool.py` path implements an explicit spatial accumulator
layout: for one 16-channel block, convolution pixel `(row,col)` occupies
`acc_base + row * conv_width + col`. It keeps up to nine convolution rows for
four pooled output rows. The full stem uses 1,008 of 1,024 accumulator rows and
15,008 of 16,384 scratchpad rows. All fourteen weight K blocks remain cached;
all A panels for a band are loaded once and shared by its channel blocks.

Adjacent bands recompute their shared convolution row. This bounded tradeoff
uses 125 convolution rows instead of 112: **49,000 computes**, **6,125 A MVINs**,
and **14 B MVINs**. The array issue floor is 784,000 cycles. This schedule is not
the reference's 43,904-compute schedule; no equality claim is made. It avoids
all host stem intermediates and spills.

For the full shape, the first four pooled rows use convolution rows 0 through 7
with top padding one; the next four use rows 7 through 15 with no top padding.
The final band uses rows 103 through 111. Pool output is exactly 56x56. Each
hardware command emits at most two pooled rows, keeping its nine-pixel window
visits within the 1,024-row command tracking limit. Phase proofs/tests also cover
odd convolution extents, bottom/right padding, and channel tails.

Stock Gemmini's reservation station conservatively models a pool command as
reading from its starting accumulator address to that bank's end. Bands may
cross banks, so an explicit fence drains stores before reusing a channel block.
There are 56 such phase drains for the full stem. A future schedule can reduce
these costs only after proving the relevant bank lifetimes.

### Exact source binding

`stem_pool_binding.py` validates the original complete flatten/reshape/transpose
layout, ordered scalar epilogue, 3x3 stride-two affine pool maps, maximum reducer,
negative-infinity padding/identity, and quantizer. The existing complete output
transition proof validates both f32 multiplications, captured zero bias, ReLU,
and final reciprocal/rounding for every possible integer accumulator value.
All 64 channels pass for this capture. ReLU makes every valid pool input
nonnegative; every window contains a valid pixel, so the hardware's zero pad
has the same maximum as the source's negative-infinity pad. Monotonicity then
permits max to exchange with the exactly proved quantizer.

`stem_pool_bundle.py` pins source, manifest, parameter blob, and bias payload;
nonzero biases, failed threshold proofs, or unmatched structures are refused.
Use `tests/fused_whole_model_probe.py --pooled-stem` for the complete composition.

### Device compilation fixes exposed by pooling

The golden typed primitive lowering had discarded the pool fields of
`gemmini.config_st`, even though those fields existed in the dialect. It now
forwards every field exactly; verification rejects bit-field overflow and
incomplete enabled geometry. A regression test compares the complete encoded
command against the ISA encoder.

The device compiler now uses `-mcmodel=medany`. Pool configuration constants
introduced LLVM literal pools whose absolute `R_RISCV_HI20` relocations could not
link at the stock 0x80000000 image base under the previous default code model.
The PC-relative model fixes those relocations without changing the ISA.

### Completed validation

Small GSIM H17/W35/Cout19 with phase overlap, odd height, and channel tails:
17,218 kernel cycles, complete numerical output and guard pass. Full-size GSIM
uses the original sequential f32 dequantization, ReLU, floating maximum pool,
and reciprocal/rounding as an independent oracle.

The source-bound full model already passes native and actual Gemmini Spike for
all 1,000 outputs exactly, with zero descriptor mismatches and a clean final
no-FSM audit. It retires **777,097,851 instructions**, down from 908,532,797 for
the packed i32 stem with host pooling. This is not a FireSim cycle measurement.
ELF `out/whole_requant_pool/build_direct/model.elf`, SHA256
`99267af7a1c66e6dc450fd7a21ee093c7b84d3021a11cb36a85a568d7e34dfa3`.

Full-shape GSIM completed successfully: **1,333,146 kernel cycles**, all
**200,704 outputs** and the guard pass against the original sequential-f32
source oracle. Probe ELF SHA256
`e62f6ea6a9ad07ab15edc8c74f2ebc2501e77efc2064bb154bceb89d956c0da6`;
final no-FSM audit passes. An additional odd-width/height, channel-tail GSIM
probe using the actual captured scales passes at 17,122 cycles. The full shape
also passes strict RV64GC+Zicntr Spike (405,314 kernel retired instructions).
The whole-model pooled variant is ready for stock FireSim measurement; no
whole-model FireSim improvement is claimed from these functional gates.
