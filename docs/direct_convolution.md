# Direct convolution golden, 2026-10-05

`mlir_oot.golden_conv.GoldenConv` lowers NHWC input × HWIO weights to NHWC
output for 3×3/pad1 convolutions at stride 1 or 2. Int8 inputs accumulate in
int32; optional int8 output uses the existing Gemmini scale/ReLU store path.
The ABI is three pointers. There is currently no bias parameter.

The schedule tiles a whole output row across 16-pixel array tiles. For each
kernel tap and channel tile, primitive MVIN reads pixels directly with stride
`stride * Cin`. Horizontal edge lanes are zeroed directly in scratchpad with
null-address MVIN; valid lanes overwrite those zeros. Top/bottom taps entirely
outside the input are skipped. Each B panel remains stationary across spatial
tiles. CPU CFG loops implement repetition. Output accumulators remain resident
until all taps finish. No host im2col, host tensor copying, RVV, or Gemmini
LOOP/FSM operations are generated.

GSIM numeric validation with host-generated independent convolution oracle:

| Input H×W×Cin / Cout | Stride | Kernel cycles | Result |
|---|---:|---:|---|
| 3×19×20 / 19 | 1 | 9,572 | All output values + 2KB output guard pass |
| 7×19×20 / 19 | 2 | 9,641 | All output values + 2KB output guard pass |

Both linked ELFs pass the no-FSM static audit. Machine-readable receipts include
compiler, target IR, object, ELF and GSIM engine hashes under `perf_records`.
These are padding/tail correctness probes, not full-model performance claims.

For the ResNet 56×56×64→64 class, the command model gives 31,872 computes,
9,296 A loads, 7,968 B loads, 896 int32 output stores and zero host im2col bytes.
The padded array issue floor is 509,952 cycles. It is not a latency prediction.
The 7×7 class currently wastes array rows because it preserves output row
boundaries; a flattened multi-row tile is needed there.

## What automatic lowering needs

The upstream layout transform must prove NHWC input/output and HWIO weights,
or insert/propagate these layouts once across the convolution graph. Current
prepared ResNet source uses OIHW × materialized NCHW im2col. It cannot call this
kernel merely because the contraction dimensions match. The binding must match
the pre-im2col convolution, verify stride/padding/dilation, fuse/replace the
im2col producer, and account for users that still require its original layout.
Constant weights should be transformed offline. Residual add/activation must
retain compatible layouts so every layer does not pay a transpose.

The schedule should be parameterized by row-strip size, stationary operand,
resident input extent and buffer lifetime. The current baseline reloads weights
for each output row and input taps for each output-channel block. The reference
uses substantially longer input/weight residency. Reaching its performance
needs accumulator-capacity-aware row strips, input residency and overlapped DMA,
plus wide int8 output stores and a flattened 7×7 tile. Measure each addition with
matching numeric precision and timing scope. This implementation establishes
direct gather and exact padding semantics; it does not claim Jack's performance.

Run `tests/gsim_conv_probe.py` with the existing Merlin Gemmini environment,
`--llvm-bin` and a fresh `--workdir`. Optional `--h --w --cin --cout --stride`
select the geometry. It compiles primitive xDSL, audits the final ELF, and
records numeric output and cycles from GSIM. FireSim must use the requested
stock `FireSimGemminiRocketConfig`; GSIM cycles are not FireSim measurements.

## First DMA optimization

`ConvShape.wide_b=True` combines up to four adjacent weight tiles in one MVIN2.
For 56×56×64→64 this reduces B-load commands from 7,968 to 1,992 with unchanged
compute count. The same 3×19×20→19 i32 edge probe passed at 9,413 GSIM cycles,
versus 9,572 for narrow loads (1.7% reduction on this small probe only).
The int8 scale=0.03125/ReLU variant combines output tiles into wide MVOUT;
it passed every output/guard check at 9,038 cycles. Its output semantics differ
from i32, so its cycles are not an equal-precision speedup comparison.

The probe supports `--wide-b --output-dtype i8 --scale 0.03125 --relu` and
`--build-only` for preparing an audited ELF for the FireSim queue without a
GSIM run. The queued ELF should retain its independent embedded output oracle.

The exact ResNet 56×56×64→64 wide-load/int8-store candidate passed the complete
200,704-element oracle and guard under Spike's Gemmini functional extension
(dim16). The final ELF is statically FSM-free. Its command counts are 31,872
computes, 9,296 A loads, 1,992 B loads, and 224 output stores. The functional
receipt is `perf_records/conv_resnet56_c64_wide_i8_spike.json`. Spike's printed
rdcycle count is retired instructions and must not be reported as FireSim
performance. The matching ELF is ready for the stock FireSim queue.

The separate 56×56×64→64 i32 baseline GSIM run ended without any kernel/output
marker under its 600-second/3M-engine-cycle budget. It supplies no layer timing
or numeric verdict; the large initialization/check harness is part of that
engine budget. Future probe failures record completion/return-code/stderr as
well as stdout, and both budgets are CLI options. The optimized int8 candidate
has full functional evidence above and is submitted as FireSim queue job 1722;
accept timing only after queue confirms simulator-loaded ELF identity.

## Proven upstream boundary bridge

`direct_conv_binding.match` proves two source families without trusting a
provenance label:

* Older prepared capture: `Cin×KH×KW×N×OH×OW` affine gather, complete row-major
  collapse/expand, then signed integer weights×pixels matmul.
* Current quantized capture: nine spatial tap slices, NCHW→NHWC transposes,
  complete row-major flattening, concatenation along K, then signed integer
  pixels×weights matmul. Every static slice is composed back to its base input;
  all nine paths must agree on the base, geometry and stride. Tap ordering is
  checked against KH/KW/Cin reduction order.

Both require a from-zero signed i8×i8→i32 contraction, one batch, exact 3×3 taps,
and stride 1 or 2. They recognize all 16 3×3 contractions in each ResNet source.
The seven-by-seven stem and 1×1 contractions are outside this matcher.

`rewrite` replaces the proven contraction with an external direct-convolution
call. Its padded activation is transposed to NHWC; the new `explicit_halo`
kernel mode consumes every original halo value, including nonzero values. It
never infers that quantization/padding commutes. Older weights require an
OIHW→HWIO transpose and their result requires restoring channel-first order.
The current source already uses HWIO-linear weights and spatial-first results,
so its replacement requires only the activation layout conversion plus views.
All other uses of the original producers remain intact; canonical DCE removes
unused gather/concat chains. The external ABI uses the standard MLIR C interface
and explicit dense descriptor checks. It has no static temporary buffers.

`tests/upstream_conv_probe.py` extracts a selected region from the real source,
compiles the rewritten host MLIR and xDSL primitive device code to RV64gc, links
and audits the final ELF, and runs an independent oracle with deliberately
nonzero halo values. Functional results (all values compared):

| Capture/region | Stride | Output values | Allocated host layout bytes | Result |
|---|---:|---:|---:|---|
| Older / conv_2 | 1 | 200,704 | 1,894,976 | PASS |
| Older / conv_12 | 2 | 100,352 | 1,381,120 | PASS |
| Current / matmul_2 | 1 | 200,704 | 1,018,240 | PASS |
| Current / matmul_12 | 2 | 100,352 | 832,128 | PASS |

All four final ELFs pass the static no-FSM audit. These are complete selected
host+device boundary executions under Spike, not whole-model or FireSim timing.
Source, target compiler and ELF identities are in the corresponding JSON files
under `perf_records`.

`python -m mlir_oot.direct_conv_bundle SOURCE --llvm-bin LLVM --output DIR`
emits rewritten source, primitive device objects plus descriptor adapters,
a linkable `direct_conv.o`, and a source-bound manifest. Apply it before the
dense-GEMM catalog build; its unselected contractions remain in the module.
Then run the existing model preparation/lowering and dense catalog on the
rewritten source and link both device objects. Final whole-model ELF auditing
still applies. Model-wide correctness and performance require subsequent runs.

The xDSL custom bodyless function printer/parser loses argument access
attributes. `serialize` emits only the external declarations in generic syntax,
retaining their `arg_attrs` property structurally across parsing; the model
function retains its usual syntax. Later printers must also preserve these
properties before one-shot bufferization, or bufferization defensively copies
already converted input/weights. Merlin's shared portable printer handles this.
The `llvm.emit_c_interface` attribute must also reach LLVM lowering.

Persistent NHWC propagation is the next performance step: carry layout facts
through pointwise activation, quantization and residual operations, transform
their indexing maps and result shapes together, and insert conversions only at
real graph boundaries or incompatible consumers. Weight transforms should be
performed offline. A layout fact must include logical axis order, physical
strides, alias/users and quantized padding semantics; an extent match is
insufficient. Both upstream families should converge to that shared layout
representation, with the explicit conversions in this bridge as the reference
semantics. Its current conversion cost is intentionally visible and must not be
advertised as a maximum-performance default.

## First real FireSim layer result

Job **1727** completed with **961,350 kernel cycles** and `GOLDEN_CONV PASS` for
56×56×64→64 3×3 stride1, wide-B loads, int8 scale0.03125/ReLU output. The actual
simulator-loaded ELF hash matched the submitted artifact. Hardware was stock
`FireSimGemminiRocketConfig`, using the same bitstream archive as the reference
(`a9a190b9…eca1`). Identity and UART hashes are recorded in
`perf_records/direct_conv_firesim1727.json`. The measured kernel is 1.89× its
509,952-cycle padded array issue floor; substantial scheduling improvement is
still required. The reference ZIP has no per-layer timing, so no exact
same-layer speedup against Jack can be claimed. This measures the direct kernel;
the upstream boundary transposes are outside this probe's timing window.

The bundle now also emits `native_oracle.c`, defining all selected MLIR adapters
and scalar convolution stand-ins for native whole-graph correctness tests.
The stride1/stride2 native stand-ins were checked against independent NumPy
oracles (200,704 and 100,352 values). These stand-ins are correctness tooling;
the RV64 bundle continues to use the xDSL primitive kernels. ABI shape/stride
violations trap directly, so the device bundle has no runtime `exit` dependency.

## Spatial flattening for small feature maps

`golden_flat_conv.py` retains the entire spatial output for one channel block.
It fills array M tiles across output-row boundaries. Each A DMA run is split at
both a source row and an array-tile boundary, so the input remains the exact
NHWC tensor including its explicit halo. HWIO reduction order and accumulator
semantics are unchanged. This is a kernel schedule change after structural
source binding, with no materialized host im2col.

`conv_schedule.select_kernel(..., flat_spatial=True)` initially selected the option when
the complete output fit the accumulator; larger maps now use complete-row bands
as described below. It derives the channel block
from capacity. There is no model-name/region-name dispatch. Both direct and
exact-requant bundles record `schedule_kind` and the selected shape. Ordinary
row schedules remain the default and the fallback for larger feature maps.

For 7×7 output, 512 input/output channels, stride1 or stride2, the schedule uses
four spatial tiles and 16 channel tiles per block (all 1024 accumulator rows).
The prior row schedule used seven partially occupied spatial tiles and four
channel tiles per block:

| Dynamic primitive count | Row schedule | Flat | Flat with wide A |
|---|---:|---:|---:|
| Compute/preload (each) |64,512|36,864|36,864|
| A DMA |16,128|5,760|1,440|
| B DMA |16,128|2,304|2,304|
| i32 store |224|128|128|
| Padded array issue floor |1,032,192|589,824|589,824|

Weights are read once (2,359,296 bytes), rather than once per output row.
Optional wide A loads use up to 64 contiguous input channels, distributed into
four scratchpad tiles by the ordinary load controller. No new opcode or host
packing is needed. Each spatial tile reserves four adjacent K tiles; source
row fragments write identical lane ranges within those tiles. This preserves
partial K tiles and does not read beyond the declared channels.

For 14×14 output with 256 input/output channels, there are 13 spatial tiles,
29,952 computes (479,232 issue floor), and 576 B DMA commands instead of 8,064.
Wide A reduces the 25 gather fragments per input-channel panel to 3,600 A DMA
commands rather than 14,400 for narrow flat loads.

### Verification and measurement

`tests/gsim_conv_probe.py --flat-spatial` supplies **nonzero halo values** and
compares every output to an independent dense integer convolution. Tests cover
both strides, signed-i32 and scaled/ReLU-i8, partial spatial/channel/feature
tiles, and output guards. `--static-inputs` embeds deterministic data bytes in
the ELF so multi-megabyte scalar initialization loops do not consume the GSIM
cycle budget. Embedded assembly contains data directives only.

The complete 7×7C512 i32 flat/bn16 probe passed all 25,088 outputs+guard at
**1,012,868 GSIM kernel cycles**. Strict RV64GC+Zicntr Spike independently passed.
Final linked ELF audit has no FSM instructions. See
`docs/perf_records/flat_conv_gsim.json` for hashes and additional tail probes.
FireSim performance and complete-model validation are separate gates; the
~6.12M-cycle old hardware profile cannot be compared as a same-simulator result
to this GSIM measurement.


The wide-A version passed the same full-shape GSIM oracle at **939,677 cycles**
(7.2% below narrow flat) and strict Spike. It is queued as FireSim **1757**;
ELF SHA256 `0b999e20f2e20fead72edd139fdb83b28401ee496471a3fad2ac36bc384df794`.
The opt-in selector now chooses `spatial_flat_wide_a` after capacity checks.

The initial complete-model integration with narrow flat schedules passed all
1,000 original captured outputs exactly on native scalar standins and actual
Gemmini Spike. Its 777,760,389 retired instructions mostly reflect unchanged
host work and are not a hardware cycle estimate. Final ELF audit passed. See
`docs/perf_records/resnet_flat_conv_spike.json`. The same integration with wide
A is validated separately before whole-model hardware promotion.

The complete wide-A variant also passed all 1,000 original outputs exactly on
native scalar standins and actual Gemmini Spike, with zero descriptor mismatches
and final ELF zero-FSM audit. The receipt is
`docs/perf_records/resnet_flat_wide_conv_spike.json`.

### Scratchpad bank placement

The stock configuration has four scratchpad banks of 4,096 rows (recorded in
our ZIP analysis). `separate_b_bank=True` places the B channel block at row
8,192, with the A panels in bank zero. Accumulator placement and every primitive
count remain identical. A/B footprints are checked for overlap and bounds;
there is no assumed asynchronous completion or weakened lifetime dependency.

For the late 7×7 C512 i32 fixture, this placement reduced GSIM kernel cycles
from **939,677 to 769,976**, an additional 18.1%, with all 25,088 outputs and the
guard exact. Strict Spike also passes. Queue job **1758** replaces the unstarted
1757 candidate. Its ELF SHA256 is
`8fc6572ea8551fdb04442110af1cba7459e4f61021ca3caa915ff4af33768384`.

The 14×14 C256 i8 fixture uses the captured positive scale
`0.0011732920538634062` and ReLU. Narrow/wide A schedules pass all 50,176 outputs
and guards at **791,611 / 701,344 GSIM cycles**. These shape-specific checks
support selecting wide A for both spatial sizes. See
`docs/perf_records/flat_conv_bank_gsim.json`.

### Requirements for automatic scheduling

The automatic search needs independently modeled spatial tiling, channel block
size, K-panel DMA width, and operand bank placement. Choosing only GEMM M/N/K
block sizes misses both the row-fragment gather cost and repeated convolution
weight traffic. A useful resource description includes scratchpad bank rows,
accumulator footprint, and explicit A/B lifetime intervals. Source layout proof
must stay separate: the gather schedule consumes an exact NHWC halo/HWIO
contract, and store fusion consumes the separately verified rounding proof.

These experiments provide a concrete cost-model correction: a larger gather
fragment count can be profitable when the entire image fits the accumulator
and weights are loaded once. Conversely, identical command counts can have an
18% timing difference solely from bank placement. Do not infer max performance
from MAC count or primitive count alone.


The separate-B-bank version also passes the full 14×14 i8 probe at **674,159
GSIM cycles**, versus 701,344 with collocated operands. The opt-in catalog
selector now records `spatial_flat_wide_a_separate_b` and uses this placement.
The resulting complete ResNet passes all 1,000 original outputs exactly on
native scalar standins and actual Gemmini Spike, with zero descriptor
mismatches and zero final FSM instructions. See
`docs/perf_records/resnet_flat_banked_conv_spike.json`.

## Complete-row accumulator bands

For feature maps too large to retain the complete output, the same scheduler
now uses bands of complete output rows. The band base is an ordinary CPU loop
induction variable. Row-fragment coordinates and the final short band retain
the exact source halo semantics; no input rows or channels are read out of
bounds. Stores use the original full-output row stride.

`choose_band_rows` considers every capacity-safe row count, minimizes total
padded M tiles, then minimizes weight passes. This prevents a greedy maximum
row count from wasting array tiles. For example, 28×28 chooses eight rows,
not nine: eight rows pack into 14 full tiles, with a four-row final band.

| Output and channels | Band rows | Bands | Spatial tiles | Computes | B DMA |
|---|---:|---:|---:|---:|---:|
|56×56, C64|4|14|196|28,224|504|
|28×28, C128|8|4|49|28,224|576|

Both shapes have a 451,584-cycle padded array issue floor. Weight reads drop
from 56 to 14 passes and from 28 to four passes respectively. Input/output
layout and reduction order are unchanged. The selector records
`spatial_banded_wide_a_separate_b`; the exact band height is also embedded in
the target IR as `gemmini.flat_conv_band_rows`, pinned by its compilation hash.

The numerical probe includes a stride-two shape with partial K and N tiles,
multiple channel blocks, nonzero halo values, and a short final band. Full-size
probes use the captured store scales and signed inputs spanning almost the
entire i8 range, so small scales still produce nontrivial output values.


The full-size banded kernels passed GSIM at **617,005 cycles** for 56×56 C64
and **628,112 cycles** for 28×28 C128, checking all 200,704 / 100,352 i8 outputs
and guards. The actual captured scales were 0.00206922204233706 and
0.0013393994886428118, with ReLU. Near-full-range deterministic signed inputs
and weights exercise meaningful narrowing. Tail probe and full-size receipts:
`docs/perf_records/banded_conv_gsim.json`.

The complete exact ResNet with all 16 direct 3×3 kernels using spatial-flat or
row-banded schedules passes native and actual Gemmini Spike with all 1,000
original captured outputs bitexact, zero descriptor mismatches, and zero FSM
instructions in the final ELF. Receipt:
`docs/perf_records/resnet_banded_conv_spike.json`.

The 16 convolution kernels now require 487,872 compute commands and a 7,805,952
cycle padded array issue floor, compared with 612,864 commands / 9,805,824 cycles
for the original row schedules. This is a convolution-only analytical count;
whole-model timing still includes dense kernels, stem, host layout/epilogue
work, and runtime overhead. No 22M whole-model performance claim follows from
these standalone gates.
