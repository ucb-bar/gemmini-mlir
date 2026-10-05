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
