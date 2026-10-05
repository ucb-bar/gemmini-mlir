# Pointwise software pipeline study (2026-10-05)

Reference: own extracted ZIP analysis at `/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-studies/exo-comparison/q1013_analysis.md`, 1x1 `fx_conv_2` section. No private source folders accessed.

## Baseline evidence

FireSim job1725, exact stock FireSimGemminiRocketConfig, M3136 N64 K64, i8 scale0.125/ReLU with bias: **105258 kernel cycles**, every output+guard passes. Kernel ELF SHA610f2a86e26af94734cfe1bbe5d9ce8420e7b5ce55cf74ae14d43376d1ee1247. Corresponding GSIM result104931cycles; issue floor50176cycles. Baseline bm16/bn4 caches B, uses wide A/B transfers and wide stores, reuses stationary B across row tiles, but has no double buffering. This is a synthetic layer geometry and is not a full ResNet performance result.

## Candidate mechanism

Existing `pipeline_m` alternates A and accumulator slots, but begins the next block's loads only after all current computes and stores. New optional `prefetch_m` prepares the next block's A and bias after the first half of current K computes. The CPU loop has an explicit initial fill, alternating pair body and final drain. The next block uses independent A/accumulator storage. Bias loads issue through load unit2. Hardware dependency tracking still orders each storage reuse; no new hardware LOOP commands or inner fences are emitted.

Optional `banked_m` places A slots at scratchpad rows0/4096 and cached B at8192, with accumulator slots0/512, matching the reference's physical-bank separation. Stock `gemmini_params.h` declares4 scratchpad banks of4096rows; the ZIP's matched configuration has2 accumulator banks totaling1024rows. Two accumulator slots limit bm to8 when bn4. This means the comparison also measures the tradeoff between stationary-weight reuse at bm16 and DMA overlap at bm4/bm8.

## Validation

Existing 28 unit tests pass. GSIM edge tests validate odd block counts and drains: 144x64x64 unbanked bm4 passes at 5,449 cycles; 208x64x64 banked bm4 passes at 6,720; 48x64x64 banked bm1 passes at 2,353. Every probe checks all outputs and guard bytes, and every final ELF passes the zero-FSM audit.

Full 3136x64x64 measurements use identical i8 bias/scale/ReLU semantics:

| Schedule | GSIM kernel cycles | Stock FireSim kernel cycles |
|---|---:|---:|
| Baseline bm16 | 104,931 | 105,258 (job 1725) |
| Prefetch bm4 | 94,304 | Pending job 1736 |
| Prefetch bm8 | 96,414 | Not submitted |
| Prefetch with separate banks, bm8 | 91,250 | **92,101 (job 1733)** |
| Prefetch with separate banks, bm1 | Full shape not run | **53,624 (job 1734)** |

Job 1733 is DONE with all 200,704 outputs and guard bytes passing, exact ELF and bitstream identity verified. Its 12.50% reduction (1.143x speedup) reproduces the GSIM gain on hardware. These i8 epilogue measurements do not establish the performance of the current upstream catalog's i32-output schedule; that needs its own measurement or a proven epilogue fusion.

The old toolchain Spike rejected the generic layer probe CRT at tohost 1337, with and without explicit memory range. This is not used as a kernel correctness result; GSIM and FireSim perform the numerical checks.

## Needed automatic lowering abstractions

The automatic scheduler needs an explicit lifetime for each A and accumulator slot, bank-aware local allocation, and a schedule position for next-block DMA/bias initialization. Simply marking a loop double-buffered does not express when transfers must issue. Prologue/drain generation must handle odd block counts and partial row tiles. Cost modeling must weigh overlap against smaller output blocks and stationary-weight reuse. Selection must use measured kernel cycles and preserve the exact scale/bias/ReLU semantics.

## Device compilation observation

LLVM object disassembly has substantial stack traffic in these generated kernels. Static counts over the entire function (including setup/drain, so not dynamic performance attribution): baseline bm16 1527 instructions/108 stack accesses; prefetch bm4 1447/150; prefetch bm8 2623/199; banked bm1 458/71. The bm4 prologue reserves416 stack bytes. Device compilation currently uses clang `-O2`. These counts justify inspecting register allocation and constant-descriptor lifetimes before assuming the remaining gap is purely DMA/array latency. A future automatic scheduler should report hot-loop spill counts and instruction footprint alongside Gemmini primitive counts; static whole-function counts alone cannot establish the cause of stalls.

## Bank-separated bm1 result

Job 1734 completed with all 200,704 values and guard bytes passing: **53,624 kernel cycles**, a 49.05% reduction versus baseline and 6.87% above the 50,176-cycle array compute floor. The actual simulator ELF and bitstream match the committed identities. Small output blocks expose overlap effectively on this exact geometry despite less stationary-weight reuse. The final ELF has zero FSM instructions. This does not yet establish the same gain for the upstream i32-output catalog or other K values.
