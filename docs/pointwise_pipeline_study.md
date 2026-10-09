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
| Prefetch bm4 | 94,304 | 94,089 (job 1736) |
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

## Captured-scale A/B numeric fixture correction

The first captured-scale probes (jobs1745/1746) used the original small integer operand fixture. At the exact captured scale0.0038317402359098196 with no bias and ReLU, all200,704 expected outputs rounded to zero. Both jobs were canceled while still queued; no measurement is claimed.

Replacement jobs1752 (bm16 baseline) and1753 (banked prefetch bm1) use amplitude13 deterministic signed-i8 operands. Expected outputs now contain28 distinct values in0..58, with102,625 nonzero entries. All200,704 values and guard bytes are checked. Device object hashes remain identical to the original captured-scale kernels; only harness inputs/oracle changed. Final ELF no-FSM audits pass. Added `--input-amplitude` is bounded1..21 so both deterministic input patterns remain signed-i8 representable; default1 preserves existing fixtures.

## Captured-scale hardware comparison

Both corrected-fixture jobs completed with all200,704 varied outputs and guard
bytes passing, final zero-FSM ELFs, and actual staged ELF/bitstream hashes
verified. Bias is disabled, ReLU enabled, and output scale is the exact
source-proven0.0038317402359098196.

| Schedule | Stock FireSim cycles |
|---|---:|
| Current bm16, job1752 |90,323|
| Banked prefetch bm1, job1753 |51,922|

The source-compatible candidate cuts kernel cycles42.52% (1.740x speedup),
ending3.48% above the50,176 compute floor. This supports selection for this
exact source-bound epilogue/shape, not unrestricted promotion to other shapes.

## Late convolution hardware gate

Job 1758 measured **793,557 kernel cycles** for the H7/W7/C512 3x3 direct convolution with spatial flattening, four M tiles, bn16, 64-wide A loads, and B at scratchpad row 8192 on a separate bank. All 25,088 i32 outputs and guard bytes passed; staged ELF/bitstream identities and final zero-FSM audit passed. GSIM measured 769,976 cycles for this artifact. This is a standalone kernel measurement; no same-hardware baseline or whole-model speedup is inferred. Receipt: `docs/perf_records/late_conv_banked_firesim1758.json`.

## Large-N dense hardware gate

Stock FireSim jobs 1763/1764 compared M8/N2048/K2048 i32 GEMM with bm1/bn64 and no bias. Wide B loads (64 columns per group) plus cached A reduced kernel cycles from **1,445,879** to **925,141**: **36.02% fewer cycles**, **1.563x** speedup. All 16,384 outputs and guards passed in both, with final zero-FSM ELFs and staged ELF/bitstream identities verified. Candidate GSIM measured 927,138 cycles. This is a representative kernel gate for Tiny/Smol dense scheduling; whole-model speedups and other shapes still require independent validation. Receipt: `docs/perf_records/large_n_dense_firesim1763_1764.json`.

## Original-capture hoisted whole ResNet

Job 1767 measured **2,461,226,911 forward cycles** with all 1,000 original outputs bit-exact. Final zero-FSM ELF and actual staged ELF/stock bitstream identities passed. This is 7.58% below pooled-stem job 1750; the variant combines weight hoisting with selected bank/flat/pooled schedules, so the improvement is not attributed solely to hoisting. Receipt: `docs/perf_records/resnet_original_hoisted_firesim1767.json`.
