# Pointwise software pipeline study (2026-10-05)

Reference: own extracted ZIP analysis at `/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-studies/exo-comparison/q1013_analysis.md`, 1x1 `fx_conv_2` section. No private source folders accessed.

## Baseline evidence

FireSim job1725, exact stock FireSimGemminiRocketConfig, M3136 N64 K64, i8 scale0.125/ReLU with bias: **105258 kernel cycles**, every output+guard passes. Kernel ELF SHA610f2a86e26af94734cfe1bbe5d9ce8420e7b5ce55cf74ae14d43376d1ee1247. Corresponding GSIM result104931cycles; issue floor50176cycles. Baseline bm16/bn4 caches B, uses wide A/B transfers and wide stores, reuses stationary B across row tiles, but has no double buffering. This is a synthetic layer geometry and is not a full ResNet performance result.

## Candidate mechanism

Existing `pipeline_m` alternates A and accumulator slots, but begins the next block's loads only after all current computes and stores. New optional `prefetch_m` prepares the next block's A and bias after the first half of current K computes. The CPU loop has an explicit initial fill, alternating pair body and final drain. The next block uses independent A/accumulator storage. Bias loads issue through load unit2. Hardware dependency tracking still orders each storage reuse; no new hardware LOOP commands or inner fences are emitted.

Optional `banked_m` places A slots at scratchpad rows0/4096 and cached B at8192, with accumulator slots0/512, matching the reference's physical-bank separation. Stock `gemmini_params.h` declares4 scratchpad banks of4096rows; the ZIP's matched configuration has2 accumulator banks totaling1024rows. Two accumulator slots limit bm to8 when bn4. This means the comparison also measures the tradeoff between stationary-weight reuse at bm16 and DMA overlap at bm4/bm8.

## Validation

- Existing28 unit tests pass after schedule changes.
- GSIM144x64x64, bm4, prefetch without bank separation: PASS5449cycles, every output+guard, final ELF noFSMpass.
- GSIM208x64x64 bank-separated bm4 edge/drain: PASS6720cycles, all outputs+guard.
- GSIM48x64x64 bank-separated bm1 odd-block drain: PASS2353cycles, all outputs+guard.
- Full3136 shape unbanked bm4: GSIM PASS94304cycles (10.1% below104931baseline); unbanked bm8: PASS96414cycles (8.1% below baseline). Both validate all200704 outputs+guard and final noFSM. Bank-separated bm8 GSIM still running. FireSim1733 measures bank-separated bm8;1734 measures bank-separated bm1;1736 measures measured-GSIM-winner unbanked bm4 after full-model jobs. Final ELF audits pass for compiled candidates. Do not promote a candidate before numeric validation and a hardware A/B win.
- Old toolchain Spike rejected the generic layer probe CRT at tohost1337, with and without explicit memory range. This is not used as a kernel correctness result; GSIM and FireSim perform the actual numerical checks.

## Needed automatic lowering abstractions

The automatic scheduler needs an explicit lifetime for each A and accumulator slot, bank-aware local allocation, and a schedule position for next-block DMA/bias initialization. Simply marking a loop double-buffered does not express when transfers must issue. Prologue/drain generation must handle odd block counts and partial row tiles. Cost modeling must weigh overlap against smaller output blocks and stationary-weight reuse. Selection must use measured kernel cycles and preserve the exact scale/bias/ReLU semantics.
