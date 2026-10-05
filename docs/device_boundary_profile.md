# Device-boundary attribution without changing model lowering

`python -m mlir_oot.golden_device_profile` is an optional final-link diagnostic. It reuses the existing optimized `model.o`, runtime objects and device catalog object and emits `--wrap` adapters. The ordinary build is unchanged. No profiling operations enter MLIR, so adjacent elementwise fusion and layout rewriting see the same source as the unprofiled build.

The direct ResNet build has 17 dense kernel symbols and 16 direct-convolution C-interface adapters. Dense wrappers time the three-pointer device kernel ABI. Direct wrappers time the four-pointer C-interface adapter, including its descriptor checks, because adapter+kernel can share a partial-link object and wrapping that object's internally resolved kernel reference is unreliable. All 54 dynamic device calls must be observed. `merlin_run_multi` defines the forward interval; the exit wrapper emits the compact dump after model output and metrics, with console batching enabled.

`PROFILE_SUM` reports forward, device-boundary, host-gap counters, call count and overflow. `PROFILE_CALL` records ordinal, symbol ID, preceding host gap, and device-boundary counter. `PROFILE_TAIL` is the final host gap. The parser requires every declared boundary, exact expected call count, sequential ordinals, no overflow, and conservation of both device and host intervals. The build receipt maps IDs to symbols, hashes every reused object and records the exact final ELF; the final zero-FSM audit is mandatory.

Instrumented timings remain diagnostic. Wrapper overhead is mostly assigned to host gaps, and code placement may change. Do not use profiled timing as the uninstrumented performance claim.

## Validated direct ResNet profile

Base ELF: 4ab852cd4b8b8037fe309076062bca23f3ff8e7f051d2d3269dfbd9ab6f4e14c.
Profile ELF: 4367b3b79576a42e80e46e2aa0977ba7ca417da793f45cbc52fcb9a696696d4e.

Actual Gemmini Spike: all 1,000 output words match the captured oracle bit-for-bit, rank mismatches 0, all 54 calls and 33 unique boundaries present, interval conservation passes. The wrapper forward counter is 1,193,352,331. The original harness metric is 1,193,352,377 versus 1,193,349,750 without profiling (+2,627 retired instructions). Device-boundary counter 4,724,660; host-gap counter 1,188,627,671. These are functional Spike instruction counts, not hardware performance. FireSim job 1741 measures the same profile with the verified stock config.

Largest instruction-count gaps precede the stem (131,594,925) and the first pointwise contraction after the stem (137,117,632). Per-call gaps identify where to inspect host layout, padding, pooling and quantization code; they do not identify which individual operation is responsible without further source attribution.

## Automatic-loop infrastructure consequence

Keep profiling at existing call boundaries or after optimization. Inserting opaque profiler calls before every upstream op can inhibit fusion and measure a different program. Preserve provenance from each device call to its source region so a host gap can be mapped to intervening operations. Include final ELF identity and wrapper overhead in benchmark records.

## Unprofiled hardware baseline for attribution

Stock FireSimGemminiRocketConfig job 1737 completed with **3,529,465,283 forward cycles** for the mixed direct16+dense38 ResNet artifact (ELF SHA `4ab852cd4b8b8037fe309076062bca23f3ff8e7f051d2d3269dfbd9ab6f4e14c`). All 1,000 output words are bit-exact against the captured integer oracle; rank mismatches are zero. Both actual staged ELF and actual bitstream hashes were checked before teardown. The immutable job-bound receipt is `docs/perf_records/resnet_direct_firesim1737.json`.

This is 21.85% fewer cycles than fused-host job 1731 (4,516,405,461) and 37.87% fewer than initial baseline job 1730 (5,680,463,426). It remains far above the 22M goal. The paired boundary-profile job 1741 uses unchanged model/device objects with final-link wrappers; use its hardware host gaps and completed device calls to attribute the remaining cost. Spike retired-instruction shares are not hardware cycle shares.

## Verified hardware attribution (job 1741)

All 1,000 output words are bit-exact; actual staged ELF/bitstream identities match. All 54 completed device calls cover the expected 33 symbols, with zero overflow and exact interval conservation. The labeled per-call record is `docs/perf_records/resnet_direct_boundary_profile_firesim1741.json`.

| Interval | Hardware cycles | Forward share |
|---|---:|---:|
| Wrapped forward | 3,541,290,278 | 100% |
| Completed device calls | 61,849,668 | 1.75% |
| Host gaps, including wrapper overhead | 3,479,440,610 | 98.25% |

Harness METRIC cycles is 3,541,290,794, versus the unprofiled 3,529,465,283: +11,825,511 cycles (0.335%). Final-link instrumentation perturbs timing, so individual intervals should be interpreted with that limitation. No host-gap figure identifies a single op: each gap includes preceding epilogues and next-call preparation.

Largest host gaps precede ordinal1 (282,134,606), ordinal5 (255,400,597), stem ordinal0 (232,533,325), ordinal8 (173,283,746), ordinal11 (157,251,743), and classifier ordinal53 (152,500,167). Eliminating host format/quantization/layout work is the principal whole-model priority. Device work itself is still above the 22M goal: directconv13/14/15 each take about 6.12M cycles, and directconv7..12 each about 2.34–2.40M; those deeper convolutions are the next device schedule targets.

## Fused epilogue whole-model result

Job 1743 (source-proven fused27 epilogues) completed at **3,146,164,937 forward cycles**, with all 1,000 output words bit-exact against the same original oracle, zero rank mismatches, and actual staged ELF/bitstream hashes verified. This is 10.86% below mixed direct baseline 1737 and 44.61% below initial baseline 1730. Receipt: `docs/perf_records/resnet_fused27_firesim1743.json`. The 1741 attribution applies to the earlier mixed artifact; it is not a per-section profile of this fused artifact.

## Pooled-stem whole-model result

Job1750 completed at **2,663,212,150 forward cycles**, all1,000 output words bit-exact against the same original oracle, zero rank mismatches, and actual staged ELF/bitstream identities verified. This is15.35% below fused27 job1743 and53.12% below initial job1730. The artifact combines one pooled stem,27 fused epilogues,4 remaining direct convolutions and22 dense contractions. Receipt: `docs/perf_records/resnet_pooled_stem_firesim1750.json`. This is measured whole-model progress, still far above22M cycles.
