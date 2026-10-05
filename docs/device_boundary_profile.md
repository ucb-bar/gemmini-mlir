# Device-boundary attribution without changing model lowering

`python -m mlir_oot.golden_device_profile` is an optional final-link diagnostic. It reuses the existing optimized `model.o`, runtime objects and device catalog object and emits `--wrap` adapters. The ordinary build is unchanged. No profiling operations enter MLIR, so adjacent elementwise fusion and layout rewriting see the same source as the unprofiled build.

The direct ResNet build has 17 dense kernel symbols and 16 direct-convolution C-interface adapters. Dense wrappers time the three-pointer device kernel ABI. Direct wrappers time the four-pointer C-interface adapter, including its descriptor checks, because adapter+kernel can share a partial-link object and wrapping that object's internally resolved kernel reference is unreliable. All 54 dynamic device calls must be observed. `merlin_run_multi` defines the forward interval; the exit wrapper emits the compact dump after model output and metrics, with console batching enabled.

`PROFILE_SUM` reports forward, device-boundary, host-gap counters, call count and overflow. `PROFILE_CALL` records ordinal, symbol ID, preceding host gap, and device-boundary counter. `PROFILE_TAIL` is the final host gap. The parser requires every declared boundary, exact expected call count, sequential ordinals, no overflow, and conservation of both device and host intervals. The build receipt maps IDs to symbols, hashes every reused object and records the exact final ELF; the final zero-FSM audit is mandatory.

Instrumented timings remain diagnostic. Wrapper overhead is mostly assigned to host gaps, and code placement may change. Do not use profiled timing as the uninstrumented performance claim.

## Validated direct ResNet profile

Base ELF: 4ab852cd4b8b8037fe309076062bca23f3ff8e7f051d2d3269dfbd9ab6f4e14c.
Profile ELF: 4367b3b79576a42e80e46e2aa0977ba7ca417da793f45cbc52fcb9a696696d4e.

Actual Gemmini Spike: all 1,000 output words match the captured oracle bit-for-bit, rank mismatches 0, all 54 calls and 33 unique boundaries present, interval conservation passes. The forward counter is 1,193,352,331 versus 1,193,349,750 without profiling (+2581 retired instructions). Device-boundary counter 4,724,660; host-gap counter 1,188,627,671. These are functional Spike instruction counts, not hardware performance. FireSim job 1741 measures the same profile with the verified stock config.

Largest instruction-count gaps precede the stem (131,594,925) and the first pointwise contraction after the stem (137,117,632). Per-call gaps identify where to inspect host layout, padding, pooling and quantization code; they do not identify which individual operation is responsible without further source attribution.

## Automatic-loop infrastructure consequence

Keep profiling at existing call boundaries or after optimization. Inserting opaque profiler calls before every upstream op can inhibit fusion and measure a different program. Preserve provenance from each device call to its source region so a host gap can be mapped to intervening operations. Include final ELF identity and wrapper overhead in benchmark records.
