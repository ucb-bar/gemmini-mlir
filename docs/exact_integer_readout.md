# Exact integer readout fallback

`mlir_oot/exact_integer_readout.py` derives every integer accumulator threshold at which the original ordered positive f32 scale chain, nearest-even rounding and i8 clamp changes output. Monotonicity and both sides of every transition prove the complete declared i32 domain. This supports ReLU or signed saturation; bias or arbitrary nonlinear chains are not inferred.

The baseline generated readout performs binary search over 127 ReLU or 255 signed thresholds (at most seven/eight comparisons). An optional integer fixed-point estimate is independently proved within one output step over the complete union of source and estimate transitions. It then checks the adjacent exact source thresholds and corrects by at most one step. The estimate may differ; the final output is source-exact. No approximate numerical policy is involved.

For closed-recipe `matmul_25` and `matmul_48`, shift 30 and multipliers 790059 and 733955 respectively satisfy this proof. Both use 127 i64 thresholds (1,016 bytes). Tests independently compile the original ordered f32 operations with contraction disabled, exhaust every integer in all unsaturated intervals, and check threshold neighbors and full i32 extrema. Positive-scale monotonicity proves the saturated tails. Both generated RV64 objects pass zero-FSM audits.

## Storage and integration contract

The wrapper accepts A, B, external i8 output, **caller-owned i32 scratch**, and its element capacity. The caller must provide distinct, exclusively owned scratch, and bind source proof/domain and layout. No mutable global scratch or allocation is hidden. The primitive convolution writes row-major i32 accumulators and must fence before returning. The adapter then executes exact integer readout.

For the two routes, scratch is 200,704 and 100,352 bytes. Exposing scratch as a write-only allocation in the calling IR allows liveness-based reuse between sequential calls without introducing shared races. The standalone module does not edit the source binder or infer scratch lifetime.

## Analytical cost, not a hardware result

The corrected estimate uses one 64-bit integer multiply, add, shift, clamp and at most two threshold checks per output, plus scratch load and i8 store. The i32 primitive output adds three bytes per element relative to direct i8 device readout: 225,792 additional device output bytes across 75,264 elements. CPU scratch reads add 301,056 bytes; output stores add 75,264 bytes. Binary search is a correctness fallback with more dependent table accesses. These are operation/traffic counts, not measured cycle estimates. Whole-model native/Spike and stock FireSim gates remain required before selection.
