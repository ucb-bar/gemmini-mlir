# TinyLlama composed candidate: stock job 2121 rejected

Job 2121 completed on the selected stock FireSim Gemmini/Rocket bitstream. Its
inference ROI took **573,452,525 cycles**, versus **378,946,263** for champion
2085: **194,506,262 more cycles (+51.3282%)**. Keep 2085 as the best measured
whole-model implementation. This is a negative handwritten investigation result;
it grants no default-policy or fresh Phase 2 qualification.

The original 256,000-output gate passes: the trusted target runtime reports the
complete `f32le` SHA-256 and an exact one-word prefix matching the qualified
native output. That output satisfies the unchanged Torch gate
`atol=0.03125`, `rtol=0.02`. Full raw output words are absent from UART; this is
the existing complete digest gate. The final executable has no FSM instructions.
The simulator's total emulated-cycle counter includes setup and validation and
must be kept separate from the model ROI.

## What was actually compared

Both stock measurements use the same observed FPGA bit and one-entry HWDB
identity, eight tokens, 22 layers, original input/parameter identities, and the
source-owned inference bracket. The candidate is a fresh normal Merlin 52e
build with all 17 selected features, 155 device bindings, 44 typed current-source
scalar members, and a shared refined quadratic carrier. It restores the
203-chain late RNE legalization and uses an actual runtime RNE predicate per
observed point. Its scalar representation is an explicit approximate policy
with original-expression fallback; observed final bit equality is not a
universal exactness theorem.

The device kernel object is byte identical to the champion. The payload object
is reused unchanged. Host model, device adapter, startup and some runtime objects
change. The device kernel and four matched runtime/call objects are byte
identical; the other changed objects remain confounders. This comparison does
not isolate the scalar carrier.
Current source-derived memory facts and a nonexpanding empirical load/reservation
envelope were retained. Exact archived dirty-build source-to-elaboration memory
authority remains unknown.

## Functional counts identify the wrong baseline

| Executable / recipe | Functional retired count | Stock model cycles |
| --- | ---: | ---: |
| Champion 2085, original exact source observer | 122,699,096 | 378,946,263 |
| Fresh 458 exact source-observer build | 122,689,756 | Unknown |
| 52e approximate carrier before late RNE composition | 223,991,802 | Unknown |
| 52e composed carrier, job 2121 | 181,783,401 | 573,452,525 |

Restoring legalization reduces the approximate carrier's functional count by
18.8437%, but the resulting count is still 48.1538% above champion 2085.
**223,991,802 belongs to the uncomposed approximate carrier, not an exact-interval
control.** Spike counts are functional evidence and provide no hardware timing
certificate. The similar directions of the stock and instruction regressions
motivate examining the emitted host path; they do not apportion cycle causes.

## Binary and storage evidence

| Final ELF property | Champion 2085 | Candidate 2121 |
| --- | ---: | ---: |
| File bytes | 2,337,421,440 | 2,337,128,216 |
| `.text` bytes | 326,968 | 348,956 |
| `.rodata` bytes | 591,632 | 264,008 |
| `.eh_frame` bytes | 224 | 4,096 |
| `.bss` bytes | 1,126,040 | 1,126,040 |
| `.weights` bytes | 2,336,468,096 | 2,336,468,096 |
| Physical observer table bytes | 524,288 | 196,608 |

The current final link contains the actual coefficient block once in `.rodata`.
It materializes 44 point, 44 carrier, 44 source-expression fallback and 44
observer functions, totaling 45,056 named function bytes. This is an inventory,
not hot-code or dynamic-execution attribution. A point helper has a 32-byte
save frame and an out-of-line RNE call; the actual predicate is the 10-byte
`frrm`, `seqz`, `ret` body. There are 88 linked static predicate call sites.
These are structural costs that an inner-loop capsule must include.

The smaller table does not establish fewer cache misses. Current instruction,
table and live-data placement differs; no matched miss, replacement, fallback
frequency, exclusive-PC or per-region hardware attribution exists for 2121.
Call/frame/CSR service, across-call scheduling and cache/layout effects are
plausible causes that remain unmeasured individually.

## What the existing section timings establish

The older 2076/2080 profile has 179,740,352 callback cycles and 200,819,005
outside-callback cycles, with 163,498 cycles of profiler overhead. Its pre-down
and pre-o gaps are grouped by source location, not pure host cost or isolated
causes. Those earlier bodies are not bound to the current 2121 functions, so
their attribution cannot be transferred.

The separately completed affine18 capsule improves one complete original
producer-pair/observer ROI by 3.5093% but regresses its independent case by
44.1692%. Its source ROI includes two GEMMs, 90,112 i32 products and 45,056
observations. It uses a different representation and delivery path; it neither
prices the normal 44-member carrier nor licenses multiplying its saving by 22.

## Required matched successors

1. Reproduce the champion and normal exact-source observer under a common
   pinned compiler, runtime, provider, legalizer and input contract. Retain all
   changed objects and refuse unsupported numerical/effect conditions.
2. Compare exact source/interval and carrier implementations with the same
   producer, finishing observer, publication, tail, guards and runtime costs.
   Include the actual call frames, predicate, table reads and source fallback.
3. Separate call/visibility/scheduling experiments from representation changes.
   RNE may move to a wider region only with current complete FRM-stability and
   effect proof; constant-true admission or an assumed invariant is invalid.
4. Bind actual access sequences and competing live data/code to target cache
   facts before modeling misses. Calibrate with completed matching component
   RTL costs and retain independent regressions. Do not fit a model to this
   single full-model result or claim an isolated cache cause.
5. Require original full output and final-link/zero-FSM gates before promoting a separately
   frozen stock measurement.

Generic reification, helper visibility/scheduling and recipe-composition changes
belong in Merlin. Target predicate/ISA, resource facts and execution support
belong in the OOT backend. All choices must derive from current typed source,
effects, resources and explicit numerical permissions, without workload selectors.

[Negative receipt](perf_records/root_tiny_composed_stock2121_negative_20261008.json),
[champion result](perf_records/root_tiny_rne_zero_stock2085_terminal.json),
[historical profile](perf_records/root_tiny_current2076_profile_stock2080_terminal_20261007.json),
[scoped affine component](perf_records/tiny_affine18_complete_timing_20261008.json).
