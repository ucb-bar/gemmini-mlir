# Output guard compiler screens

Each row is a complete matched pair on the pinned Gemmini GSIM engine. The
timer includes predictor input/coefficient DMA, one scaled store, fence and
the entire correction scan. All outputs equal the original source; raw target
predictions, immutable inputs, dirty guards, strict Spike and final no-FSM
audits pass.

| Correction compilation | Elements | Current exact control | Candidate | Change |
|---|---:|---:|---:|---:|
| GCC, inline helper | 65,536 | 164,387 | 169,918 | +3.36% |
| GCC, inline helper | 3,072 | 8,092 | 8,932 | +10.38% |
| GCC, cold word helper | 65,536 | 164,469 | 164,511 | +0.026% |
| Clang, cold word helper | 65,536 | 164,399 | 141,216 | −14.10% |
| Clang, cold word helper | 3,072 | 8,143 | 7,770 | −4.58% |

The ordinary support harness compiles C with provider GCC. The normal ranked
device adapter is compiled with Clang. The Clang rows explicitly precompile
the portable correction into its own object with the normal strict FP flags;
the receipt pins the exact compiler, argv, C source and object. The harness
continues to use GCC.

The generic cold word helper computes the packed endpoint once and isolates
eight-byte ambiguity checks behind a function call. The first word test was
already an existential byte match; there was no exact lane-mask scan to remove.
Observed hot-loop instruction counts differed by compiler. Retired instruction
count alone did not establish timing: the GCC cold helper came close to parity
despite executing more instructions in the measured region.

Within a row both ELFs differ only in one selector byte, with identical code
and data addresses. Across rows the compiler, code placement and object
contents differ; the differences across those rows are not an isolated causal
compiler estimate. GSIM and stock FireSim have different memory regimes.

[The receipt](perf_records/residual_output_guard_compiler_capsules.json)
independently rechecks 100 pins, all complete terminal correctness markers,
the selector-byte ledgers, certificates and executable audits. The original
802,816-element pair remains a separate running experiment. These small
positive rows do not establish a whole-model or stock-cycle improvement.
