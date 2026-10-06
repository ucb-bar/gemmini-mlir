# Source-bound accumulator stripes

The capture compiler accepts `dense_accumulator_stripes=True` (CLI
`--dense-accumulator-stripes`) as an explicit, disabled-by-default schedule choice.
The admission function reads the existing typed dense schedule: complete cached A,
scaled byte wide stores, zero accumulator seed, and a narrow output tile group.
It proves complete A fits the lower two scratchpad banks, complete increasing-K B
for four output tiles fits one upper bank, the A/B live intervals are disjoint,
and each bounded M stripe fits the accumulator. Segmented inputs and incompatible
slot/coalescing options refuse this choice. The three-pointer ABI and immutable
A/B, disjoint fully written C obligations remain explicit.

The ordinary capture command reaches the admission function and emitter. Its
manifest records these owners on that command and exposes admission/emission edit
surfaces. The shared optimization solver does not choose this experimental option;
profitability remains unknown until actual measurements. No model name, source
ordinal, region name, or captured value selects the strategy.

## Original source closure

The normal 52-contraction source bundle applies the rule to four M784/N512/K128
pointwise contractions. All original numeric proofs, semantic dimensions, scales,
activation and bias policies remain unchanged. The rewritten source MLIR and all
52 adapter objects reproduce the control bytes; 48 unselected kernel objects also
reproduce the control. The projection with a larger complete A refuses the bank
bound. Six existing default dense kernel objects reproduce their previous bytes.

The separate original-input capsule completed all 401,408 outputs, guards, and
unchanged inputs on strict Gemmini Spike and the pinned GSIM engine. The common
address comparison measured 298,600 to 251,981 GSIM cycles. Actual command traces
show 648 to 162 input loads and 1,568 to 392 stores, with 12,544 compute/preload
commands unchanged; real-B mesh reloads increase from 256 to 1,024. Requested
payload bytes remain unchanged. Independent non-square i32 and scaled-i8 tail
cases passed their complete gates. These are capsule measurements, not whole-model
FireSim predictions. The earlier M3136/N256/K64 stripe experiment regressed by
48–51% and remains a separate rejected schedule.

## Whole-model qualification

Both a fresh ordinary source build and a controlled link with the frozen actual
1903 host/runtime pass the unchanged original whole-model gate: every one of the
1,000 binary32 words is exact, in native execution and actual Gemmini Spike.
The controlled link replaces only the four selected kernel leaves. Its control ELF
is byte-identical to 1903; adapters, SAT8 providers, residual, stem, mean, host,
runtime, weights and shim objects are preserved. Final executable audits reject
FSM instructions. The inherited build marker is nonunique, so the full ELF SHA
is authoritative. Whole FireSim performance remains unknown before stock execution.
