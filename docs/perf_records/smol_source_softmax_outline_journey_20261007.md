# Actual SmolVLA source provider attribution

This is a diagnostic of the measured `b15444e0b804a416389e9bf06e91b6aebb582d749f15bb4b4a87fbceeb6c9064`
executable and its existing whole Spike PC histogram. No model run was repeated
to produce the attribution. Functional instruction counts do not establish FPGA
cycles, cache misses, memory latency, or removable work.

## Exact binary and existing compiler tooling

Only `-gline-tables-only` was added to the original provider compilation. Every
allocated object section and normalized relocation remains identical. The exact
original link command was reused with that object: every allocated final ELF
byte, address, section attribute and function extent matches the measured ELF.
The debug image hash is
`b6cb772e507eb863139af346d118e5db7ec9bcd84864c5cd4bc84815e1d8a181`.

The existing generic `merlin.perf.debug_companion.verify_debug_companion`
independently admits both actual/debug objects (4 allocated sections, 1,758
relocations) and final images (7 allocated sections). Existing
`attribute_symbolized_pcs` independently conserves all leaf-function totals.
No duplicate compiler library was introduced. Target instruction/operand
decoding is confined to the OOT diagnostic owner.

Four disjoint linked owners account for **75,020,325,903** original instructions.
The compiler does not supply a leaf source line for **12,395,074,821** of them;
these remain explicit. Validated inline function ancestry can still assign an
exclusive phase when its leaf line is absent. Shared external callees remain
outside each owner; they are never added to both a caller and another row.

## Largest exclusive source phases

| Source phase | Actual instructions | Whole functional share |
|---|---:|---:|
| Softmax interval, BF16 observation, lanes and source scheduling | 18,859,605,380 | 16.032% |
| Canonical radix packing and equality witness | 9,616,477,633 | 8.175% |
| Original polynomial endpoint evaluation | 8,039,584,237 | 6.834% |
| Original rigorous product-bound fallback, including norms | 6,610,663,433 | 5.620% |
| Input descriptor gather, BF16 widening and masks | 5,127,153,984 | 4.358% |
| RMS4 point-product admission and estimate construction | 4,878,868,918 | 4.147% |
| Exact five-plane integer reconstruction | 4,567,693,824 | 3.883% |
| Endpoint interval and original source DAG | 3,800,033,225 | 3.230% |
| Final BF16 scale/quantizer observation and certification | 2,592,327,993 | 2.204% |
| Denominator ordered QK replay arithmetic | 2,352,209,280 | 2.000% |
| Produced PV endpoint span preparation | 2,228,402,799 | 1.894% |
| Original center row/column scaling | 1,872,142,848 | 1.591% |

Ordered dot replay across initial softmax QK, denominator QK and partial PV
contributes **3,260,255,388** exclusive instructions. It is insufficient to
explain the remaining whole-model gap by itself. Required source arithmetic,
observer metadata and repeated private representation work coexist within these
phases; row totals are not savings estimates.

## Current-input coverage closes two shortcuts

A separate native diagnostic adds counters after the unchanged polynomial batch
and before existing BF16 observations. All 48 source calls, 12 preparations,
23,040 degree callbacks and the original 1,600 output words pass bit exactly.
The original accuracy gate remains `atol=0.03125`, `rtol=0.02`.

| Current-input fact | Count |
|---|---:|
| Score cells | 150,994,944 |
| Masked cells / wholly masked batches | 0 / 0 |
| Bitwise identical score endpoints | 205,173 (0.1359%) |
| Bitwise identical probability endpoints | 205,515 (0.1361%) |
| Probability endpoints in the same BF16 bin | 149,298,844 (98.8767%) |
| Ambiguous BF16 probability bins | 1,696,100 |

Masked-batch skipping and equal-endpoint sharing are therefore deprioritized
without implementation. BF16 probability agreement does not close the live F32
denominator observation, whose original lane/tree/FMA order remains required.
Historical one-endpoint, derivative/row-radius and prefix-table negatives are
retained; these are not reintroduced under new names.

## Stack addressing and a function-placement hypothesis

Actual RV64GC instruction operands show **4,971,632,913** direct-SP loads/stores
inside the four owners. Pinned Spike encodings and LLVM disassembly agree at
every selected PC. Softmax interval/scheduling contributes 3,017,406,402 and
polynomial evaluation 1,132,386,177. These accesses include frame data, saves,
locals and pointer reloads; the diagnostic does not label every access a spill.
Derived stack addresses, frame aliases and cache effects remain unknown.

The next isolated screen places the complete owned softmax body out of line.
Every source operation, signature, memory effect, FENV policy, refusal and
fallback is retained. Complete source-group cost includes call and ABI overhead.
Any reusable implementation belongs in Merlin and must select structurally from
function/callgraph/liveness/effects facts. The existing source symbol is only an
explicit experimental binding. No workload name, source hash, golden value or
measured coordinate may act as a compiler selector.

That screen has now completed. The original complete 12-head source group
changes from **1,515,040,109** to **1,506,374,301** functional instructions,
a **0.571985%** reduction including all call/ABI overhead. Native all-48 original
1,600 words are bit exact; the target ELF has zero FSM instructions. The compiled
original i8/BF16-scale consumer, guards and all eight source statistics match.
This is a small independently qualified placement result. No generic placement
topic, whole-model speedup or hardware timing is promoted from it.

An architectural candidate is a closed row-domain partition across all heads,
retaining each original per-row polynomial, eight-lane reduction, tree, alpha
FMA and cross-head scale/quantizer observation. It could shorten the lifetime
of full-row private score/probability/endpoint arrays in the 123,012,928-byte
workspace. It requires complete source-use/effects/fallback closure and product
providers that accept smaller internal row domains; the current frozen M=256
callbacks do not supply that capability. Smaller footprint alone is not a
measured traffic or timing gain.

## Evidence and compiler search requirements

Receipts are under `out/artifacts/probes/source-provider-pc-attribution-20261007/`:
`equivalence.json`, `qualification.json`, `independent_reclosure.json`,
`source_phase_census.json`, and `direct_sp_census.json`. Coverage is separately
sealed in `out/artifacts/probes/softmax-endpoint-coverage-20261007/qualification.json`.
Original source owners and receipts remain unchanged.

Phase 0 needs complete typed source observations and effect/ownership closure.
Phase 1 should use existing debug-companion tooling, source-to-compiled provenance,
register/liveness diagnostics and explicit source-component placement parameters.
Phase 2 must price the entire composition, including preparation, original
denominator continuation, fallback, callback and output validation costs. This
uses the same shared compiler architecture as other models, while current
experimental revisions and frozen leaves differ; no single automatic recipe is
qualified across all three workloads.

Exact per-change token billing is unavailable to this worker. No aggregate
session meter is assigned to the Gemmini topic. Diagnostic steps, hypotheses,
negative results, ownership and measured scopes are recorded instead.
