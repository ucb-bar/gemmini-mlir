# Executed feature evidence for fast performance screening

`mlir_oot.executed_features` decodes a complete Spike PC histogram against its
pinned final ELF. It reuses Merlin's executable-section and instruction walkers,
plus the target's strict final no-FSM audit and recorded command facts. Feature
paths bind the existing shared analytical provider; this module fits no cycles.

The counted features are retired instructions and actual primitive command
classes. Unexecuted primitive instructions contribute zero. Compressed instruction
boundaries, function extents, duplicate/truncated histogram rows and provenance
mismatches are checked. Function scopes are explicit PC unions. The decoder never
follows an alias or divides a shared function's aggregate by assumed calls.

A histogram does not include register operands or temporal order. Physical array
work, requested DMA bytes, DRAM traffic, bank hazards and dispatch overlap remain
unknown. PCs outside the ELF make primitive completeness unknown. A function
filter alone proves neither harness exclusion nor a matching hardware timer.

## Independent reference evaluation

`perf_records/q1013_reference_heldout_layer_features.json` joins the 54 buffered
layer intervals from the same diagnostic ELF in Spike and stock FireSim job 1876.
Every hardware label has **heldout evaluation only** status. No reference timing
may be used to fit the independent estimator.

| Scope | Retired instructions | Stock cycles |
| --- | ---: | ---: |
| 54 convolution/matmul intervals | 11,859,857 | 20,143,339 |
| Bracketed forward, including residual/other aggregates | 12,289,279 | 22,347,798 |
| Full model interval, including uncounted glue | 12,301,541 | 22,387,449 |

The whole-program histogram contains 231,665,151 retired instructions, largely
post-inference diagnostic dumping. It cannot be paired with the 22M inference
timer. The dataset pins the strict reference replay, engine, stdout, histogram,
stock ELF/bitstream/receipt and source geometry pairing. The reference diagnostic
uses its authorized Zicntr contract; production model gates retain strict RV64GC.

`perf_records/q1013_reference_physical_function_features.json` contains separate
physical function census records. These omit outside callees and wrapper/glue
instructions, so their PC scopes are not declared equal to the summed layer
hardware intervals. Source geometry is logical shape evidence, not a substitute
for physical command operands. Reference and current captures have different
numeric/input/weight contracts.

## Verification

Focused tests cover weighted executed commands, zero execution, compressed
instructions, missing/ambiguous/outside symbols, instruction-splitting extents,
truncated/duplicate/invalid histograms, mismatched execution bindings and dead
FSM words in the final ELF. The default model build and qualified device objects
are unchanged.

`perf_records/resnet1874_executed_layer_features.json` independently joins our
70 conserved ABI boundary intervals from a fresh strict Spike replay with stock
profile job 1899. The original 1,000-word output digest, complete call order and
forward conservation pass. Its forward scope is 10,226,219 retired instructions
and 39,235,729 hardware cycles. Primitive body counts are separately attributed;
no physical work or transfer bytes are invented before operand telemetry. This
is historical control 1874, rather than a newer policy or host/runtime candidate.

## Observed CPU roles and code footprint

`mlir_oot.cpu_opcode_census` decodes RV64GC encoding classes, including compressed
instructions, FP conversions/arithmetic, integer multiply/divide, memory, branch,
CSR and custom commands. It records weighted retired counts, unique executed PCs
and their actual encoded bytes. These are observed features; they imply neither
opcode latency, instruction-cache misses nor CPU/accelerator overlap. Cache-line
footprints remain unknown without a pinned cache geometry.

`perf_records/resnet1874_cpu_operand_features.json` joins all 70 unique source
body invocations to the existing operand and stock timing dataset. CPU-class
counts conserve 6,624,960 body instructions; their measured wrapper intervals
include another 173 instructions whose hardware cost is unknown.

`perf_records/q1013_reference_grouped_cpu_operand_features.json` groups all 54
reference intervals by their 24 physical functions. This conserves 11,858,233
body instructions and the 1,624-instruction wrapper difference. An aggregate PC
histogram cannot attribute data-dependent paths separately to repeated calls;
their per-invocation CPU classes remain unknown. Every reference timing label
remains held out of coefficient fitting. `tests/cpu_feature_probe.py` re-closes
all replay, scope and source-dataset pins before exporting these records.

The stem bodies illustrate the code-footprint distinction: our historical body
touches 58,831 PCs / 174,566 instruction bytes, compared with reference stem
7,551 PCs / 24,042 bytes. Different input/numeric contracts and schedules prevent
causal attribution. Smaller code or fewer instructions alone does not establish
lower cycles; the rejected loop schedule remains negative evidence.

Validation: 88 focused tests and six subtests pass, covering encoding roles,
compressed boundaries, weighted conservation, exact footprint bytes, source
scope refusals and no-FSM gates. The exporter observes zero unknown CPU encodings
in both qualified model datasets. Qualified default model/device bytes remain
unchanged.
