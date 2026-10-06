# Faster performance screening

## Current evidence: 2026-10-06

Performance modeling now has dedicated root and calibration work. Stock job2029
ran the identical CPU21 ELF used by strict Spike and GSIM. All45 counter windows,
24 exact FP observations, rounding/exception flags and the final checksum pass.
Every retired instruction count agrees across the three engines. The final ELF
contains zero custom instructions and zero FSM instructions.

Small and large cases train the pilot models; middle sizes, both lane/stride
arms and their repeats were withheld before timing. The three arithmetic model
hypotheses and two memory hypotheses were declared before hardware labels.
The shared Merlin fitter handles statistics and domain refusals.

| Stock FireSim stream | Count-only maximum withheld error | Dependency/extent hypothesis | Maximum withheld error |
|---|---:|---|---:|
| DIV |4.41%|Issued FP operations + dependent stream operations|0.031%|
| FMA |91.17%|Retired instructions + dependent stream operations|0.260%|
| Memory |259.94%|Requested bytes + initialized address extent|38.29%|

Each error column covers two withheld cases in one controlled stream family.
Choosing the best of the predeclared hypotheses remains model selection;
these results do not qualify arbitrary mixed code or whole programs. Coefficients
describe these emitted loops, not pure FPU latency or memory service rates.
Spike mcycle remains an instruction proxy and is never fitted as hardware cycles.

The memory model is insufficient. GSIM's extent hypothesis has11.60% maximum
error while stock has38.29%; these are separate execution regimes. Large table
reads, persistent operand owners and cache/physical-traffic claims remain
unpriced. The first and second samples are retained, including substantial
differences in memory and instruction-footprint cases. They are not confidence
intervals or evidence of an assumed cache capacity.

Evidence: [root model pilot](perf_records/root_cpu_stream_model_pilot_20261006.json),
[three-engine closure](perf_records/rv64gc_cpu_service_battery_three_engines_qualified.json),
[stock terminal](perf_records/stock2029_cpu21_terminal.json).

## Features needed to rank actual compiler changes

The source-exact Tiny DIV/up pair has identical opcode counts but different
instruction order. Its small measured difference remains a held experiment.
The new OOT decoder exposes within-block FP producer/consumer instruction
distances weighted by actual sink multiplicities from the same ELF's Spike
histogram. This distinguishes the two objects. It delegates register dependency
construction to existing Merlin depgraph code; it exports no cycle prices.

Loop-carried and cross-block edges, chronological paths, integer addresses,
memory aliasing, FP exception state and resource occupancy remain unknown.
These partial features cannot authorize a reorder or establish its performance.
Fifteen operand fixtures were independently disassembled with GNU; unsupported
forms and signed branch targets were checked. Nine separate stale/missing/unit/
partition/feature corruptions are refused by the model admission probe.

The complete Tiny interval-table section separately measures1,602,024 to
1,497,396 mean GSIM cycles,6.531% lower, while instructions increase.
Its8MiB table traffic, fallback and finishing work are included. This is a useful
future held experiment for memory-sensitive ranking, not training proof or a
whole-model result. Generic source-DAG/closed-consumer promotion is in progress.

Evidence: [ordering adjunct](perf_records/tiny_source_exact_ordering_heldout_pc_adjunct.json),
[complete table experiment](perf_records/tiny_source_interval_table_complete_gsim.json).

## Next calibration and compiler work

1. Close random gather observations over64KiB,512KiB and8MiB working sets.
   Preserve initialization, requested-region counts and timing scope; do not
   convert requested regions into physical misses. Middle size stays withheld.
2. Close primitive load, resident compute and accumulator readback service
   fixtures. Finish/CPU issue costs remain part of each operational measurement.
3. Validate predictions and ordering on independent source-equivalent variants
   and complete callers. Changed ISA counts, spills, dependencies, storage,
   dispatch and finishing costs must all be covered.
4. Keep Jack's22,387,449-cycle implementation held out. The current model does
   not reproduce it well enough to claim a whole-model estimator.

## Changes useful for automatic phases

**Phase0:** export typed source DAGs, numeric context and closed-consumer proofs;
bind readonly/noescape effects and immutable views before preparing shared
operands; include exact executable and timer scope in every observation.

**Phase1:** expose generic host transforms and source interval-table parameters,
plus target schedules through the existing compiler interfaces. Prune identical
objects before timing. Report storage, alignment, preparation and finishing
costs alongside arithmetic counts. Workload/golden selectors remain forbidden.

**Phase2:** use the shared fitting, dependency and resource infrastructure with
OOT feature producers. Keep engines/regimes distinct; expose unknown features
to CCA; require withheld coverage and candidate ordering before enabling a
screen. Spend FireSim on calibration gaps and shortlisted complete programs.

The model currently licenses diagnostic arithmetic stream screening only.
Whole-model targets and automatic ranking remain unmet.
