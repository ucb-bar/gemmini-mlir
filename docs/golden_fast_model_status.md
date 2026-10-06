# Faster performance screening

## Latest results and transfer test: 2026-10-06 23:12 UTC

The stock binary64 DIV/FMA battery2043 completes all four held middle cases
and both repeats. The predeclared count-plus-chain model has0.490% maximum
held mean error; count alone misses148.50%. The directed multiply/add/narrowing
battery2044 has separate lower/upper count models with1.845%/1.877% held error.
Both added-chain fits refuse negative coefficients. No coefficients are clipped
or exported as pure FPU latencies. The DIV/FMA stream does not price Smol's
dominant directed multiply/add/narrowing loops or ResNet's input quantization.

The matched scalar/stage battery2049 completed on stock FireSim. It retains
eight identical per-lane arithmetic DAGs, operands, rounding, final flags and
opcode counts, while multiply-to-consumer spacing changes1→8 without stack
traffic. Root independently regenerated its source and exact words, reclosed
98pins, ran16 legality tests,12 native cases and27 strict target windows.
Middle sizes are held for within-battery fitting. A separate prediction uses
only the original2044 training cases and holds every new executable case out.
The original ELF-bound model correctly refuses transfer. The separately stated
hardware/signature count transfer fails:111.62% lower and170.85% upper maximum
error across the six new cases in each signature. Those frozen predictions
and failures remain unchanged. Within2049, count alone misses57.67%/88.44%;
adding emitted ordered dependency spacing reduces held mean errors to
0.0475%/0.0717%. Complete4096-operation scalar/stage calls change
128,040→59,434 lower and112,171→40,490 upper,53.58%/63.90% reductions.
Root reparses all27 hardware windows and24 output/flag rows, checks strict
instruction identity, and reproduces every shared-fit/scoring field.

A separate post-label diagnostic uses the existing Merlin grouped validator
with unchanged declared features. Both schedule variants of each input/size
are held together. Count features cannot decide any of three pairs. Ordered
dependence correctly decides the middle pair in each signature; both endpoint
folds refuse extrapolation. The complete ranking gate remains refused for
insufficient coverage and slices. A small interpolation error does not approve
a general ranker. Actual source loops must now supply the missing transfer test.

Next candidates use that evidence directly: eight independent columns of the
actual Smol bounds loop, explicit eight-lane ResNet host quantization, and Tiny
source-continuation placement. Predictions/features are frozen before timing.
Loads, stores, paired endpoints, new calls/spills and memory domains that lack
calibration remain UNKNOWN. No saved-instruction cycle rate is substituted.
One bounded3600s strict Smol normal-owner diagnostic is running after fresh
NULL-callback/allocator/ABI/noFSM review; it checks whole target functionality,
not hardware performance. Tiny's supported first-M8 lazy pair is separately
qualified for timing; its special-input source conversion refusals are retained.

The actual Tiny lazy-continuation pair is admitted as stock2053, with a frozen
pre-timing model packet. Existing elements/regions screens predict an exact
tie. An exploratory dependency/regions fit predicts6,166,110 control versus
5,171,461 candidate cycles, but approved ranking remains UNKNOWN because of
216 new cold calls, a128→224-byte frame and32→110 executed stack accesses per
call. Grouped old validation resolves only three of nine table variants and
refuses its coverage/slice gate. Root rehashes305pins and reproduces all fitted
fields using the original qualified NumPy2.5.3 environment. A first replay with
the ordinary root interpreter differed in least-squares last bits and fit hashes;
neither replay replaces the frozen forecast. Stock2052 separately holds the
current ResNet flat-loop whole candidate; its primitive features tie and CPU
issue/footprint costs remain unpriced. Both are prospective transfer tests.

[Actual Tiny forecast replay](perf_records/root_tiny_lazy_prospective_model_review_20261006.json),
[frozen prelabel packet](perf_records/source_continuation_prospective_model.json),
[ResNet source-bound release](perf_records/root_resnet_flat_current2039_stock_release_20261006.json).

Tiny's complete original M8 immutable-base pair2047 improves from6,073,417 to
5,927,830.5 mean cycles,2.397%, in four ABBA samples. All45,056 original outputs,
guards, inputs and instruction counts pass. This is a section result; the
422,018,733 whole champion remains. Generic implementation is published as
[Merlin PR44](https://github.com/ucb-bar/merlin/pull/44), one topic from main,
with24 root tests and installed-wheel/object-equivalence evidence.

Current ResNet2039's source-preserved profile2046 measures27,791,061 callback
cycles and2,965,649 outside-callback cycles,90.36%/9.64% of30,756,710 forward
cycles. Callback windows include CPU issue, transfer and waiting. The30,715,818
uninstrumented whole champion remains. Role accounting against the permitted
ZIP locates3,279,338 cycles of residual difference and2,913,932 outside timers,
about74% of the diagnostic gap. This directs the next experiments toward
residual implementations and generic host quantization lane scheduling; the
two current flat-convolution loops alone cannot close the dominant difference.
Reference and original-source numerical contracts are not interchangeable.

Phase0 needs immutable source/ELF/numerical/memory-scope bindings and declared
feature schemas. Phase1 should choose experiments that separate candidate
schedules whose current features collide, then validate held schedule ordering
across executables. Phase2 needs composed coverage with explicit unknown CPU,
DMA, memory and overlap terms, plus whole-model numeric/performance gates.
Whole prediction and a held-out reproduction of ZIP22.39M remain unqualified.

[FP64 terminal/model review](perf_records/root_fp64_stock2043_terminal_model_review_20261006.json),
[directed review](perf_records/root_fp64_directed_bounds_stock2044_terminal_review_20261006.json),
[cross-executable release](perf_records/root_fp64_matched_stage_stock_release_20261006.json),
[matched terminal/model replay](perf_records/root_fp64_matched_stage_stock2049_terminal_review_20261006.json),
[grouped ordering/coverage diagnostic](perf_records/root_fp64_stage_grouped_rank_diagnostic_20261006.json),
[Tiny stock pair](perf_records/root_tiny_stock2047_base_pair_terminal_review_20261006.json),
[Tiny lazy supported-fixture release](perf_records/root_tiny_source_continuation_lazy_short_release_20261006.json),
[bounded Smol strict release](perf_records/root_smol_prepared_owner_null_diagnostic_strict_release_20261006.json),
[current ResNet role accounting](perf_records/q1013_current2039_stock2046_profile_alignment.json).

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

Stock2030 and2031 also completed the identical gather and primitive ELFs used
by strict Spike and GSIM. Original outputs, guards, inputs, exception state,
protocol and per-window retired instructions agree across all three engines.
The33 cases across CPU21/gather3/primitive9 retain75 separate windows per engine.

| Additional stock stream | Hypothesis | Withheld mean error |
|---|---|---:|
| Indexed gather | Requested distinct64B address-region bytes |2.83%|
| Resident compute | Source padded array rows |1.88%|
| Requested loads | Requested payload bytes |8.98%|
| Raw i32 readback | Requested payload bytes |6.59%|

Each row has one middle-size case withheld, with its two repeats averaged
before fitting. First/second samples stay separate in the evidence; gather and
load differences are substantial. No confidence interval or cold-cache claim
follows. Gather extent alone misses92.41%; the requested-region hypothesis
misses1.07% on GSIM. Regions do not mean physical cache lines or misses.
The new hypotheses were committed before root inspected stock labels; gather
GSIM results had already been observed. Independent variant validation remains
necessary. Accelerator windows contain CPU issue and fenced completion, so
these fitted terms are not pure array/DMA/DDR rates. Constant gather count
features and all two-term hypotheses refuse for insufficient distinct training
points; the shared fitter's gate remains intact.

Evidence: [operational model pilot](perf_records/root_operational_service_model_pilot_20261006.json),
[gather terminal](perf_records/stock2030_gather3_terminal.json),
[primitive terminal](perf_records/stock2031_primitive9_terminal.json).

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

The source-equivalent inner-stripe-row variant retains the exact primitive
and operand order, but reduces instructions4.72% and touched code54.65%.
Complete common-address source-capsule GSIM cycles change512,059 to511,541,
only0.1012% lower. Root reclosed all127 evidence pins. This is a held validation
case for instruction-footprint/overlap sensitivity; it is not new training or
a stock whole result. The experimental compiler option stays isolated.

Evidence: [ordering adjunct](perf_records/tiny_source_exact_ordering_heldout_pc_adjunct.json),
[complete table experiment](perf_records/tiny_source_interval_table_complete_gsim.json).

[Compact-loop validation](perf_records/root_resident_inner_row_model_review_20261006.json).

## Next calibration and compiler work

1. Validate the gather screen on independent index streams, read counts and
   source-equivalent table/caller variants. Preserve requested-region counts,
   initialization and timing scope; physical misses remain unknown.
2. Measure source-equivalent primitive variants with resident B reuse and
   changed issue order. The ordinary preload/compute calibration does not
   automatically price those schedules. Finish/CPU issue costs stay included.
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

## Additional model effort after user steering

The12-case crossed gather battery completed stock FireSim2036. Its27 windows use the original byte-identical
gather and common fenced counter source. Read counts1024/4096/16384 cross
64KiB/512KiB/8MiB extents; a second seed supplies three additional independent
streams. Only four primary-seed corners train; eight middle-axis/second-seed
cases, including both repeats, remain withheld. Two-term instruction/region
and payload/region hypotheses were committed before new timing labels. Root
reclosed83pins and independently regenerated every index, expected accumulator
and checksum. Spike establishes correctness and instruction counts; hardware
timing is complete. Another full GSIM run is unnecessary
for this CPU fixture.

[Declaration](perf_records/gather_crossed_model_hypotheses_20261006.json),
[root release](perf_records/root_gather_crossed_stock_release_20261006.json),
[root model review](perf_records/root_gather_crossed_model_pilot_20261006.json).

All eight held cases resolve, but the broader test fails: requested-region
bytes alone reach40.76% maximum error and instructions plus regions40.87%.
Instructions alone reach80.75%; initialized extent alone174.31%. The earlier
fixed4096-read2.83% result does not establish transfer across read counts and
index streams. Both repeats remain visible, including differences exceeding
2x. No cache-state or confidence claim follows. Root independently reclosed
89 qualified artifacts and nine terminal pins; every original output/checksum,
actual staged ELF and stock bitstream agrees.

The next generic feature is exact sequential logical-region recurrence:
first touches and distinct intervening regions between repeated requests, with
explicit address granule and resource limits. This belongs in Merlin and is
not a cache-miss census. Any new gather hypothesis after these labels is
exploratory until independent validation. The matched table and resident-B
reuse batteries retain their own predeclared held cases.

The real Tiny table candidate has now passed full normal/native/strict-target
qualification and completed stock job2033 at424,921,379cycles, versus
422,018,733for2004:2,902,646cycles/0.6878%higher in one observation each.
Root reclosed298qualification pins, both
immutable measured-core snapshot resolutions, all256000original words and the
unchanged Torch gate, controlled host-object link, actual readonly/aligned8MiB
table bytes and final executable noFSM. No whole savings are projected from the
6.531% M2 GSIM gain. Root also reclosed nine terminal pins and reparsed the
original256000-word digest, DONE and rank0, including named actual staged
ELF/bitstream receipts. Keep2004champion; the table remains default off.
Cache cause and statistical significance are unknown.

The emitted-feature adjunct reuses the exact OOT FP producer and shared Merlin
RAW graph; root reclosed its18artifact and7implementation pins. Actual normal
helper static counts include retained fallback paths and cannot substitute for
dynamic counts. Mixed chains/branches, whole memory requests, cache misses and
cross-block dependencies remain unknown. A separate small matched table
calibration is being prepared across source section sizesM2/M4/M8 and explicit
16/18/20-bit partitions. Four corners train; five middle-axis cases are withheld
before timing, alongside complete source controls. All preparation/table/
certification/replay/finishing costs stay inside the common window. The Tiny
variant labels remain withheld from fitting.

[Whole release](perf_records/root_tiny_source_interval_whole_release_20261006.json),
[terminal review](perf_records/root_tiny_source_interval_terminal_review_20261006.json),
[actual helper features](perf_records/tiny_source_interval_model_features.json).

Untimed generic partition analysis preserves every original first45056i8
word at16/17/18/19/20bits. Their tables span512KiB to8MiB; first-source replays
are216/98/50/27/13, and requested64B regions432/794/1446/2602/4632. All22
actual source quant factors were checked on that same first input fixture;
later groups' operand distributions were not measured. Root reclosed34pins.
The unchanged allocator has a fixed absolute arena; table insertion moves
bss/stack VMAs, not that arena base. Dynamic pointers and physical misses
remain unknown. No new partition is selected from these untimed counts.

[Parameter/layout analysis](perf_records/source_interval_partition_layout_untimed.json).

Typed resident-B/K loops are also being checked on complete original and
independent tail capsules. Exact command/DMA/reuse features will validate whether
the primitive screen can distinguish their schedules. Root reclosed239pins
and used Merlin's existing feature-collision analyzer: unchanged primitive/
array/requestedDMA/reuse features imply point-error floors0.882%onH28 and
7.184%on independent tails. Retired-count ordering misranks the tail pair,
which has11.8%more instructions and13.4%fewer measured GSIMcycles. CPU issue,
address/dependency/spill and overlap pricing remain needed. Pure text reductions
did not yield proportional cycles; no unmeasured overlap coefficient is introduced.

[Source-equivalent variant check](perf_records/root_resident_source_variant_model_validation_20261006.json).

## Latest calibration and reusable features

The exact generic address-region recurrence census is published for review in
[Merlin PR41](https://github.com/ucb-bar/merlin/pull/41). Its26 tests include an
independent oracle and40,000-request trace. No target constants or cycle prices
are introduced. The OOT gather adapter reconstructs bound kernel requests from
actual source/ELF object addresses and emitted load order, then delegates census
and fitting to Merlin. All16 post-label hypotheses are retained:10 refuse
negative screening terms, while six have37.73–54.49% maximum error. The best
exploratory error remains too large, with wrapper/preparation state unpriced.
No independent-validation, physical-capacity or production claim follows.
[Exploration review](perf_records/root_gather_locality_exploration_review_20261006.json).

Matched resident-B calibration2038 now closes the same strict/GSIM/stock ELF,
15 separate windows and original i32 outputs. Both M64 strategy arms remain
held. Stock array-row demand alone ties the schedules and misses16.59%; adding
real-B preload demand predicts their ordering with5.15% maximum error. GSIM
separately gives5.45%. Instruction-only diagnostic gives4.57% stock error and
is also retained. These coefficients include CPU issue and fenced completion;
they are not pure device prices or mixed-stripe/whole-model rates. Root reclosed
107 receipt pins and104 distinct screen artifacts.
[Matched model review](perf_records/root_b_reuse_model_pilot_20261006.json).

The complete table battery finished stock2040 after root independently
reclosed145+14pins, native336/strict420mode-and-sticky gates, actual readonly
512KiB/2MiB/8MiB table bytes and27 short-harness rows. M2/M8×16/20-bit corners
train the two supported no-fixed models; five cases and both repeats are held.
All three complete source controls are excluded from fitting. The original
underidentified fixed-term declaration stays rejected. Source control,
preparation, table certificate, replay, finishing and storage are measured.
[Root release](perf_records/root_source_interval_calibration_stock_release_20261006.json).

All27 actual hardware rows, original prefix outputs, inputs, guards and flags
pass. Root independently reparsed the real UART and reclosed11 named source,
ELF and staging pins. Tables reduce the complete M2 section by3.57–13.24%, but
increase M4 by8.44–20.94% and M8 by8.54–22.66%. All three table widths regress
at the actual M8 row extent. Another whole-model table candidate is held.
The predeclared no-fixed element-count model covers all five withheld cases
with7.78% maximum mean error; elements plus logical regions reaches10.96%.
These models cover this first-helper input distribution and table-only cases;
source controls, later groups, complete variants and whole costs remain
unqualified. Adding a plausible feature did not improve validation.
[Root model diagnostic](perf_records/root_lookup_stock2040_model_diagnostic.json),
[terminal](perf_records/stock2040_lookup_terminal.json).

Binary64 dependency calibration is separately admitted as stock2043 after root reclosed84
pins, regenerated the exact rational source oracle and all source bytes,
independently reran strict Spike, and passed10 oracle/parser tests. Twelve
DIV.D/FMA.D cases cross1/8 independent lanes with1024/4096/16384 source
operations. All four middle cases and both repeats are withheld before stock
labels; count-only and count-plus-chain hypotheses have no fixed term.
All27 windows,24 full64-bit output rows, guards, immutable inputs and exception
flags pass; final ELF has no custom instructions. FP32 coefficients do not
transfer to this precision. Full fenced calls include loop and output stores;
they cannot establish pure FPU latency or SmolVLA whole-model predictions.
[Root release](perf_records/root_fp64_dependency_stock_release_20261006.json).

Dependency analysis itself is faster in
[Merlin PR43](https://github.com/ucb-bar/merlin/pull/43). Indexed RAW-edge
deduplication preserves the complete ordered graph and known/unknown latency
semantics. Root21 tests pass; seven actual register-feature reports agree.
An8192-instruction source-pair diagnostic changes host Python construction
from55.14s to0.069s, one pass per implementation with concurrent host load
unknown. This is analysis wall time, separate from target model cycles.
The one-topic main-based commit remains pending review.
[Publication review](perf_records/root_merlin_depgraph_index_upstream_review_20261006.json).

A lost-output calibration failure also led to
[Merlin PR42](https://github.com/ucb-bar/merlin/pull/42): validate the emulator
stdout destination before starting it. Seven tests include real FIFO fixtures
that refuse before opening or invoking the backend. Both PRs are isolated
one-topic commits from main7fee5cfdac and remain pending review.
