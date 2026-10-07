# Infrastructure fixes and performance changes

## 2026-10-07 19:40 UTC: exact source observations and reusable scheduling costs

Newest qualified local Merlin main is `3f6a8db27`, not a newly published remote
head. It integrates generic exact multi-output product contracts, finite-point
observation composition and exact BF16 integer observation decoding. Numerical
and effect proofs, immutable operands, disjoint output planes, source ownership
and host emitters belong in Merlin. All 1,227 source and 1,227 outside-installed
checks pass; default numerical and selection policy remains unchanged. GitHub
fetch fails DNS, so publication and current remote ancestry remain unverified.
[Installed qualification](perf_records/merlin_source_observation_topics_current_main_20261007_qualification.json).

OOT owns actual multi-plane instruction schedules, accumulator stripes, physical
output stride adapters, residency rows/banks, prefetch placement, command batch
implementation, resource refusals and simulator execution. New ResNet complete-A
lifetime/output-block separation has independently qualified complete GSIM and
a fresh installed-core whole original-output gate. Remaining-row weight slots
also preserve exact command/data ownership, but an independent shape regresses;
legal placement does not establish general profitability. These explicit options
are not workload-name dispatch or default promotion.
[Fresh whole target gate](perf_records/resnet_resident_a_output_blocks_current_installed_whole_20261007_qualification.json),
[placement cost and refusals](perf_records/resnet_resident_remaining_weight_slots_complete_20261007_qualification.json).

Merlin/CCA needs a portable way to express host-loop retention and command
batching, plus object/ELF-bound actual CPU dispatch, branch/address and stack
traffic. Existing dispatch counts and loop-offload flags cannot distinguish
Tiny schedules with identical primitive commands but different CPU costs.
Target implementations and resource checks stay in OOT. A smaller-stack variant
can still regress, so these observations must enter a complete cost model with
transfer, accelerator service and waits before automatic promotion.
[Actual Tiny costs](perf_records/tiny_retained_n_normal_native_model_20261007.json).

The current b154 Smol encoder contains no FP division; actual encoding cost is
mostly integer/branch work. Complete source-provider/evaluation/certification
costs dominate possible 1.06964% consumer recomputation savings. Reusable source
representation/preparation and observation-region passes belong in Merlin;
device scheduling alone does not eliminate their required arithmetic. The full
stock trial2113 regresses despite exact output, retaining the original numeric
gate and best historical result. Original gates and complete hardware timing
remain necessary before making a default compiler decision.
[Actual Smol census](perf_records/root_smol_exact_math_whole_function_costs_20261007.json),
[consumer cost](perf_records/root_smol_bf16_consumer_pc_price_20261007.json),
[whole stock regression](perf_records/root_smol_endpoint_stock2113_whole_20261007_qualification.json).

## 2026-10-07: invocation-local preparation and explicit layout selection

Local Merlin main `edf0e0ca4` adds a generic fully prepared model callback:
immutable private input, verified public function types and recorded selected
bytes before normal profiling/upstream lowering. No target or workload chooses
policy. Source and independently installed checks pass 17 each; 1,016 Python
and 61 runtime payloads match all release forms. GitHub publication is pending
DNS access. [Qualification](perf_records/merlin_prepared_model_transform_local_20261007_qualification.json).

The OOT physical-layout/catalog wrappers now forward an explicit channel block
to Merlin's existing layout pass. Target catalog options remain OOT; the shared
algorithm is retained in Merlin. Both process-global replacements in the fresh
ResNet recipe are removed. Its same-core control/selected LLVM, object and final
ELF are byte-identical, with original whole 0/0 output and zero-FSM gates passing.
This is compiler interface qualification, with hardware cost still pending.
[Shared recipe audit](compiler_recipe_status.md).


## 2026-10-07: current shared compiler and normal-recipe coverage

Current Merlin main is `3430c2ca9`; model2MLIR main is `7915e234`.
The allocator overflow guard and explicit LLVM helper link/inline contract
are shared infrastructure topics, independently qualified in source and an
outside package installation. They add no default numerical policy.
The earlier head and cycle snapshots below are historical records.

The three models share the normal compiler API and target backend. Explicit
experimental selections, differing pinned revisions and source numeric
contracts are not a qualified universal automatic policy. Maintain a per-option
promotion record: source legality, independent cases, actual compiler feature
selection, original whole gate, complete cost, installed delivery and default
routing status. Golden outputs may validate an option, never select its code.

The latest concrete unused shared mechanism is exact scalar BF16/math
legalization: current Smol source LLVM retains hundreds of BF16 conversions
while its recipe omits `lower_exact_math_inline`. Normal selection now produces
LLVM with no scalar BF16 arithmetic or truncation; all 1,600 native outputs
and the final zero-FSM audit pass. Full target/hardware qualification remains
pending. This is host lowering in Merlin. Sharing all five
signed-radix product outputs, accumulator stripes and device commands belongs
in OOT; source numeric/product/ownership proofs and orchestration stay in Merlin.
[Actual whole function costs](perf_records/root_smol_endpoint_whole_retirement_20261007_summary.json).

## 2026-10-07: clean upstream and handwritten branch delivery

Merlin main is `95e8142d9`; model2MLIR main is `7915e234`. Reviewed changes
reach main with one clean commit per topic. The final Merlin package passes
1,085 installed checks, preserves 995 Python modules, 28 runtime headers and
five C templates exactly, and reproduces the accepted Tiny2070 source routes.
Root independently reclosed all 2,096 package qualification pins.

The named `handwritten-implementation` Gemmini branch now includes the missing
qualified ResNet rectifier/domain/fence/panel compiler modules and explicit
Smol/Tiny experiment reproduction drivers. ResNet integration passes 64 checks;
compiler/export/audit gates pass 61 checks and six subtests. Both Tiny model
objects and final ELFs reproduce exactly; source-wide table and observation
proofs remain generic Merlin mechanisms. All original numeric gates remain.

Scoped cleanup archives 34 PR records and verifies public HTTP 404 for every
one. Eight already merged records cannot be archived; zero authored PRs remain
open in Merlin/model2MLIR. Published history is retained, and future PR creation
requires explicit approval in every repository. The 31 stale Merlin topic refs
and four merged model2MLIR refs are removed after exact local preservation.

The broader test failures identified checkout child-import ambiguity and stale
audit fixtures. Phase 0 should distinguish execution environment and oracle
identity; phase 1 should verify installed compiler/resource ownership; phase 2
should bind selected source, prepared LLVM, actual objects and complete cost.
One-source module ownership and explicit compiler/toolchain provenance prevent
editable-checkout mixtures from qualifying a different implementation.

No new FireSim result is claimed by this publication. Whole ResNet/Tiny/Smol
remain 29,698,347 / 394,765,577 / 258,621,872,969 cycles. The 22M/300M/5B goals
remain unmet. Family counts and token ledgers retain their stated deduplication
and attribution limits; publication commits are not additional performance wins.
[Delivery details](handwritten_implementation.md).

## 2026-10-07 04:26 UTC: quantization quality and typed observation boundaries

The original Smol TorchAO recipe reproduces 1,600 golden words exactly. All303
registered Linear weights store i8; 302 captured calls are integerized, while
attention remains floating. Prequantization quality was not separately measured
before. The recipe's 2.9199% relative L2 loss and 181/1,600 tensor failures are
separate from compiler fidelity against that quantized golden. Generic capture
baselines and quality reports belong in model2MLIR; task budgets and whole
fidelity admission belong in Merlin evaluation. Source arithmetic, QDQ axes,
dtypes and escapes remain explicit. No TorchAO application bug or robot task
quality result is asserted.
[Audit](smol_quantization_audit.md).

Shared typed observation-region analysis belongs in Merlin. Actual Smol
attention i8 values and scales are internal through output projection on all12
source closures. Proving all uses establishes a legal boundary; a numerical
replacement requires a separate theorem. Projection error intervals and
immutable constant integer product/prefix bounds also belong in Merlin.
Physical GQA grouping, integer planes, accumulator capacity, tiles, ABI and
device instructions remain OOT. These topics are in progress; a numerical
projection theorem and default approximation remain unqualified.

Phase0 preserves separate prequantization quality, quantized Torch and compiled
source oracles, including nonfinite values, dtype/opmath and ordered reductions.
Phase1 exposes source i32→BF16/F32 rounding, activation/channel scale axes, source
use/effect closure, exact weight bindings and overwrite epochs through views.
Phase2 permits changes to closed observation regions, representation, preparation
ownership and GQA schedules, pricing complete encoding, readback, replay and
refinement work. Local floating tolerance does not prove escaping integer bins.

Exact encoder/witness composition uses generic Merlin frontier math and OOT
target integration, improving Smol group2069→2072 by2.702942%; whole cycles are
unchanged. Stock2073 prices current masked host attention at4.974002% fewer
complete block cycles; it does not qualify a new accelerator offload or whole
gain. Model2MLIR mask PR5 at7cc1b4b passes69 source/69 installed checks with59
modules exact on actual mainbd50. No model selector or causal attribution to
current workload failure is introduced. Conservative family counts remain61
pending explicit deduplication; import repairs and review-driver refusals are
not additional compiler bug families.

## 2026-10-07 03:12 UTC: structural algorithms and consumed proofs

Generic completed-plane integer reconstruction is now a clean main-based
Merlin PR49. Defined integer prefixes, one final exact conversion, typed private
storage and original numeric policy are independent of the producer target.
Root958 source/wheel/installed modules,151 packaged-data and50 bundled source
resources match;946 source and56 installed checks pass. OOT owns the actual
target readout ABI, resources and stock run. A1MiB larger native workspace is
queried and propagated into the fresh normal pool; obsolete capacities refuse
before writes. The native ranked-versus-flattened ABI mismatch is retained and
an explicit native-only provider adapter repairs integration. This motivates a
generic ABI-version/signature admission gate; target adapter implementation stays
OOT. Experimental RMS4 policy is not relabeled an exact normal binder theorem.
[Delivery and publication](perf_records/root_merlin_fused_radix_PR49_delivery_review_20261007.json).

The shared integer-result observer ABI/use/effect proof and portable lowering
also belong in Merlin; target rounding legalization and actual simulation remain
OOT. Stock2067's14.0703% complete-helper gain becomes3.750235% on whole Tiny2070,
not an additive whole forecast.
Immutable original scales/source/table/cold paths and typed floating-escape
refusal remain mandatory. New partial integer readout enclosures are generic
Merlin math; target callback scheduling belongs OOT. First complete partial arm
loses141.37%instructions from extra replay, despite60% fewer callbacks. Do not
promote a legal numeric mechanism based on primitive count alone.

ResNet's four-panel private residency/store-reload/fence schedule is OOT. Its
17.496% local GSIM gain becomes0.91858% whole stock2068. The next missing seam
is generic propagation of existing source ReLU/clamp domains through actual
typed SSA/layout maps into finite affine coefficient proofs. Existing domain
certificates were not consumed by normal coefficient choice; full type-domain
assumptions prevented three cheaper routes. Eligibility must come from source
and producer proofs, not callee/source names or captured values. New20 focused
checks plus16existing checks pass on a clean main-based core topic. Fresh normal
and controlled whole compositions pass original1000words bitexact. The controlled
stock release now has an independently verified whole result at29698347cycles,
0.647726% below2068; no additive forecast is claimed.

Root now independently rederives all16actual normal source choices, the three
admitted32768-pair certificates and actual active finalELF entries;5921pins,
all1000 original native/strict words and all98304target pairs close. One controlled
stock2071 observation is root-verified. A target-free
phase1 fact-consumption report should expose producer proof, SSA/view path,
admitted domain, selected cost and unused facts so automatic agents can discover
this seam. Stable typed evidence schemas should distinguish one rebinding from
lists of rebindings and identify actual normal inputs despite identical copies;
experimental receipt conventions caused root review-driver refusals, not compiler
or accuracy failures. No unqualified family count is added.

### Additional requirements for automatic phase0/1/2 loops

- Phase0: test ordered floating reductions, escaping BF16 scales, partial integer
  intervals, nonfinite/refusal/rounding effects and producer-domain propagation.
- Phase1: expose whether each proved fact is consumed by actual normal lowering;
  bind emitted objects, callback ABI versions, source owner epochs and queried
  workspace lifetimes. A point and a remainder interval must have different
  semantic contracts. Source replay storage remains live when observed.
- Phase2: expose changes to producer/consumer grouping, numeric representation,
  packing, preparation lifetimes and target schedules. Include actual source
  replay/ambiguity counters, requested readouts, frames and dependencies in cost
  evidence. Preserve complete-cost negatives, original whole gates and unknown
  physical memory/overlap terms; never infer whole gain from section percentages.

The conservative reviewed61-family inventory below is unchanged. PR49 and new
private mechanisms have not yet been independently mapped into its families;
do not count every variant or test as a new bug/improvement, or claim them merged.

## 2026-10-07 01:44 UTC: reusable compiler topic and calibrated target features

Shared typed closed-mask contraction scheduling is published in Merlin PR48,
independent of target/model names, with explicit nontrapping/unobserved-flags
permission and original live reduction order. The conservative shared inventory
now61families:29Merlin bug fixes,24Merlin reusable improvements,8model2MLIR bug
fixes.55families are on main,6new families pending review. PR47 extends the
already-counted bounded host quantization family33. Private RMS/finite/floor
prototypes are not counted as merged shared infrastructure.
[Publication readback](perf_records/root_merlin_masked_PR48_publication_review_20261007.json).

Complete source-use/effect/numerical legality and portable bounded floor
conversion belong in Merlin. Gemmini SPAD hazards, primitive realization,
resources, ABI, target counters and RV64 dependence decoders belong in OOT.
Generic provenance/fit/ranking/selected-vs-fallback evidence belongs in Merlin.
Stock spacing calibration now distinguishes equal-work host schedules and
scores one held pair within3.892%; this does not price whole programs or new
cache/dispatch/fence/overlap regimes. Tiny2062 gains0.612%whole despite16.38%
compound gain; Smol2060 gains8.043%section, with whole258.622B unchanged.
These measured scope differences guide general compiler and model work.

## 2026-10-06 19:10 UTC ownership and modeling checkpoint

Generic finite BF16 source-point preparation sharing is a Merlin runtime/source
optimization, published for review in [PR39](https://github.com/ucb-bar/merlin/pull/39).
It improves the complete original Smol group8.1516%on stock2024; whole cycles
remain unknown. It extends the existing preparation family; the conservative
change inventory remains55families, including47Merlin and8model2MLIR.
Target products, ISA encodings and resource/layout facts stay in OOT.

Six source-qualified resident spatial stripe schedules are OOT; the normal
compiler propagates the existing resource-checked option. Root releases their
whole composition over actual2025with37,015rehashed pins. Stock2026finishes
30,977,892cycles,2.2706%below2025. ResNet latest whole2026 is30,977,892cycles; Tiny2004
422,018,733 and Smol whole1906 258,621,872,969 remain the other bests.

Model calibration now has dedicated work. Merlin's existing target-free fitter,
grouped validation, unknown-domain handling and ranking gate are exercised on
actual same-ELF GSIM/FireSim evidence; OOT owns the source/ISA/scope join.
Geometry holdouts give3.88%median/11.85%maximum callback error but inaccurate
host costs. A short independent primitive benchmark is being built. Physical
rates, traffic, dependencies, program generalization and variant ranking must
be qualified before automatic phase1/2loops use these prices. Jack remains held out.
[Diagnostics](perf_records/paired_engine_geometry_screen_diagnostic_20261006.json).

## What has been fixed

The work falls into three groups. Portable host performance is compiler optimization in Merlin,
alongside the infrastructure that makes target optimization usable through a normal model build.
Gemmini instructions, device schedules and resource decisions remain in the OOT compiler.

| Group | Concrete changes | Current evidence |
| --- | --- | --- |
| Frontend and shared numerical correctness | BF16/f16 opmath, source reduction/order contracts, runtime helper ABI, explicit portable expf precision policy | Verified frontend fixes are upstream on model2MLIR main. Full original Smol now passes all1,600 native and actual target words exactly, retaining atol=.03125/rtol=.02. [Target](perf_records/smol_full_double_exp_target_exact.json) |
| Merlin build and runtime infrastructure | Generic loop outlining, fresh output ownership, linked-object identities, strict compiler flag precedence in compile and link | Corrected Smol outlined host compiles in111.254s after the monolithic900s timeout. Normal math-policy build reproduces the entire passing loaded target image except its diagnostic marker. A real compiled subnormal regression catches strict flags being overridden. [Normal build](perf_records/smol_normal_host_math_policy_equivalence.json) |
| Merlin planning and dispatch infrastructure | Complete CCA preservation, exact source-bound alternatives, normal build selection, distinct implementations for equal tensor shapes | Selected measured objects reach the real device catalog. Tiny full normal build retains all155 source calls, original256,000 words and the Torch gate. Whole-program search/cost optimization is still not wired. [Tiny qualification](perf_records/tiny_resident_a_prefetch_whole_spike.json) |
| Merlin host performance | Exact bounded quantization/RNE packets, pointwise scheduling, packing/readout/mean, ordered contraction scheduling and selected BF16 widening | Stock whole-model and complete capsule gains below; these are optimizations, not merely build fixes |
| OOT Gemmini performance | Resident/banked transfers, overlap and B slots, convolution stripes, primitive issue/store scheduling, device numeric and layout bindings | General shape/layout/resource predicates; every qualified final ELF passes the all-executable-section zero-FSM audit |

## Gains that can be attributed to a controlled change

Each row is a matched experiment with its own control. Separate rows must not be added to predict a
composition. Full model means stock FireSim forward cycles; GSIM capsule means the complete stated
fixture, not a whole-model speedup.

| Owner and change | Scope | Control → candidate cycles | Reduction |
| --- | --- | ---: | ---: |
| Merlin exact ResNet host quantization packets | Stock1853→1886, whole model |40,479,548→38,603,949|1,875,599;4.6334%|
| Merlin Tiny scalar pointwise packets | Stock1846→1880, whole model |569,151,067→531,072,370|38,078,697;6.6904%|
| Merlin exact early-saturation/eight-lane readout; OOT ABI binding | Stock1874→1900, ResNet whole model |39,201,279→37,946,541|1,254,738;3.20%|
| OOT resident/banked transfers | Stock1849→1853, ResNet whole model |42,269,808→40,479,548|1,790,260;4.235%|
| OOT banked residual added to transfer control | Stock1853→1874, ResNet whole model |40,479,548→39,201,279|1,278,269;3.158%|
| OOT full-reduction convolution stripes | Stock1853→1878, ResNet whole model |40,479,548→39,754,283|725,265;1.792%|
| OOT convolution stripes added to transfer/residual control | Stock1874→1897, ResNet whole model |39,201,279→38,468,933|732,346;1.868%|
| Merlin adjacent independently proved RNE packets | Stock1880→1902, Tiny whole model |531,072,370→527,255,504|3,816,866;0.719% single-run marginal|
| Merlin/OOT qualified host/readout/residual composition | Stock1886→1903, ResNet whole model |38,603,949→36,102,704|2,501,245;6.479%|
| OOT resident-A/B-prefetch alternative | Tiny-shaped synthetic int8/amplitude21 common-address GSIM capsule |111,885→76,456|35,429;31.67%|
| Merlin integer attention packing | Stock1909, complete packing section |776,043,123→665,638,655|110,404,468;14.2266%|

Receipts: [ResNet host](perf_records/firesim1886_resnet_quant_packet_verified.json),
[Tiny host](perf_records/tiny_pointwise_packet_firesim.json),
[exact readout](perf_records/firesim1900_resnet_sat8_readout_verified.json),
[transfer](perf_records/resnet_transfer_command_policy_firesim.json),
[residual composition](perf_records/resnet_transfer_residual_composed_firesim.json),
[stripes](perf_records/resnet_resident_stripe_policy_firesim.json),
[stripe composition](perf_records/firesim1897_resnet_stripe_composition_verified.json),
[Tiny device capsule](perf_records/tiny_resident_a_prefetch_gsim.json),
[adjacent RNE](perf_records/firesim1902_tiny_adjacent_rne_verified.json),
[measured composition](perf_records/firesim1903_resnet_composed_verified.json).

The packing hardware result includes complete representation planes, steps,
reconstruction and norm metadata. It does not measure the attention contraction,
softmax, certificate or whole model.
[Packing receipt](perf_records/smol_integer_packing_1909_hardware.json).

The Tiny device alternative also passes the normal full-model original-output and zero-FSM gates.
Its first build selects one contraction. The expanded44-binding normal and controlled builds
also pass all original native/strict target output gates. Stock1911 measures the normal build
with its runtime compiler change;1912 isolates the device schedule with all1880 host/runtime
objects retained. Both hardware outcomes are pending. [Family](tiny-calibrated-family-20261005.md).

Rejected optimizations remain recorded. In particular, the complete Smol attention-head gamma
candidate takes3,077,601,494stockcycles versus2,618,580,085controlcycles(17.53% slower), despite
lower device/readback work. It is disabled. [Hardware](perf_records/firesim1895_original_attention_head_gamma_verified.json).

## Remaining gap and accounting

Current verified whole-model champions are ResNet33,500,256cycles (1992) and
Tiny422,018,733cycles (2004). Tiny's user-specified target is300M.
Smol stock1906 measures258,621,872,969cycles with full original target correctness. These do not
meet the requested22M/5B goals. Matching Jack's permitted executable remains a device scheduling
oracle, with original source/numeric gates held fixed.

There is no defensible overall infrastructure-versus-dialect percentage of effort or tokens.
Requests and threads mix both kinds of work, and the controlled performance arms are not an
additive campaign attribution. The [journey](golden_optimization_journey.md) retains measured
campaign counters, per-experiment ownership, gains, negative results and remaining abstractions.

The latest numerical-provider consistency gate is Merlin infrastructure: it
refuses stale workspace queries, native library/compile identities, incomplete
dependency coverage and conflicting duplicate pins before an opaque proof hash
enters dispatch. Exact squared-sum scalar accumulation and probability-bin reuse
are portable Merlin performance alternatives. Physical B-panel addressing,
Gemmini resource scheduling and disjoint DMA/correction overlap belong in OOT.
Every alternative remains explicit and source-qualified; a passing numeric proof
does not establish profitability. Normal packed-parameter binding needs a generic
representation contract if measured target cost supports that route.
[Current normal seal](perf_records/root_smol_normal_word_integer_sealed_reclosure.json),
[squared-sum](perf_records/tiny_scalar_squared_sum_whole_qualification.json),
[probability reuse](perf_records/prepared_probability_bins_qualification.json).

Encoder equality and private bound-producer coverage are Merlin proofs about
unchanged source values, effects, owner lifetime and source numerical order.
They save complete measured CPU work; they grant no ISA capability. Sparse
affine relation predicates are also Merlin alternatives, with complete-domain
equivalence and preserved source replay. The current original ResNet cost
regresses despite a separate signed-domain fixture win, so no automatic
selection follows. Target disassembly and DMA/cacheline scheduling stay in OOT.
The one-endpoint polynomial enclosure similarly stays disabled: a faster local
section increases complete replay cost. These results require phase 1/2 tooling
to price the full consumer, keep proof flow and dependencies explicit, and prune
compiler alternatives that emit identical actual code before hardware runs.
[Complete negative](perf_records/smol_one_endpoint_word_negative_journey.json),
[sparse source and costs](perf_records/residual_sparse_predicate_initial_checkpoint.json).

## How these changes generalize

Two new stock measurements qualify general compiler changes. Merlin's typed
normalization hoist removes repeated source math at a proved smaller affine
domain: Tiny531,072,370→461,389,700cycles against isolated1880. Merlin's borrowed
view/ownership proof plus the OOT segmented-input implementation removes copied
matrices: ResNet36,102,704→35,152,730cycles against isolated1903. These gains are
not added to different packet/schedule arms.

The source attention executor is portable Merlin runtime infrastructure with
explicit numerical/consumer closure and retained actual source fallback. A
shared private workspace pool owns allocation and release; OOT owns device
products and ranked ABI adaptation. Linking its mixed host helper needs the new
separate host-provider object hook. Device kernels still have zero unresolved
symbols. Companion LLVM machine types, imported build identities and final
symbol closure supplement logical tensor/ownership proofs; they do not replace
them. Default-disabled ordinary builds preserve prior executable bytes.

Exact binary32 floor, nonnegative multiply and positive-RHS endpoint arithmetic
belong to Merlin. Fixed RUP/RDN CPU instruction definitions belong to OOT.
The measured complete-group instruction proxy improves3.529B→3.303B without
changing196,608 quantized observations or256 scales. Row metadata hoisting
requires immutable/disjoint typed storage facts. It must not be inferred from
arbitrary C pointers. The separately qualified production compilation gain
9.624B→5.709B instructions removes function/bitcopy overhead under unchanged
normal flags; it is not composed with the row arm or measured whole cycles.

Phase1/2 tooling should expose these real seams: full source/consumer and
ownership contracts, provider compile capabilities, imported object recipes,
workspace/lifetime plans, source-view reuse opportunities and attributable
section costs. The actual emitted link and hardware regime must remain part of
candidate identity. Unknown costs and unmeasured combinations remain unknown.
Read-only source-view analysis does not itself grant a physical cache, data
reuse or an approximation policy.

Recent portable host optimizations prove ownership and representation before
changing traffic: private uniform buffers can fill their copy destinations
directly, and distinct fresh strided allocations can copy their longest common
contiguous suffix with ordinary runtime memcpy. Unknown aliases, shared roots,
dynamic extents or incompatible layouts are refused. These optional passes live
in Merlin and preserve normal pipeline coverage receipts. Their current whole
instruction reductions are0.465% and0.509%; stock gains remain unmeasured.

The exact i64 reconstruction helper also belongs in Merlin: a canonical numeric
range proof and disjoint scratch establish integer updates and one final exact
binary64 conversion independently of the producer target. Gemmini zero-tile
initialization, residency/coalescing and ISA/resource legality remain OOT. The
zero-tile/support composition loses1.7625% on the complete original bounded
capsule and is disabled; legal numeric proofs alone do not establish profitability.

Shared fast-screen fitting and grouped validation live in Merlin. Executed
Gemmini opcode/operand extraction and the optional isolated simulator hook live
in OOT. Provider pointers describe observed work; fitting/composition reuse the
existing shared calibration and resource abstractions. Missing physical traffic
or dependency/overlap facts are unknown. Independent coefficients must pass
section and whole known-reference checks plus within-workload ranking before
they can guide automatic phase1/2 selection. The current counter-only and
additive operand diagnostics both fail; neither is enabled as a cost provider.

Production transforms match typed operations, operand/result mappings, numeric contracts, layouts
and resource facts. They do not select an implementation by a model name. Source ordinals and hashes
bind a chosen implementation to the exact source; they are not profitability rules. This is an
explicit invariant in [AGENTS.md](../AGENTS.md).

- Host packet scheduling, ownership, compilation and dispatch fixes apply to any source satisfying
  their structural proof. Ordered-FMA and BF16 widening retain their source reduction contract.
- Device residency/prefetch/tiling applies when shape, scratchpad/accumulator capacity and live
  transfer intervals prove it legal. Independent nonbenchmark shapes and tails are tested.
- Numeric certificates are recomputed from each source's actual scales, rounding and value domains.
  The derivation generalizes; the resulting coefficient/table belongs to that particular operation.
- Explicit approximations require qualification against each model's unchanged accuracy criterion.
  One capture passing is not a universal accuracy claim.

Measured performance is narrower than applicability. A legal shared transform can lose on another
shape, blocking factor or composition. The normal calibrated selection path retains actual measured
costs and unknowns so those choices can be evaluated independently.


## Source-level redundancy, physical views and exact store alternatives

Merlin's typed broadcast-source math pass moves a pure smaller-domain DAG
outside its broadcast consumer before bufferization. It preserves source
operation order, casts, precision and live uses and refuses unknown effects,
dynamic/empty domains and strict scopes. The actual complete first source
normalization has a64.6175%paired GSIM mean reduction with all16,384outputs;
full256,000word native/Torch/strict gates pass, but whole stock cycles remain
pending. Allocation and traffic are part of the capsule timing; its small
allocator differs from the frozen whole allocator.

Merlin also owns static matrix-view proof and source consumer acceptance under
explicit full-write producer and read-only consumer contracts. OOT owns the
physical segmented DMA strategy, rank/byte ABI and emitted device code.
The matched original-consumed projection measures37.838%less complete GSIM
cost; the full normal source/provider route passes all1,000original words.
This is an actual compiler rewrite, not an address substitution in a bespoke
benchmark. Full stock timing remains separate.

Preserving an ordinary CPU command loop uses a reusable Merlin LLVM metadata
helper. Dynamic accumulator-row encoding and resource/range closure belong in
OOT. An explicit compiler/export option now emits the strategy through the
normal source route. Its independent complete convolution capsule improves
4.0067%; a complete current stride2geometry with independent inputs improves
1.7326%. Both normal upstream compilation and an arm retaining the actual1903
host/runtime pass all original words. The larger object-size reductions are
compiled byte counts rather than cache-miss/cycle evidence; stock1928 is pending.

Complete-domain exact multiple-readout certificates and portable packed pair
scanning belong in Merlin. Target scale/store implementation, accumulator
lifetimes and a distinct two-output device ABI belong in OOT. Numerical proof
covers every reachable output pair; actual sample disagreement counts are cost
evidence and never eligibility. Both stores, extra storage, final scan and
correction must be timed together. The complete compiler-copy readout capsules
reduce their ROIs by71.41%/70.40%. Normal preparation now proves a fresh,
sole-use uninitialized scratch producer and rewrites its actual type and
allocation to i8 before lowering. The inherited oversized-i32 diagnostic
remains separate. Actual byte extents, ownership, host/device ABI and full
original native/strict output close in the normal and frozen1903 builds;
stock1930 is queued. Generic
compiler byte-copy selection stays in Merlin; target catalog dependency and
storage guards stay in OOT.

Full Smol stock1906 now measures258.622B cycles with all original words exact.
Its384CPU BF16 contractions represent19.327B ordered f32 source FMAs; separate
head experiments are not whole-model device coverage. Merlin's generic source
analysis now closes48original groups at BF16 endpoints without live f32 escapes,
retaining each original eight-contraction/50-operation DAG. Numerical obligations
remain separate. OOT should bind
the proved group to actual device partials/ABI/resources. Live f32max, source
exp, denominator and ordered PV paths prevent an early BF16-only substitution.
This is a compiler coverage/integration task, not a new accuracy allowance.
The same-image exact native control now closes all48groups and all1,600
original outputs. The one-bin bounded and center-only policies fail113and121
outputs respectively; both remain disabled under the unchanged elementwise
gate. Actual OOT integer-product/readout checks succeed, while host certificate
cost and source replay still prevent a profitable whole attention replacement.
Generic tighter arithmetic bounds, consumer-frontier proofs and immutable
SSA/lifetime preparation commoning belong in Merlin. Packed representations,
device resources and primitive schedules belong in OOT; pointer-value caches
or model-name selectors do not supply compiler legality.
Recovering the user's earlier approximately4B route now also requires an audit
of source/workload, timing boundary, numerical criterion and emitted ISA. The
current qualified build is not a claim about the best historical implementation.

The CPU-footprint additive fast screen is also disabled. It resolves only12of24
held-out reference physical groups and has64.53%maximum resolved section error.
Whole prediction and ranking are unknown. Fitting the reference labels or
extrapolating repeated groups would not qualify it.
[Negative screen](perf_records/cpu_footprint_fast_estimate_partial_check.json).


### 2026-10-06: measured compiler capability boundary

Merlin's `SourceNumericContract` now treats standard finite classification as an
independent default-off compiler choice, alongside FMA and object copies. It
owns numeric observations, refusal and the portable executor hooks; OOT pins the
actual compiler/header/LLVM/object and executes the target representation and
complete device-product capsule. Classification alone removes29.9373% of the
matched production group's retired instructions, without changing a target
kernel, source math, original input, accepted carrier or consumer observation.
The native helper is byteidentical to the accepted full48 release. This is a
host compiler improvement; it is not a whole FireSim result.

Phase1 should expose explicit numeric/effect/interposition obligations and
actual component compilation receipts through the normal host-provider seam.
Phase2 should offer separately priced compiler capabilities and stage-cost
census, preserving complete source-consumer and fallback proofs. A library import
or smaller object is evidence for a hypothesis, never a dynamic price. Retain
static source opportunity, logical traffic, physical traffic, instruction proxy
and hardware cycles as separate fields. Reuse a prior correctness gate only
when the actual compiled object, complete inputs, contracts and observations
are proven identical; do not infer identity from shape or API alone.

### 2026-10-06: output materialization and cost scope

The Tiny two-product experiment exposes a generic compiler pipeline problem:
an unchanged loop-carried memref hides the caller's destination from public
result conversion. Merlin now offers ordinary upstream canonicalization at the
proved unique bufferization/result-conversion boundary. The exact source packet
goes from20.49% slower to52.49% faster in complete GSIM after eliminating its
65,600-byte temporary and65,536-byte output copy. This is a portable ownership
and materialization improvement; the target provides measurement and ELF audits.
Full original256000-word native/strict/Torch gates pass; hardware timing is pending.

Phase1/2 edit surfaces should include ordered pipeline anchors, buffer identity,
public destination forwarding and emitted allocation/copy costs. The alias and
ownership proof remains upstream, with explicit refusal for ambiguous anchors.
Numeric capabilities independently state interposition, errno, FP flags,
nonfinite and signedzero obligations. Merlin owns these algorithms and contracts;
OOT owns literal CPU instructions, ABI and hardware resource facts. Provider-ROI
costs must exclude post-ROI validation: the236M-instruction BF16 trunc helper
was entirely in the validation oracle, and is not a provider optimization target.

Unique Tiny ELFs duplicate about2.2GB of immutable weight payload even when
only host code changes. Baseline duplicates already share immutable inodes;
different final executable bytes cannot be hardlinked. A future generic split
of code and content-addressed payload could avoid this storage cost, but requires
explicit load/relocation/address/alignment/lifetime contracts and fresh execution
qualification. Loader and ELF layout implementations remain target-owned. No
loader change or payload movement is part of the current performance candidates.

## Source compilation and numerical proofs added at the latest checkpoint

Generic exact binary64-to-binary32 floor/ceil permission and finite domain
proofs are Merlin. Static RISC-V rounding fields are OOT. Four-cell polynomial
scheduling and rectangular tensor contraction scheduling are Merlin; CPU
register/ISA cost and hardware evidence remain OOT. Whole gains require
source-compatible compositions, including allocation, copying and replay.

The installed compiler now carries the transitive source-runtime headers and
templates in its resource manifest. Baseline and optimized provider emission
compile from the declared installed resources alone. Producer-bound coverage
uses C identifier/punctuation tokens, with no shared-library regex dependency.
These fixes generalize to every backend that consumes the portable source
certificate and do not change its numeric gate.

The exact absolute-product experiment is also portable Merlin math and private
workspace orchestration. It can trade an extra exact integer product for tighter
source-FMA bounds; the provider supplies primitive resources/costs. Its first
complete cost is negative in instructions, so no automatic policy is enabled.
The readback/MAC ratio is only a priced diagnostic hypothesis, not a promoted
shape threshold. Different timer/CPI regimes remain separate.


Independent finite source FMA batch scheduling is split at the numeric contract:
Merlin supplies the lane, source arithmetic/effects, finite prefix and private
owner obligations; OOT supplies early-clobber register constraints and explicit
RV64GC instruction order. Default behavior is unchanged. Complete original
group and all48native gates pass, but measured0.411%instruction gain is not
a hardware latency claim. Redundant producer/result validation is the next
generic proof-flow opportunity; it must preserve checked public paths.


## Current measured scope and additional integration lessons

Merlin owned changes are published as13squashed main topics; model2MLIR
eight fixes are verified upstream. The conservative family count is55
(28Merlin fixes,19reusable improvements,8model2MLIR fixes), not55performance
wins. Optional alternatives still need explicit selection and full-model gates.

Exact integer mean splits at a reusable Merlin source certificate and guarded
integer-sum finishing API; OOT emits the typed matrix-by-ones reduction and its
resource/fence contract. Full-K weight residency and paired source-stride are
OOT physical schedules. Stock2018/2020/2022 are measured controlled wins; their
composition is a new stock experiment rather than an additive forecast.

The current profile exposes selected semantic routes versus original stored
partial-link leaves as different entities. Preserve both in generic compilation
recipes and emitted-work/CCA witnesses. A renamed selected route cannot safely
reconstruct the old component from its selected objects. All executable bytes,
including unused original leaves, remain subject to final instruction audit.

Complete diagonal polynomial, denominator and reciprocal cost screens reject
legal alternatives. A valid numerical certificate and more accelerator calls
do not establish a useful implementation. Phase1 must price full preparation,
readback and refinement; phase2 must retain compatible accepted options or
report the exact source/resource/cost reason for omission. Current2013's
residual/spatial/outside costs are actually measured; new2023 section costs
remain unknown.
