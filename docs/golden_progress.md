# Primitive-only Gemmini golden: current evidence

The requested targets are ResNet-50 around 22M FireSim model cycles, full `SY_model_smolvla` near 5 billion, and TinyLlama around **300M** (user clarification on 2026-10-06), all on `FireSimGemminiRocketConfig` with **zero** Gemmini `LOOP_*` instructions in the final linked ELF. Ordinary RISC-V branch loops repeat the xDSL Gemmini primitive tile schedule.

## Latest verified whole-model results (2026-10-06 UTC)

| Model/capture | Stock FireSim forward cycles | Correctness evidence | Receipt |
|---|---:|---|---|
| ResNet exact52/wide16, transfer/banked residual, host quantization packets and exact paired readout | 34,905,135 (1930) | All 1,000 original output words exact; 3.3171% below isolated1903 control. Stock hardware and retained pre-teardown staging observations pinned; segmented inputs are a separate arm | [1930](perf_records/resnet_paired_readout_stock1930.json) |
| Full 22-layer pretrained TinyLlama, 8 tokens, source normalization hoist, ordered contractions/K2, B-prefetch and fresh writer ownership | 461,389,700 (1926) | All 256,000 original compiled words unchanged; original Torch gate passes. 13.1211% below isolated1880 control; not composed with1902/1920 | [1926](perf_records/tiny_norm_hoist_stock1926.json) |
| Full SmolVLA, explicit portable expf-via-double policy, original numeric gate retained | 258,621,872,969 (1906) | All 1,600 original output words bitexact on stock hardware; original atol=0.03125/rtol=0.02 retained. ELF and bitstream identity recorded before cleanup | [1906 stock result](perf_records/smol1906_stock_hardware.json), [normal build](perf_records/smol_normal_host_math_policy_equivalence.json) |

Every listed hardware result pins its final zero-FSM ELF, actual staged ELF,
stock bitstream and job-owned output. ResNet is a random-weight semantic capture;
Tiny uses the full pretrained checkpoint. These results do not establish the
remaining performance targets or pretrained ResNet accuracy.

[Optimization journey and token ledger](golden_optimization_journey.md) records matched gains,
regressions, ownership and actual owned-thread counters. The current DeviceRouting catalog route
uses shared calibrated selection for explicitly supplied, source-bound contraction alternatives; it does not invoke shared whole-program search. Five compiler export/selection/build commands now expose eight AST edit surfaces. A measured singleton contraction invokes the existing shared solver and emits its actual selected object: 3,234→2,357 full-fixture GSIM cycles, every output/guard exact. This does not transfer fixture prices to a whole model. [Measured selection](perf_records/golden_calibrated_source_selection_qualification.json). The normal model route verifies a 3,434-node outlined identity plan and actual catalog/final ELF closure while compiling unchanged prepared source bytes. This remains a full-source identity admission gate. The normal build now additionally accepts `--contraction-calibrations` to run the existing shared measured selector per exact source contraction and compile its winner into the real device catalog. Independent full-model native/strict target execution closes all1,241 original i32 outputs and actual final ELF/symbol bindings. Whole-graph search and whole-model costs remain unknown. [Normal build selection](perf_records/golden_calibrated_normal_model_qualification.json). [Binding](perf_records/golden_model_plan_binding_qualification.json).

The first real Tiny source calibration now closes through that normal route: resident-A/B-prefetch
measures111,885→76,456cycles(31.67%) on a common-address synthetic signed-int8 GSIM pair
with amplitude21 at the source contraction shape. The complete
normal model retains all155 calls, selecting one source contraction and preserving154 other bindings.
All256,000 original compiled words, the Torch gate, strict RV64GC Spike and final zero-FSM audit pass.
A generic Merlin fix preserves different source-bound implementations when their tensor shapes agree.
This is one selected contraction, not a31.67% whole-model improvement. Expansion across equivalent
resource-legal contractions is being qualified before hardware submission.
[Measured schedule](perf_records/tiny_resident_a_prefetch_gsim.json),
[normal full model](perf_records/tiny_resident_a_prefetch_whole_spike.json).

The expanded normal native and strict-target builds preserve all256,000 original words and the Torch gate,
with44 selected source bindings,111 unchanged bindings and five shared implementation bodies.
Stock1911 completes at529,006,294cycles. Its normal build also uses the current
runtime compiler, so its comparison with1880 includes that runtime change.
The device-only1912comparison retains every original1880 host/runtime object
and completes at529,440,142cycles, a0.307%single-run improvement versus1880.
Both preserve all256,000original words and remain slower than1902. No champion
change or causal attribution to the small runtime difference follows.
[Controlled stock receipt](perf_records/tiny_resident_a_controlled_1912_hardware.json).
Reusing the one synthetic fixture price across those
source-equivalent implementations is a calibration assumption; actual model operands, addresses,
cache state and full-program timing have not been independently timed by that fixture.

Bounded compact-convolution next-K weight prefetch is also now an explicit general OOT option.
Its original-input matched capsule measures535,839→499,034GSIMcycles(6.87%), all50,176outputs
and4,096guards exact. An independent5×5/Cin32/Cout19 i32/tail capsule also passes, and47focused
resource/order/source-policy tests pass. Only five source-qualified device kernels change in
the new whole arm. Whole-model hardware qualification is separate.
[Schedule and proof](compact_weight_prefetch_schedule.md),
[capsule](perf_records/compact_weight_prefetch_capsule.json).

[Infrastructure and optimization ownership](infra_vs_dialect.md) separates correctness/integration
fixes, portable host performance and target scheduling, with matched measurements for each gain.

### Latest integration checkpoint (2026-10-06 UTC)

Stock1926 and1930 are the new Tiny and ResNet best measured whole-model arms.
ResNet1930 exact paired readout measures34,905,135cycles,0.7043% below the
separate segmented-input1927 arm35,152,730; these gains are not added.
Tiny still needs about35% below461.390M to reach300M. The isolated constant-clamp
arm1920 measured527,211,739; it was effectively tied with1902 and is now slower
than1926. No gains are added across arms.

The complete reentrant Smol attention executor now passes all1,600 original
native words, every9,437,184 original quantized byte and12,288 escaping scales
across48 groups. One caller-owned121,963,584-byte workspace replaces mutable
global numeric state and executor allocation. The normal model builder now
accepts separately pinned host/provider objects and retained source fallbacks,
with typed ABI/import/final-link checks; the device catalog still requires
closed kernels. Default-disabled hook builds are byte-identical. Full new
pooled whole-target integration remains pending.
[Native workspace](perf_records/smol_quant_frontier_workspace_native_journey.json).

The actual normal-compiler provider now has separately closed FMA/copy, endpoint
row preparation and exact-floor capabilities. The complete original12-head
production capsule drops9,624,336,737→5,709,055,956→5,034,507,191 functional
Spike retired instructions. Full48 native execution preserves all1,600 original
words,9,437,184 compiled quantized bytes and12,288 escaping BF16scales.
[Production row/floor](perf_records/attention_prepared_rows_floor_target.json),
[full48 original consumer gate](perf_records/smol_numeric_rows_floor_full48_journey.json).

The next explicit standard finite-classification capability reduces that same
production capsule5,034,507,191→3,527,309,706instructions(29.9373%), with all
196,608 accepted carriers and guards exact, the same refinement/replay counters,
480 product calls and86,507,520logical readbackbytes. All device/driver/bridge/data
objects are unchanged. An independent565,925-check target representation gate
covers signed zeros, subnormals, infinities and both NaNclasses in five rounding
modes. Disabling the new choice reproduces the prior provider object byteexact.
The native compiled helper remains byteidentical to the accepted full48row/floor
SO; independent identity/consumer reuse is closed. These are section instruction
savings, not stock cycles or a5B whole-model claim. Normal pooled whole-target
integration is running; performance admission remains separate.
[Classification proof and matched measurement](perf_records/smol_standard_classification_complete_group.json).

An independently admitted absolute-value capability further reduces this same
complete-group instruction count3,527,309,706→3,370,349,620(4.4499%). Its default
object remains byteidentical.565,925 actual target representation checks pass
across five rounding modes; all196,608 accepted carriers/guards and refinement
counters remain unchanged. The new frozen native helper independently passes
all48 original input bindings, all1,600 original whole outputs and all9,437,184
compiled quantized bytes/12,288 escaping scales.209 live source/compiler/artifact
pins and both final executable-section audits close. No hardware timing is
claimed for this capability.
[Absolute-value proof](perf_records/smol_standard_absolute_complete_group.json),
[full48 compiled consumers](perf_records/smol_numeric_absolute_full48_journey.json).

The ordinary prepared whole Smol model now compiles and links all17 provider
objects through their actual companion LLVM ABI, with one pooled workspace and
retained source fallback. Its new physical native route passes all1,600 original
outputs with48 calls/zero fallback; forcing null workspace independently checks
the actual compiled source fallback's196,608 endpoint words and input/descriptor
ownership. Whole strict-target execution is running. Stock1944/1945 compare the
two qualified row/floor and classification complete-group capsules; they are
not whole models. Stock1947 measures only the qualified ResNet residual-output
arm against1903, without composing other wins.
[Ordinary Smol build](perf_records/smol_normal_workspace_provider_build.json),
[physical native/fallback gate](perf_records/smol_normal_workspace_provider_native.json),
[residual release](perf_records/residual_output_guard_release_closure.json).


For the same complete original12-head device group, portable exact floor and
source multiply/row invariants lower the functional Spike instruction proxy
3,528,972,523→3,494,086,004→3,303,437,300. All196,608 quantized observations,
256 scales,4,461,440 replay FMAs and480 product calls remain unchanged.
These are capsule counters, **not** stock hardware cycles or a5B whole forecast.
[Reclosed arithmetic receipts](perf_records/smol_exact_row_arithmetic_complete_group.json).
The next directed-conversion arm further reduces that counter to3,121,235,451
(5.5155% below the row arm), with all original quantized values/scales exact and
576 fewer replay FMAs. One additional internal carrier differs; original source
gold and quantization observations remain fixed. Full new whole-model/production
qualification is separate.
[Directed narrowing](perf_records/smol_directed_cast_complete_group.json).

Stock packing job1909 now verifies776,043,123→665,638,655cycles across
matched complete packing sequences (14.2266% reduction), all12output digests,
representation metadata and guards exact. This is a packing-section result,
not a whole attention or model speedup.
[Hardware receipt](perf_records/smol_integer_packing_1909_hardware.json).
Compact weight-prefetch job1910 completes at38,332,743cycles, only0.3540%
below its1897control and above the36,102,704current ResNet best. It does not
change the selected implementation. Jobs1911and1912are now complete; both
remain above the selected Tiny result.
[1910](perf_records/resnet_compact_weight_prefetch_1910_hardware.json).

Tiny broadcast-axis pointwise sharing now passes all256,000original native,
Torch and strict target words. The normal compiler control/relink is byteexact
and onlymodel.o changes. Stock1932is queued versus1880. Its actual-source
ABBA GSIM capsule is incomplete after the first control interval; candidate
timing remains unknown. [Qualification](perf_records/tiny_broadcast_packet_whole_qualification.json).

Full Smol stock1906 completed at
02:57:36Z after submission at22:52:37Z. Engine elapsed was8,918.2seconds
(about2h29m); submission to completion was about4h5m, including queue/setup.
Its258.622B forward cycles are51.724times the5B target. Correctness now has
stock whole-model evidence; the performance target remains unmet. Current-best
ResNet profile1919 completed on stock hardware:36,138,975 forward cycles conserve
30,678,196 device-wrapper cycles plus5,460,779 host-gap cycles, all70 events and
1,000 original outputs exact. This profiles the prior1903 arm, not new1927.
[Stock profile](perf_records/resnet1903_stock1919_profile.json).
The same qualified ELF
completed in the separately pinned GSIM memory regime:33,939,464forward cycles
conserve29,422,251device-wrapper cycles plus4,517,213host-gap cycles, including
the73,136tail. All1,000 original words and all70boundary events pass. This is
not stock1919 and does not repartition the stock13.72M reference gap.
[Completed profile](perf_records/resnet1903_gsim_conserved_profile.json).

- **Tiny typed source math:** Merlin's explicit
  `hoist_broadcast_source_rsqrt` feature proves a smaller affine iteration domain
  before bufferization, retaining source casts, arithmetic order and live uses.
  Actual control code repeats737,280 logical rsqrt calls for360row inputs.
  The complete original first-normalization capsule measures paired means
  2,369,163→838,269GSIMcycles(-64.6175%), all16,384outputs/128guards exact.
  Five rounding modes and identical sticky flags are independently qualified.
  Capsule allocation/traffic are timed with a small common bump allocator;
  whole frozen allocator/address context needs stock qualification. The full
  model native/Torch/strict/noFSM gates now pass all256,000original words with
  onlymodel.o changed. Stock1926 verifies461,389,700cycles versus1880;
  this is the current best Tiny result and remains above300M.
  [Source/capsule](tiny-broadcast-math-20261005.md).
- **ResNet borrowed projection input:** typed Merlin ownership/view acceptance
  and an OOT segmented resident-input schedule remove copied logical matrices.
  The matched original consumed401,408bytes+4,096guards measure
  866,791→538,815GSIMcycles(-37.838%). Unread physical owner cells in that
  isolated fixture are synthetic zeros, not original producer state. The normal
  source/catalog build now binds three accepted views and has passed all1,000
  original native/strict outputs; frozen1903 comparison stock1927 verifies
  35,152,730cycles, the current best ResNet result.
  [Measured projection](perf_records/segmented_input_original_projection_gsim.json),
  [normal compiler route](segmented_input_view.md).
- **Compact ordinary CPU spatial command loops:** the explicit OOT
  `spatial_command_loops` compiler/export option preserves the exact primitive
  sequence, first weight flip, stationary reuse, accumulator initialization and
  bounded tails. Shared Merlin loop metadata prevents LLVM from undoing the
  choice. The complete independent15,257-i32-output AB/BA capsule reduces
  paired mean4.0067%; current schedule geometries compile from120,366→39,744,
  211,352→63,498and82,490→28,850text bytes. Bytes are not cache misses or
  measured cycle savings. The complete50,176-output current stride2geometry
  capsule now measures727,093/726,232→716,175/711,970GSIMcycles(-1.7326%),
  using independent deterministic inputs. Normal upstream and frozen1903
  builds both pass all1,000original words/native/strict/noFSM. All52adapters,
  41unselected requant kernels and original host/runtime/weights are retained
  in the controlled arm;11source-derived kernels change. Fresh normal host
  bytes differ from1903 and are disclosed separately. Stock1928 is queued
  versus1903; no whole cycle gain is established.
  [Capsules](perf_records/flat_spatial_command_loop_qualification.json),
  [normal/frozen full gates](perf_records/resnet_spatial_cpu_loops_normal_whole_qualification.json).
- **Exact readout:** source-proven domain selection is now an ordinary OOT
  provider option backed by Merlin numeric proofs and explicit storage guards.
  Both actual readout families pass complete AB/BA RTL capsules and the normal
  full-source build passes all1,000original native/strict words. The normal
  runtime differs from historical1903 and is disclosed; no isolated whole
  hardware result is inferred. A further complete-domain pair proof permits
  two scaled i8stores plus a full decoder, avoidingi32readback and dot replay.
  Original75,264outputs and independent boundary/tie/conversion cases pass on
  the target. Complete readout capsules including both stores and full decoding
  reduce the two families by about71.0%/70.6%; independent convolution tails
  also pass. The normal preparation now proves the fresh, sole-use scratch
  producer before changing its actual allocation from i32 to i8. The two
  selected host allocations are 50,240 and 25,152 bytes including alignment.
  Normal and frozen1903 builds pass all1,000 original native/strict words and
  the complete zero-FSM audit. The earlier oversized-scratch build remains a
  diagnostic. Explicit compiler byte-copy selection also passes its own
  complete RTL pairs: 71.41% and 70.40% reductions for the two readout ROIs.
  The qualified typed allocation candidate is queued as stock1930; a whole
  cycle improvement remains unknown.
  [Normal domain](perf_records/resnet_normal_producer_domain_qualification.json),
  [pair proof](perf_records/resnet_exact_pair_readout_feasibility.json),
  [target store contract](paired_readout_store.md),
  [complete measured decoder](perf_records/exact_pair_readout_matched_gsim.json),
  [compiler-copy RTL](perf_records/exact_pair_readout_builtin_matched_gsim.json),
  [actual typed whole route](perf_records/resnet_paired_readout_typed_whole_qualification.json).

- **Stem ordinary CPU command loops:** the normal upstream option retains the
  original primitive schedule, pooling, input traversal, initialization and
  bounded tails. The complete original 200,704-byte output capsule improves
  1,335,580 to 1,294,977 GSIM cycles (3.0401%), with all output bytes and
  4,096 guards exact. An independent shape improves 11.53%. Compiled text
  shrinks 87.8%; that is not a measured cycle reduction. The normal build and
  stem-only frozen1903 build pass all1,000 original native/strict words and
  the zero-FSM audit. Stock1929 is queued.
  [Capsules](perf_records/stem_spatial_command_loop_capsules.json),
  [whole qualification](perf_records/stem_spatial_command_loop_whole_qualification.json).

- **Smol source coverage:** the current whole route leaves384BF16 contractions
  as ordered CPU f32 FMA loops, totaling19,327,352,832 source FMAs. This is an
  exact source-work census, not an attributed cycle total. Optimized complete
  attention-head capsules are separate experiments and are not enabled in1906.
  The source QK result remains live through f32max, source polynomial exp and
  denominator reductions; PV retains ordered f32 partial additions and scaling
  before BF16 conversion. A narrower early BF16 rounding substitution is
  therefore refused. Generic Merlin source analysis now closes 48 groups,
  each retaining eight contractions and 50 original operations, through a
  BF16 endpoint with no live f32 escape. Parent revalidation on the immutable
  original source passes all 48 groups. Group closure does not establish a
  numerical certificate, profitable implementation or whole-model speedup.
  Ordinary source-exact partitioning/binding is now integrated in Merlin;
  the full-source control preserves384roots and48typed calls. An actual group0
  read-only tap preserves every original1,600full-model output word, and its
  extracted50-op source helper matches all196,608BF16words independently.
  A fixed one-adjacent-BF16-bin interval policy reduces local source fallback
  from94,066,240to8,423,360FMAs (23.36%to2.09%of402,653,184sourceFMAs),
  with53local word changes inside the proved bound. Exact-policy control is
  retained. Independent random/mixed/all-masked full-shape source checks pass.
  The original whole-model atol=.03125/rtol=.02 gate now rejects this policy:
  113of1,600outputs fail. The identical compiled image and writer ABI with
  exact endpoint reconstruction passes all1,600original words bitexact;
  all48endpoints also match independently compiled original groups. This
  isolates numerical propagation, rather than a grouping or ABI error.
  Exact reconstruction replays5,507,717,696source FMAs(28.497%); the rejected
  bounded policy replays429,497,664(2.222%). A single separate center-only
  reconstruction retaining the full source DAG also fails121of1,600outputs.
  Neither approximation is enabled and the gate remains unchanged.
  [Paired control and rejection](perf_records/smol_source_group_bounded_exact_journey.json).
  Main QK/PV products have independent actual target integer-plane checks;
  this does not establish whole-model target dispatch or profitability.
  The M256/head0 target screen verifies1,802,240i32readout words, but its
  prepared-polynomial path retires456,096,296instructions versus232,955,195
  for source CPU code. These are instruction counts, not FireSim cycles.
  actual plane packing, readouts, certificate scans, source replay and host
  softmax must be timed together before promotion.
  [Local feasibility](perf_records/closed_bf16_endpoint_feasibility.json),
  [independent reclosure](perf_records/closed_bf16_endpoint_independent_reclosure.json).
  [Parent revalidation](perf_records/smol1906_source_group_parent_reclosure.json).

The user reported an earlier approximately4B Smol implementation. Recovering its
provenance and fast route is now a priority. The 258.622B result describes the
current qualified build and does not claim the best historical Smol result.
The owned historical610plan has cached prefix KV/masks, flow state and timestep
inputs, with no image input; it represents a cached-prefix denoising trajectory.
Its arithmetic3.3B-per-step average does not establish image-forward equivalence.
Separate Exo3.22–3.72B reports remain unresolved in permitted owned artifacts.
[Historical scope](perf_records/historical_smol_cached_prefix_scope.json).
Separate earlier cycle reports require workload, timing boundary, accuracy and
ISA comparison before their performance can be transferred to this build. The
original atol=.03125/rtol=.02 gate remains fixed; bitexact whole output is beyond
that required criterion.
The retained owned job610 reports33.085B cycles and declares a ten-step
trajectory with an older atol=.06/rtol=.05 gate. Its actual ELF contains FSM
instructions. Dividing by ten gives an arithmetic average, not a separately
measured step; the sibling harness's ELF differs, so exact linked timing scope
is not established by that harness. This artifact is not a drop-in replacement
for1906. The separate near4B Exo reference source and gate remain unclosed in
permitted owned copies. [Historical audit](perf_records/historical_smol_job610_audit.json).

- **Merlin host copies:** optional private uniform-fill copying removes four
  copies and two temporaries under a complete ownership/order proof. Frozen1903
  runtime whole validation preserves all1,000 original words and reduces
  retired instructions9,246,562→9,203,553. A separate longest-common-contiguous-
  suffix rewrite preserves fresh distinct allocation roots and lowers three
  strided copies to ordinary contiguous memcpy calls:
  9,246,562→9,199,529instructions. These are0.465%/0.509% instruction changes;
  neither is a measured stock-cycle improvement. The actual post-bufferization
  transfers are i8 with256/512/1024byte suffixes. Earlier f32 semantic view
  offsets are element offsets and do not authorize direct i8 device loads.
  [Uniform proof](perf_records/resnet_uniform_fill_copy_whole_qualification.json),
  [suffix proof](perf_records/resnet_contiguous_suffix_copy_whole_qualification.json).
  The border+uniform+suffix composition also preserves every original word
  and frozen runtime:9,246,562→9,156,449instructions(-0.975%).
  [Composition](perf_records/resnet_host_copy_composition_qualification.json).
- **OOT resident-A DMA coalescing:** general resource/layout proofs combine
  legal adjacent input tiles without changing allocated cells or reduction
  order. Complete original capsules measure288,847→277,410 and
  563,901→552,245GSIMcycles, with all outputs and guards exact. Stock1922 is
  queued against the frozen1897 control; whole-model gain and composition with
  current1903 remain unknown.
- **Exact Smol reconstruction:** a new explicit Merlin helper accumulates
  canonical weighted i32 groups in separately owned i64 scratch, then converts
  once to binary64. Its existing absolute-prefix proof prevents overflow and
  rounding. Constant weights inside each case produce constant shifts in
  ordinary CPU codegen without shifting negative signed C values. Original
  full-head native and strict target words and replay/readback counts remain
  exact. Complete unchanged-device RTL pairs reduce cycles7.3715% for the
  variable-weight arm and14.4479% for its separate constant-case arm. The
  latter retains30calls/47,185,920readback bytes and all65,536 original words;
  one complete-head stock comparison is queued as1924 versus1917. Neither result is a
  full-head or whole-model cycle forecast.
  The earlier composed encoded-zero/support arm was rejected:
  original bounded paired GSIM mean+1.7625%, despite exact outputs and guards.
  [Selected exact head](smol-exact-i64-reconstruction-20261005.md),
  [negative journey](smol-encoded-zero-groups-20261005.md).
- **Fast cycle estimation:** shared Merlin validation now requires grouped
  held-out absolute errors and within-workload ranking, with unresolved or
  out-of-domain features remaining unknown. The counter-only independent fit
  predicts45.95M for Jack's measured22.39M and fails by105.25%; it is disabled.
  Isolated copied-engine operand telemetry preserves stdout and the complete
  PC histogram byte for byte and binds70 actual current entries and54+16
  reference entries. It supplies command geometry and requested payload,
  while physical DRAM traffic and overlap remain unknown. The first additive
  operand fit also refuses nonnegative terms. The CPU-role census now closes
  all70 actual current bodies and24 reference physical groups covering54
  measured calls, with zero unknown encodings. Stem touched instruction bytes
  are174,566 current versus24,042 reference. This is observed code footprint,
  not cache misses or cycles. Command dependencies remain unknown.
  [Counter check](perf_records/counter_only_fast_estimate_reference_check.json),
  [operand fit refusal](perf_records/additive_operand_fast_estimate_fit_rejected.json),
  [telemetry](../support/gemmini_spike_telemetry/README.md).

Stock reference job1876 now reproduces Jack's permitted ZIP reference at **22,387,449 cycles**,
with all1,000 reference logits passing its self-check and54 buffered layer timings.
The reference has different numerical coefficients and physical boundaries; this result does not
qualify our original-source model. [Same-stock reference](perf_records/q1013_diagnostic_reference_firesim.json).

[Remaining ResNet gap](resnet_current_gap_status.md) records the measured historical
section comparison and the unresolved attribution of the current13.72M-cycle gap.
The qualified current-best1903 profile is queued as1919; historical1874 timings are not
reported as current section costs.

## Current compiler work and ownership

The OOT dialect is a general compiler backend. Production decisions use input semantics,
shapes, layouts, numeric contracts and hardware capabilities; model names, captured IDs and
benchmark constants are confined to experiment selection/receipts. See [AGENTS.md](../AGENTS.md).

Merlin now owns complete requantization transition proofs, CPU integer readout and guarded
contiguous/packed mean code generation. OOT delegates to those owners and retains device
schedules, instruction lowering, resource facts and target ABI wrappers. The extraction
preserves all proof and generated C bytes for five independent cases and both actual source
readouts; generic numeric tests moved to core. [Extraction receipt](perf_records/generic_numeric_owner_extraction.json).

Structural bounded host RNE legalization and common tensor permutation proofs now also live
in Merlin; the OOT provider delegates with unchanged qualified object/C bytes. The general
resident compiler policy chooses the same seven kernels from shape/padding/resource facts,
without a resident source-ID selection list, and reproduces the qualified device object and
rewritten IR byte for byte. [Policy equivalence](perf_records/resnet_resident_compiler_policy_equivalence.json).

Generic source-bound post-offload callbacks and explicit full-write/result-identity contracts
now let the normal model builder expose fresh output ownership. All155 Tiny calls are covered,
with native and actual Spike outputs unchanged. Isolated hardware arm1842 measures607,616,796 cycles versus610,246,484 in1835 (0.431% lower); device bytes are unchanged and prefetch is off. The separately gated ownership+prefetch composition1846 verifies569,151,067 cycles,1,946,440 (0.341%) below1841, with identical device bytes. This is a single-run small improvement, without a variance-adjusted claim.
Eight-output/K2 plus device prefetch arm1841 now verifies571,097,507 hardware cycles,0.300% below1839; this small difference is one run, not a variance-adjusted claim.
The current1837 four-output/K4 hardware profile conserves155 calls and attributes611,478,982
interior cycles to219,073,155 device and392,405,827 host; intervals include all intervening
operations after fusion, not one source operation. [Profile](perf_records/tiny_current_k4_profile_firesim1837.json).

The next ResNet resident channel-loop/grouped-row composition passes all1,000 outputs in native
and actual final-ELF Spike, with an identical1836 host object and only seven device kernel
objects changed. Stock FireSim1844 verifies43,514,726 cycles,1.582% slower than1836; that arm was rejected. That target composition remains qualified; newer host-packet arm1886 is now best in the table above. The rejected1844 arm retires9,467,083 instructions versus9,946,365, demonstrating that fewer Spike instructions did not imply a FireSim improvement. [Qualification](perf_records/resnet_resident_channel_loop_spike.json).

The1849 profile control has22,989,952 calculated padded mesh geometry cycles:
8,893,440 pointwise,7,990,272 direct,5,193,216 residual,784,000 pooled stem
and129,024 classifier. All70 kernels are bound to the actual1850 source profile,
which measures33,814,043 device cycles plus8,489,121 host cycles. This occupancy
describes the selected arithmetic and tiling, not every possible exact algorithm.
Pinned RTL can execute shorter WS waves when operand extents, garbage D and
transposer state allow it. This padded count is not a mandatory cycle floor.
Exact variable-wave issue accounting and alternative epilogues are being
investigated alongside host and transfer improvements.
[Current geometry](perf_records/resnet1849_current_issue_geometry.json).

General complete-M input residency and a shape/resource/command-cost comparison
of resident and banked transfer families are now implemented as explicit,
default-off OOT compiler options. Five actual capsules pass outputs, guards,
final noFSM and strict RV64GC Spike; the paired wide contraction saves22.69%
fenced GSIM kernel cycles, while total harness cycles increase. The whole
transfer-policy arm preserves1849 host/shim/weights bytes and all1,000 original
output bits in native and actual Spike, changing24 dense kernel objects.
Stock FireSim1853 verifies40,479,548 cycles,1,790,260 (4.235%) below1849.
This qualified the transfer arm. The independent residual arm1854 verifies
40,981,079 cycles,3.049% below1849. Both strategies compose in a candidate
with all1,000 original native and actual Spike words exact. The1861 UART counter39,201,279
lacks complete actual staging proof and remains historically **unverified**. Its strict immutable
rerun1874 now verifies39,201,279 with all1,000 original words exact and staged ELF/bitstream
closed:1,278,269cycles(3.158%) below1853. This was the best before1886 host packets.
[Verified composition](perf_records/resnet_transfer_residual_composed_firesim.json).
No additive whole-model gain is inferred.
[Schedule](multirow_resident_a.md),
[whole gate](perf_records/resnet_transfer_command_policy_spike.json).

The additional reference-parity worker identified a missing complete-input,
full-reduction weight-resident convolution schedule for wider feature maps.
The general stripe policy passes40 focused tests and independent/tail capsules.
H56/C64 saves15.89% fencedGSIM; H28/C128 saves19.22%, full outputs/guards exact.
All52 source numeric proofs are retained; only six device objects change versus1853.
Original1,000 whole native+strictSpike words remain exact. Stock1878 verifies39,754,283cycles,
725,265(1.792%) below1853 with actual staging closed. Target composition control1874 is39,201,279; the newer1886 host-packet arm is38,603,949.
The independently qualified stripe+transfer+residual composition now verifies38,468,933cycles
in stock1897,1.868% below1874 and0.350% below1886(single-run evidence);
its host/runtime/startup/weight objects remain identical1874. No additive gain is inferred.
[Verified stripe composition](perf_records/firesim1897_resnet_stripe_composition_verified.json).
[Isolated hardware](perf_records/resnet_resident_stripe_policy_firesim.json).
[Residency strategy](resident_stripe_conv.md).

Stock QK64 exact source replay measures1,137,275→910,407 cycles for one versus
four independent outputs (1856/1857),19.95% lower, all1,024 original words exact.
Generic Merlin selected BF16 widening includes allocation and conversion in
each replay capsule. QK4 LHS-only stock1872 measures828,986cycles,8.94% below1857.
PV4 stock1873 verifies168,142cycles; strict control1875 verifies180,776,
a 6.989% matched reduction, with all64 original words exact and staged hardware pinned.
[Strict PV control](perf_records/ordered_attention_pv_control_firesim.json). Both-operand M1 widening is rejected. QK16 passes original native/strictSpike/zeroFSM and stock1883 now verifies814,343cycles,1.766% below LHS-widened QK4 and10.552% below1857. [QK16](perf_records/ordered_attention_qk_lhs16_firesim.json). These are replay capsules, not a5B full
SmolVLA result. [Hardware pair](perf_records/attention_qk64_replay_firesim.json),
[complete widening costs](perf_records/ordered_replay_selected_widening_spike.json).

Smol's full native and actual target gates are now exact with the explicit generic math policy.
The first optimized RV64GC host compile timed out at900seconds on a large monolithic function.
Generic loop extraction retains the full native original gate bitexact all1,600 outputs. An initial77.4-second prototype also altered original function attributes; it is not final policy timing. The corrected generic helper-only inlining policy preserves original attributes, compiles through the normal pipeline in111.254seconds and passes all1,600 original native outputs exactly. Before the math policy both full actual targets failed89/1,600 with identical output bytes; these failed receipts remain historical evidence. The newer source-order schedule retired134,730,816,466instructions versus327,506,537,392; neither is a hardware-cycle result. All first-layer Q/K/V/output/fc1/fc2 integer A/B/C boundaries match the same accepted native path, which did not certify separately carried floating scales. [Six-stage trace](perf_records/smol_target_six_stages.json). A separately labeledO0 correctness ELF also builds,
passes the final no-FSM audit and runs in Spike. Reusable host fusion/compilation
scalability and cheaper explicitly gated device attention remain work. A source-derived post-vision
suffix now passes all1,600 original final bits on actual target execution with the old failing runtime
objects frozen; this localizes the remaining discrepancy to the vision prefix. The later fifth-block expf diagnosis is documented below. [Frozen-runtime suffix](perf_records/smol_postvision_suffix_frozen_runtime.json).
The diagnostic supplies a captured intermediate and is separate from whole-model qualification; the exact scalar vision
attention has19.33B MACs and cannot by itself establish the5B target.

General remaining-row B-slot prefetch selects seven pointwise kernels from shape/resource facts.
Its matched full-range capsule saves5.36%GSIM; all1,000 whole native/Spike words are exact with
unchanged1853 host/runtime/weights/shim. Stock1888 verifies40,148,896cycles,330,652(0.817%) below its1853 control, but slower than current1886 best; composition is unmeasured. Generic Merlin bounded RNE packets
save42.49% warm cycles on a complete quantization/packing capsule; the whole arm preserves original
1,000 words and all13 checked nonhost objects; stock1886 verifies38,603,949cycles. The first redundant dead-branch arm was
rejected before hardware. Tiny two-lane scalar pointwise packets save22.09% warmGSIM on the full
capsule; stock1880 now verifies531,072,370wholecycles,38,078,697(6.6904%) below1846, all256,000 original words/Torch gate exact and actual staging closed. [Hardware](perf_records/tiny_pointwise_packet_firesim.json). Other queued arms remain pending experiments.


The Smol diagnosis closed the first four isolated vision blocks, post-layernorm and
the language/action suffix against original captured states with the original failing runtime
frozen. The fifth vision block (index4) passes natively and diverges on the target; the next
isolated block passes. This localizes a reproducible failure without certifying the whole model.
Replacing only expf with diagnostic native results restores all786,432 original fifth-block bits, with zero lookup misses. The generic optional `(float)exp((double)x)` policy restores this entire block and full native/actual-target all1,600 final bits. Lookup stays diagnostic-only. The normal build reproduces all executable/runtime/weights/input bytes and loaded segment addresses, sizes and permissions, differing only in the diagnostic marker; its gate is explicitly inherited by that complete equivalence, without claiming a new replay. Stock1906 is the single qualified hardware baseline, cycles pending. [Full target](perf_records/smol_full_double_exp_target_exact.json), [normal build](perf_records/smol_normal_host_math_policy_equivalence.json), [causality](perf_records/smol_block4_expf_causality.json).
[Localization](perf_records/smol_later_vision_block_localization.json),
[corrected block2 and rejected diagnostic input](perf_records/smol_vision_block2_corrected_target.json).

The complete original attention-head hardware pair rejects the gamma certificate candidate:
stock1894 control2,618,580,085cycles versus1895 candidate3,077,601,494cycles,17.53% slower.
Both preserve all65,536 original output bits and staged identities. Reduced device/readback work
does not pay for CPU certificate work; this candidate remains disabled.
[Matched hardware](perf_records/firesim1895_original_attention_head_gamma_verified.json).

A two-digit approximate attention path passes the first head's local tolerance but fails106/1,600
unchanged original whole-model comparisons in native execution. It is rejected without target or
hardware admission. Local accuracy does not establish full-model accuracy.
[Full original gate rejection](perf_records/smol_two_digit_full_native_rejected.json).

The generic early-saturation/eight-lane exact integer readout candidate preserves every
original ResNet output in native and strict target execution. Only two CPU adapters change;
the unmodified control link reproduces verified1874 byte for byte. Whole retired instructions
fall10,222,806→9,389,018. Stock1900 now verifies37,946,541cycles versus the matched1874
control39,201,279:1,254,738cycles(3.20%) saved, all1,000 original words exact with staged
ELF/stock pins closed. This was the whole-model best before1903; it does not include1897 stripes.
[Verified readout hardware](perf_records/firesim1900_resnet_sat8_readout_verified.json).

The following narrative retains the earlier experiment sequence; the table above is current.

The exact uniform-zero/scale-aware quantize-round pass now applies to the two
remaining ResNet quantizers. All1,000 outputs remain exact in native and actual
Spike, whose retired instructions fall19,286,601 to14,254,689 (26.09% fewer).
FireSim1781 verifies 55,239,221 cycles; instruction counts are not hardware cycles. The separately
proved explicit-RNE/clamp legalization reduces the same whole model further to
11,355,701 Spike instructions, retaining all1,000 golden words. All107,415
boundary checks across the five rounding modes pass. FireSim1786 measures this
candidate at 49,673,153 cycles. Blocked64 mean reduces this to 48,780,534 in1789;
both remain explicit host lowering options. The source-
bound dense group/bank schedule variants also pass complete golden gates, with
representative GSIM367,749 versus447,124cycles (17.75% fewer); whole-model bank
timing with the strongest host path remains pending.

Tiny's combined scalar host, quantization fusion and bounded clamp/RNE candidate
passes the full native and actual Spike output gate, with all 256,000 outputs
unchanged. It retires 271,019,239 instructions, 60.08% fewer than the earlier
678,959,918 build. FireSim1792 verifies 866,822,103 cycles, 15.86% fewer than1788 and51.85% fewer than1747.
Its actual legalized model object participates in the normal build identity.
See [target gate](perf_records/tiny_scalar_host_quant_rne_spike.json).
The explicit scalar activation polynomial replaces22 activation expf chains while
retaining every normalization expf call. All256,000 output bits remain unchanged
in native and actual Spike on this capture; the approximation makes no universal
bit-exact promise. It retires218,423,000 instructions (19.41% fewer than1792).
FireSim1800 verifies791,638,514 cycles,8.67% fewer than1792 and56.03% fewer than1747.
[Gate](perf_records/tiny_scalar_quant_rne_act_poly_spike.json).
Explicit fused polynomial evaluation with ClangO3 retains all captured output bits
and the original Torch gate, retiring209,494,756 Spike instructions (4.09% fewer
than1800's build). FireSim1806 verifies764,493,870 forward cycles,3.43% fewer
than1800 and57.53% fewer than1747. This remains an explicitly selected activation
approximation with capture-specific accuracy evidence.
[Gate](perf_records/tiny_fused_activation_poly_o3_spike.json),
[hardware](perf_records/tiny_fused_activation_poly_o3_firesim1806.json).
The fresh profile for this compiled model preserves all existing objects and
passes full output/count/conservation checks in Spike and FireSim1809. Hardware
interior765,151,605 cycles divides into219,056,004 device and546,095,601 host;
all155 calls are conserved. Its765,152,026 whole-forward timing adds658,156
observed cycles (0.0861%) to the unprofiled1806 control. This profile predates the1816 scalar accumulator improvement. Host intervals before
attention output projections and down projections include all intervening CPU
operations. See [fresh profile](perf_records/tiny_fused_current_profile_firesim1809.json).

Fresh ResNet profile1801 attributes47,057,935 interior cycles to35,450,518 device
and11,607,417 host cycles, with all70 primitive calls conserved. Instrumentation
and placement add38,118 observed whole-forward cycles (0.081%);47,020,321 remains
the unprofiled profile control. The captured classifier tail's wide-B/cached-A GSIM screen
is23.26% faster with exact outputs; it is now composed with destination reuse in1812.
See [current profile](perf_records/resnet_current_leaf_profile_firesim1801.json)
and [selected geometry](perf_records/resnet_current_issue_geometry.json).
Exact pointwise destination reuse for the padded pre-stem passes complete native
and Spike gates, reducing retired instructions4.32%. Composed with the classifier schedule,1812 verifies46,316,907 cycles,1.50% fewer than1795. A proposed external result-alias wrapper failed a live-alias regression
and was removed. A branchless readout correction also remains rejected: four
completed GSIM passes on50,176 captured values are24.47% slower by median; the
600-second probe timed out before its full pass marker and smaller readout.
See [pre-stem gate](perf_records/resnet_prestem_destination_spike.json),
[alias diagnosis](perf_records/prestem_destination_alias_diagnosis.md) and
[rejected screen](perf_records/resnet_branchless_readout_rejected_gsim.json).

Confirmed frontend fixes are upstream on model2MLIR main: precision fixes at
050009e, SDPA scale/causal/options semantics at69c0370, and half-precision
matmul f32 accumulation plus initialized rank-2 outputs at46851eaf. The latest
commit passes160 focused and related tests, including45 native execution cases.
Fresh SmolVLA capture retains the original weights, inputs, golden and manifest
identities. The fresh complete gate still fails47/1600 outputs, with relativeL2 .016156828 and maxabs .096718788. Explicit backend arithmetic
compatibility experiments remain separate from default frontend semantics.

The selected ResNet schedule's mesh issue geometry totals22,805,632 cycles,
already above22M before DMA and CPU work. A source-proven nonnegative skip-domain
experiment reduces this to22,466,944; it has not been promoted into a graph.
This floor describes the selected schedule, not all possible exact algorithms.
See [mesh geometry](perf_records/resnet_exact52_mesh_geometry.json) and
[domain evidence](perf_records/residual_source_domains.json).

## Reference and schedule

The original 22,387,449-cycle evidence was Jack's `resnet50_nofsm_q1013.zip` manifest. Its ZIP contains an ELF, disassembly, FireSim bundle and manifest, but no source. We inspected the ZIP and extracted the ELF/disassembly into `out/reference_q1013/`; no Jack folder was opened. `python -B -m mlir_oot.reference_profile ELF DIS -o static_schedule.json` gives static instruction counts for the function symbols. It finds 72 model-related symbols, 42 four-byte aliases and 30 larger bodies. Among those bodies there are 13,764 static `PRELOAD_CMD`, 8,359 `COMPUTE_AND_STAY_CMD`, 5,405 `COMPUTE_AND_FLIP_CMD`, 2,178 `LOAD_CMD`, 1,537 `LOAD2_CMD`, and 909 `LOAD3_CMD` instructions. Static counts are not dynamic issue counts or cycle attribution. The archive's ELF includes six `LOOP_WS*` commands in a linked library routine; the ZIP therefore cannot meet the stricter final-ELF zero-FSM criterion even though the measured path may not execute that routine.

Our `no_fsm_audit.py` scans all executable RISC-V ELF sections after linking, decodes variable-length instructions, and refuses every `LOOP_*` funct, unknown Gemmini funct, and invalid Gemmini funct3. It must pass on every candidate submitted for a zero-FSM claim. The GSIM probe ELFs pass it; the archived reference fails it.

The prior Claude session's own [q1013 analysis](/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-studies/exo-comparison/q1013_analysis.md) resolves the reference control flow. On the FireSim `argc=0` path, **all 53 convolutions run as Exo Gemmini kernels**, with zero host im2col calls and zero executed hardware-loop commands. The alternate host im2col/vendor path is reached only with `argc>=3`; its linked `sp_tiled_matmul_ws` contains the six static loop commands. The q1013 stock bitstream matches our pinned stock `FireSimGemminiRocketConfig` tar byte for byte. Its full-forward timing window covers FC, argmax and dequantization and is comparable to the prior lean-board whole-model window within the report's approximate 2% hardware uncertainty. q1013 issues 1,155,264 computes against an ideal 1,083,136, or 1.067×, at about 19.4 cycles per compute.

The report attributes the prior 38.33M-to-22.39M gap to FC on Gemmini (~3.5M estimated), packed stem K with pooled readout (~3.3M), residual add as identity matmul with a real D input (~2.7–3.3M), overlapped double-buffered load/compute and 16×64 stores (~3.5–3.8M), and whole-output-row 3×3 tiling (~2.3–2.6M). These are **estimates on the old program**, not measured gains in this branch. The lean configuration forces D to garbage, so the residual form requires stock.

## Implemented path

`mlir_oot/golden_gemm.py` emits verified xDSL `gemmini.*` target IR for a dense int8 GEMM schedule. It supports i8 or i32 output, edge tiles, i32 bias, store scaling and ReLU. `mlir_oot/golden_tuning.py` searches scratchpad- and accumulator-legal tile blocks using a primitive-command and DMA-request model. This is a geometric model, not a fitted FireSim cycle predictor. `tests/gsim_gemm_probe.py` links the final ELF, audits it, and compares against a CPU reference; `--embed-expected` computes large references on the host to avoid spending GSIM time on scalar oracle loops.

Passing GSIM checks include 16×16×16, 16×16×32, 17×19×20, 80×80×17, 144×16×16, 16×144×16, and 17×19×20 with i8, bias, ReLU, and 0.25 scale variants. For 16×144×16 i32, analytical tuning selected `bm=1,bn=9` and measured **1,248 kernel cycles** versus **1,388** for the initial `bm=4,bn=4` probe, on the same pinned GSIM engine. This is a kernel probe, not a ResNet cycle result.

The optional 16×64 accumulator store is bit exact on the simulator. For 144×64×64 with a tuned 9×4 output block it measured **8,142 kernel cycles** versus **8,414** with 16×16 stores. For 16×64×16 it was slower (658 versus 613), so it remains a shape-selectable choice. All four final ELFs pass the no-FSM scan.

Holding B stationary across output-row tiles (`reuse_b`) passes numerical and final-ELF checks. At 144×64×64 it reduces GSIM kernel time from 8,142 to **6,273 cycles**; at 512×64×64 it reduces **25,360 to 17,727**. Preloading a whole small B matrix into scratchpad once per call (`cache_b`) reduces the latter again to **16,575**. The 17×19×20 edge/bias/scale/ReLU probe passes with both options, but `cache_b` raises its time from 776 to 802 cycles, so caching remains optional. The analytical model now counts one B load per cached panel, rather than per output block. These are simulator kernel measurements; there is still no full ResNet FireSim result for this implementation.

The q1013 trace's 16×64 MVIN/MVIN2 panel form is now supported for compatible channel extents. The dialect verifier admits up to four DIM-wide blocks for scratchpad MVIN, with the accumulator MVIN bound kept at DIM. `wide_a` and `wide_b` preserve the same panel payload but reduce RoCC load commands; the tuner counts commands separately from panel bytes. With a 16×4 output block, wide stores, stationary/cache B and both wide loads, 512×64×64 passes the numeric and linked-ELF checks at **15,607 GSIM kernel cycles**, compared with 16,575 using narrow loads. The edge 17×64×64 biased/scaled/ReLU case also passes, including a 2048-byte output guard. Alternating two accumulator/A slots (`pipeline_m`) passes but measured 16,732 versus 16,674 at 8×4 on this shape, so the upstream selector leaves it off. Jack's interleaving occurs inside compute bursts; this simple alternation does not reproduce that schedule.

`golden_resadd.py` implements the stock-only identity-matmul form with a real D input and wide stores. Its 16×16, 17×19 and 17×64 cases pass GSIM, the final linked-ELF audit, and a 2048-byte output guard check (431, 3,192 and 1,837 kernel cycles respectively). A tile-grouping bug had treated a final partial row as full when `bm=1`; it is fixed and covered by `test_golden_groups.py`. The kernel stages edge and non-64-byte-pitch stores in 1024-byte aligned scratch before copying valid elements into dense output. The odd-width edge path is correct but much slower than the full-tile path, which is acceptable for the ResNet channel dimensions that are multiples of 64.

### Device compilation

`golden_device_lower.py` is the only translation of golden `gemmini.*` operations to RoCC inline asm. It verifies the target module, encodes only exact primitive operations, and refuses an unsupported Gemmini op. CPU repetition remains ordinary LLVM CFG. `golden_device_compile.py` writes the target xDSL module, lowered LLVM dialect module, LLVM IR and RV64GC object, records their hashes and compiler arguments, and audits the object. The probe harnesses then link the object with their runtime and audit **every executable section of the final ELF** again, since a library could introduce forbidden LOOP commands after object compilation. The 32×16×16 target-to-LLVM run produced the same linked ELF hash and 409 GSIM kernel cycles as the former direct emitter; 17×64 residual likewise retained its original linked ELF hash and 1,837 cycles. This is a real compile path from the xDSL target module, rather than separately authored target and LLVM schedules.

For a supported single-layer capsule, run `python -m mlir_oot.golden_upstream capsule.interface.mlir --llvm-bin LLVM_BIN --workdir OUT`. The output includes `kernel.gemmini.mlir`, `kernel.llvm.mlir`, `kernel.ll`, `kernel.o`, `device_compile.json`, and `upstream_binding.json`. The receipt records the SHA-256 of both compiler binaries and all generated code artifacts. After linking the model runtime, run `python -m mlir_oot.no_fsm_audit final.elf`; the object audit alone cannot certify a linked program.

FireSim queue job **1670** failed before simulation because its Chipyard config lacked the requested hardware key. Job **1673** used our own stock Chipyard worktree and the existing `alveo_u250_firesim_gemmini_rocket_stock` entry, but timed out in `INFRASETUP` after its leading kill exceeded 90 seconds; it never booted the ELF or produced a cycle measurement. The queue records it as `TIMEOUT` at 1080 seconds. A nearby unrelated queue job also failed, so more submissions on the same slot are deferred until the infrastructure can complete setup.

## Upstream whole-model lowering

Running this OOT `gemmini-opt --convert-iface-to-gemmini --emit-command-buffer` on each shipped capsule yields zero commands and an explicit decline:

| Capsule | First blocking result |
| --- | --- |
| `SY_model_resnet50` | Estimated 65,404,026,705 CPU-lane straight-line evaluations exceed the 400,000 budget. Its conv matmuls remain f32 after dequantization despite `prov.quantization = "int8_static_act_int8_weight"`. |
| `SY_model_smolvla` | Estimated 9,395,352,591 CPU-lane straight-line evaluations exceed the same budget. An xDSL multi-result `linalg.generic` printer/parser mismatch was normalized first so the full 11 MB file parses. |
| `SY_model_tiny_llama` | Estimated 3,031,111,489,682 CPU-lane straight-line evaluations exceed the same budget. |

These estimates count element evaluations in a hypothetical unrolled scalar program; they are **not** predicted hardware cycles. The looped GEMM removes one code-size bottleneck for device contractions, but whole-model lowering still needs quantized conv recognition and direct input gathering, tensor layout propagation, scalable host-operation loops, residual add/reduction kernels, ABI and weight binding, and a correctness oracle before any full-model FireSim run can be called golden. The current ResNet capsule's f32 path is a different program identity from Jack's int8 static recapture, so equality of cycle numbers alone would be misleading.

One upstream geometry optimization is now implemented in `lowering/plan.py`: a stride-one, unpadded NHWC 1×1 `merlin_iface.conv2d` uses its original input as the GEMM lhs, because the physical C row pitch is identical to the matrix K row pitch. The real GQ1 capsule still emits a valid command buffer and xDSL target artifact, now with no `im2col_recipes`; GQ2's padded 3×3 path retains its recipe. This eliminates an unnecessary derived tensor for this upstream case. It does not convert the shipped f32 ResNet graph to the int8 q1013 program or route an entire model through the golden kernel yet.

`golden_upstream.py` is a strict single-layer bridge from typed `merlin_iface.conv2d` to this compiled golden kernel. It derives dimensions, dtype, bias, scale and ReLU through `Builder`, refuses a gather or unsupported epilogue, records its source tensor to device pointer binding, tunes legal tiles, and compiles through `golden_device_compile.py`. The real GQ1 capsule generated `M=36,N=16,K=16`, `bm=3,bn=1`, and the object that this bridge compiled passed a numerical GSIM probe at **535 kernel cycles**, including a final linked-ELF no-FSM audit. The bridge is not yet the whole-model runtime or a general 3×3 convolution lowering.

A synthetic 1×1 NHWC capsule with `M=512,N=64,K=64`, bias, 0.125 store scale and ReLU selected 16×4, stationary/cached B, wide A/B loads and 16×64 stores through the upstream bridge. The **exact object compiled from that capsule** passed the linked-ELF audit, numerical GSIM and the output guard at **17,822 kernel cycles**. The 15,607-cycle figure above omits this bias/scale/ReLU work and is a different kernel; neither number is a full ResNet layer timing in FireSim.

A larger `M=3136,N=64,K=64` probe with bias, 0.125 store scale and ReLU passes at **104,931 GSIM kernel cycles**, with 3,136 mesh compute commands and a 50,176-cycle peak array floor. The analytical model also reports unique tensor bytes and a distinct upper request volume; dividing repeated request bytes by DRAM bandwidth is not a lower bound because cache and stride-zero bias broadcast can satisfy repeated requests. The model remains uncalibrated to FireSim.

`upstream_quant.py` now invokes Merlin's existing QDQ-aware integer contraction rewrite, writes the transformed IR and a source/content-bound receipt, and validates that this OOT backend parses the result. On the shipped `SY_model_resnet50` capsule it rewrites **54** contractions (53 convs and FC), makes all **54** named i8×i8→i32 matmuls mesh eligible, and hoists activation quantization ahead of **50** pure gathers; three strided-hole gathers are refused and one calibrated static activation is reused. The pre-gather option changes numeric granularity, so its receipt marks model accuracy verification as required. `frontend/mixed_matmul.py` fixes xDSL's synthesized body for i8×i8→i32 named matmul by widening both inputs before the multiply; `linalg_reader.py` places rewritten contractions by their current operand dtype even when `prov.orig_dtype` remains f32. The full transformed ResNet now parses/verifies and gives an explicit mixed-lane decline: its remaining host ops cost about **470,139,986** scalar element evaluations in the current unrolled emitter, above the 400,000 budget. That count includes zero-runtime tensor views and allocations, so it is a code-size estimate, not a cycle forecast. The top counted families are `tensor.empty` 107.8M, generic 80.2M, reshape views 135M combined, and dequantization 63.5M. Whole-model looped/fused host lowering and weight binding remain unimplemented.

## Generalized integer contractions and model arithmetic

`contraction_patterns.py` now checks typed indexing maps, iterator order, i8-to-i32 signed extension, multiply and add wiring, shapes, and a zero accumulator before accepting a contraction. The same matcher handles ordinary 2D GEMM and dense batched 3D attention GEMM. On the prepared model IR it finds **54/54 ResNet**, **367/367 SmolVLA** (303 2D, 64 batched), and **200/200 TinyLlama** (155 2D, 45 batched) integer contractions. This is contraction recognition, not whole-model code coverage. The 302 SmolVLA 2D `linalg.generic` contractions, previously routed to the host because only named `linalg.matmul` was recognized, are now eligible for 2D mesh scheduling. Rank-3 contractions have a separate primitive-only batched kernel with ordinary CPU batch repetition.

The inventory now separately counts every top-level operation in each prepared function, making the model coverage gap explicit:

| Prepared model | Top-level graph ops | Device contraction ops | Other graph ops still needing graph lowering |
| --- | ---: | ---: | ---: |
| ResNet-50 | 3,619 | 54 | 3,565 |
| SmolVLA | 32,386 | 367 | 32,019 |
| TinyLlama | 12,988 | 200 | 12,788 |

Many remaining ops are constants, `tensor.empty`, or reshape views and should collapse under allocation and alias planning; the rest include quantization, gathers, reductions, softmax, transposes and elementwise computation. This is an operation coverage ledger, not an estimate of required runtime commands. The whole-model coverage flag remains false for every model.

`golden_model_inventory.py` writes an auditable arithmetic census from those exact shapes. The numbers below count integer contractions only and assume a 16×16 array with one fully pipelined output row per cycle; they exclude loads, stores, host work, and all model overhead:

| Prepared model | Exact contractions | Integer MACs | MAC/256 floor | Padded compute-command issue floor |
| --- | ---: | ---: | ---: | ---: |
| ResNet-50 | 54 | 4.089B | 15.973M | 17.330M |
| SmolVLA | 367 | 130.873B | 511.221M | 526.368M |
| TinyLlama | 200 | 8.281B | 32.348M | 64.741M |

Jack's 22.387M ResNet run is 1.29× the padded issue floor. The requested 5B SmolVLA target is about 9.5× its integer padded issue floor, but the old full-policy runs spent very large time in scalar softmax and other host operations. Arithmetic alone cannot establish feasibility. TinyLlama's short rows waste half or more of a 16-row array in many layers, making its padded issue floor about twice its pure MAC floor.

There is a structural ResNet layout mismatch to solve before model performance can approach Jack's schedule: the prepared `conv_0` is `64×147 · 147×12544`, and `conv_2` is `64×576 · 576×3136`. The left operand comes from OIHW weights and the right from a materialized NCHW im2col gather. Jack's high-performance kernel instead consumes spatial output rows and output-channel columns, and gathers each convolution from resident input without a host im2col. Merely compiling the current prepared matmul would preserve the expensive gather and wrong physical output layout. A convolution-specific tensor layout and direct-gather lowering is required.

### Model-wide device object compilation

`golden_contraction_upstream.py` selects one exact rewritten operation, chooses a legal schedule from its shape, and compiles it through the xDSL Gemmini to LLVM to RV64GC path. The small 2×17×19×20 batched probe passed complete numerical GSIM, output guard, and linked-ELF audit at **1,492 kernel cycles**. The **exact upstream SmolVLA `matmul_384` object** (batch 15, M50, N64, K113) passed the complete numeric oracle and linked-ELF scan at **83,398 GSIM kernel cycles**. Its first harness run timed out at 180 seconds before readback; the same ELF passed with a 600-second timeout. The exact upstream TinyLlama `matmul_194` (batch 32, M8, N64, K8) passed at **11,736 GSIM kernel cycles**. These kernel timings are on pinned GSIM, not FireSim model cycles.

`golden_device_catalog.py` now builds one model-scoped object from all recognized contractions, deduplicating identical schedules and assigning content-derived unique symbols. Both ordinary and batched specializations receive stable distinct symbols, preventing duplicate definitions when hundreds of operations are linked. The source SHA, every operation-to-symbol binding, per-kernel schedule, compiler hashes, and object audit are recorded. Complete catalogs compiled with **54/54 ResNet contractions → 21 kernels (133 KB)**, **367/367 SmolVLA → 26 kernels (117 KB)**, and **200/200 TinyLlama → 8 kernels (29 KB)**; all three object audits passed. The TinyLlama catalog object itself was linked into a final ELF and its `matmul_194` symbol passed the full numerical GSIM oracle, output guard and final-ELF no-FSM audit at **11,683 kernel cycles** before the configuration-once refactor. This validates the deduplicated multi-kernel object's symbol linkage as well as its arithmetic. The catalog currently covers the integer contractions only. Tensor allocation, layouts, quantization epilogues, host nonlinearities, complete model execution, correctness verification, and FireSim per-group timing remain open.

The batched implementation now configures and flushes Gemmini once per batched call, then repeats primitive tile work through an ordinary CPU loop and fences at the end. On the real TinyLlama `matmul_194` shape, complete numerical GSIM and the linked-ELF audit pass at **9,282 cycles** versus 11,736 before, a 1.264× speedup. The full TinyLlama catalog was rebuilt, linked, audited and numerically checked after the change at **9,338 cycles**. The real SmolVLA `matmul_384` improved **83,398 → 81,607** GSIM kernel cycles with complete numerical and ELF audit checks. The two-batch edge case was 1,506 versus 1,492 and remains a candidate for size-based selection. The [optimization log](golden_optimization_log.md) records comparable observations and the abstractions needed to automate them.

An optional short-row A-resident schedule now preloads one M≤16 A tile before sweeping several N blocks. On the same 8×512×256 i32 synthetic projection and pinned GSIM, full numerical and final-ELF checks passed at **26,941 cycles versus 31,947** (1.186×). The analytical count reduces A MVIN commands from 128 to 16 without changing compute count. The model rule selects it only with at least four N blocks; among the prepared model catalogs this currently applies to TinyLlama's 8×5632×2048 and 8×32000×2048 projections. Those exact layers have not been timed in FireSim or GSIM, so the 1.186× figure is a shape-specific probe result.
The updated TinyLlama catalog still covers 200/200 contractions with eight distinct kernels and binds all eight through Merlin. A typed dynamic A scratchpad address keeps cached-A K repetition in an ordinary CPU loop. The catalog object is now **31,856 bytes**, versus **953,672 bytes** with K unrolled and **28,432 bytes** before cached A. On the same pinned synthetic GSIM probe, the looped kernel passes full numeric, guard and final-ELF audits at **26,948 cycles**, seven cycles above the unrolled form and **1.186× faster** than the 31,947-cycle baseline. The typed verifier checks a declared address range; a future automatic lowering pass must establish that range from loop bounds before using the dynamic form. These are contraction-level results, not a TinyLlama model cycle result.

The isolated Merlin integration worktree now has an external catalog route in its existing whole-model offload compiler. The route checks exact prepared source bytes and provenance region/type bindings, adapts dense rank-2 and whole-batch rank-3 pointers, and compiles an RV64 shim. On the exact prepared sources, rewrite plus shim compilation bound **54/54 ResNet contractions (21 symbols), 367/367 SmolVLA (26 symbols), and 200/200 TinyLlama (8 symbols)**. Set `DeviceRouting.catalog_builder` to `mlir_oot.golden_device_catalog.merlin_builder(LLVM_BIN)` to compile from Merlin's final prepared file before offload; its TinyLlama smoke run built and bound all 200 calls. Set `DeviceRouting.final_elf_audit` to `mlir_oot.golden_device_catalog.final_elf_audit` to require the no-FSM policy on the final image. This is device-code binding, not whole-model lowering or execution. FireSim jobs 1709–1711 are probe jobs, not model performance results.

The upstream ResNet preparation now enables Merlin's exact 1×1 im2col identity-view pass after QDQ contraction rewriting. Its matcher was extended to named `linalg.matmul`, the form QDQ actually emits. On the exact capsule it replaced **33 copies** spanning **7,200,256 int8 elements**, refused **17 nonidentity windows**, and retained **54/54** mesh contractions. A newly compiled 21-kernel catalog passed its no-FSM object audit and its exact-source Merlin rewrite/shim binding routed all 54 calls. The copy elimination has no FireSim cycle measurement yet; 3×3 and strided windows still materialize or require a direct-gather schedule.

For an actual Merlin model build, set `DeviceRouting.prepared_transform` to `mlir_oot.golden_device_catalog.merlin_identity_view_transform` alongside the catalog-builder and final-ELF-audit callbacks. The preparation hook runs before catalog compilation and source hashing. Its ResNet callback smoke proved 33 views and compiled all 54 contractions to 21 audited kernels; a runtime regression test checks that catalog selection and offload bind the transformed bytes.

The complete ResNet capture now compiles to an RV64GC ELF with all54 contractions linked and a passing final no-FSM audit (SHA18823d02831d0b98…). The same LLVM host graph plus identical device ABI shim, using scalar int8 reference kernels, reproduces all1000 captured integer-reference outputs exactly: relative L2=0, max error=0, argmax713 matches. This proves host lowering/layout/ABI for the captured random-initialized model, not hardware execution, pretrained quality, or22M performance. Actual Gemmini Spike execution is now running before a complete-model FireSim measurement.

## Current exact candidates and compilation contract

Tiny1816's default-off scalar accumulator schedule preserves each f32 contraction's increasing-K multiply/add order, including nonzero initial values. FireSim verifies648,210,569 cycles,15.21% fewer than1806 and64.0% fewer than1747. Four independent output accumulators improve this to615,651,105 in1821,5.02% fewer than1816; every output bit is unchanged. Partial K-unroll2 improves this further to612,282,210 in1828,0.547% fewer than1821. Two-output1825 measures622,639,906 and is rejected in favor of four outputs. All155 device contractions and the previously selected activation approximation remain unchanged. Partial K-unroll4 is queued as1832. An additional Clang loop-unroll flag emits a byte-identical model object and was rejected without duplicate simulation.

The guarded quantized mean replaces a proved canonical Q/DQ serial mean with an integer sum and an exhaustive sum certificate. For the original49-value ResNet reduction, eight of12,496 totals require exact floating-point replay; the certificate covers every signed-i8 input sequence. Full native and actualSpike retain all1,000 original output bits at10,816,839 retired instructions,1.54% fewer than1812. FinalELF zeroFSM passes. FireSim1819 measured46,680,853cycles,0.79% above1812; this arm is not the best recipe and its driver omitted explicit host scheduling. See [proof and binding](guarded_quantized_mean.md) and [whole-model gate](perf_records/resnet_guarded_quantized_mean_spike.json).

Normal build markers now include the actual linked device/matrix object bytes in link order, with length and domain separation. Object paths do not affect identity. Final and staged ELF SHA256 remain authoritative for both new and historical runs. The prepared source, ABI, target object, native standin, source numeric policy and final ISA audit must close over the same compilation. Default-off compiler transforms remain independently selectable.

SmolVLA retains the existing elementwise gate by explicit user direction. Original Torch MKL VML high-accuracy sine/cosine dispatch differs from scalar libm at three BF16 query entries, which propagate into attention. SLEEF was investigated but is not the selected Torch backend. A bounded integer-position RoPE lookup policy is under qualification; unrelated timestep trig remains separate, and the original weights, inputs and golden remain immutable.

The packed NHWC mean variant proves the existing transpose and consumes the
physical layout directly. Exact unsigned16-lane sums process eight channels per
word with no interlane carry, keeping source f32 fallback for ambiguous sums.
Full native and actualSpike preserve all1,000 original words at10,214,792
instructions,7.02% below1812.23 focused tests pass. Stock1824 verifies44,507,889
cycles,3.91% below1812. The bankedmatmul11 arm1823 independently retains
a host object byte-identical1812 and all52 source/numeric proofs, measuring
46,127,051cycles. The combined packed-mean plus bankedmatmul5/8/11 candidate
passes all original native and actualSpike outputs and zeroFSM; its host object
is byte-identical1824. Stock1829 verifies43,969,384cycles,1.21% below1824. See
[combined gate](perf_records/resnet_three_banked_packed_spike.json).

SmolVLA source-compatible softmax initially returned NaNs in full preparation:
the pinned xDSL parser numerically interprets unquoted dense float hex literals,
turning the maximum initializer's negative infinity into4286578688.0. Merlin's
portable printer now emits quoted raw bytes for dense floating attributes and
recognizes splats by byte equality. Repeated xDSL/upstream parsing and native
tests retain infinities, signed zeros, NaN payloads and subnormals;44 related
tests pass. The trusted191,535-value softmax fixture is exact through full
preparation. The fresh whole-model gate remains pending; this does not admit
SmolVLA to hardware or relax its original elementwise criterion.

The explicit fresh-output descriptor interface passes all1,000 original native
and actualSpike outputs, zeroFSM and7 focused ownership/refusal tests. Caller
contracts reproduce the qualified host MLIR and compiled bridge byte-identically;
stock1831 is pending. See [ownership contract](descriptor_writer_contract.md).

Resident-input direct convolution reduces the matched H14/W14/C256 capsule from
675,375 to533,649 GSIM cycles with all50,176 outputs and guards exact. A DMA
block-stride field had been silently dropped by lowering; its corrected encoding
now has a regression gate. Five source-bound whole-model routes are qualifying
with separate preserved scale proofs. This is a capsule gain, not yet a whole
FireSim result. See [resident study](resident_convolution_study.md).


The general dense banked command-cost candidate passes all1,000 original native
and final-ELF Spike values. Its host object/LLVM match1836 byte for byte; only
four pointwise device objects change. Independent176x256x64 GSIM numeric/guard
checks measure29.00% fewer kernel cycles, and95x48x80 tails pass. Whole-model
stock1849 is queued; its slightly higher Spike instruction count is not a cycle
claim. [Candidate gate](perf_records/resnet_banked_command_policy_spike.json).

An explicit OOT support provider now exposes the curated harness/GSIM command to
current Merlin orchestration, removing the legacy Python registration dependency.
A real numeric/guard capsule retains byte-identical final ELF and1,919 GSIM kernel
cycles. [Infrastructure evidence](perf_records/current_core_gsim_provider.json).


Fresh1849 profile1850 verifies42,303,592 forward cycles (33,784 above unprofiled
1849). Conserved interior42,303,164 comprises33,814,043 device and8,489,121 host
cycles; all70 calls and original1,000 output words pass. The pre-stem host gap
is4,017,289; gaps before matmul26/49 are1,955,979/962,942 and include the two
integer-readout epilogues plus any other intervening CPU work. These are host
intervals, not isolated operation costs.

The qualified composition now combines1886 generic host quantization packets,1874 banked
residual schedules and1900 exact eight-lane readouts. The1886 control and1900 readout ELFs
are reproduced byte for byte; all semantic catalog proofs/bindings and unchanged host/runtime/weights
objects are pinned. Native and actual strict target execution preserve all1,000 original words.
Stock1903 verifies36,102,704cycles versus1886 38,603,949 (6.479% lower), all1,000 original words exact with staged stock identity closed. This is the current verified ResNet best, 4.859% below prior1900; gains are measured as a composition.
[Composition hardware](perf_records/firesim1903_resnet_composed_verified.json).
[Composition qualification](perf_records/resnet_qualified_packet_composition_spike.json).

Tiny adjacent independently proved RNE results now use separate CPU floating temporaries under an
explicit CPU policy. The general Merlin pass preserves strict source chains/aliases and refuses
unsafe motion. Actual full capsuleGSIM256,637→247,568cycles(3.53%) with identical instruction count;
whole native/target256,000 words and original Torch gate pass. Stock1902 verifies527,255,504cycles versus1880 531,072,370 (0.719% lower), a single-run marginal gain.
[Adjacent RNE hardware](perf_records/firesim1902_tiny_adjacent_rne_verified.json).
The fresh155-boundary profile1901 targets1880; older1837 attribution above remains historical.
[Qualified next Tiny arm](tiny_adjacent_rne_packets.md).

The source-stride resident convolution uses a general layout/stride/resource rule and fixes execute-stride propagation in target lowering. Its original-input paired capsule measures727,070→577,376GSIMcycles (20.59% lower), every50,176 output and4,096guard exact; an independent non-square/tail case also passes. Both normal and controlled whole builds preserve all1,000 original words on native/strict target. Only one of52 device kernels changes, with every original1897 host/runtime object retained in the controlled arm. Stock1914 is queued versus1897; whole-model gain is unknown. [Qualification](perf_records/source_stride_resident_conv_whole_qualification.json).

Three-digit approximate attention also fails the unchanged full Smol gate (121/1,600), despite only27 changed first-head words. A separate zero-encoding-error diagnostic fails144/1,600 when accumulation is widened; exact source-ordered f32 replay at the identical384-route seam reproduces all1,600 bits. This closes integration sanity and proves accumulation rounding alone is sufficient to fail this original gate. No approximate route is promoted. [Precision negative](perf_records/smol_three_digit_full_native_rejected.json), [paired source-seam diagnosis](perf_records/smol_source_replay_vs_wide_accumulation.json).


## Checkpoint 2026-10-06 09:15 UTC

Verified stock whole-model bests: ResNet1930 **34,905,135 cycles**,
Tiny1926 **461,389,700 cycles**, Smol1906 **258,621,872,969 cycles**.
Targets remain unmet: ResNet about22M, Smol about5B and Tiny300M.
The newer clarification from Jack concerns Smol single-digit billions and
Tiny300M; matching token/batch/timer scope has not been supplied.

Smol's general source min/max capability closes282 pins and preserves the full
original48-call native consumer gate. Its complete12-head group uses
**3,243,485,384 Spike instructions**,3.764% below the separately qualified ABS
control. A zero/NaN library guard variant is6.485% slower and rejected.
These are instruction counts, not stock cycles. Stock1944 separately verifies
**10,524,980,809 cycles** for the earlier row/floor complete group; it is a
section result and does not establish new whole-model timing.
[Min/max proof](perf_records/smol_standard_minmax_complete_group.json).

The retained resident command candidate is released for one stock comparison
with controlled1903 after independent483-pin reclosure and original1,000-word
native/strict gates. Five selected resident objects change;52 adapters and47
other kernels remain byte identical. It is not composed with1930 or1947.
[Whole qualification](perf_records/resident_commands_normal_whole_qualification.json).

The Tiny norm+packet4+constant-lifetime candidate is released for one stock
comparison with1926. All256,000 original words and the original Torch gate pass;
all executable sections are zero-FSM. Its complete two-row scalar section saves
10.302% GSIM cycles; whole Spike instructions increase1.379%. Whole stock
performance remains unknown.
[Whole qualification](perf_records/tiny_fma_constant_whole_qualification.json).

## 2026-10-06 scratch recovery and resumed optimization

User requested scratch space and continuation of the performance goal. Retained all
historical artifact paths and contents with hash-verified same-filesystem hardlinks
for frozen identical completed copies. Root deduplicated 24 Tiny weight/object/capture
copies (42.524 GiB); Tiny deduplicated one completed baseline ELF
(another was already linked, no double attribution); reference deduplicated five
owned historical simulator copies. Total physically reclaimed by these actions:
48896086016 bytes (45.538 GiB). Scratch available at root closure: approximately
58 GiB. No model data, goldens, ZIP, proof content or useful live process was deleted.
New builds resumed. Historical hardlinked artifacts are immutable; use fresh output
directories for later builds, or copy before changing a shared file.

Generic infrastructure follow-up: compilation artifacts should support immutable
content-addressed weight/capture storage and independent writable build outputs,
with physical-storage accounting and safe retention manifests. This belongs in
Merlin orchestration, independently of target. These storage savings are unrelated
to model runtime cycle performance. Receipts: root_frozen_tiny_storage_dedup.json,
tiny_frozen_baseline_storage_dedup.json, owned_reference_spike_space_dedup.json.

Performance work resumed: reference has a local M784/N256/K512 full-K weight-cache
GSIM win (599,463 to 434,406 cycles, original output/guards/inputs exact), pending
independent shape/tail/resource/default checks before normal binding; Tiny has a
local actual hoisted normalization pure-f32 multiplication packet win (~26.33%),
pending whole-model validation. Neither is a whole-model FireSim performance claim.
Stock ResNet job 1957 remains queued and the ordinary minmax SmolVLA whole-model
functional target remains live, with their existing owners and collectors.
