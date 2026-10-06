# Golden compiler optimization journey

This is the index of hypotheses, compiler changes, measured outcomes and rejected experiments.
The [chronological log](golden_optimization_log.md) retains earlier work and
[current progress](golden_progress.md) lists qualified whole-model champions.
Target instruction/device schedules belong in OOT. Shared host/numeric/global/build/runtime
mechanisms belong in Merlin. Production choices derive from semantics, shape, layout, numerical
contracts and resources; source IDs identify bindings and experiments only.

## Measured changes and pending compositions

### Absolute values and ordinary physical compilation (2026-10-06 08:20 UTC)

- **Absolute values — Merlin capability, OOT actual ISA proof:** the same
  complete original12-head production group drops3,527,309,706→3,370,349,620
  retired instructions(4.4499%). Default-disabled object bytes, all15 device
  product objects, source/data/bridge/driver,196,608 accepted carriers/guards and
  replay counters remain unchanged. Independent actual target565,925 paired
  checks cover both precisions, all BF16 words, directed/random float values,
  signed zeros, subnormal/infinite/NaN classes and five rounding modes. New
  NaN-payload/interposition/errno/flag/nontrapping obligations are explicit.
  All48 frozen native calls pass the unchanged1,600-word whole gate and every
  compiled original consumer byte/escaping scale.209 pins reclosed. These are
  instruction and functional observations; FireSim cycles remain unknown.
  [Receipt](perf_records/smol_standard_absolute_complete_group.json).
- **Actual physical ABI and pooled whole compilation — Merlin:** the normal
  whole build closes all17 provider objects and final source/compiler/ABI/link
  identities. A generic integer return-range parser accepts Clang's non-ABI
  value fact while retaining ABI attribute refusals.24 focused tests pass,
  one unavailable tool capability skips. Native execution through the actual
  new physical wrappers uses one workspace and48 calls/zero fallback, all1,600
  outputs exact. The actual compiled source fallback separately passes196,608
  endpoint words with immutable inputs/descriptors. Whole strict target remains
  live, with final ELF zero-FSM already closed.
  [Build](perf_records/smol_normal_workspace_provider_build.json),
  [native/fallback](perf_records/smol_normal_workspace_provider_native.json).
- **ResNet residual-output release — Merlin legality/math, OOT instruction:**
  complete original802,816-output Clang capsule1,996,045→1,656,089GSIMcycles
  (17.031% lower); GCC negative and first incomplete long run retained.321pins,
  independent aliases/guards/source binding and all1,000 whole original outputs
  close. Stock1947 admits only its controlled1903 arm, with no composition.
- **Required gates:** latest core docs and structure checks pass after the
  classification/absolute-value capability and imported ABI fix. Actual measured
  whole champions remain1930(34.905M ResNet),1926(461.390M Tiny),1906(258.622B
  Smol); targets22M/300M/5B remain unmet.

### Whole hardware and production compilation checkpoint (2026-10-06 07:00 UTC)

- **Tiny source normalization hoist — Merlin:** stock1926 preserves all256,000
  original words/Torch gate and reduces the isolated1880 arm531,072,370→461,389,700
  cycles (13.1211%). This is the new best; user target is now300M. The527,211,739
  constant-clamp arm1920 is effectively tied with1902 and is not composed here.
  [1926](perf_records/tiny_norm_hoist_stock1926.json).
- **Segmented borrowed inputs — Merlin legality, OOT device view/schedule:**
  stock1927 preserves all1,000 original words and reduces isolated1903
  36,102,704→35,152,730 cycles (2.6313%), new ResNet best.
  [1927](perf_records/resnet_segmented_inputs_stock1927.json).
- **Current stock attribution:** profile1919 closes all70 boundaries with
  30,678,196 device and5,460,779 host-gap cycles. The aligned ZIP1876 class
  comparison retains a host-counter scope mismatch; it does not assign every
  difference to a compiler cause. This profile belongs to prior1903, not1927.
- **Smol exact floor and source multiply specialization — Merlin:** portable
  binary32 floor, nonnegative scalar multiply, positive-RHS interval multiply
  and proved immutable row-factor/reciprocal hoisting preserve all196,608
  original quantized values,256 scales and4,461,440 replay FMAs. Complete-group
  strict Spike instruction proxy3,528,972,523→3,494,086,004→3,303,437,300;
  row changes improve5.4563% relative to floor. Native elapsed time is functional
  evidence only.94 distinct mathematical/source/build files and audits of all
  executable sections are retained and rehashed by the reclosure driver.
  [Archive](perf_records/smol_exact_row_arithmetic_complete_group.json).
- **Directed narrowing — Merlin hooks/proofs, OOT CPU encoding:** complete-group
  instruction proxy3,303,437,300→3,121,235,451 (5.5155% lower); probability stage
  1,398,679,908→1,216,478,078.13,197 independently derived rational conversion
  cases pass in all five target rounding modes (65,985 checks), alongside5,316
  existing directed arithmetic checks. Replay falls by576 source FMAs to
  4,460,864, with all196,608 original quantized observations and256 scales exact.
  One additional unobserved internal BF16 carrier changes (126 versus125).
  The old implementation carrier-reference assertion correctly refused; its
  failed run is retained. A fresh independent native carrier reference closes
  target implementation agreement while leaving the original source gold,
  input data object and quantization oracle unchanged.68 distinct files are
  rehashed. Full48 new-policy/production/stock qualification remains pending.
  [Archive](perf_records/smol_directed_cast_complete_group.json).
- **Reentrant source executor and normal host-provider link — Merlin:** caller
  owns one121,963,584-byte workspace; all48 groups preserve all1,600 original
  whole outputs,9,437,184 quantized bytes and12,288 escaping BF16 scales with
  506,370,304 replay FMAs.23,040 native product callbacks and4,152,360,960 logical
  i32 readback bytes describe work, not physical traffic or hardware time.
  Actual O0/O2 source-fallback/ownership and ranked RV64GC final-link gates pass;
  an inert host-provider hook preserves previous default ELF bytes. Source,
  companion LLVM, object, compiler/dependency and ordered final-link identities
  stay explicit; opaque LLVM pointers do not prove logical dtype/rank/ownership.
  New pooled full-model target execution is pending.
- **Compilation regression retained:** the new production descriptor executor
  initially emitted no terminal PASS from its final puts call, so collection
  refused. A formatted marker relink closed the numeric gate. Broad normal
  `-fno-builtin` leaves per-value memcpy and fmaf calls, unlike older capsules;
  fmaf itself is hardware fmadd+return, not a software arithmetic routine.
  A matched static-helper control/capability pair subsequently measures
  9,624,336,737→5,709,055,956 instructions (40.68% lower), same device/driver/
  inputs/counters. Source FMA and fixed bit-copy capabilities are separately
  admitted, with default bytes unchanged. Exact floor/row changes are not yet
  composed with this production pair. No5B stock claim follows.
- **Residual correction compiler recipe:** complete65,536-pair Clang capsule
  164,399→141,216 GSIMcycles (14.10% lower), independent shape8,143→7,770
  (4.58% lower). Original802,816-output GCC-inline capsule instead regresses
  1,995,900→2,052,178 (2.82%); it is rejected. The normal optional three-argument
  route uses the existing catalog/FreshWriter infrastructure, not model-name
  selection. Real whole compilation exposed a source-seal ordering bug that
  synthetic base callbacks had missed; its original failed attempt is retained
  and the stateful callback fix is being qualified before target admission.
  [Compiler screens](perf_records/residual_output_guard_compiler_capsules.json).
- **Typed preparation opportunities — Merlin:**48 source calls request432
  logical BF16 views; exact root SSA/dtype/static-coordinate identities identify
  144 unique views and96 groups of four reads.113,246,208 repeated logical input
  bytes are an opportunity, not measured physical traffic or saved cycles.
  Different query quarters stay distinct; no pointer cache or physical reuse is
  enabled by this read-only analysis. Format, numerical/effect, dominance and
  complete consumer lifetime contracts are still required for emission.

None of these separate capsule, compiler or stock measurements are added to
form a whole-model prediction. Stock1922 DMA coalescing38,312,898 remains above
the ResNet champion; stock1924 earlier exact head2,101,389,170 is a scoped head
result, not the current complete-group or whole Smol measurement.

### Latest compiler and measurement work (2026-10-06 UTC)

New closed-source attention work belongs in Merlin: ordinary function
partition/binding retains48groups/384roots. Generic interval primitives now
pass35boundary/refusal tests, including a discovered signed-zero bin mismatch.
Actualgroup0 independent compiled source closes196,608BF16words. The fixed
≤1BF16step policy changes53local words and reduces local source fallback from23.36%
to2.09%, but fails113of1,600whole outputs under the original gate. An exact
endpoint control reuses the identical host image/ABI, closes all48endpoints
against independently compiled original source, and preserves all1,600words.
It replays28.497%of19.327Bsource FMAs; the rejected whole bounded arm replays
2.222%. One center-only source-DAG arm also fails121outputs. No numerical
policy is promoted. [Paired controls](perf_records/smol_source_group_bounded_exact_journey.json).
Actual M256/head0 device products/readouts are independently exact; prepared
source-polynomial interval evaluation reduces target instructions6.90%, but
still retires456.10Mversus232.96Msource CPU instructions. Dynamic interval
sign specialization instead regresses4.91%and is rejected. Neither count is
a hardware cycle prediction. Tighter exact bounds, closed quantized consumer
frontiers and immutable source-operand preparation commoning are concrete next
compiler opportunities; legality and cost remain obligations.
Prepared rigorous L2 row norms now preserve all196,608original group0words
while reducing exact source replay94,066,240→69,035,648FMAs(26.61%lower).
The generic header passes13independent exact-rational bound/refusal tests;
this is replay-work evidence, not hardware cycles or a whole-model result.
[Bound screen](perf_records/prepared_l2_exact_group_screen.json).

The two-predictor residual prototype shares resident input panels in OOT;
Merlin supplies the complete tuple proof and512-byte decoder. All65,536signed
source pairs pass, with complete both-readout+decoder GSIM164,367→153,249cycles
(6.764% reduction),15resource/order/refusal tests and8,192guards. Original
802,816-byte captured-layer independent11chunk implementation instead
regresses16.69%; the resident-input large follow-up is incomplete. Both remain
disabled. A source-derived two-scale bracket shares a five-chunk integer
producer and reduces the common-address full-domain capsule23.32%, with an
independent three-panel drain case20.05%lower. The original full-layer pair is
still running. Merlin proves the joint relation and optional first-output
ambiguity predicate; OOT owns the unchanged accumulator and both stores.
Existing private fresh writer storage can represent the second output without
global workspace. [Shared producer](perf_records/residual_shared_affine_bracket_capsule.json),
[large negative](perf_records/residual_joint_large_footprint_rejected.json).

The optional Merlin broadcast-axis packet pass shares identical tensor extracts
across two rows while retaining scalar operation order.25native/refusal/default
tests pass. An actual original1880 read-only tap preserves all256,000words and
captures the first45,056-byte gate output and its input tensors. Actual-source
capsule passes native/strict/128guards with all45,056outputs exact. Its full
ABBA GSIM run times out at1800seconds after only the first control interval
(5,828,044cycles); candidate timing is unknown. The normal/frozen whole
qualification now passes all256,000original compiled words/Torch/native/strict,
with a byte-exact control relink,155device bindings preserved and onlymodel.o
changed. The isolated stock1932experiment is queued versus1880 to resolve timing, with no
local speedup claim. [Whole qualification](perf_records/tiny_broadcast_packet_whole_qualification.json).
Shape/name selection is
confined to this source-binding experiment, outside production pass policy.

Stock1909 confirms integer packing776,043,123→665,638,655cycles,14.2266%lower,
all12digests/metadata/guards exact. This is a section result.
[Hardware](perf_records/smol_integer_packing_1909_hardware.json).
Stock1910 compact weight-prefetch measures38,332,743cycles,0.3540%below its
1897control but above current1903=36,102,704. The champion remains unchanged.
[1910](perf_records/resnet_compact_weight_prefetch_1910_hardware.json).
Stock1911normal resident-A family measures529,006,294cycles. The controlled
1912device-only arm measures529,440,142versus531,072,370for1880(0.307%lower).
Both retain all256,000original words and are slower than1902=527,255,504;
the current Tiny choice stays unchanged.
[Controlled hardware](perf_records/tiny_resident_a_controlled_1912_hardware.json).

| Change and owner | Complete evidence | Decision |
| --- | --- | --- |
| Typed source-bound profiler ABI — Merlin | Ordinary LLVM lowering and RV64GC/Spike compiled five f32 output events without the previous diagnostic ABI mismatch | Measurement infrastructure, not a performance gain |
| Border-only fresh destination initialization — Merlin | Isolated original pre-stem slice reduces instructions27%; actual frozen1903 runtime whole adds666instructions, all1,000 outputs exact | Whole arm rejected. Legacy capsule memset was byte-only at158,700bytes; whole libc uses words plus tail. Capsule result does not transfer across runtimes |
| Private uniform fill/copy folding — Merlin | Normal optional feature removes four copies/two slab allocations; frozen1903 all1,000 native/strict words exact/noFSM;9,246,562→9,203,553instructions |0.465% instruction reduction; hardware held, no cycle projection |
| Broad scalar memref copy expansion — existing Merlin feature | Frozen1903 all1,000 exact/noFSM,9,246,562→10,718,894instructions |15.92% instruction regression, rejected; removing copy helpers produced scalar byte loops |
| Contiguous-suffix strided copies — Merlin | Proved distinct fresh allocation roots, static common contiguous suffix, original byte representation; three selected copies, all1,000 native/strict exact/noFSM;9,246,562→9,199,529instructions |0.509% instruction reduction; hardware held; memcpy runtime retains alignment/tail handling |
| Cached-A adjacent DMA coalescing — OOT | Complete original288,847→277,410 and563,901→552,245GSIMcapsules; independent i32 tails/explicit B slots pass. Controlled1897 changes24device objects, freezes all host/runtime/weights, original1,000 exact | Stock1922 queued; first3.5Mcycle capsule truncation retained, complete6M rerun closes the second case; whole composition unmeasured |
| Encoded-zero device/support composition — OOT device, Merlin numeric metadata/consumer | Original bounded32×64 head, all four4,096-value reconstructions and192guards exact; paired GSIM mean+1.7625%, warm+2.0098% | Rejected, no complete-head target/stock admission. Large pair hit wall budget after control only; no invented candidate result |
| Exact i64 radix reconstruction — Merlin | Canonical absolute prefixes≤2^53, separate disjoint initialized scratch, defined negative multiplication, one exact final conversion; compiled/UBSan proof tests and full65,536 native/strict head exact | Separate complete GSIM pairs: variable-weight paired mean−7.3715%; constant-case−14.4479%. Includes scratch/reset/finish/encoding/readbacks. One selected complete-head stock comparison being prepared; no full-model forecast |
| Scoped primitive operand telemetry — OOT observer/provider | Copied optional engine preserves stdout and complete PC histogram byte identity for independent tails, ZIP reference and current1874; actual entry/class conservation closes54+16 reference and70 current scopes | Requested DMA payload and nominal padded work observed. Physical traffic, busy cycles and overlap unknown; production engine untouched |
| Grouped fast screen validation — Merlin | Shared target-free feature-pointer fit/unknown domains, absolute held-out checks plus interval ranking; real same-ELF timing pairs and ZIP labels excluded from fit | Counter-only fit predicts45.95M vs22.39M(+105.25%), rejected. Additive instruction/array/load/store fit cannot identify nonnegative terms, rejected |

Evidence: [uniform](perf_records/resnet_uniform_fill_copy_whole_qualification.json),
[copy expansion](perf_records/resnet_expand_memref_copy_negative.json),
[suffix](perf_records/resnet_contiguous_suffix_copy_whole_qualification.json),
[encoded-zero journey](smol-encoded-zero-groups-20261005.md),
[counter validation](perf_records/counter_only_fast_estimate_reference_check.json),
[operand refusal](perf_records/additive_operand_fast_estimate_fit_rejected.json),
[observer qualification](perf_records/spike_operand_telemetry_qualification.json).

The first additive fit's refusal is useful evidence: complete packet counts alone
do not establish CPU issue/array overlap, serialized readout cost or physical
memory services. The next reusable observations are CPU opcode classes and
declared command dependencies. Reference labels remain evaluation-only.

The CPU-role census now records70 source-bound current bodies and24 physical
reference groups covering54 measured calls, with zero unknown encodings. It
conserves body counters plus their separately stated wrapper deltas. Reference
shared-function counts are never divided into guessed per-call paths. The
actual stem touched code is174,566bytes current versus24,042reference; the
whole selected body unions are2,772,286 versus599,496bytes. Unordered PC
histograms do not establish cache misses, fetch chronology or CPU/array overlap.
The selected exact constant-case Smol head is queued as stock1924 versus1917.
[CPU census](executed_feature_census.md),
[exact head qualification](smol-exact-i64-reconstruction-20261005.md).

| Change and owner | What changed | Matched evidence | Outcome and next gate |
| --- | --- | --- | --- |
| Resident/banked transfer policy — OOT | Complete-M A residency; shape/resource/command/traffic comparison | Stock1849→1853:42,269,808→40,479,548 whole forward cycles; original1,000 words exact | Qualified transfer arm;1886 later became best; [receipt](perf_records/resnet_transfer_command_policy_firesim.json) |
| Banked residual M prefetch — OOT | Disjoint resident residual panels and next loads | Stock1849→1854:42,269,808→40,981,079 whole cycles; original1,000 exact | Separate positive arm; [receipt](perf_records/resnet_residual_banked_m_firesim.json) |
| Transfer+residual composition — OOT | Compose the two qualified strategies with identical host/runtime | Stock1853→1874:40,479,548→39,201,279cycles; all1,000 original words exact and actual staging pinned | Qualified target composition,3.158% below1853; 1886 later became best; old1861 remains historical unverified; [hardware](perf_records/resnet_transfer_residual_composed_firesim.json) |
| Complete-input convolution stripes — OOT | Full reduction weight residency and bounded output stripes | H56/C64:619,364→520,943GSIM; H28/C128:630,670→509,443GSIM; full outputs/guards exact | Stock1853→1878:40,479,548→39,754,283 whole cycles(1.792% lower). Stock1897 composition now verifies38,468,933 versus1874 39,201,279(1.868% lower), all1,000 words exact;0.350% below prior1886 best(single run). [Composition](perf_records/firesim1897_resnet_stripe_composition_verified.json); [study](perf_records/resident_stripe_conv_gsim.json), [journey](perf_records/resident_stripe_optimization_journey.json) |
| Remaining-row B slots — OOT | Two explicit disjoint next-K slots above complete resident A; interval and bank legality | M196/N512/K1024:540,046→511,096GSIM; all100,352 outputs+guards; independent i32tails pass | Seven general source selections; all1,000 whole native+Spike words exact; stock1888 verifies40,148,896cycles,0.817% below1853 but slower than1886 best; composition unknown; [study](perf_records/remaining_b_slots_prefetch_gsim.json), [journey](perf_records/remaining_b_slots_optimization_journey.json) |
| Exact bounded RNE packets — Merlin; target binding OOT | Independent scalar packet schedule, complete tensor proof, padding/tail/live-output preservation | Stock1864 warm quantization/packing traversal:3,042,623→1,749,755cycles; all150,528 inputs/158,700 destination bytes+guards | Capsule gain42.49%; Stock1853→1886:40,479,548→38,603,949wholecycles(4.6334% lower); all1,000 original bits/staging closed, then ResNet best; [hardware](perf_records/firesim1886_resnet_quant_packet_verified.json); [capsule](perf_records/bounded_rne_packet_cpu_firesim.json), [whole gate](perf_records/resnet_quant_packet_spike.json) |
| Tiny fresh writer ownership+device prefetch — Merlin/OOT | Proved complete destination writes plus existing device B prefetch | Stock1841→1846:571,097,507→569,151,067 whole cycles; original256,000 words exact and Torch gate passes | Qualified earlier Tiny control; single-run small gain, no variance claim; [receipt](perf_records/tiny_expanded_writer_prefetch_firesim1846.json) |
| Exact scalar pointwise packets — Merlin | Two/four independent SSA lanes; original operation order and exact tails | Tiny2049-output capsule:328,373→255,845GSIM(two),268,247(four); outputs/guards exact | Stock1846→1880:569,151,067→531,072,370cycles(6.6904% lower), all256,000 original words/Torch gate exact, staged ELF/bitstream closed; then Tiny best; [hardware](perf_records/tiny_pointwise_packet_firesim.json); [capsule](perf_records/tiny_pointwise_packet_gsim.json), [whole gate](perf_records/tiny_pointwise_packet_spike.json) |
| Ordered FMA replay outputs — Merlin | Independent outputs retain zero seed and increasing-K source FMA | QK64 stock1856→1857:1,137,275→910,407cycles; all1,024 original words | Capsule19.95% lower; [receipt](perf_records/attention_qk64_replay_firesim.json) |
| Selected BF16 pre-widening — Merlin | Hoist LHS conversion including allocation/copy, retain ordered replay | QK4 stock1857→1872:910,407→828,986cycles; all1,024 original words. PV4 stock1873:168,142cycles, all64words | QK gain8.94%; stock QK16 now814,343cycles(1.766% below QK4), all1,024 exact; [QK16](perf_records/ordered_attention_qk_lhs16_firesim.json). Strict PV4 control1875 verifies180,776,6.989% paired gain; [PV control](perf_records/ordered_attention_pv_control_firesim.json); [QK](perf_records/attention_qk64_lhs_widen_firesim.json), [PV](perf_records/attention_pv192_lhs_widen_firesim.json) |
| Upstream typed ordered-FMA schedule — Merlin | Structural maps/wiring/positive-zero proof;384 BF16 contractions, tile8 and both operand widening | Both unoutlined and outlined LLVM produce all1,600 original Smol bits exactly in native; normalO2RV64GC/noFSM build completed | Before the math policy, actual replay56792b45…dad8 failed89/1,600 with byte-identical old target outputs;134.731B versus327.507B retiredinstructions, no hardware-cycle claim. New full-target gate closes below; [historical receipt](perf_records/smol_upstream_ordered_fma_native.json) |
| BF16 runtime calling convention — Merlin | Build lowered scalar helpers with matching LLVM compiler ABI |17 minimal actual target edge tests pass; older ABI-corrected whole replay still failed89/1,600, maxabs.115166 | Necessary runtime correction; remaining failures were later fixed by explicit math policy below; [historical failed whole target](perf_records/smol_corrected_runtime_spike_failed.json) |
| Whole-model declared-call preservation — Merlin; catalog closure OOT | Explicit typed external permissions; source-order graph nodes, effect-only calls and declaration ABI/attributes retained | Generic13 new plus47 regression checks PASS; 3,434-node outlined identity cover and actual catalog/final ELF closure; original prepared bytes retained | Identity admission gate only, selected_plan_controls_emission=False; shared search/timing remain unresolved; [binding](perf_records/golden_model_plan_binding_qualification.json) |
| Smol suffix diagnosis — experiment only | Derive original source suffix, supply captured post-vision state, freeze original failing runtime objects | Actual target all1,600 suffix final bits exact against original golden; older complete models failed89/1,600 | Vision prefix localized further: first4 isolated target blocks exact; fifth native-exact/target failure causally restored by native expf oracle and optional float-exp-via-double; full-target math-policy gate now passes; lookup remains diagnostic-only; [causality](perf_records/smol_block4_expf_causality.json); next block/postLN exact; [localization](perf_records/smol_later_vision_block_localization.json) |
| Explicit portable libm precision policy — Merlin | Default native emits nothing; opt-in expf-via-double adds a normally hashed runtime object and linker interception | Full original native/actual RV64GC target all1,600 bits exact, zero gate failures; normal API entire loaded image equivalent except diagnostic marker | Stock1906 is the single qualified full baseline, cycles pending. No universal correctly-rounded libm or errno/fenv claim; [full target](perf_records/smol_full_double_exp_target_exact.json), [normal build](perf_records/smol_normal_host_math_policy_equivalence.json), [admission](perf_records/smol_first_exact_stock_baseline_admission.json) |
| Shared calibrated selection — existing Merlin planner; OOT generation/emission | Prices exact source-bound alternatives from pinned full-fixture GSIM receipts; solver winner controls actual compiled object | Independent17×73×65:3,234→2,357GSIMcycles; all1,241outputs+2,048guards exact, strict target/noFSM | Normal model build now compiles actual selected object into its catalog, all1,241 original i32 outputs/finalELF closed; [normal-path qualification](perf_records/golden_calibrated_normal_model_qualification.json);27.1% fixture gain; complete measured ranking, physical floor/internal engine occupancy/whole-model costs UNKNOWN; [receipt](perf_records/golden_calibrated_source_selection_qualification.json) |
| Explicit resident-A/B-prefetch alternative — OOT; normal equal-shape implementation dispatch — Merlin | Shape/resource-legal overlap alternative enters shared measured selector and actual normal model catalog; same tensor shapes retain distinct selected implementations | Common-address synthetic int8/amplitude21 GSIM111,885→76,456cycles(31.67%); native/strict single-selected full model all256,000 original words/Torch gate/noFSM pass | Expanded44-selected native and strict-target gates pass with155calls/five bodies. Stock1911 normal measures529,006,294cycles;1912 freezes all1880host/runtime and measures529,440,142cycles(0.307%below1880), both above1902best. No champion change. A single fixture price reused across equivalent physical code is a calibration assumption, not measured model operands/address/cache/full-program cost; [capsule](perf_records/tiny_resident_a_prefetch_gsim.json), [single-selected full gate](perf_records/tiny_resident_a_prefetch_whole_spike.json), [journey](tiny-calibrated-resident-a-20261005.md) |
| Bounded compact-convolution weight lookahead — OOT | Alternate disjoint bank2/bank3 panels; preserve increasing-K and all arithmetic/transfer counts | Original-input common-address535,839→499,034GSIMcycles(6.87%), all50,176outputs+4,096guards exact; independent grouped/tail475outputs pass;47focusedtests pass | Explicit default-off general option, full original model gates/hardware separate; rejected full-K/BN2 variants retained; [capsule](perf_records/compact_weight_prefetch_capsule.json), [schedule](compact_weight_prefetch_schedule.md) |
| Explicit compiler policy precedence — Merlin | Apply caller compiler flags after recipe defaults in compilation and linking | Real native executable checks fast-math macro and actual subnormal result;20 focused build/math/runtime/catalog tests pass(2 capability skips) | Correctness/infrastructure fix, no cycle-gain claim. Link-time compiler options also select startup floating-point behavior;1bc88bf28 |
| Early saturation and scalar readout packets — Merlin; ABI binding OOT | Test exact source transitions before estimation, retain input-load/output-store order in bounded8lane packets | Full50,176/25,088 readouts:1,546,262→1,130,840 and756,032→560,759GSIMcycles; allvalues+guards exact; whole original1,000 words exact and unchanged1874 control link byte-identical | Stock1874→1900:39,201,279→37,946,541cycles(3.20% lower), all1,000 original words exact and staged stock pins closed. Then best; separate from1897 stripes. No capture selector or numeric relaxation; [hardware](perf_records/firesim1900_resnet_sat8_readout_verified.json), [qualification](perf_records/resnet_exact_readout_packets_qualification.json) |
| Qualified host/target/readout composition — Merlin/OOT |1886 host packets plus1874 banked residual and1900 saturation/readout packets; source/ABI/proofs exactly matched |1886 and1900 control links byte-identical; all1,000 original native/target words exact; unchanged host/runtime/weights objects pinned | Stock1886→1903:38,603,949→36,102,704cycles (6.479% lower), current verified best; composition measured, no additive gain inference; [hardware](perf_records/firesim1903_resnet_composed_verified.json); [qualification](perf_records/resnet_qualified_packet_composition_spike.json) |
| Adjacent independent RNE results — Merlin | Explicit CPU packet with independent earlier inputs, complete typed lane proofs and distinct floating temporaries | Complete Tiny capsule256,637→247,568GSIMcycles(3.53%); same108,797instructions; full256,000 original target/native words exact | Stock1880→1902:531,072,370→527,255,504cycles (0.719% single-run marginal gain), current verified best; [hardware](perf_records/firesim1902_tiny_adjacent_rne_verified.json); current1880 profile1901 has155 conserved boundaries; root42 numeric/compiler regression tests pass; [journey](tiny_adjacent_rne_packets.md) |
| Source-stride resident convolution — OOT | Preserve execute A stride and KHWIO source order, reuse complete channel planes selected by layout/stride/capacity/traffic | Original-input GSIM727,070→577,376 (20.59% lower), independent non-square/tail case passes; normal/controlled whole native+strict target all1,000 original words exact | Stock1914 queued versus1897; only one kernel changes, original host/runtime retained; no whole gain yet. [Qualification](perf_records/source_stride_resident_conv_whole_qualification.json) |
| Complete CCA artifact — Merlin | Preserve all10 facets, full fields, scope/provenance/unknowns; reject contradictory views | Complete roundtrip/refusal and legacy artifact tests pass; integration tests124PASS across CCA, accounting and bounded packets | Compiler analysis correctness; no measured inference-cycle change; implementation150f2af09 |
| Ordinary effective lowering recipe — Merlin | Actual pass pipeline/gates, selected features, prepared source/runner/schedule/executable hashes, redacted environment and returned LLVM identity | Actual upstream lowering/native execution plus secret redaction and failed/refused reused-workdir cases pass; prior success is removed before new validation | Reproducibility/infrastructure improvement; no emitted IR or existing build-identity change. Scope excludes full imported toolchain closure and later host/device compilation; coree39059379/e531045ca |
| Finite norm adjacency — Merlin | Explicit proved nonnegative finite binary64 increments preserve all seven certificate metadata fields | Same-buffer original Q/K16×64, four complete case/order pairs:1,103,888→816,603GSIMcycles(26.03% lower); native/strict target and all metadata/planes/steps/reconstruction/guards exact | Capsule only; default callers unchanged; whole head/model impact unknown. Distinct representation-only requirements are qualified separately below; [receipt](perf_records/attention_finite_norm_pair_gsim.json) |
| Explicit representation-only norm requirements — Merlin | A distinct four-field metadata type omits unconsumed square/sqrt/reconstructed-max work when the absolute-product bound is independently available | Complete original Q/K16×64 case/order pair totals1,121,764→761,887GSIMcycles(32.08% lower), all required metadata/planes/reconstruction/guards/native/strict target exact; original default wrapper object bytes unchanged | Capsule only, independent control from the seven-field adjacency pair; no additive or full-head gain inference. Nontrapping arithmetic/unobserved exception flags are explicit; [receipt](perf_records/attention_representation_norm_pair_gsim.json) |
| Ordinary effective compilation recipe — Merlin | Records actual ordered compile/link argv, compiler executable hashes, input/output identities and redacted project environment; stale success removed before validation | Real ordinary upstream→RV64GC build and exact Spike output; link inputs replay byte-identically; failed command cannot claim a retained older object as output | Infrastructure observation, no emitted code or performance change; explicitly incomplete header/library/imported compiler closure; core7fb0fec7d |
| Primitive semantic-field coverage — OOT | Preserve all ConfigEx activation/shift/scale/stride/transpose/stride-only fields and Flush skip; refuse unsupported fields and malformed values |16focused checks decode actual lowered inline-assembly operands and compile/audit the object; previously qualified default and source-stride objects byte-identical | Fixes silently ignored instruction parameters; no cycle-gain claim; bd34e6e |
| IEEE constant clamp grouping — Merlin | Typed independent maximum/minimum groups use one NaN guard; freeze prevents poison-based control-flow UB; strict/fenv/unsupported contracts refused |13focused frontend tests; native57,876 raw cases, strict165,360 five-mode rawbits/fflags and1,520 directed GSIM cases pass; measured capsule object unchanged after freeze | Interleaved capsule253,765.33→249,014GSIMcycles(1.872% lower); whole native/target qualification in progress, no hardware gain; coref586fc25b/8ead5e46b |
| Source-stride residue input layout — OOT | Group physical input rows by stride residue, preserve logical input bijection and source K order, retain legal16-column panel and exact live resources | Original input774,520→712,682GSIMcycles(7.984% lower), all25,088 outputs+4,096guards; independent non-square/channel-tail case exact; prior queued1914 object unchanged | Explicit default-off normal compiler option; only an incremental second stride-2 kernel changes; whole original-source qualification complete, hardware pending; [layout](source_stride_residue_layout.md) |
| Exact radix group proof — Merlin; resident product-sum provider — OOT | Prove weighted integer exactness and per-group signed accumulator range; reduce equal-weight plane pairs in resident i32 accumulator and read each group once | Independent323i32-output capsule+guards exact; full original head native/strict all65,536 bits exact, same replay counts;54→30 calls and84,934,656→47,185,920 readback bytes | No digit approximation; stock1917 queued against1915 norm-only and1894 control, full-model effect unknown; [qualification](perf_records/attention_grouped_full_head_qualification.json) |

Each hardware receipt pins the final ELF, actual staged ELF, hardware configuration and output.
GSIM kernel cycles, Spike retired instructions, analytical geometry and full hardware cycles have
different scopes. Gains from separate arms are never added to predict a composed model result.
Queue admissions above are pending experiments, not accepted performance results.

## Rejected hypotheses and the lesson

| Hypothesis | Evidence | Decision |
| --- | --- | --- |
| Fewer instructions and full-A dense stripes should be faster | Matched full3136×256×64 GSIM:204,570control→308,168BN4 /303,445BN16, despite fewer instructions; all802,816 outputs+guards exact | Reject48–51% regressions; archive prototype and keep qualified banked transfer policy. Need actual overlap/bank-service costs. [Receipt](perf_records/resident_stripe_dense_rejected_gsim.json) |
| BF16 widening both operands at M1 pays for itself | All packing included; QK672,081 versus385,298 retired instructions; PV126,771 versus69,393 | Reject this M1 choice; selected-operand widening is a separate proved strategy. Instruction metric only. [Study](perf_records/ordered_replay_selected_widening_spike.json) |
| Opaque packet helper placement alone will remove dead quantization | First whole packet build retains two quant maps,11,209,182 versus10,195,770 instructions, although all1,000 outputs exact | Do not enqueue. Generic source use/effect liveness removes the dead NCHW branch before helper insertion; corrected whole qualifies at10,053,314instructions, then hardware decides. [Diagnostic](perf_records/quant_packet_dead_branch_diagnostic.json) |
| Fewer instructions certify a resident convolution-loop gain | Stock1844 whole43,514,726 versus1836control42,837,088cycles | Reject hardware regression; instruction count is an explanatory metric only. [Receipt](perf_records/resnet_resident_channel_loop_firesim1844.json) |
| A two-chunk affine predictor plus rare exact source corrections beats the39-chunk exact device residual | Device-only capsule19,817 versus164,566GSIMcycles; complete65,536-prefix correction293,843cycles versus164,566, and full-domain correction687,416 | Reject78.56%/317% complete-cost regressions; source-pair certificate utility remains opt-in. CPU scanning dominates; full802,816-output timeout has no final correctness closure. [Study](perf_records/residual_affine_cpu_correction_rejected.json) |
| Searching affine coefficients and a small constant seed gives a cheaper full-domain exact residual |4,605candidates up toq512 find none;36,861up toq4096 first exactp2609/q2180/seed0 needs39chunks, matching existing schedule | No promoted change. Only the stated ratio neighborhood/seeds were searched; not a proof against other algorithms. [Search](perf_records/residual_affine_scale_search.json) |
| Reducing attention device planes offsets a stronger CPU certificate | Complete original-head stock1894→1895:2,618,580,085→3,077,601,494cycles; all65,536 original bits exact | Reject17.53% hardware regression; fewer device bytes/commands do not establish complete performance. [Hardware](perf_records/firesim1895_original_attention_head_gamma_verified.json) |
| Two-digit attention passing a local head gate establishes whole-model acceptance | First original head passes local tolerance; full native source fails106/1,600 original comparisons,maxabs.14355785,relativeL2.02077864 | Reject without target/hardware promotion. Original tolerances/inputs/golden remain fixed. [Whole-model negative](perf_records/smol_two_digit_full_native_rejected.json) |
| Three-digit attention removes the whole-model accuracy failure | First-head27/65,536 changed words; full native fails121/1,600, maxabs.167664. A separate zero-encoding-error wide accumulation fails144/1,600; same-seam ordered f32 replay is all1,600 bitexact | Reject precision sweep; accumulation rounding alone explains a sufficient failure. Preserve source ordering or certify/replay ambiguity. [Negative](perf_records/smol_three_digit_full_native_rejected.json), [diagnosis](perf_records/smol_source_replay_vs_wide_accumulation.json) |
| One local-LSB residual approximation is accepted by the original whole-model gate | Source-derived complete65,536pair proof has53 mismatches/max1LSB; changing only one residual alters all1,000 final words,maxabs1.2304 | Reject at the unchanged exact gate; no target or hardware admission. This is separate from the earlier costly exact-correction branch; [receipt](perf_records/residual_affine_approx_native_rejected.json) |

Guarded signed-digit attention now executes primitive device partials, certificate computation and
exact replay together for a complete original first-vision head on the target, with all65,536 output bits exact.
The factored signed-prefix bound retires1,483,561,398instructions versus a fair1,185,308,895 control;
a cheaper gamma bound improves to1,430,369,073 but still loses20.67% in that metric.
These candidates are not admitted as a performance winner. Paired stock FireSim1894/1895 now measures
2,618,580,085 versus3,077,601,494cycles: the gamma candidate is17.53% slower and rejected.
The stronger bound also passes all786,432 original first-block
output bits across all12 heads in strict target execution at17,319,092,950 retired instructions.
[Archived target costs and qualification](../out/artifacts/probes/attention-prefix-guard-20261005/README.md)
retain every losing variant, actual kernel calls, input/ELF pins and source snapshots.
[Hardware decision](attention_certificate_hardware.md) pins the completed fair pair.
A whole-model gate remains separate;12 heads in one block are not12 model layers. Full original first-vision
head audits do not imply all model layers have equal fallback rates. Account for digit uploads,
partial readback, guard evaluation and replay, and preserve the original Smol elementwise criterion.

## Reference hardware comparison

Stock1876 reproduces the permitted ZIP reference at22,387,449 full forward cycles with
its original1,000-logit self-check and54 layer timings. This reference-only result is separate
from our original-source qualification. It provides a same-hardware schedule comparison,
not model equivalence: source numeric coefficients and physical input/output boundaries differ.
[Reference receipt](perf_records/q1013_diagnostic_reference_firesim.json).
The [shape-paired hardware intervals](q1013_paired_hardware_intervals.md) reconcile all54 layers
against the1850 profile of1849, with class/whole conservation and explicit numerical scope.
These are historical measured cost locations; they do not causally partition the remaining1886 gap.

The later1899 profile binds the unchanged1874 model and reconciles its39,235,729
interior cycles against the same reference. The current1903 best36,102,704 still
has13,715,255 cycles to remove. Its object-exact70-boundary profile is qualified
and measured as1919; current stock attribution is recorded above.
Reference other/uncounted51,717 is time outside printed layer timers, not total
CPU time. [Current gap and measured historical locations](resnet_current_gap_status.md).


## Source, storage and command-loop checkpoint (2026-10-06)

| Change | Owner | Complete measured evidence | Whole qualification |
| --- | --- | --- | --- |
| Hoist repeated source row math from broadcast | Merlin typed pass; OOT capsule/binding | First original16,384-output normalization mean2,369,163→838,269GSIMcycles(-64.6175%); five rounding modes/sticky flags and128guards exact; allocator caveat retained | All256,000 original words/Torch/native/strict/noFSM; frozen1880model.o-only stock1926 queued |
| Borrow segmented projection input | Merlin view/ownership/acceptance; OOT DMA/ABI | Original consumed401,408bytes+4,096guards; common-address866,791→538,815GSIMcycles(-37.838%); unread owner cells disclosed synthetic | Three accepted views through normal source/catalog, all1,000original native/strict words; frozen1903stock1927 queued |
| Retain ordinary CPU spatial command loops | Merlin no-unroll metadata; OOT exact bounded address/schedule | Independent15,257i32outputs mean-4.0067%; complete50,176-output current stride2geometry with independent inputs mean-1.7326%; first automatic-unroll attempt retained negative | Normal and frozen1903 all1,000original words pass;11kernels change/all52adapters and original host/runtime retained; stock1928 queued |
| Source-bound exact readout range | Merlin numeric proof; OOT producer binding/ABI guards | Both complete original capsules15–24%reductions by pair/context; native/strict1,000words pass normal route | Normal runtime change disclosed; no isolated whole stock claim |
| Two exact i8readouts with decoder | Merlin complete-domain certificate/portable scan; OOT store plan/accumulator lifetime and preparation ABI | Original75,264bytes+2,048guards/eightABBAcalls exact; complete compiler-copy readout ROIs improve71.41%/70.40%; independent convolution tails pass | Normal preparation now allocates actual i8 scratch; selected allocations50,240/25,152bytes including alignment; normal/frozen1903 all1,000 original words/native/strict/noFSM pass; stock1930 queued |
| Retain stem ordinary CPU command loops | Merlin no-unroll metadata; OOT exact stem/pool schedule and address proof | Original200,704output bytes+4,096guards exact;1,335,580→1,294,977GSIMcycles(-3.0401%); independent shape-11.53%;87.8%text reduction is not cycle evidence | Normal upstream and stem-only frozen1903 all1,000 original native/strict words/noFSM pass; stock1929 queued |
| First exact full Smol hardware baseline | Merlin ordered source arithmetic/explicit host math policy; OOT catalog | Stock1906:258,621,872,969 forward cycles; all1,600original words bitexact;8,918.2seconds engine elapsed | Performance51.724times5B goal; optimized head capsules remain separate from whole route;384BF16 CPU contractions/19.327B source FMAs motivate coverage work |

Current1903profile completed in GSIM at33,939,464forward cycles, exactly
29,422,251device-wrapper+4,517,213host-gap. Original1,000words, all70events
and final audit pass. Stock1919 is now complete; memory regimes remain distinct.
ResNet and Tiny whole-model champions are unchanged; Smol now has its first
qualified stock baseline above. Source-work counts do not assign measured
cycles to those operations.

Generic Merlin analysis now retains the original DAG through 48 BF16 attention
endpoints, with eight contractions and 50 operations per group and no live f32
escape. Source, uses, maps, ancestor context and ordered-FMA contracts are
revalidated before extraction. Numerical certification, actual provider call
coverage and whole-model profitability remain separate obligations. The user's
earlier approximately4B Smol result is being audited to recover its fast route;
258.622B is the current qualified implementation, not a historical-best claim.
The retained owned job610 audit closes the actual ELF/UART/gate/plan identities
and finds FSM instructions plus an older numerical gate. Its declared ten-step
trajectory does not establish a separately measured3.31B step. The separate
near4B Exo implementation remains an open provenance comparison, without
reading private reference folders. [Audit](perf_records/historical_smol_job610_audit.json).

The CPU-footprint fast-model diagnostic retains a negative result: only12of24
reference physical groups resolve, with64.53%maximum section error. Repeated
physical groups outside the training domain and missing host/residual costs
leave whole prediction and ranking unknown. The screen is disabled. No
reference labels were used to fit its coefficients. Compact-code candidates
are therefore timed rather than selected from this rejected model.

Receipts: [norm](perf_records/tiny_source_broadcast_math_complete_capsule.json),
[projection](perf_records/segmented_input_original_projection_gsim.json),
[loops](perf_records/flat_spatial_command_loop_qualification.json),
[normal readout](perf_records/resnet_normal_producer_domain_qualification.json),
[pair](perf_records/exact_pair_readout_matched_gsim.json),
[pair compiler-copy](perf_records/exact_pair_readout_builtin_matched_gsim.json),
[paired typed whole](perf_records/resnet_paired_readout_typed_whole_qualification.json),
[stem](perf_records/stem_spatial_command_loop_capsules.json),
[stem whole](perf_records/stem_spatial_command_loop_whole_qualification.json),
[normal spatial loops](perf_records/resnet_spatial_cpu_loops_normal_whole_qualification.json),
[Smol stock](perf_records/smol1906_stock_hardware.json),
[Smol source revalidation](perf_records/smol1906_source_group_parent_reclosure.json),
[profile](perf_records/resnet1903_gsim_conserved_profile.json),
[screen](perf_records/cpu_footprint_fast_estimate_partial_check.json).

## Token accounting

[Measured snapshot at2026-10-06T07:17:03Z](perf_records/golden_token_usage_20261006T071703Z.json)
contains counters for the eight explicitly owned root/worker/descendant sessions.
Ownership is supplied by the root spawn mapping; copied session metadata IDs are not used.

| Bucket | Cumulative measured campaign traffic |
| --- | ---: |
| Uncached input |38,184,042|
| Cached input reads |1,819,432,960|
| Output |7,567,875|
| Reasoning output, already included in output |3,015,712|
| Raw input+output total, including cached reads once |1,865,184,877|

The raw total includes repeatedly read cached context. Uncached input plus output totals
45,751,917; it does not count cache reads or reasoning twice. Tokens do not establish dollar spend.
The separately observed goal counter is42,118,654; its accounting semantics are not exposed, so it
is retained rather than silently equated to raw session totals.

The latest snapshot records completed-request deltas since05:43:24Z:
2,225,183uncached input,490,693output and102,475,776cached reads.
The [prior snapshot](perf_records/golden_token_usage_20261006T054324Z.json) retains the earlier window.
Threads mix OOT, shared compiler, numerical debugging and orchestration. Exact OOT-only or
per-optimization token allocation is **unavailable**, not zero. Experiment starts, finishes and
transitions should carry counter checkpoints and request-span attribution; elapsed-time prorating
is not proof. Earlier Claude traffic is not reconstructed from this Codex snapshot.

The [ownership summary](infra_vs_dialect.md) records measured host-versus-device performance
comparisons. No overall effort or token percentage is inferred from those nonadditive arms.

Reusable counter reader: Merlin `merlin.agentreport.tokens.read_codex_tokens` (9fcace968).
The explicit campaign driver and retained token-count-only evidence are under core
`out/artifacts/perf-studies/golden-optimization/`; the tracked snapshot records source prefix hashes,
counter-evidence hashes, observation times, agent mapping and the driver hash.

## Automatic convergence and compiler ownership

The [target-agnostic phase1/2 recommendations](/scratch/agustin/tmp/merlin-golden-integration-20261004/docs/design/agent_compiler_performance.md)
map existing CCA/schedule/planner/timeline/edit-contract machinery to concrete integration gaps.
An OOT catalog does not establish shared whole-program search. Declared optimization surfaces must
resolve to real reachable compiler owners, and a selected plan must account for emitted executable
work. Host/global transformations require explicit Merlin edit grants. Numerical gates, source
inputs, hardware identity and trusted grading remain fixed.


## 2026-10-06: production host compilation and new stock results

ResNet1930 exact paired readout is the current stock champion at34,905,135cycles,
with all1,000 original outputs exact:3.3171% below frozen1903 and0.7043% below the
separate segmented-input1927 arm. Stock1928 spatial command loops35,722,259 and
1929 stem command loops35,966,658 remain slower than the champion. These arms
are not composed and gains are not additive. The original Clang residual guard
capsule did not reach the candidate timer or terminal marker within1,800seconds;
its candidate cycles remain unknown. The unchanged candidate is being replayed
with a longer budget; completed GCC negatives remain rejected.

For SmolVLA, actual ordinary compiler FMA/copy capabilities reduce the matched
complete12-head group9.624B→5.709B retired instructions. Endpoint row invariants
plus exact floor reduce it to5.0345B, with all48 original live inputs,1,600 outputs,
9,437,184i8 values and12,288 escaping scales exact in independent native gates.
No normal whole-model hardware performance follows from those functional gates.

Merlin05e119b95 adds a separately selected standard finite-classification builtin
contract. Ordinary no-builtin compilation had emitted costly library classifiers
for eligibility checks. The choice requires standard classification, unobserved
interposition and exception flags, and nontrapping execution; prior FMA/copy
permission does not imply it. Defaults preserve library semantics and reproduce
this production control object byteexact. The full original target group uses
3,527,309,706instructions versus5,034,507,191(−29.9373%), all196,608 accepted
carriers and guards exact, same8 refinement counters and4,461,440source FMAs.
Only provider.o changes; device, driver, bridge and original data stay fixed.
565,925 independent target checks cover all BF16 patterns and F32/F64 exponent,
mantissa and random boundaries under five ambient rounding modes.194live pins
are rehashed and every executable ELF section is audited for forbidden/unknown
accelerator instructions. Five old NaN-classification imports remain; their cost
has not been isolated. The native SO is byteidentical to accepted full48row/floor,
so its independent original consumer qualification is reused by actual identity,
without another execution or wall-time claim. All counters here are functional
Spike retired instructions, not FireSim cycles. WholeSmol remains258.622B stock
and the5B target is unmet.

The Tiny division census also refused a tempting incorrect optimization:
991,232 dominant pre-down reciprocal results pass through three separately
rounded products beforei8 observation. A direct quotient-to-integer threshold
cannot preserve this DAG. An optional bounded reciprocal/refinement experiment
retains every source multiply and replays the original division when the final
bin is ambiguous. It has no production selection or hardware result yet.


### Owned token observation2026-10-06T07:48:33Z

Eight explicitly owned threads: uncached input39,152,925, cache
input1,860,482,560, output7,716,380, reasoning
3,073,691(already included in output). Raw input+output is
1,907,351,865; uncached input+output is46,869,305.
Since07:17:03Z completed-request window: uncached input968,883,
output148,505, cache input41,049,600. These counters
cover mixed compiler, target, correctness and orchestration work; exact OOT-only
and per-optimization attribution remain unavailable. Goal tracker43,233,797
is a separate observed metric with unexposed cache semantics. No unrelated
sessions were read and no elapsed-time proration or dollar cost was inferred.
[Raw owned-thread ledger](perf_records/golden_token_usage_20261006T074833Z.json).


### General numeric min/max and measured promotion, 2026-10-06 09:15 UTC

Merlin1c5b1d1dc adds explicit min/max capabilities with separately proved
unobserved signed-zero/NaN-payload distinctions; defaults retain library calls.
The actual general emitter reproduces the measured candidate object and ELF.
859,360 target operand pairs across five rounding modes pass the declared
observation contract; all48 original live consumer inputs,9,437,184 i8 values,
12,288 BF16 scales and1,600 final words remain exact. The complete group saves
3.764% instructions. Preserving library zero/NaN paths adds6.485%; reject it.
The282-pin closed receipt preserves both arms. No whole performance follows.

Fresh stage measurements place soft intervals at1.294B of3.250B instrumented
instructions, dot bounds779M and consumer/refinement422M. Instrumentation adds
0.196%; nested child scopes must not be double counted. Generic exact-zero-error
hoisting saves3.307% in a separately qualified complete group; its full48 gate
is pending. Q-row preparation reuse is rejected: best matched arm adds0.379%.

Phase1/2 tooling should expose interval proof strategy and certificate tightness
as explicit portable choices, with exact source replay counted as part of cost.
Prepared producer proofs can remove repeated eligibility checks only when they
bind actual immutable spans, formats, lifetimes and generations. No unchecked
boolean, shape equality or model name establishes those facts. Target resources
and CPU ISA emission remain in OOT. Paired measurement tools must also account
for optimizer-eliminated harness work before claiming instruction savings.

The new portable exact byte checker passes native and RV64 functional properties,
but its preliminary cost comparison is invalid: the compiler hoists repeated
pure byte checks while retaining word checks, and an outlined helper drops
alignment knowledge. That evidence is retained and the harness/codegen are
being corrected before promotion; no model-performance gain is claimed.

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

### 2026-10-06 10:02 UTC resumed qualification checkpoint

Root integrated general typed f32 multiplication packets (core709345057) and
private producer-bound FMA row-domain admission (corea22d31fd7), composing all16
endpoint/word/zero-error/domain combinations. All native product/frontier/word
cases pass;40 scalar packet tests pass with explicit real upstream paths, and
repository structure/docs checks pass. Initial dummyupstream-path failures are
environment failures, not silently treated as test passes. Default routing stays
inert and default generated bytes remain the original SHA.

Tiny isolated pure-multiply whole candidate preserves all256000 original words,
originalTorch tolerance,155sourcebounddevicecalls and finalzeroFSM. Normal current
control reproduces1926 exactly; onlyhostmodel.o changes. Functional retired
instructions144498799→142343799 (−1.491%); hardwarepending. Original champion
461389700cycles stays the verified best. Root reviewed/reclosed168filepaths
across Tiny and the Smol word capsule before release; original gates unchanged.

Smol word-space complete original group now passes the actual compiled54-op
original rowconsumer on RV64GC:786432i8values+1024scales (fourrepetitions of
original256rows toexercise the original1024rowABI), inputbytes/descriptors/guards
exact, noFSM. Instructions3243485384→3014749382 (−7.052%); nothardwarecycles.
Two oldunobservedcarrier differences are retained. Independentfull48native
original1600words/Torch +9437184consumeri8values/12288scales exact alreadyclose;
146oldcarrier differences and60.5% moreexactsourcePVreplay are retained separately.
A table10enclosure prototype passes thesameconsumer but costs3263954378instructions
(+0.631%versuscontrol); do notpromote it. Initial consumertrap was minimized and
fixed in afreshharness by linking the unchanged Merlin allocator. A separate
libcputs successmarker failed toprint despite rc0; anew matched printfharness
prints observedPASS. Old failures preserved; no changes to originalconsumer,
sourcepolynomial, goldens or production fallback/routing.

Owned tokenledger refreshed onlyfrom8explicit owned rolloutfiles, cutoff
2026-10-06T10:02:05Z:uncachedinput42509478, cacheread2018702720,
output8371543 (reasoning3338787 alreadyincluded),
rawinput+output2069583741, uncached+output50881021. Exact OOT-only and
peroptimization attribution remain unavailable; no proration or unrelatedsessions.
Goal tracker is separate:47241576tokens/96704seconds atthischeckpoint.


### 2026-10-06 10:59 UTC measured continuation

Scratch cleanup remains complete (45.538 GiB physically reclaimed by our
hash-verified actions); approximately89 GiB now available on scratch. Original
model data, goldens, ZIP, historical paths and live experiments remain retained.
Fresh outputs are used for every new build.

**Verified stock bests:** ResNet1930 remains34,905,135cycles (ZIP1876 is22,387,449);
Smol1906 remains258,621,872,969cycles; Tiny1967 is now451,221,105cycles versus
1926's461,389,700, a2.204% improvement with all256000 original words/Torch gate.
The target goals remain unmet. Root rehashed the owned terminal evidence;
current source/ELF and stock metadata stay bound to the original measurement.

ResNet1957 finished35,728,234cycles,1.037% below1903 but slower than1930; retain
this result without composing it into the champion. Full-K B capacity is integrated
(OOT2b51f56),491pins independently reclosed and50tests passed; stock1971 admitted,
whole cycles pending. The source-derived resident weight packet policy is also
integrated (OOT841eac2): matched kernel536180→468824GSIMcycles,12.56% improvement;
prior complete-panel prefetch499263 at the same seam. Only5normal device leaves
change,47others/source proofs/adapters unchanged. Root492whole+279capsule pins and
98tests pass; stock1974 admitted once vs1903. Independent small shape6359→7858
is a23.6% regression, so this remains explicit and has no universal profitability
policy. Whole-model improvements are not inferred from local timings.

Ordinary whole Smol minmax execution completed all1600 original words exactly,
rank0/DONE/rc0,220253190030 retiredinstructions versus rowfloor306168018359,
28.06% fewer. This remains above the original instruction baseline and is not
promoted for stock whole-model performance. Word capsule1968 is queued; all
original compiled consumer/gate evidence retained.

Core consumer-derived typed L1 norm requirements (dd5d0f57c) eliminate dead
squares/root work only under immutable all-zero RHS error admissions and typed
L1 consumers. Original complete group3086875912→3067137022instructions,0.639%
fewer; original accepted196608carrierwords and8stats unchanged, full native1600
exact. Certified-row endpoint retention (0b48951f5) independently reduces the same
control to2904713292instructions,5.901%, preserving complete initialized certified
rows and refinement/source replay counts. Full native1600exact. Root integrates
all36valid endpoint/word/zero-error/domain/norm/retention policy combinations:
504tests PASS. Root168immutable source/header/ELF/native evidence pins reclosed.
A historical mutable-source pin failure is preserved explicitly; original git
bytes match and are rebound to immutable evidence, not relabeled as unchanged.

Generic build fix61e2628dc adds separate late linker flags after objects/runtime
libraries. A real static-archive failure reproduces before the fix;9tests pass,
and actual RV64GC helper ELF is byte-identical to correctly ordered manual linking.
Generic LLVM outline+merge policyf93e22c04 preserves original exported symbols,
source arithmetic, observable stores and function-address rules;16executed tests
and core structure/docs checks pass. Tiny isolated M2 outline cycle gain is0.901%,
while62.39%whole text shrink is only code-size evidence. Independent current M2
broadcast scheduling measures9.761%fewer GSIMcycles; neither has a new whole stock
claim. A fresh composition with the verified1967 pure-multiply champion is being
qualified, with original256000/Torch/noFSM/155devicebindings required.

Preserve negatives: representative counted-N panel+3.20% and grouped DMA
interleave+3.57% despite smaller code/fewer spills or plausible queue benefits.
The unpromoted polynomial table's exact integer division-to-shift change passes
original compiled786432i8+1024scale consumer, inputs/guards/noFSM and16native
source/quotient tests. Full group instructions regress1.212% versus control.
A small synthetic helper pair improves GSIM1.388%; it does not establish an
actual group or whole gain. Unpromoted source is archived outside the installed
core runtime/test tree, with failures and recipes preserved. Uniform source-radius
feasibility widens replay4.705x; column-specific alternative2.629x. Both preserve
original compiled consumer but lack cost proof. A complete-cost screen for the
column-specific proof is now authorized; no empirical threshold or stock claim.

**Phase1/2 tooling:** use existing Merlin content_store immutable byte copies
for compiler source closures and model artifacts; pin original git blobs plus
physical frozen files, never only mutable checkout paths. Keep explicit compiler
policy prerequisites and typed consumer requirements/certificate-state transitions
in the agent edit surface. Score complete source-bound sections and then whole
compositions; reconcile dynamic commands, replay counts, host text/instructions
and measured cycles separately. Static shrink, queue hypotheses, instruction
counts, unique bytes and requested traffic cannot substitute for physical timing.
Derive applicability from semantics/layout/resources; keep target implementation
in OOT and reusable host/numeric/build mechanisms in Merlin.

Owned tokenledger cutoff2026-10-06T10:59:12Z (8explicitownedthreads only):
uncachedinput43721954, cacheread2085313152, output8644828,
reasoning3463950 alreadyincluded in output; rawinput+output2137679934,
uncached+output52366782. Exact OOT-only and per-optimization
allocation remain unavailable; no proration or unrelated session access. Separate
goal tracker48731274tokens at this checkpoint. Evidence is in
perf_records/golden_token_usage_20261006T105912Z.json.

## 2026-10-06 11:20 UTC — qualified compositions and general build fix

Whole stock bests are ResNet34,905,135 (1930), Tiny451,221,105 (1967),
Smol258,621,872,969 (1906). No target is met. Scratch has83GiB free and
scratch2 has5.4GiB free at this check; no further deletion was needed.

**Tiny normal host compositions.** The source-exact LLVM loop-extract/mergefunc
policy with current1967 pure-multiplypacket4 passes original256000 native,
Torch, strictSpike, finalELF noFSM and all155 source-bound call references.
Fresh normal control IR/object/relink reproduces1967 byteexact. Onlymodel.o
changes. Full retiredinstructions142343799→142195690 (−0.10405%), which is
not a hardware prediction. It is admitted once as stock1975 versus1967.
The isolated1926 outline arm increases instructions0.02268%, and is retained
without a separate hardware submission. The complete original two-row preDown
GSIM capsule measures−0.900895%; whole host object text shrinks62.39%, a
separate static metric. [Composition](perf_records/tiny_loop_multiply_composed_whole_qualification.json),
[isolated attribution](perf_records/tiny_loop_outline_isolated_whole_qualification.json),
[1975 admission](perf_records/stock1975_tiny_outline_multiply_admission.json).

The independent broadcastrow2 policy with current1967 pure-multiplypacket4
also closes every original whole-model gate and1967 normal control. Its complete
M2/all5632-channel source preDown capsule measures−9.76097% warmABBA GSIM,
including allocation/finalcopy and source operation/rounding order. Full model
retiredinstructions142343799→142342304 (−0.001050%); actual whole cycles are
UNKNOWN. It is admitted once as1976 versus1967, without outline or rejected1954
FMApacket4. Root104 current pins and finalELF were independently rehashed.
[Local cost](perf_records/tiny_broadcast_predown_m2_journey.json),
[normal whole qualification](perf_records/tiny_broadcast_multiply_whole_qualification.json).

**Smol portable source certificates.** Default-off separable_source_radius
proves the source-zero-seeded ordered-FMA error with upward rowgamma×L1 and
each source column maximum. Private exact reconstructed products, zero admitted
representation errors/no uncertain positions, stableRNE and a uniform finite
source-prefix envelope are required. Unknowns keep the checked bound. Source
arithmetic and original elementwise gate remain unchanged. The wider enclosure
increases replay4,461,440→11,730,560FMAs, yet complete original consumer provider
ROI instructions3086875912→3032543604 (−1.7601%). Original54-op source consumer
786432i8+1024BF16scales and all1600 whole outputs are exact. Three old intermediate
carrier values differ but are unobserved by the actual source consumer. The old
carrier-equality diagnostic failure is preserved; no numerical gate changed.
[Source proof and actual costs](perf_records/separable_source_radius_qualification.json).

Merlin integration3f0c86375 preserves all prior word/zero-error/domain/norm/row
policies. All60 valid7-policy combinations compile and execute against independent
source oracles, with770 tests PASS. The radius substitution precedes typed L1
replacement so the whole initialized product-row statement retains its proof.
Fresh L1+certified-row+radius composition source snapshots are frozen before builds.
Original consumer/full1600 native gates pass; complete ROI2879340410 instructions
versus3086875912 (−6.7232%), with480devicecalls/86,507,520readback bytes unchanged.
Same-ELF whole-program PC totals3545895757→3338365845 reconcile exactly: inclusive
local evaluate_products−125061674 and group_provider−92260334; encoder unchanged,
extra replay/rounding library costs retained. These are instruction attribution,
not isolated-stage causal cycles or whole hardware gains. Root148 composition,
PC and admission pins reclosed; no radius hardware request.
[Composition](perf_records/radius_l1_certified_rows_qualification.json),
[PC conservation](perf_records/radius_l1_certified_rows_pc_comparison.json).

The next RHS reuse investigation rejected repeated refinement scans: actual
certify_frontier makes zero evaluate_products/dot_bounds calls; RHS preparation
runs96 times only in initial2QK+6PV×12heads. Reuse across the four query quarters
needs typed source owner/view/format/epoch and physical lifetime proof. Equal
shapes, values or public pointers do not establish reusable preparation. Diagnostic
RHS attribution must disclose noinline/timer overhead before any profitability
claim; no production cache or workload selector has been enabled.

**ResNet queue and spatial proof.**1971 full-K capacity and1974 residentweight
packets remain queued versus1903, with durable owner collectors. Root imported
1974admission and closed70 local-loop/broadcast/admission pins. The next explicitly
selected halo-free resident-plane layout proves all original H7/C512 output cells
and increasingHWIOK order. Current fragment reload count2016→328 and requested
Abytes369664→68096; Bbytes2359296 unchanged, compute36864 andi32stores128 unchanged.
Three shifted planes6048SPADrows/1024ACCrows fit. Independent nonsquare3×5/C32/N67
terminal capsule8076→7995GSIM (−1.00%), all1005 outputs/guards/input bytes exact.
Original section terminal and retained-loop ablation are PENDING; initial3M
budgets stopped during postROI validation and are preserved as INCOMPLETE.
No whole speed, physicalDRAM or reference-i8 equivalence is inferred from these
logical request counts. No new flat-layout stock release yet.

**Device/host build infrastructure.** Root reproduced a real generic compiler
bug: caller/provider translation units with equal basename overwrite one .o,
causing duplicate symbols and a missing function. Merlinfd3069220 introduces
shared deterministic named_object_paths in the pure build recipe, used by both
contract and layer builders. Existing unique names/recipe/link order are preserved;
colliding sources receive link-position ordinal names and imported objects are
reserved. Real native builds execute both source units in either support order,
repeatELFhashes match, and imported kernel bytes remain unchanged.42 relevant
build/handoff/oracle tests PASS,1 existing optional skip;2 historical target-backend
integration tests are outside the selected harness-only provider capabilities.
The initial missing-provider/load-address/legacy-error-class failures and the
pre-fix real collision failures were observed, not silently reported as passing.
All structure/docs checks PASS. This generic orchestration belongs in Merlin;
compiler/ISA/ABI/linker resources stay target-declared inOOT. No model2MLIR source
changed this interval and no unauthorized main push occurred.

**Phase1/2 tooling.** Continue using exact source-compatible controls, explicit
candidate compositions and default-off policies. Shared operand preparation needs
an immutable handle and caller-owned lifetime in portable IR/ABI; target packing
formats/resources/implementation stayOOT. The existing content_store should freeze
source/tool dependencies before compilation and deduplicate bytes, rather than
adding another store. Attribution should distinguish source operations, immutable
preparation, exact replay, endpoint certificates, accelerator requests and actual
physical traffic. Actual complete original consumer/correctness gates, cost scope,
terminal completion and executable closure must travel with each candidate. Global
search must retain negative interactions and verify fresh compositions; fewer
instructions or smaller code alone do not select a hardware winner.

**Owned tokens.** Explicit8-thread ledger at11:19:48UTC records44,519,893uncached
input,2,112,114,944cache-read,8,755,549output,3,501,723reasoning (subsetoutput),
2,165,390,386rawinput+output or53,275,442uncached+output. The20m36s completed-request
window contains797,939uncachedinput+110,721output (908,660),26,801,792cache-read,
37,773reasoning subset,27,710,452rawtotal. ExactOOT-only/per-optimization allocation
remains UNAVAILABLE; no proration. Separate active goal tracker49,632,438 tokens.
[Ledger](perf_records/golden_token_usage_20261006T111948Z.json).

At11:27UTC, the completed RHS diagnostic records92,951,781 encode plus
68,195,232 metadata instructions (161,147,013;5.574% of perturbed ROI), with
0.403% total instrumentation/codegen overhead disclosed. Root146 RHS/build/1976
admission pins reclosed. The source mapping then identifies soft_details call
~1,296,968,766instructions, dot_bounds~508,331,531 andcertify_frontier~278,765,395
in the existing composed PC profile. Debug-only final relink preserves all102
allocated sections/address/size/type/bytes; mapping closure remains under review.
These nested inline costs overlap and must not be added to whole function totals.
Prepared private source-domain/interval checks are the next evidence-led direction,
with original fallback and numerical/source/consumer gates retained.
[Measured RHS scope](perf_records/attention_rhs_preparation_cost_census.json),
[generic build fix](perf_records/generic_object_naming_compile_fix.json).

The new ResNet resident-plane matched source kernel is now terminal GSIM764114→
718722 (−5.94%), preserving all25088i32 outputs/4096guards/2384384inputbytes.
The independent existing flat loop-only control765872 (+0.23% versus original),
so new layout is−6.16% versus that stronger control. IncreasingK/compute/store
geometry stays fixed; source instructions rise140448→214084 while touched static
instructionbytes shrink69602→13992. Neither count alone explains cycle gains.
Normal52-binding/native/strict/sourcecontrol qualification is in progress, with
111 focused source-cell/tail/resource/default-identity tests PASS. No hardware
release, whole speed claim or original ZIPnumeric-contract substitution.

## 2026-10-06 11:39 UTC — completed stock group and stronger section controls

Stock1968 is DONE/rc0 at7,075,784,369 provider ROI cycles. All786432original
compiled54-op consumer i8words+1024BF16scales, guards, descriptors, inputbytes
and eight statistics pass; exactELF5527a93a...d0ca6/UART/stagedELF/stockbitstream
are pinned. Job elapsed520.10s includes setup/execution; it is not the slow
wholeSmol run. Raw WORKSPACE_GROUP_INSTRUCTIONS is a legacy label for mcycle;
on stock hardware these are CYCLES.1944(10.525B) and1945(8.026B) have different
recipes and are not isolated word-policy controls. WholeSmol best remains1906
258.6Bcycles; no48×extrapolation, whole5B claim or route promotion.
[Terminal](perf_records/stock1968_word_consumer_terminal.json).

Root318 exact capsule/debug/stock1968 evidence pins reclosed. New ResNet flat
planes imported528b5e2 preserving all previous compact/capacity/packet probe
choices, and106 source-cell/packet/retained-loop/default/export tests PASS.
Agent normal52bundle changes only47/50 kernels; all52adapters/sourcebindings,
other50objects and rewrittenIR are unchanged. Full native/strict1000 gate and
frozen1903 control are still pending. No flat-plane stock release yet.
[Complete capsules and ablation](perf_records/resident_flat_planes_capsules.json).

Merlin debug_companion importede8efe0a6f;18actual compiler/symbolizer and negative
binary/source testsPASS. It admits identical allocated ELFbytes/addresses/flags
and normalized relocation symbol identities, refuses unsupported formats and
conserves everyPC including missing source metadata. RISC-V production4object
allocated sections/1213relocations and102finalallocated sections match, enabling
existing3338365845instructionPCprofile mapping with no new target execution.
Generic utility remains target-independent, tools/source/ISA/numerical gates
caller-owned. Tests moved to DSE per local performance AGENT.md; structure/docs
checksPASS. Sourceconstructor f32_interval_endpoint.h:17 costs215208045innermost
instructions, including soft subtraction/addition and separate PV endpoint
contexts. Nested source/callsite counts overlap and cannot be summed as disjoint
costs. These are instruction counts, not hardwarecycles.
[Actual source mapping](perf_records/composed_provider_debug_source_attribution.json).

PreparedSoftmaxIntervalDomain is the next default-off portable proof under
implementation. It must derive finite scaled-score/difference envelopes from
actual plan and source domain, establish a finite nonnegative polynomial upper
bound, and execute the original lane/tree/two-chunk schedule on that upper bound
withalpha=1 to prove every prefix. Runtime active endpoint order/finiteness and
alpha[0,1] are admitted once; all-masked, cutoff, signed-zero, NaN, underflow,
overflow and unsupported-source cases retain checked fallback. Only typed private
softmax consumers may elide checks; signedPV has a separate domain and remains
checked. No global unchecked interval constructor or golden-derived range.

Tiny separate exactly-two-multiply packet family retains current>=3 champion
eligibility/default. New56native/default/refusal testsPASS; actualsource16384F32
outputs and81920five-FRM/flags comparisons, plus independent3×7+tail1260rawspecial
cases close. Actual source GSIM complete-cost pair is still live; no fullmodel
composition or hardware release for this prospective policy.

## 2026-10-06 12:23 UTC — stock results and generic buffering repair

ResNet1974 resident weight issue packet2 completes at34,830,614stockcycles,
all1000originalwords exact, stagedELF/hardware/UART pins closed. This is3.5235%
below1903 and a marginal0.2135% below1930. It changes five resident leaves and
does not compose1930 paired readout. FullK capacity1971 completes35,623,567cycles
(1.3272% below1903) and remains above the best. Latest section attribution is
UNKNOWN: the conserved1903 profile cannot assign1974's remaining12.44M gap
against the permitted ZIP1876 reference22,387,449cycles.
[1974](perf_records/stock1974_resident_weight_terminal.json),
[1971](perf_records/stock1971_full_k_capacity_terminal.json).

Tiny1975 generic loop organization composed with1967 completes450,035,885cycles,
all256000original compiled words/Torch gate exact, staged stock hardware pins
closed. The0.2627% marginal single-run gain is not a variance estimate. Broadcast
arm1976 and flat-resident ResNet1980 remain separate queue arms. Root freshly
reclosed1196 owned capsule/terminal/whole pins, including complete945-pin paired
flat normal qualification plus249-pin original/independent capsule qualification.
Paired-flat changes47 paired producer and50 directi8 producer; actual1930 paired
routes are24/47, with24capacity-refused and unchanged. All52adapter/source/numeric
bindings and other50kernels remain unchanged; control1930ELF byte reproduced.
Root explicitly released one controlled4abec57c...150af0 stock arm to Recovery.
An initial1903-only recloser correctly refused this1930 control; the existing
explicit-control pointwise recloser passes. No correctness gate was weakened.
[1975](perf_records/stock1975_tiny_outline_multiply_terminal.json),
[paired whole](perf_records/resident_paired_stores_normal_whole_qualification.json),
[reclosure](perf_records/root_paired_flat_tiny1975_reclosure.json).

The separate exactly-two-multiply packet initially loses20.4861% complete-source
GSIM cycles despite exact outputs. Emitted code reveals a65,600-byte allocation,
65,536-byte final copy and small metadata allocations. A loop-carried memref
identity obstructs output forwarding. Merlin efbf3374b inserts ordinary upstream
canonicalization before buffer-results-to-out-params under explicit unique
pipeline anchors; upstream alias/ownership analysis retains authority. No pointer
no-alias promise or relaxed FP contract is added. With packet4, actual complete
sourceABBA means489282→232454.5GSIMcycles (52.4907% lower), zero allocation/copy,
all16384source values/81920five-FRM checks and1260independent special/tail values
pass. This is a section result. Full normal1967-controlled qualification closes
all256000native/strict words/Torch gate and155bindings; hardware release awaits
immutable archive/root review. Both the losing and repaired candidates are saved.
Root56packet integration tests PASS with explicit real upstream tool paths;
the initial inherited dummy-path failures remain observed environment failures.
[Negative](perf_records/tiny_two_multiplications_negative_journey.json),
[repair](perf_records/tiny_two_products_direct_destination_journey.json).

Smol private prepared softmax intervals are admitted in Merlin a718557b1 using
source-derived finite score/difference/polynomial/prefix bounds and original
runtime endpoint/alpha checks. SignedPV remains checked. Original consumer,
48nativegroups/1600words and default provider/ELF byte identity pass. Complete
group2,879,340,410→2,818,251,584retiredinstructions (2.1216% lower). A separate
finite floor capability in Merlin7ba9df600 preserves signedzero and original
library nonfinite calls, explicitly requires unobserved interposition/errno/FP
flags.74native cases and470600target checks across five rounding modes pass;
complete group drops2,879,340,410→2,804,104,382instructions (2.6130% lower), with
the unchanged compiled54-op consumer786432i8words/1024BF16scales exact, same eight
statistics/480devicecalls/86507520logicalreadbackbytes. Fresh48native/1600original
outputs pass; old unobserved carrier differences are retained, no rebaseline.
Fresh floor+private-soft composition is being checked, not inferred additively.
Hardware cycles for these changes remain UNKNOWN.
[Private intervals](perf_records/prepared_softmax_interval_qualification.json),
[floor](perf_records/smol_builtin_floor_qualification.json),
[355-pin reclosure](perf_records/root_soft_floor_stock1971_1974_reclosure.json).

The apparently hot BF16 trunc helper is outside the measured provider ROI:
all236079104instructions/15738880calls belong to post-ROI original consumer
validation. Actualprovider.o has zero undefined trunc calls. Whole-program
callee totals cannot price provider opportunities. Root unchanged1930 census
separately conserves9335007whole-program instructions and8755275forward metric;
active_mlir_ciface_forward has1118490instructions, deadpublicforward has zero.
SHAupdate421452 occurs afterforward; sharedmemcpy300352 is not wholly forward.
22030unknown instructions remain visible; current1974 section cycles UNKNOWN.
[ROI scope correction](perf_records/bf16_helper_scope_census.json),
[unchanged ResNet census](perf_records/root_resnet1930_pc_census.json).

Phase1/2 should expose ordered pipeline anchors, typed buffer identity/ownership,
and emitted allocation/copy evidence as generic editable/compiler cost surfaces.
Compiler capability contracts should carry errno/interposition/nonfinite/signedzero
and rounding obligations. Scope-labelled PC evidence must separate production ROI
from validation and preserve unknowns. Actual CPU issue order/dependencies require
emission evidence and matched timing: generic two-FMA math belongs in Merlin;
literal CPUISA/ABI hooks belong in OOT. Search must measure fresh compositions,
retain negative interactions and exact source-compatible controls. No workload
name or golden hash becomes a compiler profitability predicate.

Owned8-thread ledger12:23:14UTC:46024771uncachedinput,2190221440cache-read,
9068021output,3637175reasoning (subsetoutput),2245314232rawinput+output or
55092792uncached+output. Since11:19:48UTC,1504878uncachedinput+312472output
(1817350),78106496cache-read and135452reasoningsubset. ExactOOT-only/per-change
allocation remains UNAVAILABLE, no proration. Separate activegoal tracker51453684.
[Ledger](perf_records/golden_token_usage_20261006T122314Z.json).

At12:27UTC, the complete normal two-product/bufferidentity Tiny arm independently
recloses107pins,256000originalnative/strict words, originalTorchgate and final
all-executable noFSM audit. Strict135853970vs1967's142343799instructions is4.5593%
lower, not a stock estimate. Root releases one stock arm b52cb58c...1fb1f against
1967, without outline/broadcast composition. Source-compatible outline addition
is the next independently qualified composition. Paired-flat ResNet is admitted
once as1982,1976 runs and1980 queues; Recovery is the sole queue owner.
[Tiny whole](perf_records/tiny_two_products_direct_destination_whole_qualification.json),
[root closure](perf_records/root_tiny_two_product_whole_reclosure.json).

Root fresh floor+private-soft composition completes the original12head group and
full48native1600originalwords. All786432i8observations/1024scales, guards, inputbytes,
eightstatistics and480devicecalls/86507520logicalreadbackbytes remain unchanged.
Control prepared-soft provider.o/ELF reproduce db4e5b39...4ff77 exactly; candidate
643ccd32...03f8c lowers completeROIinstructions2818251584→2739859187 (2.782%
below prepared-soft alone),4.846% below2879340410composednumericcontrol.442pins
include frozen numerical dependencies and all fresh native/target artifacts.
These are observed composition counts, not added isolated percentages. Native
firstcommandsentry remains inherited model-object compile evidence; actual fresh
compile/link commands and reused object hashes are disclosed. Hardware cycles
remain UNKNOWN. A matched stockgroup comparison is being prepared, not a slow
wholeSmol release or5Bforecast.
[Actual composition](perf_records/smol_private_soft_floor_composition_qualification.json).
