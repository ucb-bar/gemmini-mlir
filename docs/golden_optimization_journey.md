# Golden compiler optimization journey

This is the index of hypotheses, compiler changes, measured outcomes and rejected experiments.
The [chronological log](golden_optimization_log.md) retains earlier work and
[current progress](golden_progress.md) lists qualified whole-model champions.
Target instruction/device schedules belong in OOT. Shared host/numeric/global/build/runtime
mechanisms belong in Merlin. Production choices derive from semantics, shape, layout, numerical
contracts and resources; source IDs identify bindings and experiments only.

## Measured changes and pending compositions

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
| Explicit resident-A/B-prefetch alternative — OOT; normal equal-shape implementation dispatch — Merlin | Shape/resource-legal overlap alternative enters shared measured selector and actual normal model catalog; same tensor shapes retain distinct selected implementations | Common-address synthetic int8/amplitude21 GSIM111,885→76,456cycles(31.67%); native/strict single-selected full model all256,000 original words/Torch gate/noFSM pass | Expanded44-selected native and strict-target gates pass with155calls/five bodies. Stock1911 normal and1912 all1880host/runtime-controlled arms are queued; hardware outcomes remain unknown. A single fixture price reused across equivalent physical code is a calibration assumption, not measured model operands/address/cache/full-program cost; [capsule](perf_records/tiny_resident_a_prefetch_gsim.json), [single-selected full gate](perf_records/tiny_resident_a_prefetch_whole_spike.json), [journey](tiny-calibrated-resident-a-20261005.md) |
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
has13,715,255 cycles to remove. Its current section attribution is pending.
Reference other/uncounted51,717 is time outside printed layer timers, not total
CPU time. [Current gap and measured historical locations](resnet_current_gap_status.md).

## Token accounting

[Measured snapshot at2026-10-05T23:58:49Z](perf_records/golden_token_usage_20261005T235849Z.json)
contains counters for the eight explicitly owned root/worker/descendant sessions.
Ownership is supplied by the root spawn mapping; copied session metadata IDs are not used.

| Bucket | Cumulative measured campaign traffic |
| --- | ---: |
| Uncached input |25,653,177|
| Cached input reads |1,280,198,656|
| Output |4,931,343|
| Reasoning output, already included in output |1,884,279|
| Raw input+output total, including cached reads once |1,310,783,176|

The raw total includes repeatedly read cached context. Uncached input plus output totals
30,584,520; it does not count cache reads or reasoning twice. Tokens do not establish dollar spend.
The separately observed goal counter is26,960,347; its accounting semantics are not exposed, so it
is retained rather than silently equated to raw session totals.

The latest snapshot records completed-request deltas since23:11:24Z:
1,511,881 uncached input,288,924 output and60,884,864 cached reads.
The [prior snapshot](perf_records/golden_token_usage_20261005T231124Z.json) retains the earlier window.
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
