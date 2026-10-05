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
| Resident/banked transfer policy — OOT | Complete-M A residency; shape/resource/command/traffic comparison | Stock1849→1853:42,269,808→40,479,548 whole forward cycles; original1,000 words exact | Qualified transfer arm; later1886 host packets are now best; [receipt](perf_records/resnet_transfer_command_policy_firesim.json) |
| Banked residual M prefetch — OOT | Disjoint resident residual panels and next loads | Stock1849→1854:42,269,808→40,981,079 whole cycles; original1,000 exact | Separate positive arm; [receipt](perf_records/resnet_residual_banked_m_firesim.json) |
| Transfer+residual composition — OOT | Compose the two qualified strategies with identical host/runtime | Stock1853→1874:40,479,548→39,201,279cycles; all1,000 original words exact and actual staging pinned | Qualified target composition,3.158% below1853; newer1886 host-packet arm is best; old1861 remains historical unverified; [hardware](perf_records/resnet_transfer_residual_composed_firesim.json) |
| Complete-input convolution stripes — OOT | Full reduction weight residency and bounded output stripes | H56/C64:619,364→520,943GSIM; H28/C128:630,670→509,443GSIM; full outputs/guards exact | Stock1853→1878:40,479,548→39,754,283 whole cycles(1.792% lower); original1,000 exact, six source kernels changed and host/runtime unchanged. Composition stock1897 queued versus1874; [hardware](perf_records/resnet_resident_stripe_policy_firesim.json); [study](perf_records/resident_stripe_conv_gsim.json), [journey](perf_records/resident_stripe_optimization_journey.json) |
| Remaining-row B slots — OOT | Two explicit disjoint next-K slots above complete resident A; interval and bank legality | M196/N512/K1024:540,046→511,096GSIM; all100,352 outputs+guards; independent i32tails pass | Seven general source selections; all1,000 whole native+Spike words exact; stock1888 verifies40,148,896cycles,0.817% below1853 but slower than1886 best; composition unknown; [study](perf_records/remaining_b_slots_prefetch_gsim.json), [journey](perf_records/remaining_b_slots_optimization_journey.json) |
| Exact bounded RNE packets — Merlin; target binding OOT | Independent scalar packet schedule, complete tensor proof, padding/tail/live-output preservation | Stock1864 warm quantization/packing traversal:3,042,623→1,749,755cycles; all150,528 inputs/158,700 destination bytes+guards | Capsule gain42.49%; Stock1853→1886:40,479,548→38,603,949wholecycles(4.6334% lower); all1,000 original bits/staging closed, new ResNet best; [hardware](perf_records/firesim1886_resnet_quant_packet_verified.json); [capsule](perf_records/bounded_rne_packet_cpu_firesim.json), [whole gate](perf_records/resnet_quant_packet_spike.json) |
| Tiny fresh writer ownership+device prefetch — Merlin/OOT | Proved complete destination writes plus existing device B prefetch | Stock1841→1846:571,097,507→569,151,067 whole cycles; original256,000 words exact and Torch gate passes | Qualified earlier Tiny control; single-run small gain, no variance claim; [receipt](perf_records/tiny_expanded_writer_prefetch_firesim1846.json) |
| Exact scalar pointwise packets — Merlin | Two/four independent SSA lanes; original operation order and exact tails | Tiny2049-output capsule:328,373→255,845GSIM(two),268,247(four); outputs/guards exact | Stock1846→1880:569,151,067→531,072,370cycles(6.6904% lower), all256,000 original words/Torch gate exact, staged ELF/bitstream closed; new Tiny best; [hardware](perf_records/tiny_pointwise_packet_firesim.json); [capsule](perf_records/tiny_pointwise_packet_gsim.json), [whole gate](perf_records/tiny_pointwise_packet_spike.json) |
| Ordered FMA replay outputs — Merlin | Independent outputs retain zero seed and increasing-K source FMA | QK64 stock1856→1857:1,137,275→910,407cycles; all1,024 original words | Capsule19.95% lower; [receipt](perf_records/attention_qk64_replay_firesim.json) |
| Selected BF16 pre-widening — Merlin | Hoist LHS conversion including allocation/copy, retain ordered replay | QK4 stock1857→1872:910,407→828,986cycles; all1,024 original words. PV4 stock1873:168,142cycles, all64words | QK gain8.94%; stock QK16 now814,343cycles(1.766% below QK4), all1,024 exact; [QK16](perf_records/ordered_attention_qk_lhs16_firesim.json). Strict PV4 control1875 verifies180,776,6.989% paired gain; [PV control](perf_records/ordered_attention_pv_control_firesim.json); [QK](perf_records/attention_qk64_lhs_widen_firesim.json), [PV](perf_records/attention_pv192_lhs_widen_firesim.json) |
| Upstream typed ordered-FMA schedule — Merlin | Structural maps/wiring/positive-zero proof;384 BF16 contractions, tile8 and both operand widening | Both unoutlined and outlined LLVM produce all1,600 original Smol bits exactly in native; normalO2RV64GC/noFSM build completed | Actual full target replay56792b45…dad8 fails89/1,600 with byte-identical old target outputs;134.731B versus327.507B retiredinstructions, no hardware-cycle claim; [receipt](perf_records/smol_upstream_ordered_fma_native.json) |
| BF16 runtime calling convention — Merlin | Build lowered scalar helpers with matching LLVM compiler ABI |17 minimal actual target edge tests pass; older ABI-corrected whole replay still fails89/1,600, maxabs.115166 | Necessary runtime correction, **whole accuracy unresolved**; [failed whole target](perf_records/smol_corrected_runtime_spike_failed.json) |
| Whole-model declared-call preservation — Merlin; catalog closure OOT | Explicit typed external permissions; source-order graph nodes, effect-only calls and declaration ABI/attributes retained | Generic13 new plus47 regression checks PASS; 3,434-node outlined identity cover and actual catalog/final ELF closure; original prepared bytes retained | Identity admission gate only, selected_plan_controls_emission=False; shared search/timing remain unresolved; [binding](perf_records/golden_model_plan_binding_qualification.json) |
| Smol suffix diagnosis — experiment only | Derive original source suffix, supply captured post-vision state, freeze original failing runtime objects | Actual target all1,600 final bits exact against original golden; complete full models still fail89/1,600 | Vision prefix localized further: first4 isolated target blocks exact; fifth native-exact/target failure is causally restored by native expf oracle and optional float-exp-via-double; fulltarget gate pending; [causality](perf_records/smol_block4_expf_causality.json); next block/postLN exact; [localization](perf_records/smol_later_vision_block_localization.json); [receipt](perf_records/smol_postvision_suffix_frozen_runtime.json) |
| Shared calibrated selection — existing Merlin planner; OOT generation/emission | Prices exact source-bound alternatives from pinned full-fixture GSIM receipts; solver winner controls actual compiled object | Independent17×73×65:3,234→2,357GSIMcycles; all1,241outputs+2,048guards exact, strict target/noFSM | Normal model build now compiles actual selected object into its catalog, all1,241 original i32 outputs/finalELF closed; [normal-path qualification](perf_records/golden_calibrated_normal_model_qualification.json);27.1% fixture gain; complete measured ranking, physical floor/internal engine occupancy/whole-model costs UNKNOWN; [receipt](perf_records/golden_calibrated_source_selection_qualification.json) |
| Early saturation and scalar readout packets — Merlin; ABI binding OOT | Test exact source transitions before estimation, retain input-load/output-store order in bounded8lane packets | Full50,176/25,088 readouts:1,546,262→1,130,840 and756,032→560,759GSIMcycles; allvalues+guards exact; whole original1,000 words exact and unchanged1874 control link byte-identical | Whole retiredinstructions10,222,806→9,389,018; stock1900 admitted, hardware outcome pending; no capture selector or numeric relaxation; [qualification](perf_records/resnet_exact_readout_packets_qualification.json) |
| Qualified host/target/readout composition — Merlin/OOT |1886 host packets plus1874 banked residual and1900 saturation/readout packets; source/ABI/proofs exactly matched |1886 and1900 control links byte-identical; all1,000 original native/target words exact; unchanged host/runtime/weights objects pinned | Stock1903 versus1886 pending, no additive gain inference; [qualification](perf_records/resnet_qualified_packet_composition_spike.json) |
| Adjacent independent RNE results — Merlin | Explicit CPU packet with independent earlier inputs, complete typed lane proofs and distinct floating temporaries | Complete Tiny capsule256,637→247,568GSIMcycles(3.53%); same108,797instructions; full256,000 original target/native words exact | Stock1902 versus1880 pending; current1880 profile1901 has155 conserved boundaries; root42 numeric/compiler regression tests pass; [journey](tiny_adjacent_rne_packets.md) |
| Complete CCA artifact — Merlin | Preserve all10 facets, full fields, scope/provenance/unknowns; reject contradictory views | Complete roundtrip/refusal and legacy artifact tests pass; integration tests124PASS across CCA, accounting and bounded packets | Compiler analysis correctness; no measured inference-cycle change; implementation150f2af09 |

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

Guarded signed-digit attention now executes primitive device partials, certificate computation and
exact replay together for a complete original first-vision head on the target, with all65,536 output bits exact.
The factored signed-prefix bound retires1,483,561,398instructions versus a fair1,185,308,895 control;
a cheaper gamma bound improves to1,430,369,073 but still loses20.67% in that metric.
These candidates are not admitted as a performance winner. Paired stock FireSim1894/1895 is queued
to measure complete original head costs. The stronger bound also passes all786,432 original first-block
output bits across all12 heads in strict target execution at17,319,092,950 retired instructions.
[Archived target costs and qualification](../out/artifacts/probes/attention-prefix-guard-20261005/README.md)
retain every losing variant, actual kernel calls, input/ELF pins and source snapshots.
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

## Token accounting

[Measured snapshot at2026-10-05T22:15:56Z](perf_records/golden_token_usage_20261005T221556Z.json)
contains counters for the eight explicitly owned root/worker/descendant sessions.
Ownership is supplied by the root spawn mapping; copied session metadata IDs are not used.

| Bucket | Cumulative measured campaign traffic |
| --- | ---: |
| Uncached input |22,760,989|
| Cached input reads |1,160,923,904|
| Output |4,349,897|
| Reasoning output, already included in output |1,629,678|
| Raw input+output total, including cached reads once |1,188,034,790|

The raw total includes repeatedly read cached context. Uncached input plus output totals
27,110,886; it does not count cache reads or reasoning twice. Tokens do not establish dollar spend.
The separately observed goal counter is23,489,700; its accounting semantics are not exposed, so it
is retained rather than silently equated to raw session totals.

The latest snapshot records completed-request deltas since21:26:19Z:
1,427,625 uncached input,275,029 output and57,610,880 cached reads.
The [prior snapshot](perf_records/golden_token_usage_20261005T212619Z.json) retains the earlier window.
Threads mix OOT, shared compiler, numerical debugging and orchestration. Exact OOT-only or
per-optimization token allocation is **unavailable**, not zero. Experiment starts, finishes and
transitions should carry counter checkpoints and request-span attribution; elapsed-time prorating
is not proof. Earlier Claude traffic is not reconstructed from this Codex snapshot.

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
