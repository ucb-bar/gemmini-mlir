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
| Resident/banked transfer policy — OOT | Complete-M A residency; shape/resource/command/traffic comparison | Stock1849→1853:42,269,808→40,479,548 whole forward cycles; original1,000 words exact | Qualified transfer arm; composed1874 is now best; [receipt](perf_records/resnet_transfer_command_policy_firesim.json) |
| Banked residual M prefetch — OOT | Disjoint resident residual panels and next loads | Stock1849→1854:42,269,808→40,981,079 whole cycles; original1,000 exact | Separate positive arm; [receipt](perf_records/resnet_residual_banked_m_firesim.json) |
| Transfer+residual composition — OOT | Compose the two qualified strategies with identical host/runtime | Stock1853→1874:40,479,548→39,201,279cycles; all1,000 original words exact and actual staging pinned | Selected ResNet best,3.158% below1853; old1861 remains historical unverified; [hardware](perf_records/resnet_transfer_residual_composed_firesim.json) |
| Complete-input convolution stripes — OOT | Full reduction weight residency and bounded output stripes | H56/C64:619,364→520,943GSIM; H28/C128:630,670→509,443GSIM; full outputs/guards exact | Stock1853→1878:40,479,548→39,754,283 whole cycles(1.792% lower); original1,000 exact, six source kernels changed and host/runtime unchanged. Composition stock1897 queued versus1874; [hardware](perf_records/resnet_resident_stripe_policy_firesim.json); [study](perf_records/resident_stripe_conv_gsim.json), [journey](perf_records/resident_stripe_optimization_journey.json) |
| Remaining-row B slots — OOT | Two explicit disjoint next-K slots above complete resident A; interval and bank legality | M196/N512/K1024:540,046→511,096GSIM; all100,352 outputs+guards; independent i32tails pass | Seven general source selections; all1,000 whole native+Spike words exact; stock1888 queued; [study](perf_records/remaining_b_slots_prefetch_gsim.json), [journey](perf_records/remaining_b_slots_optimization_journey.json) |
| Exact bounded RNE packets — Merlin; target binding OOT | Independent scalar packet schedule, complete tensor proof, padding/tail/live-output preservation | Stock1864 warm quantization/packing traversal:3,042,623→1,749,755cycles; all150,528 inputs/158,700 destination bytes+guards | Capsule gain42.49%; corrected whole stock1886 queued after original1,000 exact; [capsule](perf_records/bounded_rne_packet_cpu_firesim.json), [whole gate](perf_records/resnet_quant_packet_spike.json) |
| Tiny fresh writer ownership+device prefetch — Merlin/OOT | Proved complete destination writes plus existing device B prefetch | Stock1841→1846:571,097,507→569,151,067 whole cycles; original256,000 words exact and Torch gate passes | Selected Tiny best; single-run small gain, no variance claim; [receipt](perf_records/tiny_expanded_writer_prefetch_firesim1846.json) |
| Exact scalar pointwise packets — Merlin | Two/four independent SSA lanes; original operation order and exact tails | Tiny2049-output capsule:328,373→255,845GSIM(two),268,247(four); outputs/guards exact | Two selected; whole stock1880 queued with unchanged1846 device/runtime; [capsule](perf_records/tiny_pointwise_packet_gsim.json), [whole gate](perf_records/tiny_pointwise_packet_spike.json) |
| Ordered FMA replay outputs — Merlin | Independent outputs retain zero seed and increasing-K source FMA | QK64 stock1856→1857:1,137,275→910,407cycles; all1,024 original words | Capsule19.95% lower; [receipt](perf_records/attention_qk64_replay_firesim.json) |
| Selected BF16 pre-widening — Merlin | Hoist LHS conversion including allocation/copy, retain ordered replay | QK4 stock1857→1872:910,407→828,986cycles; all1,024 original words. PV4 stock1873:168,142cycles, all64words | QK gain8.94%; QK16 stock1883 queued. Strict PV4 control1875 verifies180,776,6.989% paired gain; [PV control](perf_records/ordered_attention_pv_control_firesim.json); [QK](perf_records/attention_qk64_lhs_widen_firesim.json), [PV](perf_records/attention_pv192_lhs_widen_firesim.json) |
| Upstream typed ordered-FMA schedule — Merlin | Structural maps/wiring/positive-zero proof;384 BF16 contractions, tile8 and both operand widening | Both unoutlined and outlined LLVM produce all1,600 original Smol bits exactly in native; normalO2RV64GC/noFSM build completed | Actual full target replay56792b45…dad8 fails89/1,600 with byte-identical old target outputs;134.731B versus327.507B retiredinstructions, no hardware-cycle claim; [receipt](perf_records/smol_upstream_ordered_fma_native.json) |
| BF16 runtime calling convention — Merlin | Build lowered scalar helpers with matching LLVM compiler ABI |17 minimal actual target edge tests pass; older ABI-corrected whole replay still fails89/1,600, maxabs.115166 | Necessary runtime correction, **whole accuracy unresolved**; [failed whole target](perf_records/smol_corrected_runtime_spike_failed.json) |
| Whole-model declared-call preservation — Merlin; catalog closure OOT | Explicit typed external permissions; source-order graph nodes, effect-only calls and declaration ABI/attributes retained | Generic13 new plus47 regression checks PASS; 3,434-node outlined identity cover and actual catalog/final ELF closure; original prepared bytes retained | Identity admission gate only, selected_plan_controls_emission=False; shared search/timing remain unresolved; [binding](perf_records/golden_model_plan_binding_qualification.json) |
| Smol suffix diagnosis — experiment only | Derive original source suffix, supply captured post-vision state, freeze original failing runtime objects | Actual target all1,600 final bits exact against original golden; complete full models still fail89/1,600 | Localizes residual discrepancy to vision prefix; first-block target check running; [receipt](perf_records/smol_postvision_suffix_frozen_runtime.json) |
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
These are measured cost locations; they do not causally partition the remaining1874 gap.

## Token accounting

[Measured snapshot at2026-10-05T21:26:19Z](perf_records/golden_token_usage_20261005T212619Z.json)
contains per-thread counters for the eight explicitly owned root/worker/descendant sessions.
Ownership is supplied by the root spawn mapping; copied session metadata IDs are not used.

| Bucket | Cumulative measured campaign traffic |
| --- | ---: |
| Uncached input |21,333,364|
| Cached input reads |1,103,313,024|
| Output |4,074,868|
| Reasoning output, already included in output |1,526,318|
| Raw input+output total, including cached reads once |1,128,721,256|

The large raw total includes repeatedly read cached context. Uncached input plus output totals
25,408,232; it does not count cache reads or reasoning twice. Tokens do not establish dollar spend.
The separately observed goal counter is21,765,307; its accounting semantics are not exposed, so it
is retained rather than silently equated to raw session totals.

The latest snapshot records completed-request deltas since20:38:06Z:
1,578,169 uncached input,273,001 output and51,554,432 cached reads.
The [earlier snapshot](perf_records/golden_token_usage_20261005T203806Z.json) retains the initial window. Threads mix OOT,
shared compiler, numerical debugging and orchestration. Exact OOT-only or per-optimization token
allocation is **unavailable**, not zero. Future experiment starts, finishes and task transitions
should carry counter checkpoints and request-span attribution; elapsed-time prorating is not proof.
Earlier Claude traffic is not reconstructed from this Codex session snapshot.

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
