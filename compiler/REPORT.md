The compiler preserves the supplied seed and adds checked frontend movement/residual lowerings, connected schedules and resource-selected dense/convolution alternatives. Source axes, addresses, storage effects and completion edges survive to their consumers. Incumbent and transformed schedules use the same terminal encoder. Merlin still owns host tensor lowering, runtime and CPU arithmetic. Decisions use source types, geometry and selected hardware facts, never capsule or model names.

The selected roster comprises 109 core cases, 40 scalability cases, three connected public source programs and six target model programs. The current integrated functional evidence is:

| Scope | Demonstrated result | Evidence boundary |
|---|---|---|
| Core | All 109 case flags pass: 102 accelerator numerical executions, six native screens, one expected explicit requant decline | Separate physical/model gates remain |
| Scalability | 40/40 L2 pass | 31 complete modeled timings, five fast estimates only, four UNKNOWN timings |
| Connected public programs | 3/3 L0/L1/L2 pass | Physical qualification is separate |
| Small native source models | 6/6 required L0/L1/L2/L3 pass; 36/36 declared eligible units across 2,283 source operations | Candidate-only native placement/dispatch verification |
| Full-model kernel replay | 68/68 kernels build; 159/159 visible calls exact | Replay does not establish complete-root execution |

Exact immutable source digests and request identities are retained in `../artifacts/integration-functional-{core,scale,programs}.json`, `../artifacts/physical/final-repairs/summary.json` and `../artifacts/integration-model-{resnet50,tiny_llama,smolvla}.json`. Their snapshots differ; the final evidence summary reports these identities separately. The later RHS-residency promotion has focused validation and an actual emitted-IR impact census, rather than an inferred broad pass.

| Target program | Kernels built | Visible calls exact | Contractions device/declared |
|---|---:|---:|---:|
| ResNet50/model | 24/24 | 45/45 | 54/54 |
| TinyLlama/prefill | 8/8 | 22/22 | 3675/3675 |
| TinyLlama/decode | 8/8 | 22/22 | 3675/3675 |
| SmolVLA/prefix_encode | 14/14 | 36/36 | 3259/3259 |
| SmolVLA/flow_denoise | 14/14 | 34/34 | 2036/2036 |
| SmolVLA/action_decode | 0/0 | No device replay | Empty contraction population |

The fresh promoted-RHS replay source is `734259a3cdf6e4fe93f5af9e63135331a53657a5ee7f9b9a3598070fcbb5b4f1` (`../artifacts/dense-rhs-model-{resnet50,tiny_llama,smolvla}.json`). Action decode's empty contraction population supplies no numerical proof. Source-group identities, layouts, numerical boundaries, invocation counts and work rankings appear in `../artifacts/model-coverage-map.json`. All-operation coverage remains UNKNOWN where a complete native source-demand report is absent.

Fresh background full-root evidence demonstrates ResNet50 linking and execution: all 1,000 declared outputs exactly match the quantized reference, with 314,678,713 reported complete-root cycles. The record binds compiler source `208a92f9ede272a6bff316401c676cfc7a2a5578763bd5246d6bdffde5dd77eb` and newly linked ELF `c0ab580a86e529d2910bffbf015918accb7ae7fd470361f68ed99e04bd225f37`; it is not inferred from replay. Its cycle field does not expose a qualified timing-provider binding, so no matched-provider whole-model speedup is inferred. The five other target programs remain unverified, and all-operation coverage remains UNKNOWN (`../artifacts/model-gate-latest-frozen.json`). The earlier 314,893,192-cycle root record is retained separately in `model-gate-current-1400.json`. No whole-model convergence is claimed.

The earlier required-original campaign records 100 barrier passes and meets the expected decline. Standard operator cases carry numerical execution evidence; the six native source entries carry the candidate-only verification described above. Qualification is 98/101: three identity-scale edge sources remain structurally unqualified because their source requires `acc_scale=1.0` while the trace predicate requires a nonidentity CONFIG_ST. The compiler preserves the source scale and adds no dummy commands. Exact diagnostics are in `../artifacts/identity-scale-qualification-gap.json` and `../artifacts/physical/final-summary.json`. The newer functional integration preserves all 94 comparable published original LLVM products byte-for-byte (`../artifacts/physical/functional/ir-comparison.json`); that observation alone does not assert cross-engine ELF certificate reuse. The changed RHS-residency original receives a separate numerical and gSIM check. Required physical correctness does not calibrate performance.

Promoted transformations consume actual checked products in emission:

| Complete region and transformation | Before → after modeled cycles | Change |
|---|---:|---:|
| Residual M1K5728N2048: capacity-selected segmented panel | 7,719,487 → 3,546,828 | −54.05% |
| Connected sibling feed: capacity selection | 14,054,840 → 9,983,842 | −28.97% |
| TL down M16K5632N2048: capacity selection | 7,617,813 → 3,512,700 | −53.89% |
| Convolution C512H7: invariant coordinates | 3,821,295 → 2,334,049 | −38.92% |
| Connected sibling feed: frontier issue order | 9,983,842 → 8,091,586 | −18.95% |
| Residual / TL down: frontier issue order | 3,546,828 → 2,874,989 / 3,512,700 → 2,856,759 | −18.94% / −18.67% |
| Multihead: shared-RHS stacking | 334,547 → 320,601 | −4.17% |
| QK / PV: shared-RHS stacking | 27,815 → 16,056 / 29,741 → 17,818 | −42.28% / −40.09% |
| Valid convolution CI64, 56×56: source-proven gather | 5,287,836 → 5,126,583 | −3.05% |
| Valid convolution CI128, 28×28: held-out gather | 3,403,257 → 3,371,794 | −0.925% |
| Dense M3136K128N512: RHS residency | 1,630,512 → 1,587,261 | −2.65% |
| Dense M12544K64N256: RHS-residency development geometry | 2,162,256 → 2,071,981 | −4.18% |
| Dense M12544K256N64: RHS-residency development geometry | 2,121,425 → 1,821,169 | −14.15% |
| Dense M3136K512N128: fresh refined-rule validation | 1,615,362 → 1,322,001 | −18.16% |
| Connected KV heads: RHS residency | 320,601 → 304,773 | −4.94% |

These are matched complete provisional Spike timing-model estimates, not measured physical speedups. Model `b3ce9aaf7e8e2227e781f5d0bb3bd597f73ab12cd8c67aff64a5cd7f54347b11` remains untuned. Reports pin model, facts, comparison domain, scope, original and transformed products, emitted hashes, numerical evidence and keep/reject decisions. See `../artifacts/{dense-capacity-paired-model,dense-frontier-paired-model,connected-frontier-confirmation}.json`, `conv_gather_report.json`, `conv_valid_gather/report.json` and `dense-rhs-impact/guard_final_report.json`. The RHS selector requires strict total LHS+RHS byte reduction and at least two M and two N tiles per output panel. The square candidate fails the byte filter. Small one-dimensional panels restore the incumbent: a connected residual regression of 2,952,637→2,961,266 cycles motivated this guard, and guarded emission exactly restores the off product. Earlier validation shapes became development evidence after this refinement. The 3,136×512×128 geometry was reserved before reading its timing and validates the refined rule. The 109-case core census builds every declared frontdoor; only the separately checked beyond-capacity original changes from residency off.

Resource views keep their denominators explicit. The fresh RHS-resident GEMM reaches 60.73% of structural compute peak from 49.70%, with 100% padding efficiency and a 1.65× compute-floor gap (802,816 structural cycles versus1,322,001 complete modeled cycles). Requested reads fall 3,244,032→1,671,168B; modeled CPU issue reservations fall 559,906→459,117 cycles. These are model-derived diagnostics. C512H7 useful peak attainment rises 11.82%→19.35%, defined by useful MACs/(256 structural MAC/cycle × complete modeled elapsed). Padding efficiency remains 76.56%; compute-only gaps are 8.46×→5.17×. Modeled CPU issue reservations fall 1,800,642→633,387 cycles, while requested device bytes stay 4,400,640 read/25,088 write. The connected sibling path reaches 1.70% useful structural peak with 6.25% padding efficiency and a 3.68× logical-read roofline gap. These are structural ratios and modeled reservations, not measured engine occupancy. Activity intervals overlap and are never summed into elapsed time. Physical queue pressure, engine occupancy and scoped host wait remain UNKNOWN (`../artifacts/resource-views.json`, `dense-rhs-impact/resource_views.json`).

The complete-source host profile identifies reduction/softmax and elementwise work as the largest categories: 48.30% and 36.43% of 31,972,806 raw instrumented ticks. These counters are not elapsed-cycle percentages. The public host owner does not consume the candidate's generic LLVM CSE hook; identical public ELF/cycles correctly reject that supposed optimization. Existing source/CPU contracts remain selected explicitly. No unavailable host arithmetic interface is replaced with a receipt-only adapter (`../artifacts/host-category-profile-summary.json`, `host_memory_cse/matched-collected.json`).

Functional additions cover source-exact i8/i32 views, raw IEEE f32 transport, rank-two through rank-four bijective permutations, live destination observers, dependent sequences, multiple outputs and exact quantized residual edges. Matchers reject arithmetic permutation bodies, projected maps, dynamic shapes and invalid rounding equivalences. Tests include source-bound native builds, signed-extreme/IEEE bit-pattern ISA execution and current native required-tier checks. The portable pinned compiler-only Z3 package allows proof construction in relocated broker candidates. Source hashes and limitations are retained under `../artifacts/{integer_movement,float_movement,float_generic_permutation,source_lowering}/` and `functional-activation.json`.

Optional alternatives remain available through `TUNING_KNOBS.json`. Adaptive scratchpad partitioning wins one PR06 comparison by 11.21% but lacks a distinct affected timing holdout and stays off. Source peeling, disjoint readout, interior specialization and bounded channel unrolling stay off because gains are negligible, unsupported, or mixed. Factor-two unrolling improves valid convolution by 1.24% and its holdout by only 0.105%; padded cases regress. K=32 loses complete latency despite fewer fences. Ragged stacking wins M1 and loses M13, so it remains off without a distinct M1 holdout. Array-weight persistence and wide-load coalescing remain refused where selected execution laws are missing. Missing timing and resource attribution remain UNKNOWN, never zero cost or numerical failure.

The final functional repairs preserve the original transpose proof assertions while retrying only timeout/canceled UNKNOWN results in a fresh solver context within a shared 20-second budget. SAT and persistent UNKNOWN still refuse. Original JC and representative serial/batched transpose products remain identical. The host adapter additionally hoists only bounded, constant-one, nonescaping copy descriptors from acyclic blocks to function entry. Original JC native preflight now accepts a 704-byte static frame, previously refused as a 496-byte dynamic frame. Two independent/dependent multioutput regressions build with 144/128-byte static frames. This changes storage placement, not Merlin CPU arithmetic or preflight rules (`../artifacts/transpose_proof_retry/summary.json`, `host_static_alloca/evidence.json`).

The optional `MIXED_PERMUTED_DENSE_PREPARATIONS` path absorbs uniquely consumed static transpose/reshape preparations into a device group, materializes fresh private buffers through checked movement, and passes their rank-two aliases to the ordinary dense schedule. Original public axes and both live JC outputs remain intact. Default-off callback buffers exactly match all 11 pre-edit JC callbacks; enabled grouping uses nine regions. Local source/DMA checks pass, but public whole-call execution/performance evidence is recorded separately. The actual enabled whole-source path builds with a 640-byte static frame and both public outputs. The public JC A/B nevertheless emits the identical ELF and identical 97,320,353 complete provisional modeled cycles in both arms: this public owner does not consume the grouping option. The option remains off (`../artifacts/permuted_dense_views/{integration_report,final_consumer_report,public_comparison}.json`).

`../artifacts/final-validation-summary.json`, `../OPTIMIZATION_RULES.json` and `../ITERATION_NOTES.md` record source-bound validation, regional rules, rejected alternatives and remaining work. Full-root execution beyond the demonstrated ResNet snapshot, all-operation placement, three trace qualification conflicts and declared resource-acceptance probes remain explicit limitations.

The final observed gate records SmolVLA’s three roots as unavailable (no build or first failing group exposed) and TinyLlama’s two roots pending. This is missing full-root evidence, not a numerical failure or zero cost (`../artifacts/model-gate-final-observed.json`).
