# Fresh10 compiler continuation

This run continues the supplied seed in one editable compiler. Source semantics,
geometry, layout, numerical proofs and selected RTL capacities drive its choices.
There is no capsule/model-name schedule lookup or CPU contraction fallback.
The previous run's report remains outside this compiler in `SEED_NOTES/REPORT.md`;
its certificates and timing are not claimed as this run's evidence.

## Implemented behavior

The frontend retains original axes, initialized reductions, exact integer tensor
views and live outputs through source grouping. It now lowers exact rank-one to
rank-six integer concatenations with disjoint DMA output coverage, bounded positive
stride integer extract/insert slices, runtime column bias, and source-proved
weighted residual quantizers. Each residual proposal must match all 65,536 i8
input pairs under the original ordered binary32 operations; its actual emitted
DMA packing and shared integer product implement that proof. Unsupported or
unproved floating/nonlinear/control seams retain Merlin's host tensor lowering,
runtime and arithmetic. Contiguous slice-to-dense composition is available as an
unselected alternative with real source/LLVM products.

Dense lowering now has registered shape/resource alternatives, adaptive full-LHS
and double RHS storage, bounded segmented K frontiers and two issue orders.
Selected short-row frontiers require M<=16 and ceil(K/16),ceil(N/16)>=128;
within that legal class N/K<=3 selects column prefetch. Larger ratios retain K
prefetch. Within that existing K-prefetch region, M>1 selects a deeper RHS panel
when independently legal; M1 and column-prefetch retain their measured incumbent.
The deeper region has one positive complete-cycle observation and remains a
provisional tuning hypothesis. Other geometries keep the existing retained/streaming/compact
paths. Legality and profitability remain distinct. Exact finite interpreters
check actual source-row identities, reduction prefixes, initialized accumulators,
unique output cover and ownership/completion edges; unsupported finite domains
retain the original symbolic checker/refusal. No numeric tensor oracle or sampled
iteration certifies a schedule.

Retained full-K dense panels now share readout epochs when exact storage effects,
uniform CONFIG_ST words and selected RTL ordering facts admit the transformation.
The checked product reaches the common encoder; closing completion fences remain
before RHS or accumulator reuse. Unsupported effect patterns keep the incumbent.

Convolution adds retained 3x3 row rings, direct nonoverlapping patch panels and
positive-stride 1x1 resident panels. Selected rolling-window readout epochs batch
disjoint completed-K accumulator panels only when the exact CONFIG_ST words
match; mixed scales retain incumbent completion. Original NHWC coordinates, source K order,
matrix seeds, bias, per-column rounding and outputs survive the common terminal
encoder. Attention shares a literal RHS or packs short independent rows/GQA heads
only when exact source and workspace predicates admit it. Their terminal readout now maps joined rows directly into the original disjoint outputs, preserving exact accumulator row displacement, configuration and completion; unsupported readout effects retain the joined-buffer path. Pooling uses bounded
rectangular ACC windows with the original scale/RNE/activation/clamp order.
Residual DMA epochs batch disjoint ACC windows with complete seed/add/store
phase drains. Hardware-loop/sequencer instructions are not used.

Host alternatives use the existing Merlin source/CPU contracts, including scalar
fusion and alias-proved bulk row copies. Their realized LLVM recipes and exact
native arithmetic checks are recorded; `HOST_OPTIMIZATIONS=baseline` remains
selected because the native model owner exposes no candidate-host-code selection
hook and local code-size reductions do not establish complete-call speedups.

## Demonstrated performance

These are complete-call **modeled, provisional Spike timing** comparisons with
matched provider/model/facts/input/scope identities and exact numerical checks.
They are not physical speedups or whole-model latency claims.

| Representative geometry | Incumbent cycles | Selected cycles | Evidence |
|---|---:|---:|---|
| Dense M16 K5632 N2048 | 7,617,813 | 2,820,766 | retained → segmented/column frontier |
| Dense M1 K2048 N2048 | 1,307,762 | 1,060,744 | retained → segmented/column frontier |
| Fresh dense M16 K2048 N2048 | 1,322,691 | 1,076,033 | frozen aspect rule holdout |
| Dense M1 K2048 N32000 | 19,673,813 | 18,458,713 | K frontier retained at high aspect |
| Connected decode sibling feed | 14,054,840 | 7,803,585 | complete connected region |
| Dense M16 K2048 N11264 | 6,598,830 | 5,997,143 | deeper K-prefetch frontier; provisional region |
| Dense M196 K256 N1024 | 495,502 | 487,051 | retained panel readout epochs |
| Dense M784 K256 N1024 | 1,411,302 | 1,384,931 | retained panel readout epochs |
| Fresh dense M3136 K128 N512 | 1,630,512 | 1,585,481 | frozen epoch-rule holdout |
| 3x3 convolution C512 H7 | 3,821,295 | 2,239,690 | row ring plus readout epochs |
| 3x3 convolution C64 H56 | 4,929,904 | 4,674,733 | row-ring incumbent → readout epochs |
| Stride-two 1x1 per-column convolution | 3,387,485 | 3,095,384 | direct sampled source rows |
| Short-row GQA decode | 2,345,322 | 547,745 | shared RHS row packing |
| Short-row GQA decode, latest matched pair | 506,994 | 412,195 | direct original-output readout; 18.70% |
| Nonoverlapping 4x4 patch core | 4,699 | 1,078 | source patch panel |
| Resident 1x1 column-tail core | 2,156 | 2,016 | full-K resident panels |

The five convolution epoch comparisons improved complete modeled calls by
0.54%–5.18%, with exact numerics. Deeper dense panels regressed three
column-prefetch shapes by 0.40%–1.19%; the regional policy preserves their
incumbent LLVM rather than selecting those losses. The pooling issue variant has only a calibrated **fast_estimate**:
4,655,599→4,654,901 cycles, with a wide [2,266,937,5,911,724] band.
It is selected according to the requested fast-estimate rule, with substantial
ranking uncertainty. Missing complete timing stays UNKNOWN. Broad all-row RHS
packing, resident readout epochs and raw conv/pool fusion lost against their
incumbents and remain unselected. See [rule support](OPTIMIZATION_RULES.json)
and [trial identities](docs/evidence/fresh10-20261029/index.json).

For the final GQA readout pair, the complete-call structural roofline gap falls from 15.47x to 12.58x (32,768-cycle floor). Floor/estimate attainment rises from 6.46% to 7.95%, with 50% useful/padded work. Device-requested reads fall from 1,454,080 to 1,183,744 bytes and writes from 608,256 to 337,920 bytes; CPU fetch requests rise from 298,436 to 340,876 bytes. These are requested/model boundary bytes, not measured bandwidth. Baseline scoped interval capture is incomplete, so no engine-overlap improvement is inferred. Actor reservations overlap and remain distinct from engine occupancy.

Useful compute attainment, padding efficiency, scoped resource reservations,
traffic, queue pressure and lifetimes are kept in the trial feedback. For example,
the JM column frontier has 6.25% padding efficiency and modeled structural peak
attainment increasing 1.334%→1.545%; the fresh M16 holdout has 100% padding
efficiency and 24.362% modeled structural attainment. The denominator is source
useful MACs divided by structural peak MACs/cycle times modeled whole-call cycles.
These are not measured processing-element occupancy. Overlapping resource
intervals are never summed into elapsed latency. Host wait attribution and actual
engine occupancy remain UNKNOWN where the provider supplies none.

For the retained M784 panel trial, useful compute attainment increases from
56.88% to 57.97% of modeled structural peak over the complete call; the fresh
M3136 holdout increases from 49.24% to 50.64%. Those denominators include packing,
submission and synchronization in the modeled call. They are not measured engine
occupancy. All three paired epoch calls improve modeled elapsed cycles.

Descriptive same-request correctness observations provide 81 complete-model/gSIM
pairs: median absolute percentage error 13.63%, maximum 97.70%, and 13 cases
without complete modeled cycles. These small correctness workloads do not
validate ranking accuracy on large schedules. No fitted runtime or calibration
authorization is inferred from those observations.

## Correctness and deployment evidence

The selected public inventory is 109 core cases (101 originals plus eight
calibration), 40 scalability cases and three connected program cases, plus the
six model programs below. Public graph coverage maps include host regions and
source operation counts/MACs/static boundary bytes; these are not dynamic
invocation counts, measured traffic or complete-model timing.

The final dense-epoch milestone returned 40/40 scalability L2 numerical passes.
Its aggregate `complete`/`passed` flags remain false, and some tier records are
skipped; this does not become a full qualification claim. All three
connected program checks returned L0/L1/L2 passes. Twenty-seven of these 40 final scalability cases have complete modeled cycle estimates; the other thirteen retain explicit UNKNOWN complete timing with available fast estimates kept separately. Core Spike checks do not
certify an unrun L3 tier. The later integrated phase-four core check returned all 109 declared accepted
outcomes, including the one expected decline, with no numerical failures; its
aggregate all-pass remains false because physical and structural qualification are
separate. The later affected-core run passed 50/51 cases; the remaining micro-model build hit a native output-owner freshness error. Both the micro-model and its source coverage case passed L2 on a repeat of the exact same frozen candidate. The original failure and resolving repeat remain in the evidence index. Final changed-byte reconciliation is recorded separately.

The required original campaign returned 94 gSIM numerical passes, six native
candidate required-tier verifications and one expected decline. Ninety-eight
requests fully qualified; three numerically correct executions failed instruction
trace qualification because an identity-scale source was required to contain a
nonidentity CONFIG_ST. No artificial scale or dead instruction was introduced to
satisfy that predicate. The [physical ledger](docs/evidence/fresh10-20261029/ledger.json)
names each request, executable digest and exact structural outcome. The
[phase-three reconciliation](docs/evidence/fresh10-20261029/physical_phase3_ledger.json)
renews 18 changed physical programs and six native views, retaining 76 matching
LLVM/input products; cross-engine execution digests are not compared for equality.
This audit establishes emitted-product identity, not a new physical measurement. A certificate
is reused only when the executable bytes match; these counts describe that
campaign, not an automatic certificate for later compiler edits.

Phase-four renewal additionally passed both changed convolution kernels on gSIM
and both changed CNN/multimodal native source views. Its ledger finds no unmatched
LLVM/input products against that development milestone. Later retained-dense
epoch changes receive their own reconciliation rather than inheriting this claim.

The final phase-five reconciliation renews 34 changed gSIM kernels and all six native source views. It reports 94 original gSIM numerical passes, six native required-tier passes, one expected decline, the same three trace conflicts, and zero unmatched emitted LLVM/input products. The decode-step native request initially failed with a harness output-directory collision; its separate repeat passed. The [final ledger](docs/evidence/fresh10-20261029/physical_phase5_ledger.json) retains the winning completed requests and the failed request remains on disk. The final attention readout option changes no core source product, as established by the full source/CLI census.

Twenty-four changed correctness kernels also have fully matched, structurally qualified before/after gSIM cycle pairs. All improve, by 0.65%–22.67%; for example M12 K48 N48 is 1,051→957 measured complete-call cycles. These are small correctness-shape observations under the same simulator/input/facts/scope, not large-shape or whole-model speedups. Seven unmatched timing identities and three trace conflicts are excluded from performance comparisons. See [physical cycle pairs](docs/evidence/fresh10-20261029/physical_phase5_cycle_pairs.json).

| Model | Program | Compile / link | Visible kernel numerics | Contraction census |
|---|---|---|---|---|
| ResNet50 | model | 24 kernels built; whole-chain link UNKNOWN | 45/45 calls exact | 54/54 device |
| TinyLlama | prefill | 8 kernels built; whole-chain link UNKNOWN | 22/22 calls exact | 3675/3675 device |
| TinyLlama | decode | 8 kernels built; whole-chain link UNKNOWN | 22/22 calls exact | 3675/3675 device |
| SmolVLA | prefix_encode | 14 kernels built; whole-chain link UNKNOWN | 36/36 calls exact | 3259/3259 device |
| SmolVLA | flow_denoise | 14 kernels built; whole-chain link UNKNOWN | 34/34 calls exact | 2036/2036 device |
| SmolVLA | action_decode | no selected kernel; whole-chain link UNKNOWN | no kernel replay | 0/0; no offload claim |

The visible endpoint used one input and no shape variants. Its passing
`model_numerics` gate is based on kernel replay; full-chain execution/numerics,
recurrent-state deployment, sealed checks and whole-model timing are not inferred.
The contraction census does not establish all-operation accelerator coverage.
The resident pointwise integration was subsequently replayed successfully on all
24 ResNet kernels/45 calls. Exact source identities and full endpoint reports are
linked from the compact evidence files.

The final selected dense-epoch milestone was replayed again: all 68 built kernels
and 159 visible calls pass exactly across the six programs. The final endpoint
reports are included in the portable evidence index; their kernel-only scope and
one-input/no-shape-variant limitations remain the same.

An instrumented connected ResNet source profile attributed 31,972,806 raw counter
ticks primarily to its f32 pooling reduction and elementwise work. These are
instrumentation counter ticks, not elapsed timing-model cycles. The preparation
CSE trial merged two ResNet preparations but preserved the exact final ELF and
97,627,861 modeled cycles; TinyLlama complete timing remained UNKNOWN.

Two additional alternatives are implemented and disabled. Source window-product
composition consumes eight real convolution windows in full authored-CLI builds,
preserving source interfaces and exact outputs. The native program endpoint
consumed the same ELF in both arms and returned identical 2,975,295 modeled cycles;
its public source-composition interface has no custom window-product route.
A separate vocabulary output-lookahead pipeline is implemented in an isolated variant. Its exact source/storage and large ragged integer checks pass, but complete timing is UNKNOWN and the older fast estimate uses different calibration coefficients. It remains unselected; fewer commands are not a speedup.

The direct-readout admission census covers every one of the 149 public core/scalability sources; only the large GQA source changes. Eleven mixed-source on/off CLI pairs emit identical LLVM. Forced actual proof-budget refusals restore byte-identical incumbent LLVM and all 43 interpreted command packets/fences. The integrated GQA check reproduces the same 412,195 modeled cycles; the visible TinyLlama kernel replay also passes again.

Weighted packing epochs likewise reach actual emitted packing kernels in full
CNN and decode-step CLI builds, preserving all-pair quantizer proofs and dense
schedules, but have no matched elapsed timing. Neither receives a speedup claim.

Host cleanup also uses the installed bounded instcombine convergence option with
fixpoint verification. This fixes real compilation failures on three composed
sources; six paired source builds and native arithmetic checks pass. It is a
functional repair without a latency claim.

## Rank and dependency integration

Exact integer transpose and last-axis i32 bias addition now accept positive static
ranks 1–6. Structured source maps, axes and effects remain intact until the shared
terminal DMA lowering. Bias retains modular i32 arithmetic; transpose retains
signed i8/raw i32 values. Malformed domains and overflow promises refuse before
emission. Independent checks interpret 1,770 transpose programs and 13 generated
bias LLVM programs, including tails and signed overflow. Existing rank2–4
transpose LLVM and rank2 bias LLVM stay identical. Ten named public regressions
pass L2; fifteen public/connected authored CLI products also stay identical.
See [rank integration](docs/evidence/fresh10-20261029/rank6_integration.json).
New synthetic rank geometries have local exact-program checks, without an
additional physical execution claim.

The selected convolution input-dependency path changes the actual typed rolling
IR and consumes that transformed IR in emission. Selected hardware first-reader
RAW dependencies order primitive input transfers; real completion drains remain
before storage reuse. An exact reverse projection verifies original source
addresses, coverage and lifetimes. Matched complete provisional model pairs are
4,674,733→4,635,206, 3,095,006→3,083,024 and 817,432→817,364 cycles.
Device traffic is unchanged. The largest case's useful compute structural floor
divided by modeled whole-call cycles is38.64%→38.97%; it is not measured PE
occupancy. Per-resource reservations, overlap, queues, CPU work, traffic and
model/provider identities remain in the [comparison](docs/evidence/fresh10-20261029/conv_input_dependencies.json).
All three L2 numerical executions pass. Six native authored CLI products stay
identical. The combined ResNet replay builds24/24 kernels and passes45/45 visible
calls, with54/54 contractions placed on device. This is kernel replay on one
input, without complete-chain execution or elapsed-latency qualification.

The selected attention refinement specializes original member fragments only
when every actual terminal tile covers the proven full joined row range.
Its matched complete provisional model improves412,195→411,727 cycles (0.11%);
requested device traffic is unchanged. General tiles keep identical fallback
LLVM.128 paired actual terminal packet checks,80 independent output checks and
the named large GQA L2 execution pass. This small modeled gain is uncertain
physically. See [static fragments](docs/evidence/fresh10-20261029/attention_static_fragments.json).
Both dependency and static-fragment options have explicit off/on knobs and use
the incumbent terminal encoder. Physical qualification is recorded in the
[new reconciliation](docs/evidence/fresh10-20261029/rank6_dependencies_physical_ledger.json):
both changed convolution kernels pass gSIM and all six native source views pass
their required tiers. Across the retained original campaign there are94 gSIM
numerical passes, six native verifications, one expected decline, the same three
instruction-trace qualification conflicts, and zero unmatched non-native emitted
LLVM/input products. Artifact identity alone is not a new physical measurement.
Matched complete-call gSIM cycles on the two changed correctness shapes improve
5,515→5,456 (1.07%) and52,382→52,185 (0.38%); these are small-shape observations,
not measured large-case or model speedups. Exact simulator/input/facts/scope and
ELF identities are in the [physical pairs](docs/evidence/fresh10-20261029/rank6_dependencies_physical_pairs.json).
The combined GQA L2 check reproduces411,727 modeled cycles. Four actual CLI
pairs prove combined emission identical to the isolated winning alternatives.

## Future transport and finite operand proofs

The selected future-row convolution schedule improves three matched complete
provisional model pairs: 4,635,206→4,587,208, 3,083,024→3,008,063 and
817,364→761,327 cycles (1.04%, 2.43%, 6.86%). Their roofline gaps are 2.54×,
1.67× and 1.69×. Useful compute structural-floor attainment over modeled
whole-call cycles becomes 39.38%, 60.05% and 59.32%; padding efficiency stays
87.5%. Resource reservations and overlap remain modeled, while physical engine
occupancy and host wait attribution remain UNKNOWN. Full traffic, queue,
storage, CPU and provider views are in the
[comparison](docs/evidence/fresh10-20261029/conv_future_rows.json).

The actual transformed typed IR stages the next row after the last spatial
panel's complete-K products launch and before its readout. Spare row slots,
relocated immutable weights and selected cross-queue ordering facts establish
legality. Exact reverse projection retains original source addresses and real
completion before every reuse. A capacity guard refuses before effects.
Five independent source geometries, mutation checks and all three named L2
executions pass. All six native source CLI products remain identical. The two
changed original correctness kernels pass required gSIM; matched complete-call
cycles improve 5,456→5,432 and 52,185→51,718. These are small-shape measurements,
not measured large-case or whole-model gains. See the
[physical pairs](docs/evidence/fresh10-20261029/conv_future_physical_pairs.json).

Shared-LHS source checking now preserves actual pointer checks while projecting
instantiated commands into the existing finite dense verifier. Bounded exact
DMA interval checks remove an optional Z3 dependency from epoch construction.
Three affected original cases pass L2 and required gSIM. The single-panel
restriction now refuses from the same planner's actual blocks before expensive
group construction; four CLI products stay identical. Resident RHS/LHS rewrites
also use bounded source and address proofs, fixing a reproduced legal no-Z3
build failure. Their four public LLVM products match the incumbent, with 25
independent integer tests and 17 refusal tests. These are functional repairs
without a new performance claim. See the
[integration record](docs/evidence/fresh10-20261029/future_finite_integration.json).

The reconciled original campaign retains 94 gSIM numerical passes, six native
verifications and one expected decline, with zero unmatched non-native
LLVM/input products. Three historical instruction-trace conflicts remain.
Current ResNet kernel replay passes 24 kernels/45 calls. A fresh SmolVLA rebuild
passes 28 kernels/70 calls, including all 3,259 prefix contractions on device;
the earlier isolated build refusal does not reproduce. Kernel replay does not
establish complete-chain timing or native all-operator placement coverage.

## Commuted bias inputs

The frontend now derives activation and column-bias roles from their original
typed indexing maps and accepts either input order at ranks 1–6. It retains
the existing modular i32 arithmetic and DMA lowering. Twelve actual emitted-LLVM
numerical cases, fifteen no-Z3 CLI builds and seven malformed-source refusals
pass. Canonical rank1/2/6 products and all six native source CLI products remain
byte-identical. The named coverage-MLP and PF02 bias checks pass Spike L2;
the native required execution tier is not established by that development screen.
This is a functional extension without a performance claim. See the
[integration evidence](docs/evidence/fresh10-20261029/commuted_bias_integration.json).

The latest full-model verdict still reports a SmolVLA prefix build refusal
(13 built, one refused). A separate fresh kernel replay builds all 28 SmolVLA
kernels and passes 70 calls, but does not resolve that full-model gate. Its cause
remains unexposed by the public full-model diagnostic.

## Connected host alternative

A matched provisional complete model comparison rejects the existing round-even
host option: 97,627,861→97,649,530 cycles (+0.0222%). A separate bounded-RNE
source option actually transforms one proven clamp/rounding chain and changes
the native ELF; L0/L1/L2 pass. Its complete timing remains UNKNOWN because the
owner reports opaque completion effects in emitted assembly, and its fast model
has no decoded accelerator stream. It stays isolated and unselected. Native host
builder dependencies are outside the selected candidate inventory; the trial
uses the existing source/CPU option and unchanged Merlin implementation. See
[actual owner and timing evidence](docs/evidence/fresh10-20261029/connected_host_bounded_rne.json).

## Remaining limits

Complete selected-model latency, source-complete physical coverage beyond each
reported native scope, actual host/wait attribution and timing on cutoff neighbors
remain unobserved. Larger patch/pointwise source/effect/resource checks establish
construction and exact semantics, not their runtime speed. Bounded exact proofs
retain explicit limits instead of accepting an unchecked schedule. Native owner
source-composition requests are only considered applied when their actual report
shows transformed products; a selected recipe or import alone grants no credit.
