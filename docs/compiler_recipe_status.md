# Shared compiler, optimization records and promotion status

## One compiler and one target backend

All three models use Merlin's shared typed preparation, upstream lowering, host
compilation, runtime and device linking, plus the same Gemmini OOT backend.
Operation semantics, shapes, numerical contracts and hardware resources can
select different transformations within that compiler. Target encodings,
device schedules, ABI glue and simulation belong in OOT. Portable host, packing,
requantization, ownership and orchestration mechanisms belong in Merlin.

The current measured champions use different experimental settings, earlier
source revisions and retained objects. A single fresh current-version build and
automatic selection policy has not been qualified across all three. Captured
source identities authenticate bindings; model names, provenance IDs and golden
values must never choose production transformations.

Separate experiment drivers currently choose pass and schedule options and
bind source-specific providers or retained objects. This manual orchestration
does not establish that ordinary compilation automatically discovers every
winning recipe. The shared preparation callback removes global function
replacement, but it does not supply the legality proofs or selection policy.

To qualify one general compiler, rebuild all three entire models from current
source through the same supported pipeline and installed packages; report every
selected or refused transform and its semantic/resource justification; prohibit
model-name dispatch and unexplained object substitution; then pass the unchanged
whole accuracy gates, final zero-FSM audits and whole stock performance gates.
Independent shapes, tails and refusal cases must also qualify each reusable
transformation. Different decisions justified by the input IR remain valid
within that one compiler.

| Workload | Best verified whole stock cycles | Current qualification boundary |
| --- | ---: | --- |
| ResNet50 | 28,649,233 (2109) | Fresh installed 3f6a8db27 build regenerates every linked object leaf and passes all 1,000 original words exactly plus final zero-FSM. Whole stock timing and automatic policy qualification remain open |
| TinyLlama | 378,946,263 (2085) | Fresh installed cdf117e3 build regenerates every linked object leaf and passes all 256,000 original compiled words exactly plus the unchanged Torch gate and final zero-FSM. A successor using the normal 451849ba7 mask-contract API is building; whole stock timing remains unknown |
| SmolVLA | 258,621,872,969 (1906) | Stock2113 completes at 324,229,555,204 cycles, a 25.3682% regression, with all 1,600 original words exact. Fresh exact-math b154 passes the full functional gate but has no stock timing |

ResNet's original experiment replaced global compiler functions. The explicit
layout parameter and invocation-local prepared-model callback now reproduce
byte-identical fresh current-core LLVM, host object and final ELF against the
same-core legacy control. All 1,000 original words and the final zero-FSM audit
pass. The new shared hook is local Merlin commit `edf0e0ca4`, with 17 source and
17 independently installed checks; all 1,016 Python and 61 runtime files match
source, wheel, source archive and installation. Publication is pending GitHub
DNS access. [Normal interface qualification](perf_records/root_resnet_shared_preparation_20261007_qualification.json),
[installed compiler qualification](perf_records/merlin_prepared_model_transform_local_20261007_qualification.json).

The fresh Smol exact-math build now passes its own full target run, with all
1,600 original words bitwise exact, zero rank mismatches and final zero-FSM.
Its functional counter is 117,635,197,150; no new FPGA timing or isolated pass
savings are attributed. The prior native-only receipt remains unchanged.
[Whole target qualification](perf_records/root_smol_exact_math_normal_whole_target_20261007_qualification.json).

## What the optimization records retain

The [journey](golden_optimization_journey.md), [current evidence](golden_progress.md)
and [ownership notes](infra_vs_dialect.md) retain each hypothesis and change,
source/compiler/object identities, source legality and numerical/effect/ownership
obligations, actual selected options, independent shapes and refusal cases,
original whole output gates, complete measured windows, gains, regressions and
pending checks. Rejected candidates remain beside their successors.

Functional counters, analytical estimates, GSIM sections and whole FireSim
results retain distinct labels. Callback timers include host adapters, command
issue, transfers, accelerator service and waits; they do not establish a pure
host/accelerator percentage. Section savings cannot be added to manufacture a
whole-model result. Experiment selection is recorded separately from default
compiler routing and automatic promotion.

The campaign meter at 2026-10-07 20:35:32 UTC is **108,738,933 aggregate tokens**.
Per-topic/OOT-only token allocation and billing are unavailable. This aggregate
must not be attributed to Gemmini dialect work alone.

## What phase 0/1/2 must expose

| Phase | Target-independent tooling and abstraction |
| --- | --- |
| 0 | Complete typed producer/consumer observations, numeric/effect/ownership contracts, exact prepared operand identity, source-to-object traceability and explicit lowering refusals |
| 1 | Invocation-local stage interfaces, immutable operand preparation, bounded multi-output products, source rounding legalization, host lane schedules and installed compiler qualification |
| 2 | Searchable semantic/resource options, unused-feature reports, complete producer-to-consumer cost models, actual spills/traffic/issue/wait measurements, calibrated section screens and unchanged whole hardware gates |

Each promoted option needs a record of applicability, checked refusals,
independent cases, complete cost and its whole-model effect. The agent must be
able to edit reusable Merlin passes/runtime and target OOT scheduling/lowering
through the same compiler interfaces. The analytical model must include setup
and validation outside the region when admitting a bounded simulator run: an
80M-cycle single-issue run cannot reach timing after a 115M-instruction input
check. Preserve original complete input checks and budget them explicitly.

Local Merlin main cdf117e3d now includes the generic explicitly contracted
borrowed-buffer lane schedule, with 96 source and 96 independent installed
checks and complete package byte closure. This extends the supported common
feature system; it does not automatically choose the experimental best recipe.
[Qualification](perf_records/merlin_borrowed_pointwise_current_main_20261007_qualification.json).

Stock2113 completed at 19:17:56 UTC; the queue is now idle. New FireSim
submissions still refuse because execution UID2621 lacks the `firesim` group.
The attempted ResNet submission created no job. GitHub fetch also fails DNS,
so new local topics are not claimed published. Prepared packet admission, actual
hardware execution and publication status are recorded separately.


## 2026-10-07: shared source-observation topics and remaining promotion work

### 20:35 UTC: normal effect forwarding and fresh object closure

Newest qualified local Merlin main is `451849ba7`, on `3f6a8db27`. The existing
typed `MaskEffectContract` now passes unchanged through `lower_model`,
`lower_model_file`, the normal bare-metal model builder and the Zephyr builder.
Selection still requires the explicit mask feature and scalar schedule; no
numerical or effect permission is inferred. All 69 source checks pass with six
unavailable target-support skips. The 58 affected normal lowering/build/runtime
checks pass in an independent installation. Seventeen legacy curated-target
routing cases run in source only because installed core intentionally supplies
no default target or reference contracts. All 1,020 Python and 61 runtime files
match source, wheel, source archive and installation. Four independent normal
default model lowerings emit byte-identical LLVM across old source, current
source and outside installation; all 2,881 parent qualification pins reclose.
GitHub fetch still fails DNS at 20:35:32 UTC, so this is qualified local main.
[API qualification](perf_records/merlin_model_mask_effects_current_main_20261007_qualification.json),
[default emission](perf_records/merlin_model_mask_effects_default_emission_20261007.json).

Fresh installed `3f6a8db27` ResNet now regenerates every final linked object leaf:
stem, all convolution/requantization and residual/domain providers, mean,
classifier, host adapters, weights, runtime and harness. All 1,000 original words
pass exactly in native and actual Spike; final zero-FSM passes. The receipt closes
3,937 pins. Fresh installed CDF Tiny also regenerates all object leaves and all
155 source bindings, with 256,000 original compiled words exact in native and
actual Spike, the original Torch `atol=0.03125, rtol=0.02` gate passing, and final
zero-FSM. Its receipt closes 1,348 pins. These builds retain explicit earlier
source ingredients and manual selections, and have no new whole FPGA timings.
The Tiny 451 successor removes CDF's extra mask-contract lowering through the
supported normal API; its full gates remain pending.
[Fresh ResNet delivery](perf_records/resnet_fresh_all_target_leaves_installed3f_whole_20261007_qualification.json),
[fresh Tiny delivery](perf_records/tiny_fresh_all_target_leaves_installedCDF_whole_20261007_qualification.json).

The remaining automatic-discovery gap is concrete: Tiny's early prepared-model
hook sees no closed scalar-observer proofs; the source after ordinary polynomial,
FMA, elementwise fusion and generalization has 22. A selected invocation-local
typed stage at that point must discover proofs from current IR and reify their
original helper arithmetic before ordinary lowering. The diagnostic checkpoint
is an experiment; supported integration and a fresh whole-model gate are required
before claiming that retained recipes have been eliminated.
[Delivery journey and phase recommendations](perf_records/tiny_current_shared_delivery_journey_20261007.md).

Complete Tiny command batching costs 2,016,652.5→2,012,174.0 GSIM cycles
(−0.22207594%), despite a −14.0806% native instruction result. Arm spread is large
relative to this small change; no default or whole-model promotion follows.
Smol's actual source phases assign 18.860B instructions to softmax interval/
BF16/lanes/scheduling, 9.616B to canonical packing and 8.040B to polynomial
endpoints. Ordered QK/PV dot replay totals only 3.264B. Source coverage finds all
150,994,944 score cells active, so omitted masked/cutoff cells supply no savings
on this capture. Complete softmax outlining passes original outputs but saves
only 0.5719854% functional instructions on one full group.
[Tiny complete cost](perf_records/tiny_retained_n_complete_cost_20261007.json),
[Smol source costs](perf_records/smol_actual_source_phase_census_20261007.json),
[endpoint coverage](perf_records/smol_actual_softmax_endpoint_coverage_20261007_qualification.json),
[complete outlining](perf_records/smol_source_softmax_outline_complete_20261007_seal.json).

Root independently reclosed 12 source packets and 6,274 distinct declared pins,
then mirrored the new records byte exactly. Source counts, GSIM sections,
functional counters and stock model cycles keep their original scopes.
[Source reclosure](perf_records/root_common_compiler_records_20261007_reclosure.json),
[mirror identity](perf_records/compiler_current_record_mirror_identity_20261007_second.json).

Local Merlin main `3f6a8db27` now integrates three clean topics on cdf117e3:
an exact multi-output integer-product family, elimination of duplicate finite
point observations and exact BF16-to-integer observation decoding. Each has
explicit numerical, effect and storage contracts, independent shapes and
refusal cases. All three remain default-off; they choose no workload names.
The four production module ASTs match the independently measured prototypes.
All 1,227 source and 1,227 outside-installed checks pass; 1,020 Python and 61
runtime files match source, wheel, source archive and installation. Twelve
default emission cases remain byte-identical to the prior core. Root recloses
2,881 qualification pins. These installed mechanisms do not establish a fresh
whole-model performance result or automatic profitable selection.
[Compiler qualification](perf_records/merlin_source_observation_topics_current_main_20261007_qualification.json),
[default and source closure](perf_records/merlin_source_observation_topics_source_reclosure_20261007.json).

The earlier full ResNet successors used installed cdf117e3. One verifies the
shared eight-lane host recipe; the other
freshly generates all 52 target kernels, including 11 explicitly selected dense
output-block changes. Original 1,000-word 0/0 and final zero-FSM gates pass.
Their receipts close 859 and 4,847 pins respectively. Input capture/source
bindings remain pinned earlier inputs; this is not a fresh current model2MLIR
capture. Neither inherits stock2109's timing.
[Shared host successor](perf_records/root_resnet_current_eight_lane_installed_whole_20261007_qualification.json),
[fresh target successor](perf_records/resnet_resident_a_output_blocks_current_installed_whole_20261007_qualification.json).

Stock2113's timer is 324,229,555,204 model cycles, rather than the simulator's
324,535,367,732 total including other work. Its all-1,600-word original digest
is exact and its queue duration is 11,516.48 seconds. The best historical
258,621,872,969 result stays the champion; differing whole recipes prevent
single-transform attribution. The exact-math b154 candidate was not measured
on this FPGA run. Physical peak-capacity proof remains unknown even though the
tested run completed correctly.
[Whole stock receipt](perf_records/root_smol_endpoint_stock2113_whole_20261007_qualification.json),
[original UART](perf_records/root_smol_endpoint_stock2113_whole_20261007_uart.txt).

For phase 1/2, existing dispatch counts and loop-offload flags miss CPU command
batching, address work and spills: Tiny variants emit the same 101,861 primitive
commands but execute different CPU instruction and stack counts. Expose portable
host-loop retention/batching choices and ELF-bound executed opcode/stack metrics;
retain target resource checks and actual command generation in OOT. A spill-only
model is insufficient because one smaller-stack variant regresses. Smol's
complete cost likewise limits family and observer gains; improving callback or
refinement counts is not a whole cost theorem.
[Tiny actual native cost model](perf_records/tiny_retained_n_normal_native_model_20261007.json),
[Smol complete source observations](perf_records/smol_source_observer_norm_journey_20261007.md).
