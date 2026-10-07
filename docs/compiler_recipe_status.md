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
| ResNet50 | 28,649,233 (2109) | Fresh installed cdf117e3 host builds pass all 1,000 original words; a successor also freshly generates all 52 target kernels and passes the whole 0/0 gate. Neither successor has whole stock timing |
| TinyLlama | 378,946,263 (2085) | Observer and command-batching section candidates have independent gates. A fresh installed cdf117e3 whole build with all target leaves is in progress; current-head automatic qualification remains open |
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

The campaign meter at 2026-10-07 19:40:57 UTC is **107,256,545 aggregate tokens**.
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

The new full ResNet successors use the previous installed cdf117e3, not the
new 3f6a8db27 package. One verifies the shared eight-lane host recipe; the other
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
