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

| Workload | Best verified whole stock cycles | Current qualification boundary |
| --- | ---: | --- |
| ResNet50 | 28,649,233 (2109) | Best source recipe pins Merlin eb15a85; fresh shared-interface successor now passes the full original 0/0 target gate, with hardware cost pending |
| TinyLlama | 378,946,263 (2085) | Generic observer topic is reproduced, with eleven other object leaves retained; a fresh full current-head compiler build remains required |
| SmolVLA | 258,621,872,969 (1906) | New source-bound f90 whole executable passes all 1,600 original words in functional target execution; stock2113 remains pending |

ResNet's original experiment replaced global compiler functions. The explicit
layout parameter and invocation-local prepared-model callback now reproduce
byte-identical fresh current-core LLVM, host object and final ELF against the
same-core legacy control. All 1,000 original words and the final zero-FSM audit
pass. The new shared hook is local Merlin commit `edf0e0ca4`, with 17 source and
17 independently installed checks; all 1,016 Python and 61 runtime files match
source, wheel, source archive and installation. Publication is pending GitHub
DNS access. [Normal interface qualification](perf_records/root_resnet_shared_preparation_20261007_qualification.json),
[installed compiler qualification](perf_records/merlin_prepared_model_transform_local_20261007_qualification.json).

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

The campaign meter at 2026-10-07 17:37:19 UTC is **104,048,865 aggregate tokens**.
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

New FireSim submissions currently refuse because execution UID2621 lacks the
`firesim` group. Existing job2113 continues independently. Prepared packet
admission and hardware execution status are recorded separately.
