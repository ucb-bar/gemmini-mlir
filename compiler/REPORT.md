# Fresh10 compiler continuation — 2026-10-29

Recorded r8 evidence (new continuation checks are tracked below):

| Model | Program | Compile/link evidence | Numerical evidence | Contraction census |
|---|---|---|---|---|
| ResNet50 | model | 24 extracted kernels built; full root UNKNOWN | 45 selected kernel calls exact | 54/54 device |
| TinyLlama | prefill | 8 extracted kernels built; full root UNKNOWN | 22 selected kernel calls exact | 3675/3675 device |
| TinyLlama | decode | 8 extracted kernels built; full root UNKNOWN | 22 selected kernel calls exact | 3675/3675 device |
| SmolVLA | prefix_encode | 14 extracted kernels built; full root UNKNOWN | 36 selected kernel calls exact | 3259/3259 device |
| SmolVLA | flow_denoise | 14 extracted kernels built; full root UNKNOWN | 34 selected kernel calls exact | 2036/2036 device |
| SmolVLA | action_decode | No extracted contraction kernels; full root UNKNOWN | No kernel replay obligation; full chain UNKNOWN | 0/0 |

These visible model checks use one input and zero shape variants. Extracted-kernel
builds and replays do not establish root-program linkage, end-to-end numerics or
whole-model timing. The census counts contractions, not every source operator.
Full reports and source identities are in `../artifacts/integration/r4/` and the
latest `../artifacts/integration/runtime_*.model.json`. All three packaged model
checks pass: 68 extracted kernels built and 159 selected calls bit-exact. The model
region profile is `../artifacts/integration/model_region_profile.json`.

The R4 public sweep has 109 expected core outcomes: 108 numerical executions and
one correct declared refusal. All 40 public scalability cases pass numerical
execution. The earlier scalability aggregate has skipped L0/L1; the explicit required-tier
result for the runtime package is recorded separately in
`../artifacts/integration/runtime_scalability.response.json`. Missing or skipped
tiers are not passes. All three public connected programs pass L0/L1/L2 on the packaged runtime
(`../artifacts/integration/runtime_programs.response.json`); changed repeat
coverage also passes its affected-program check. All six source-core numerical
screens pass on the packaged runtime, with L3 unverified by that development request. Current request summaries
remain under `../artifacts/integration/`; unrun tiers are never counted as passes.

All 101 required original physical requests were collected: 100 numerical
executions pass, and the one expected refusal passes its contract. Ninety-eight
outcomes qualify their required checks; three identity-scale cases retain a
structural trace gap despite passing numerical execution and their L3 barrier.
They declare source scale 1.0 while the trace requirement demands a nonidentity
store scale. The compiler preserves source numerics and emits no fabricated
scale commands. See `../artifacts/integration/identity_scale_structural_gap.json`.
Six changed source cases have R4 physical checks; the two repeat-affected cases
also pass another required physical check. The direct LLVM/ABI comparisons and native
tier results are retained separately. Cross-engine execution digests do not have
a supplied equality contract and are not used to claim certificate identity.

The implementation keeps the seed's shared terminal encoder and Merlin host
lowering. General source, shape and hardware predicates select actual checked
schedules: segmented K pipelines, strict-traffic RHS reuse across M, independent
shared-RHS row products, resource-guarded convolution row groups/readout epochs,
and disjoint residual batches. Source axes, pointer bindings, exact epilogues and
completion edges reach the shared transformations. Static dense f32 permutation,
concat, rectangular slice and singleton-axis repeat use exact full-width integer
transport; floating arithmetic remains under its original host contract. Repeat
is enabled for required device coverage, with complete cost UNKNOWN.

Representative retained complete-call comparisons follow. Values are provisional
Spike modeled cycles under matched source/input/model/facts/scope pins. They are
not physical measurements, calibrated speedup guarantees or whole-model results.

| Region | Before | Retained | Change | Partial floor gap | Useful peak attainment |
|---|---:|---:|---:|---:|---:|
| Decode attention q32/kv4/s1/l2048/d64 | 2,345,322 | 456,578 | −80.53% | 6.54× | 7.18% |
| Dense M16/K5632/N2048 | 7,617,813 | 2,877,199 | −62.23% | 3.96× | 25.06% |
| Convolution 56×56, C64/N64, 3×3 | 4,934,651 | 2,727,681 | −44.72% | 1.51× | 66.22% |
| Dense M3136/K512/N128 | 1,615,362 | 1,322,001 | −18.16% | 1.65× | 60.73% |
| Residual convolution M196/K256/N1024 | 611,224 | 495,511 | −18.93% | 2.47× | 40.50% |
| Prepared connected ResNet | 97,627,861 | 97,305,621 | −0.33% | 190.36× | 0.53% |

The partial floor gap uses the maximum provided structural compute and logical
request service floor, which omits sustained bandwidth, contention and host costs.
Useful-compute peak attainment divides source useful MACs by 256 structural peak
MACs per cycle times complete modeled region cycles. The exact detail pointers,
padding fractions and matching scope/model/fact hashes are in
`../artifacts/integration/retained_performance_table.json`. This is not measured processing-element occupancy.
Scoped load/execute/store reservations overlap; they must not be summed into
elapsed cycles. Missing host wait, queue-pressure and physical occupancy
attribution remains UNKNOWN. Conv resource/activity comparisons are recorded in
`../artifacts/conv_current/readout_activity_comparison.json`.

`../OPTIMIZATION_RULES.json` indexes hypotheses, actual predicates/options,
before/after product identities, matched results and keep/reject reasons. Dense
completion epochs now use the supported single-M-tile guarded path described below.
Stride-two spatial panels and conv/pool fusion remain available choices; incomplete
paired timing or held-out coverage retains their incumbents. Counted convolution groups and large attention row coalescing were
rejected on complete-cycle regressions. Host recipe trials retain the incumbent
where the realized executable was identical, slower or unobserved.

Selected software SHA256:
`3c1b8aa1c77775e1c0e590af871f05e038743d7000a99860fb12843f7053372f`.
Selected library manifest SHA256:
`ba21c662335c5f62d6b62246a0bc5f35131e0b0533e3f27bda8f1e6aa3665907`.
Compiler sources are in this directory; the inherited seed report is preserved
at `../SEED_NOTES/REPORT.md`. Private grading and independent final hardware
validation are not inferred from public development evidence.

The portable Python entrypoint packages z3-solver 4.16.0 from the supplied
runtime under `third_party/`, with MIT notice and SHA provenance. Its Python
sources are unchanged; ordinary `strip --strip-unneeded` removes unused native
symbol/debug data while preserving all 940 defined dynamic exports. Solver
controls and identical LLVM emission validate this packaging change. No solver
UNKNOWN is treated as an admitted proof.

Native staging accepts the 32,924,398-byte r8 package. The retained runtime is a
conservative symbol/import closure including every authored module and option,
all version 6–24 provenance rosters, dynamic pass targets, source-discovered host
registries and literal source resources. All 102 available core/calibration LLVM
products match the tested R4 incumbent. Sixty-two authored imports, 88 knob-reader
values and representative actual lowerings pass without ambient Merlin fallback.
Reader acceptance does not prove execution of every alternative. The closure audit
is `../artifacts/host_current/package_closure/summary.json`; native resource/model
checks are tracked under `../artifacts/integration/runtime_*` and
`symbol_package_scale_probe.*`. The native scale-emission probe accepts all five
shapes under the enforced 1,048,576-byte function-code limit; the maximum is
205,287 bytes. This proves resource acceptance only: probe numerics, object
compilation and RTL are not attempted, and static semantic obligations remain
UNKNOWN. Exact source/product pins are in `../artifacts/integration/runtime_resource_summary.json`.

Unused reviewed sources and historical evidence remain under
`../compiler_source_archive/` with a byte-preserving manifest. They are available
for authoring and are not loaded by the submitted compiler. The runtime keeps the
same semantic checks, shared encoder, host contracts and source-driven policies.

Continuation under the current public contract extends exact f32 singleton-axis
repeat to ranks 2–6, including row and column expansion. Independent source and
destination pitches reach the same raw integer DMA encoder; original source load
completion, read-only accumulator lifetime and final store completion remain.
Twenty-four geometry checks, arbitrary-bit execution of emitted packets for twelve
geometries and negative controls are in
`../artifacts/attention_current/repeat_generalization/REPORT.json`. Six old repeat
kernels and their LLVM remain byte-identical. New geometry coverage is local
execution evidence, not physical qualification or a timing result.

The guarded segmented completion-epoch path is enabled only for one M tile,
after original source/operand proofs and exact RHS-consumption checks. It retains
pure staging, initialization, readouts and final completion. Complete provisional
matched Spike comparisons support QKV 1,363,950→1,340,002 cycles, fused gate/up
5,813,261→5,708,415, down 2,877,199→2,828,627 and the complete sibling region
8,146,072→8,017,292. Multiple M tiles keep their incumbent. The 102 available
non-model core/calibration emissions were byte-identical before the separate stride-convolution promotion; all ten focused epoch scalability cases pass L2. Required skipped tiers remain incomplete; vocabulary complete timing is UNKNOWN. Exact provider and parent identities are in
`../artifacts/dense_current/epochs/promotion_current.json`. These small modeled
differences have no qualified accuracy band or measured speedup claim.

Whole-program command-buffer publication now uses the unchanged public outer
schema and exact source/emitted mixed-program validation. A valid compiled
whole-program plan may have an empty top-level command list while its device
regions retain actual accelerator command buffers. The standalone kernel
builder remains unchanged. Five full source-to-artifact-and-buffer chains
select real RAW_BIT_REPEAT device callbacks; malformed plans are rejected before
publication. Evidence is
`../artifacts/frontend_current/whole_program_publication/checks.json` and
`../artifacts/attention_current/repeat_generalization/full_compiler_checks.json`.


Continuation stride integration: the native pointwise panel route now accepts raw i32 output while preserving original attributes and the shared full-i32 encoder. CONV_STRIDE_PANELS=on selects the existing geometry/capacity-admitted unpadded 1x1 stride-two route; all refusals retain the incumbent. Raw small-core measured gSIM complete-call cycles1109→617 (-44.36%), matched simulator/harness/source; complete provisional Spike1114→613. The larger mixed-column i8 case is byte-identical to the existing opt-in route and passes L2, complete provisional Spike3387485→3095340 (-8.62%). Raw numeric local cases include ragged channels, batch and bias; no empirical cutoff or capsule lookup was added. Evidence and exact source/product/provider identities are in artifacts/conv_current/stride_raw_general/. Broader shape timing and full-root model gates remain separate.
