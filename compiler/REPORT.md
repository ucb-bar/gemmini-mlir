# General fresh10 Gemmini compiler

The compiler continues the supplied seed and uses the selected editable Merlin
library. It selects implementations by typed source semantics, geometry, layout,
numeric boundary and RTL resource facts. Capsule and model names do not select
code paths. The seed report is retained separately in `SEED_NOTES/REPORT.md`.

## Six-program evidence

| Model | Program | Device kernels built | Sampled exact calls | Contraction census device / required | Complete host link / chain numerics |
| --- | --- | ---: | ---: | ---: | --- |
| ResNet50 | model | 24 / 24 | 45 / 45 | 54 / 54 | UNKNOWN / UNKNOWN |
| TinyLlama | prefill | 8 / 8 | 22 / 22 | 3675 / 3675 | UNKNOWN / UNKNOWN |
| TinyLlama | decode | 8 / 8 | 22 / 22 | 3675 / 3675 | UNKNOWN / UNKNOWN |
| SmolVLA | prefix_encode | 14 / 14 | 36 / 36 | 3259 / 3259 | UNKNOWN / UNKNOWN |
| SmolVLA | flow_denoise | 14 / 14 | 34 / 34 | 2036 / 2036 | UNKNOWN / UNKNOWN |
| SmolVLA | action_decode | 0 device kernels | 0 sampled calls | 0 / 0, host program | UNKNOWN / UNKNOWN |

These are visible-input endpoint kernel replays, not complete-chain executions.
All endpoint `program.runs` arrays are empty. The endpoint reports its model
numerics gate passing because its sampled kernels are exact; this does not
independently establish host-chain numerical correctness or final runtime linking.
The six-program compile-and-link obligation therefore remains incompletely
demonstrated. Local complete-source RV64 objects build for representative mixed
programs, with runtime imports retained. Their runtime archives are not staged
locally; no replacement host tensor arithmetic or invented runtime is supplied.

Exact records are outside the submitted compiler at
`artifacts/final73_model_resnet50.json` (refreshed after the stride-one promotion),
`artifacts/final65_model_tiny_llama.json`,
`artifacts/final65_model_smolvla.json` and
`artifacts/host_lowering/object-scope.json`. The evidence join
`artifacts/final_evidence.json` preserves source and report identities.

## Implemented compiler behavior

`backend/frontend.py` and `source_groups.py` preserve source axes, exact numeric
boundaries, all externally used outputs and original operation identities through
device grouping. The compiler handles signed dense and batched products, direct
and window convolutions, live accumulator initializers, runtime i32 bias,
readout epilogues, residual sums, pooling, transposes and strided movement.
W8A8 contraction semantics remain i8 multiplication with modular i32 accumulation;
scaled readout preserves the declared f32 scale, half-even rounding, clamp and
activation order. Unsupported integer-shift requantization is explicitly refused.

The source movement adapters implement static rank-1 through rank-4 i8 slices,
concatenation and unit-stride insertion. They check actual source/destination
address projections, injectivity, cardinality, ownership and completion with the
selected arithmetic and storage facts. Insertions partition the output into
disjoint boxes, preserving arbitrary runtime initializer values outside the
inserted rectangle. Concatenation preserves ordered input positions, including
repeated read-only operands. Actual materialized outputs have distinct storage;
unknown original aliases do not become invented alias permissions.

Dense and convolution alternatives consume checked shared scheduling products
before the common terminal ISA encoder. Dense paths retain operands where legal,
use capacity-bounded segmentation for long reductions, and group disjoint stores.
Convolution paths retain input reduction panels, issue contiguous window row runs,
reuse resident weights and group stores only where actual ordering and ownership
facts permit. Configuration, source addresses, storage effects and completion
edges remain available until their consuming transformations run. The RTL does
not establish causal ready-readout, so required producer/readout drains remain.

Host floating-point normalization, nonlinear operations, control and unsupported
movement use Merlin's shared host lowering, CPU recipes, preparation and runtime.
`host_runtime.py` supplies source routing and ABI glue, not new tensor arithmetic.
`mixed_placement.py` checks source-to-device correspondence and emits an exhaustive
source partition. A source-group compilation failure is fatal. Outer mixed-program
command buffers intentionally have no invented accelerator commands; their real
device subprograms and complete compiled source task remain attached.

## Public correctness and physical evidence

The selected inventory is 152 cases: 101 original correctness, 8 calibration,
40 scalability and 3 connected programs. The complete core development check
screened 109 / 109 successfully, including the expected unsupported-requantization
decline. All 40 scalability cases and all 3 connected programs passed L2 in the
recorded checks. Changed mixed-source executables were checked again after
integrating the movement adapters. The final wide-weight rule preserves 101 of
102 ordinary executable LLVM modules. The changed `SY_multi_block_beyond` module
passes focused L2 numerics and its renewed required L3 check. All six original
mixed-source programs retain identical LLVM output. Evidence is joined by emitted
bytes and focused checks rather than a compiler source hash.

Required original physical records cover 101 / 101 cases. Of these, 98 are
certified, including the one expected decline; three remain structurally
unqualified. All 94 executed ordinary operator cases passed RTL numerics. The six
mixed-source original cases also passed their required native tiers. The three
unqualified cases are `CH_residual_conv_edge_m17k17n17`,
`CH_residual_linear_edge_m17k17n17` and
`CH_sibling_feed_edge_m17k17h17n17`. Each declares identity scale 1.0, passes RTL
numerics, and fails the structural expectation that a declared scale produce a
nonidentity `CONFIG_ST` scale. The compiler preserves the source scale; it does
not manufacture extra scaled commands to satisfy this expectation.

Separately, two calibration cases lack the requested resident-reuse structural
fields despite passing development numerics. Missing physical tiers and
structural qualification are not numerical failures. The ordinary physical
ledger is `artifacts/original_physical_evidence44.json`; later mixed-source records
are `artifacts/integrated48_{smol,resnet}_physical.hydrated.json`. The evidence join
names the unchanged four mixed-source records and explicit executable identity
checks. Earlier cross-provider digest comparisons are superseded by
`artifacts/physical39/execution_identity_same_provider.json`.
The final changed-byte renewal is
`artifacts/weight_panels/regional62/original_sy_physical.hydrated.json`;
its measured gSIM cycles fall from 20,949 to 18,989.

The actual seven-program source partition accounts for 2475 original operations:
410 compiled device operations, 1546 noncompute operations, 477 hardware-unsupported
operations, 21 unresolved proof obligations and 21 unresolved resource obligations.
This denominator comprises the six original mixed sources and the repeated-source
program; it is distinct from the full-model contraction census. See
`artifacts/mixed_integration48/source_binding_validation.json` for source-bound
partitions and corrupted-join rejection checks.

The mandatory five-shape convolution scalability probe passed its emission
resource budget. Maximum canonical code cost was 94,286 bytes against the
1,048,576-byte limit. This is emission/code-growth evidence, not execution evidence
(`artifacts/final42_scaleprobe.summary.json`).

## Performance evidence and scope

Matched complete Spike estimates support the retained regional schedules. One
large convolution, `RN_conv3x3_s1p1_c512_h7_i8`, falls from 3,350,812 to 2,337,336
cycles (30.25%). Its modeled roofline gap is 5.176 and useful-compute peak
attainment is 19.32%; padding efficiency is 76.56%. Modeled CPU issue, ALU,
execute-controller and shared-memory reservations are respectively 25.47%, 20.21%,
34.17% and 28.18% of the same elapsed window. Reservations overlap and cannot be
summed. These are model diagnostics, not measured PE occupancy. Host-wait
attribution and measured engine occupancy remain UNKNOWN. Exact identities and
resource views are in `artifacts/conv_transfer_regional16_scalability.utilization.json`.

`TL_att_decode_q32kv4s1l2048d64_raw` reaches 437,444 complete modeled cycles
versus 681,323
for its earlier retained implementation (35.79% lower). A partial cohort with seven UNKNOWN cases
improves from 160,064,716 to 150,612,109 modeled cycles (5.91%);
this is not a full-suite or full-model aggregate. Matched gSIM witnesses include
3290 versus 6062 cycles for `G_conv2d_c24x3x3_k3x3s1_n24_raw` and 3795 versus
4054 cycles for `CH_kv_heads_edge_q6kv2s13l13d9`. These measured improvements
apply only to those witnesses and their recorded executables.

Operator complete estimates use provisional model `b3ce9aaf…`; native connected
program estimates use provisional model `132c41bd…`. They are separate comparison
domains and are not combined. `JC_resnet50_prepared_connected` remains expensive
at 97,320,353 complete modeled cycles (190.4 times its roofline bound), with
substantial host work. Complete full-model timing is UNKNOWN. The model has not
been certified accurate for small schedule differences. Fast estimates are
order-blind and cannot decide changes in transfer width, overlap or prefetch.

The promoted `regional_traffic_wide` rule consumes two actual checked schedule
products. It selects the capacity-legal full-K 8×32 weight bound only with uniform
readout, at least two actual counted M panels and at least 5% lower checked input
traffic; these predicates do not grant legality. The rule was frozen before six
native training/held-out/validation checks, with no result-driven rule edits:

| Actual M / K / N | Incumbent cycles | Selected cycles | Change |
| --- | ---: | ---: | ---: |
| 784 / 256 / 1024 | 1,411,302 | 1,160,950 | −17.74% |
| 512 / 512 / 512 | 841,632 | 743,014 | −11.72% |
| 3136 / 128 / 512 | 1,630,512 | 1,345,095 | −17.50% |
| 3136 / 512 / 128 | 1,322,001 | 1,322,001 | identical LLVM |
| 12544 / 64 / 256 | 2,162,256 | 1,870,826 | −13.48% |
| 12544 / 256 / 64 | 1,821,169 | 1,821,169 | identical LLVM |

All six pass L2 under the same complete provisional b3 model, facts, domain and
scope. Exact capsule identities, transformed products and emitted hashes are in
`artifacts/weight_panels/regional62/report.json`. The capsule whose name ends
`12544x64x64` actually declares N=256; the table follows its source types. The
512-cube case has modeled compute-floor gap 1.417, useful peak attainment 70.56%
and padding efficiency 100%. The 3136/128/512 case has gap 1.675 and attainment
59.68%; these are bound/model ratios, not PE occupancy. Additional reservations,
traffic boundaries and unknown host attribution are retained in
`artifacts/weight_panels/regional62/utilization.json`.

The full 40-case scalability LLVM audit finds five changed operator executables,
all matching tested native outputs; 35 retain incumbent LLVM. The complete medium
residual program passes L2 but regresses from 2,964,530 to 2,973,437 modeled cycles
(+0.30%). This small connected-region regression is retained explicitly alongside
the larger operator gains; no full-model speedup or universal winning geometry is
claimed. The frozen rule retains the one-M-panel 49/512/2048 incumbent. See
`artifacts/weight_panels/regional62/promotion.json` and
`artifacts/final65_residual_program.hydrated.json`.

Optional schedules and source/CPU alternatives remain explicit tuning choices,
including capacity-checked 8×8 and 8×32 weight-panel bounds.
Wider decode geometry, wider adaptive long-K panels, medium-M weight retention,
dense LHS unrolling and prepared convolution tap schedules
were not promoted after regressions or missing complete timing. Pool-resident
and convolution M-group trials remain opt-in because their current comparisons
do not establish a general crossover against the default. Source-selected host
rounding and immutable preparation trials are recorded separately in the
iteration ledger. A prepack-only ablation on the medium residual program gives
2,963,988 versus 2,964,530 complete modeled cycles (0.0183%); exact-preparation CSE
alone leaves cycles unchanged. This small result does not justify a general
default. The optional 8×32 weight bound gives 559,929 versus 611,224 cycles
(8.39% lower) on `CH_residual_conv_resnet50_model_m196k256n1024`, but regresses
the 49×512×2048 raw case by 0.32%. The promoted counted-panel rule selects the
former executable and retains the latter incumbent.
The former has modeled compute-floor gap 2.790, useful peak attainment 35.84%
and padding efficiency 94.23% (see `artifacts/weight_panels/wide60/utilization.json`).
Dense-view trials leave the two timed connected regions unchanged;
layout propagation and intermediate-storage reuse retain unchanged code because
the original software permissions are not admitted. Selection receipts alone do
not establish lower latency.
An ablation of stock host elementwise fusion on the earlier regional incumbent
passes numerics but increases connected ResNet cycles from 97,320,353 to
115,403,195 and medium residual cycles from 2,964,530 to 3,259,575, under complete
model132. Fusion remains enabled; these ablations are not a paired timing claim
for the subsequent wide-weight schedule.

The continued construction adds opt-in dense address-invariant hoisting and a
checked convolution M-group policy. The dense path transforms the actual shared
command SCF consumed by emission and preserves every device operation, address,
operand and completion edge. Three large projection checks pass L2. Complete b3
estimates change 3,546,828 to 3,546,001; 3,512,700 to 3,511,985; and 7,005,040 to
7,004,317 cycles. These gains below 0.03% do not justify a default under the
provisional model. The original exact-token copy-pipeline contract is not admitted
by these counted asynchronous compute schedules; this is an address optimization,
not a claim of DMA overlap.

The broad convolution M-group trial passes both native checks, improving the
56x56x64 stride-one case from 4,791,983 to 4,725,457 complete modeled cycles but
regressing the stride-two case from 886,432 to 895,152. It remains opt-in. The frozen
stride-one refinement is now selected by default. Its fresh original-shape gSIM
witness passes numerics and structure at 31,610 measured cycles versus 31,761
for the incumbent under the same RTL provider (0.475% lower). The 102-module
audit changes only that uniform-i8 aligned 3x3 module. All 40 scalability sources
compile: four changed stride-one outputs match prior checked native products,
and the other 36 are unchanged. All nine complete mixed-source LLVM outputs
remain identical. Legality is independently checked using the same
source, storage and epoch products, and both alternatives retain the common
terminal encoder. See `artifacts/dense_hoist68/comparison.json`,
`artifacts/conv_mgroup67/comparison.json` and `artifacts/conv_stride1_71/`.

An instrumented connected ResNet run passes L2 and realizes the requested stock
operation profile. Reductions/softmax account for 48.34% and elementwise work for
36.47% of raw counter ticks. These diagnostic ticks are not elapsed cycles; the
instrumented whole-call timing is UNKNOWN (`same_elf_observation` ValueError),
and the uninstrumented 97,320,353-cycle control is retained. The authored scalar-loop setting leaves the medium program
executed ELF and 2,973,437 modeled cycles unchanged; this does not establish
application through the original-owner CPU route. The connected comparison
is pending.
No profiled build is promoted from counter totals. Exact observations and category
denominators are in `artifacts/source_profile69.summary.json`.

`ITERATION_NOTES.md`, `artifacts/` and `variants/` retain hypotheses, parent and
library identities, pass options, actual transformed products, emitted hashes,
checks and keep/reject reasons. No startup profiling, simulation during ordinary
compilation, per-capsule lookup, private evaluator access or reference answers are
used. The selected CCA bank was disabled for this round and was not used.
