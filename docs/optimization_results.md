# Optimization results

Updated 2026-10-08 UTC. This ledger records gains, regressions, rejected ideas and
unfinished experiments for the `handwritten-implementation` branch.

The backend emits primitive configuration, transfer, preload, compute and fence
instructions. Device objects and final executables must pass the zero-FSM audit.
Production choices derive from current IR, numerical contracts and target facts;
workload names and reference outputs do not select compiler behavior.

## Whole-model measurements

These are the best verified results on the stock FireSim Gemmini Rocket
configuration. They are whole-model counter observations, not sums of section
estimates. Recent compiler and component experiments have not replaced them.

| Workload | Best cycles | Goal | Evidence |
| --- | ---: | ---: | --- |
| ResNet50 | 28,649,233 | 22,000,000 | [Stock run 2109](perf_records/root_resnet_key_batch16_stock2109_qualification.json) |
| TinyLlama | 378,946,263 | 300,000,000 | [Stock run 2085](perf_records/root_tiny_rne_zero_stock2085_terminal.json) |
| SmolVLA | 258,621,872,969 | 5,000,000,000 | [Stock run 1906](perf_records/smol1906_stock_hardware.json) |

The original correctness gates remain unchanged: exact outputs for ResNet50;
elementwise `atol=0.03125`, `rtol=0.02` for the language models. A bit-exact claim
requires comparison of every original output word. These gates check the
captured computation; they do not establish downstream task accuracy.

Benchmark scope matters. TinyLlama uses the retained full 22-layer forward
capture with 256,000 output values. SmolVLA includes its retained prefix and one
denoising step, producing 1,600 values. Jack's ResNet reference reaches
22,387,449 cycles with an already quantized INT8 image; our original whole timer
includes FP32 input quantization. Jack's language-model recipes were not supplied,
so equivalent sequence lengths and decoding work have not been established.

## Measured gains and regressions

Negative percentages mean less time or fewer cycles. Each row identifies its
measurement scope; component measurements do not predict whole-model savings.

| Optimization | Measurement | Result | Decision and generalization | Evidence |
| --- | --- | --- | --- | --- |
| Batch an exact residual predictor's input keys | ResNet50 whole model, stock FireSim | 28,728,702 → 28,649,233 cycles (**−0.277%**) | Measured in the best explicit recipe. Selection requires a source-derived exact certificate and legal resource use. | [Whole comparison](perf_records/root_resnet_key_batch16_stock2109_qualification.json) |
| Recognize an exact zero quantizer bin | TinyLlama whole model, stock FireSim | 380,396,343 → 378,946,263 cycles (**−0.381%**) | Measured in the best explicit recipe. The sufficient bin proof and original fallback generalize to eligible integer observations. | [Whole comparison](perf_records/root_tiny_rne_zero_stock2085_terminal.json) |
| New 48-source attention endpoint composition | SmolVLA whole model, stock FireSim | 258,621,872,969 → 324,229,555,204 cycles (**+25.368%**) | Regression retained. Outputs pass, but this historical comparison changes more than one transform; no single cause is assigned. | [Whole regression](perf_records/root_smol_endpoint_stock2113_whole_20261007_qualification.json) |
| Repartition compact convolution transfers and output blocks | Complete convolution callbacks, GSIM | 685,067.5 → 470,524.5 cycles (**−31.317%**) on the source case; **+21.376%** on an independent case | Explicit candidate. The best ResNet recipe already uses this family. Geometry and resource checks establish legality; measured service costs must establish profit. | [Complete component](perf_records/resnet_compact_family_complete_20261008.json) |
| Shared affine representation for a producer pair and integer observer | Complete TinyLlama producer/observer capsule, GSIM | 8,589,793 → 8,288,352 cycles (**−3.509%**); independent case **+44.169%** | Explicit candidate; no whole promotion. Shared tables and observer proofs generalize, while table/cache costs require independent measurements. | [Complete component](perf_records/tiny_affine18_complete_timing_20261008.json) |
| Retain interval facts through subset refinement | Complete original SmolVLA 16-row slice, native timing | 242.300191 → 241.463285 ms (**−0.345%**) | Small component gain, with extra metadata and allocations. It does not explain the whole-model gap. | [Complete cost](perf_records/smol_real_frontier_narrowing_20261008.json) |
| Change input quantization/padding and initialization | Complete ResNet input-prefix capsule, GSIM | 1,911,460.5 → 1,417,055.5 cycles (**−25.865%**); independent case **+14.499%** | Capsule gain only. Its SDK control uses byte-wise clearing for the tested length; the current whole-model runtime already uses word stores. No current whole gain is assigned. | [Capsule timing](perf_records/resnet_complete_input_prefix_gsim_20261008.json) |
| Batch adjacent integer quantization packets | Complete ResNet input prefix, Spike retired instructions | 1,026,939.5 → 997,369 (**−2.879%**); independent cases **−4.439% / −1.018%** | Explicit generic loop-scheduling candidate. Both executables use support objects identical to the current whole runtime. Default stays unchanged; instruction savings are not measured Rocket cycle savings. | [Current runtime component](perf_records/resnet_current_runtime_packet_batch_20261008.json) |

## Compiler improvements awaiting performance validation

| Change | Verified so far | Remaining gate | Owner |
| --- | --- | --- | --- |
| Share scalar approximation tables before normal lowering | All 22 current TinyLlama observers compile with one 196,608-byte table; uncertified cells retain source fallback. Fresh helpers match 991,232 integer observations in each of four host rounding modes. This candidate disables a conflicting lane-packet transform. | Composition with lane scheduling, the original whole output gate, linked runtime predicate and complete timing. Four-mode replay covers supplied observer inputs; upstream producers were replayed in RNE. | Merlin |
| Compose scalar observations with tensor lane scheduling | The reusable immutable tensor-insertion and bounded-loop route passes 581 source plus 581 independently installed checks. All 44 current packetized TinyLlama observations compile upstream with one table and the original scheduling features. Fresh helpers match 991,232 integer observations in each of four host rounding modes. Whole native and RV64 Spike execution match all 256,000 original output words, with linked runtime predicate and zero FSM. Default LLVM remains unchanged. | Complete cost and a stock whole measurement. The first functional counter comparison regresses but changes observer policy and core/runtime; it also omitted a previously selected late RNE legalization. A composed, matched successor is required. | Merlin |
| Batch adjacent bounded integer observation packets | The generic typed output-axis scheduler passes 609 source plus 609 independently installed checks, with package identity and default LLVM controls preserved. | Whole delivery and stock timing; the measured complete prefix improvement remains an instruction-count result. | Merlin |
| Preserve source joins through common-subexpression elimination | The current SmolVLA structural experiment merges three equivalent Q/K/V quantizers into one while retaining their source identities and all original calls. | Final public compiler integration, effect admission, original whole output gate and executable timing. | Merlin |
| Project unused immutable weight arguments | Current prepared ResNet accounting identifies 53 removable obsolete parameter tensors, totaling 106,240 logical bytes. Captured buffers remain until immutable ownership is proved. Normal native/Spike execution and 564 source plus 564 independently installed checks pass. | Actual whole delivery and measured benefit. | Merlin |
| Generate compact convolution candidates from typed command traces | Source/resource legality, independent cases, upstream lowering and zero-FSM object checks pass. | A complete cost selector and a fresh stock whole run. | OOT dialect |

The table-sharing [normal compilation](perf_records/tiny_current_normal_scalar_all22_20261008.json)
and [coefficient coverage](perf_records/tiny_current_normal_scalar_coverage_20261008.json)
are separate from the [installed compiler qualification](perf_records/scalar_carrier_current_normal_installed_20261008.json).
The fresh [22-member numerical replay](perf_records/tiny_current_normal_scalar_numeric_20261008.json)
retains the supplied-input and RNE producer scope. The separately qualified
[tensor lane composition](perf_records/scalar_tensor_lane_composition_20261008.json)
addresses the packetization conflict. The fresh
[44-member normal compilation](perf_records/tiny_current_normal_packetized44_20261008.json)
preserves the original scheduling features. Its separate
[44-member numerical replay](perf_records/tiny_current_normal_packetized_numeric_20261008.json)
passes with independently authenticated current source and lane coverage.
The new [native whole gate](perf_records/tiny_normal_whole_native_20261008.json)
and [RV64 whole gate](perf_records/tiny_normal_whole_spike_20261008.json) pass
all 256,000 outputs. [Linked admission](perf_records/tiny_normal_whole_rv64_link_20261008.json)
and [readonly table storage](perf_records/tiny_normal_whole_table_storage_20261008.json)
are closed separately. The functional counter is 223,991,802 versus an older
122,689,756 control, an 82.568% increase. This changes observer policy, core and
runtime, and omits previously selected late RNE legalization; it is not an
isolated transform comparison or a FireSim cycle result. The whole numerical
PASS is retained while a correctly composed successor is prepared.
The weight projection has a compact
[qualification summary](perf_records/immutable_weight_projection_20261008.json).
The input-prefix [runtime attribution](perf_records/resnet_input_prefix_runtime_attribution_20261008.json)
records why its measured capsule gain cannot be credited to the current whole build.
The generic [packet scheduling qualification](perf_records/bounded_rne_packet_scheduling_20261008.json)
retains the explicit candidate and unchanged default-emission scope.
Detailed current-source findings are in the
[FX and lowering audit](fx_quantization_fusion_audit.md).

## Rejected ideas and lessons

| Experiment or diagnosis | Finding | Consequence |
| --- | --- | --- |
| Infer whole gains from a fast cache microbenchmark | A constructed lookup case improves more than the complete original execution frequency. | Price actual preparation, allocation, lookup, refinement and fallback together. |
| Assume a legal transfer-saving schedule is profitable | Independent convolution and observer cases regress despite source-case gains. | Separate legality, candidate generation and measured profit. |
| Remove unused arguments using final LLVM liveness alone | Capture state ownership, generated callers and retained packed spans can remain observable. | Project the complete ABI from prepared SSA and explicit ownership, with one checked remapping. |
| Treat all unused captured weights as zero stubs | The 54 large retained tensors are captured buffers. | Require immutable ownership evidence; manifest kind alone cannot grant removal. |
| Attribute a capsule gain to a different runtime baseline | The input-prefix capsule and current whole executable link different `memset` implementations. | Bind the actual support-library bodies before transferring a performance claim. |
| Treat compilation of an approximation as accuracy validation | A shared table can lower successfully while its numerical and final-output gates remain pending. | Keep compilation, numerical permission, output validation and timing as distinct gates. |
| Rely on the compiler host's ambient rounding mode | A reproduced xDSL float serialization case changes a binary32 word across modes. | Require checked compiler-host numerical admission; the general dependency fix remains open. |
| Replace a native test library at a reused path | `dlopen` can return the retained old image, producing a false scheduling failure. The isolated case passes unchanged. | Bind test library paths to the actual LLVM identity; keep the original failure and reproduction. Production loader hardening is a separate infrastructure task. |
| Use BF16 probabilities for both the PV numerator and denominator | The whole SmolVLA experiment fails 122 of 1,600 outputs; its first source group changes 458 integer codes and nine escaping BF16 scales. | Reject the candidate under the existing gate. Preserve the original unrounded denominator and seek a cheaper certificate. |
| Replace one LLVM transform without preserving its composed legalizations | A new TinyLlama recipe omitted the existing late RNE legalization; the public rewrite recognizes 203 eligible chains in its emitted LLVM. | Qualify the complete composed recipe before attributing performance to the new observer representation. |

## Where reusable improvements belong

| Repository | Responsibility |
| --- | --- |
| Merlin | Host code generation, packing, requantization, fusion/sharing, numerical contracts, ownership, dispatch and compilation orchestration |
| model2MLIR | Capture fidelity, quantization correspondence, axis/layout preservation and importer decomposition |
| Target OOT dialect | Device operations and instructions, hardware resources, target ABI, schedules and execution support |

For automatic optimization, phase 0 needs exact graph-to-executable and benchmark
scope accounting. Phase 1 needs composable representation, ownership and source
observation contracts. Phase 2 needs complete costs, analytical bounds calibrated
against hardware, independent regressions and immutable correctness gates.

The [scoped effects and joint attention design](perf_records/smol_scoped_effect_and_joint_attention_plan_20261008.md)
records concrete next changes: bind provider effects to source/object/ABI and
target facts, partition FP rounding epochs at unknown callbacks, and test a
distinct joint numerator/denominator representation under explicit numerical
permission. The [current source/object audit](perf_records/smol_scoped_effect_and_joint_attention_plan_20261008.json)
preserves all 407 SmolVLA calls and proves the two intervening integer callbacks'
instruction census. It does not grant global effect permission, whole accuracy
or a speedup. The selected target has no admitted normalization capability.

The [attention precision and cost comparison](attention_precision_and_cost.md)
records why these captures differ: SmolVLA's 1,024-token vision attention has
about 3,351 times TinyLlama's rectangular attention work in its eight-token
capture, with different floating arithmetic and rounding boundaries. The
selected integer Gemmini and Rocket FPU do not provide native BF16 attention.
The [device audit](perf_records/smol_attention_device_cost_audit_20261008.json)
closes actual command counts and legal operand residency. Requested-byte
reductions remain candidates without measured hardware benefit.

## Recording a new experiment

Add one row with the change, baseline, candidate, engine, exact timed scope,
correctness result, independent cases, promotion decision and receipt link.
Record regressions and refused candidates alongside gains. Preserve original
receipts and add corrections separately. Do not sum section gains, relabel
instruction counts as hardware cycles, or transfer results between different
inputs, numerical policies or runtime bodies without a checked comparison.

The [detailed journey](golden_optimization_journey.md) and
[historical experiment log](golden_optimization_log.md) retain the full record.
Exact billed tokens by optimization or repository are unavailable; aggregate
session meter observations are recorded separately and are not billing data.
