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

## Compiler improvements awaiting performance validation

| Change | Verified so far | Remaining gate | Owner |
| --- | --- | --- | --- |
| Share scalar approximation tables before normal lowering | All 22 current TinyLlama observers compile with one 196,608-byte table; uncertified cells retain source fallback. The candidate currently disables a conflicting lane-packet transform. | Numerical replay, composition with lane scheduling, the original whole output gate, linked runtime predicate and complete timing. | Merlin |
| Preserve source joins through common-subexpression elimination | The current SmolVLA structural experiment merges three equivalent Q/K/V quantizers into one while retaining their source identities and all original calls. | Final public compiler integration, effect admission, original whole output gate and executable timing. | Merlin |
| Project unused immutable weight arguments | Current prepared ResNet accounting identifies 53 removable obsolete parameter tensors, totaling 106,240 logical bytes. Captured buffers remain until immutable ownership is proved. Independent small normal native/Spike execution passes. | Released package qualification, actual whole delivery and measured benefit. | Merlin |
| Generate compact convolution candidates from typed command traces | Source/resource legality, independent cases, upstream lowering and zero-FSM object checks pass. | A complete cost selector and a fresh stock whole run. | OOT dialect |

The table-sharing [normal compilation](perf_records/tiny_current_normal_scalar_all22_20261008.json)
and [coefficient coverage](perf_records/tiny_current_normal_scalar_coverage_20261008.json)
are separate from the [installed compiler qualification](perf_records/scalar_carrier_current_normal_installed_20261008.json).
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
