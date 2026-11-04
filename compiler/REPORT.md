# Authored Gemmini compiler — evidence as of 22:25 UTC

| Model | Program | Compile/link evidence | Numerical evidence | Contraction census |
|---|---|---|---|---|
| ResNet50 | model | 24 device kernels built | 45/45 visible kernel calls exact | 54/54 |
| TinyLlama | prefill | 8 device kernels built | 22/22 visible kernel calls exact | 3675/3675 |
| TinyLlama | decode | 8 device kernels built | 22/22 visible kernel calls exact | 3675/3675 |
| SmolVLA | prefix_encode | 14 device kernels built | 36/36 visible kernel calls exact | 3259/3259 |
| SmolVLA | flow_denoise | 14 device kernels built | 34/34 visible kernel calls exact | 2036/2036 |
| SmolVLA | action_decode | 0 device kernels required | no device replay obligation | 0/0 |

These are the model endpoint's final visible isolated kernel replay results, all passing on compiler source a0b8758c49f34bd856f65358a531dddd47a8567c42ff1759f55315348456847f. They do not establish complete host-program linking, end-to-end execution, sealed-input results or all-operator accelerator coverage. Exact endpoint records are in docs/evidence/final-visible-model-replays.json.

## Construction

The compiler is authored in backend/, compiler.py and the selected editable compiler_library/. One terminal encoder handles compact, retained, segmented, streaming, spatial convolution, shared products, movement, standalone epilogues and pooling alternatives. Typed source plans retain axes, byte/element views, initialization, output observers, local ownership and completion before emission. Placement derives from typed source bodies and selected numerical/ISA facts. Host tensor preparation, floating arithmetic, runtime and CPU lowering use Merlin's existing implementation.

Selected paths include capacity-bounded retained dense panels, checked load/compute/store epochs, exact column-palette readout, flattened spatial convolution with retained RHS N phases, horizontal convolution runs, standalone striped accelerator pooling, shared-input products and integer layout transfers. Capacity refusals preserve a supported alternative; no contraction CPU fallback is used. Full-head batches consume the selected counted-batch schedule and ordinary dense encoder, but the public connected TinyLlama source does not select a whole-head rewrite. Explicit alternatives and rejected trials remain in TUNING_KNOBS.json and workspace variants/.

## Physical capsule evidence

Physical qualification belongs to exact executable and source identities. Current byte reconciliation, individual gSIM responses and native-owner group identities are retained in workspace ab/. Unrun or unavailable tiers remain incomplete. The standalone pooling partial-tail and ranked-batch adapters have additionally passed required original L3 checks. The frozen final compiler implementation manifest is 6ef115b86f6f48d33b09d3d4e4c1bc8889970779a338498d7f048cf4f67672bb (1151 implementation/library/tuning/data files; an authored manifest, not a grading digest). Final canonical emission reconciles 87 original programs directly with passing L3 source/LLVM identities. Four source-owner group programs have also passed fresh L3 execution from the frozen compiler, with matching earlier group source/LLVM/object/toolchain identities. See docs/evidence/final-original-physical-closure.json and final-native-physical-results.json.

Three original edge capsules retain a structural conflict: CH_residual_conv_edge_m17k17n17, CH_residual_linear_edge_m17k17n17 and CH_sibling_feed_edge_m17k17h17n17 declare identity scales while their trace check requires a nonidentity CONFIG_ST scale. Their numerical gSIM executions pass. The compiler preserves the declared scales. All six original source/model cases have separately passed fresh required candidate-native L0/L1/L2/L3 verification from the frozen compiler (package 43fb45cf3ef6f608d7e578c1b1cc7dd8b45fd9d319dde30f11d83a0479f44d80). Their feedback is opaque and supplies no executable digest; these are candidate-only native qualification results, not general transformation equivalence. Together these records cover 97 positive originals; the three structural conflicts remain unqualified, and the expected refusal is separate.

## Measured complete-call comparisons

All rows below use matched original inputs, gSIM simulator identity and toolchain; lower cycles are better. These measurements do not establish large-model speedups.

| Original witness / transformation | Before cycles | After cycles | Reduction |
|---|---:|---:|---:|
| CI64 3×3 convolution, spatial full-K epochs | 70615 | 35755 | 49.37% |
| CI24 3×3 convolution, spatial full-K epochs | 9199 | 6480 | 29.56% |
| Convolution operand-capacity boundary, horizontal row runs | 569191 | 194666 | 65.80% |
| Edge convolution/pool, striped pooling | 82061 | 80460 | 1.95% |
| Residual convolution/pool, striped pooling | 203253 | 199523 | 1.84% |
| Bias/scale/ReLU/pool, striped pooling | 1066 | 797 | 25.23% |

Full records: workspace ab/conv-spatial-epochs-measured-pairs.json, ab/trial-conv-horizontal-measured.json and ab/pool-stripes-measured-pairs.json. These contain provider, source, lowered-program and executable identities. Model-derived utilization and reservation intervals are diagnostics retained in ab/utilization-current-experiments.json; physical processing-element occupancy and host wait attribution are unknown. Overlapping resource intervals are not summed as elapsed cycles.

## Remaining evidence

The official qa/verdict.json still contains an 18:38 rejected snapshot; that forbidden data directory was removed earlier. The latest completed automatic grade describes source 887d221f..., rather than the frozen compiler. It reports pending physical gates. The records here are explicit per-snapshot evidence.

The latest immutable milestone passed the numerical checks for 100 positive originals, eight calibration cases, 40 scalability cases and three connected programs, plus the expected refusal. All ten final native physical replays have passed. Three originals remain structurally unqualified, and full host-program linking, sealed model inputs and all-operator coverage remain unestablished.

The final compiler keeps retained dense scheduling because the segmented alternative regresses the K8208 spill case and lacks complete policy identity/validation support. Source-specific fusion obligations identify runtime floating channel parameters, reductions, nonlinear operations and permutations; these are proof gaps. The authored placement census records device/unknown regions of 16/120 for the public CNN, 6/12 for decoder and 9/31 for multimodal, with 41950/43776/10676 static boundary bytes respectively. These are static counts, separate from transferred bytes and host time. The source quantization census retains 25 unresolved fusion proofs and six boundaries without a supported integer path.

Some complete timing models remain UNKNOWN due to posted-load mapping or opaque host effects. Calibrated fast estimates are labeled development evidence; they are not reported as measured performance.

No private evaluator, reference kernel, answer or reference performance was accessed. Hardware facts SHA256 b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af. Target completion and numerical legality are separate from these execution observations.
