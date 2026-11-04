# General Gemmini compiler

This continues the supplied seed in one compiler, with editable shared modules in `compiler_library/` and a common terminal ISA encoder. The frontend carries typed source axes, quantization, physical addresses, initialized accumulators, storage ownership, live outputs and completion edges through their consuming transformations. Merlin supplies host tensor lowering, arithmetic, preparation and runtime integration.

Selected paths include checked capacity/retained/segmented dense schedules, wide operand transfers, grouped RHS prefetch, ready readout, resource-admitted geometry selection, exact convolution row runs and flattened gathers, resident weights, compute/store epochs, shared-input/head batching, signed pooling, exact residual/readout epilogues, and integer transpose/slice/concat/insertion. `tuning.json` records selected alternatives; `TUNING_KNOBS.json` exposes reusable trials. Selection uses source geometry, layout, numerics and resource facts rather than capsule identities. Incumbents and alternatives use the same terminal encoder.

## Validation and remaining obligations

Final development checks pass 109/109 core L2 barriers (including one declared refusal), 40/40 scalability L2 cases, and all three public source programs at L0/L1/L2. The scalability formal completion flag remains false because 33 whole-program ELF members skip inapplicable L0/L1. Core checks contain 102 explicit native numerical passes (94 executable originals and eight calibration cases); source/model screens do not prove full-model execution. Detailed results are in workspace `artifacts/final-verification.json`.

Fresh model-package rebuilds pass ResNet (24/24 kernels, 45/45 visible calls) and TinyLlama (16/16 kernels, 44/44 calls). SmolVLA's first rebuild built 27/28 kernels with 69/69 executed calls exact; one prefix product had optional streaming proof unavailability. The final narrow fix retains that refusal and constructs the original independently verified compact accelerator schedule. Fresh source compilation and forced-refusal verification pass; the final SmolVLA native replay is pending in `artifacts/final3-smolvla-model-check.json`. The preceding integrated snapshot built all 68 kernels and replayed 159 calls exactly. The table records that demonstrated package coverage; it does not assert the pending final replay passed.

The mandatory proof solver now resolves through the explicitly supplied public runtime when absent from the compiler's Python environment. Missing solver/proof still fails; numerical and storage verifiers are unchanged. After this dependency fix, 135/135 executable core/scalability LLVM outputs match retained evidence exactly. The later narrow optional-route catch preserves two checked representative outputs byte-for-byte. No CPU fallback is introduced.

Package checks establish device-kernel builds/replays and a contraction census, not whole-program compile/link, sealed-input qualification, all-operation placement or full-chain execution.

| Program | Kernels built | Exact visible calls | Device contractions / census | Full-chain execution and cycles |
|---|---:|---:|---:|---|
| ResNet50 model | 24 | 45/45 | 54/54 | UNKNOWN |
| TinyLlama prefill | 8 | 22/22 | 3675/3675 | UNKNOWN |
| TinyLlama decode | 8 | 22/22 | 3675/3675 | UNKNOWN |
| SmolVLA prefix encode | 14 | 36/36 | 3259/3259 | UNKNOWN |
| SmolVLA flow denoise | 14 | 34/34 | 2036/2036 | UNKNOWN |
| SmolVLA action decode | 0 | 0/0 | 0/0 | UNKNOWN |

Every full-chain run array is empty. Action decode has no device replay image; its empty contraction census proves no execution. Exact identities/receipts are in workspace `artifacts/current6-model-program-status.json`, `artifacts/final2-*-model-check.json` and the pending final SmolVLA replay. Final release status in workspace artifacts records the collected outcome.

Manual gSIM numerical witnesses cover 94 executable originals; six source-screening originals and the declared refusal are separate. Exact emitted-LLVM correspondence is reconciled separately from physical-wrapper ELF certification. Three edge originals (`CH_residual_linear_edge_m17k17n17`, `CH_residual_conv_edge_m17k17n17`, `CH_sibling_feed_edge_m17k17h17n17`) declare identity source scales but require a non-identity scale instruction: their physical numerics pass while those structural requirements remain unsatisfied. The compiler preserves source arithmetic. Formal final certification and full-model execution remain pending the selected campaign window.

## Selected performance evidence

Pairs compare complete calls/regions with matched inputs, timing provider, model revision, hardware domain and scope. Spike figures are provisional modeled elapsed cycles. gSIM measures only its named original; large physical rankings remain UNKNOWN.

| Change and representative | Evidence | Before → after cycles |
|---|---|---:|
| Checked full-K/segmented panel expansion, GU M16 K2048 N11264 | Complete Spike model | 4,952,245 → 3,740,506 (−24.47%) |
| Same rule, reserved JM M1 K2048 N2048 | Complete Spike model | 942,109 → 718,259 (−23.76%) |
| Same rule, ragged PD M1 K2048 N1000 | Complete Spike model | 542,140 → 440,078 (−18.83%) |
| Same rule, output projection M16 K2048 N2048 | Complete Spike model | 953,623 → 732,907 (−23.15%) |
| Same rule, QKV M16 K2048 N2560 | Complete Spike model | 1,172,545 → 896,658 (−23.53%) |
| Same rule, JT M1 K2048 N32000 | Complete Spike model | 13,879,272 → 10,385,121 (−25.18%) |
| Grouped RHS prefetch/transfers, TL-down | Complete Spike model | 2,856,759 → 1,838,040 (−35.66%) |
| Same bundle, connected sibling region | Complete Spike model | 8,024,455 → 6,980,566 (−13.01%) |
| Capacity-bounded convolution compute groups, batch4 56×56 | Complete Spike model | 3,194,149 → 3,141,720 (−1.64%) |
| Same compute-group rule, small original SY | Measured gSIM | 21,743 → 21,328 (−1.91%) |
| Contiguous readout, original convolution tail | Measured gSIM | 1,760 → 1,159 (−34.15%) |
| Direct tile-uniform dense column readout, aligned original | Measured gSIM | 1,239 → 709 (−42.78%) |

The segmented-family policy changes six of 21 surveyed dense scalability interfaces, leaving 59 original and five calibration interfaces byte-identical. All paired outputs pass. The JT long-vocabulary matched complete pair passes L2; request-bound cycles are 4,096,128 (gap 2.54) and modeled useful-compute peak attainment is 2.47%. These are model diagnostics, not physical occupancy. The four-row convolution rule has six complete model pairs and three reserved holdouts. Its small physical witness admits three rows; actual four-row physical ranking is UNKNOWN.

GU useful-compute peak attainment rises 29.11% → 38.55%: source MACs divided by 256 MACs/cycle times modeled complete-call cycles. Padding efficiency remains 100%. Selected elapsed/ideal read-request cycles is about 2.59, a partial structural bound rather than a complete physical roofline. JM attainment rises 1.74% → 2.28% with only 6.25% useful padded arithmetic. Modeled resource reservations and actor overlap are diagnostic, overlap in time and must not be summed. They do not measure PE occupancy. Unattributed queue, host-wait and lifetime costs stay UNKNOWN. Full views: `artifacts/dense-family-system-views.json`, `artifacts/system-views/summary.json`.

Selected owner-built ResNet connected execution remains 97,072,127 modeled cycles, control ELF `889136f4df12c14dd733843171be255a556f7bb412e5079c626afe8c551e0e3a`, about 189.9× its compute bound. Isolated dense gains establish no connected-region gain. Tiny connected numerics pass but timing is UNKNOWN at `same_elf_observation`: the owner reports unsupported or indirect completion effects. Both remain visible in `artifacts/current6-connected-summary.json`.

## Trials retained without promotion

Two-slot LHS pipelining regresses large affected cases. Alternate K caps and several storage/placement choices regress or tie. Segmented ready readout and column-palette reuse improve less than 0.1% on tested cases and remain off. Wider segmented panels remain optional: six small complete-model gains of 0.16–0.64% are outweighed by a 3.56% ragged-output regression; JT timing remains UNKNOWN. No new cutoff was learned from those tiny differences. Rejected trials retain incumbents and exact source/emission/result identities.

Host options use Merlin source/CPU contracts. Roundeven is slightly slower; bounded RNE timing is opaque. Preparation CSE, dense input views and optional movement composition leave owner-built final executables unchanged, so native gains are not claimed. Layout propagation and allocation-lifetime reuse explicitly report `permission_not_admitted`. Host profile ticks are not elapsed cycles. Optional frontend transformations without native application evidence remain off.

Source placement extends beyond contractions, while unresolved floating/nonlinear seams and host-region eligibility remain explicit in the workspace census. Calibration has same-executable median absolute error 11.87%, p90 29.32%, maximum 97.7%; this does not qualify a large-workload fit or turn small modeled differences into measured wins.

Workspace `ITERATION_NOTES.md`, `OPTIMIZATION_RULES.json`, `artifacts/` and `submission/docs/library_inventory.md` retain ownership, actual pass order/options, source/transformed-product hashes, numerics, timing provenance, rejected alternatives and remaining coverage. Evidence paths refer to the shared workspace, not extra executable package members.
