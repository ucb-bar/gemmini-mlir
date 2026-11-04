# Compiler development report

The compiler in `submission/` continues the supplied fresh10 seed. Source types,
axes, addresses, storage effects and hardware capacities select reusable paths;
there is no capsule/model-name dispatch. Merlin supplies host tensor lowering,
preparation, runtime and CPU arithmetic. All accelerator alternatives use the
same terminal encoder and original numerical boundaries.

The final source/file identities are recorded in
`artifacts/implementation-manifest.json` and `artifacts/current-source-manifest.json`.
Those manifests are outside the compiler tree, avoiding a self-referential hash.

## Implemented behavior

* Dense lowering connects checked residency, segmented K pipelines, source-checked
  wide loads, regional storage/schedule selection and independent shared-RHS row
  batching to actual emitted programs. Capacity/refusal paths retain the incumbent.
* Convolution lowering retains source rows or stride-residue planes with joint
  input/RHS capacity planning and two accumulator slots. A narrow-output path
  packs disjoint KX-trimmed source planes using original MVIN commands, preserving
  KY/KX/CI reduction order, raw i32 or original bias/scale/ReLU readout, all tails
  and completion edges. Its profitability predicate is OW<=DIM/2; storage legality
  is separately checked. Paired timing support is strongest at widths seven/eight.
* For one-image packed calls with multiple output-column phases, immutable input
  planes are staged once and retained across phases. Plane/RHS disjointness and
  source/output alias refusal establish legality. Batch-two and single-phase
  controls preserve incumbent emission. If final packed preparation refuses after
  reserving capacity, one read-only retry rebuilds dependent planning products
  with the incumbent budget before emission.
* The typed frontend recognizes exact rank-two through rank-four integer tensor
  permutations, including valid generic affine permutations. It preserves live
  outputs and their original SSA/ABI bindings, lowers the complete command graph
  first, then binds direct plans to actual consumed instruction ranges. Fourteen
  source-derived products pass schema/source/actual-LLVM checks and RISC-V object
  builds; local mapping/graph checks cover 27,744 outputs. Native timing of this
  newly admitted direct path is UNKNOWN.
* Quantized source grouping, literal-boolean LLVM repair, solver/runtime dependency
  initialization and pre-emission proof-timeout fallback preserve the existing
  working frontend and model behavior. Unsupported semantics are explicitly
  refused; contractions are not replaced with CPU arithmetic.

`TUNING_KNOBS.json` exposes consumed alternatives. Dense frontier 32 remains the
incumbent after 64-column regressions. Connected convolution/pool reuse is
implemented in an isolated variant but stays off: two original physical witnesses
regressed. Host reduction and immutable-preparation alternatives did not establish
selection and a complete native gain, so the existing CPU contracts/defaults remain.
Exact selection predicates, support and rejected trials are in
`OPTIMIZATION_RULES.json`, `ITERATION_NOTES.md` and the library inventory.

## Complete-call performance

These comparisons use complete matched kernel-call scopes, including dispatch,
packing commands and completion. Spike values are provisional model estimates;
they do not establish full-model speedups or measured processing-element occupancy.

| Transformation / geometry | Before cycles | After cycles | Evidence |
|---|---:|---:|---|
| Packed rows, convolution H8 CI64 CO80 | 29,474 | 18,541 | Measured gSIM, -37.09% |
| Retained rows, convolution H56 CI64 | 4,540,239 | 2,644,566 | Complete Spike estimate, -41.75% |
| Joint capacity, convolution H14 CI256 | 4,303,253 | 2,716,884 | Complete Spike estimate, -36.86% |
| Stride-residue planes, H56 CI128 stride2 | 888,593 | 682,343 | Complete Spike estimate, -23.21% |
| Packed rows, convolution H7 CI512 CO512 | 2,082,798 | 1,780,448 | Complete Spike estimate, -14.52% |
| Retain those planes across column phases | 1,780,448 | 1,492,822 | Complete Spike estimate, -16.15% |

The last comparison has identical model/facts/scope/input identities and complete
accounting. Requested device reads fall 3,457,024 to 2,435,584 bytes; requested
writes stay 25,088 bytes. Useful MACs remain 115,605,504 and padded-work efficiency
76.5625%. The compute floor is 451,584 cycles at 256 i8 MAC/cycle: the modeled gap
falls 3.94 to 3.31, and useful compute peak attainment rises 25.36% to 30.25%.
These ratios use complete modeled elapsed time as denominator. Resource booking
intervals overlap and must not be summed. Host dispatch/wait attribution, hardware
queue highwater and measured engine/PE occupancy remain UNKNOWN where unpublished.

Detailed paired identities, pass order/options, before/after IR/emission hashes,
local lifetime proofs, decisions and utilization views are under
`artifacts/conv_row_{resident,capacity,strided}/`,
`artifacts/conv_packed_rows_selected/`, `artifacts/packed_input_once/` and
`artifacts/packed-row-utilization.json`. The new multi-phase large product has no
measured RTL result under the selected large-case routing.

## Coverage and qualification

The prior immutable regional snapshot reports 109/109 passing development rows
(including its expected refusal) and 40/40 scalability L2 rows. This is not a
blanket current-source certificate. The packed-row focused request reports 18/18
passing submitted-artifact screens; the large focused request has three L2 passes.
The fallback overlay has three numerical/L2-barrier passes. Parent integration
checks bind all sixteen published ordinary products of the focused request to
unchanged final LLVM; two opaque model products remain UNKNOWN. Six representative
large convolution products match their selected candidate LLVM exactly.

The required physical evidence ledger includes 98 historical qualified
observations. Its binding view finds 94 identical published ordinary LLVM products,
six opaque model identities UNKNOWN and the declared expected refusal. It does
not turn those observations into 98 current certificates. Three original
identity-scale cases pass numerical execution but retain structural conflicts:
the public source requires scale 1.0 while the trace rule demands a nonidentity
readout scale. No scale, dummy configuration or grading contract was changed.
See `artifacts/exact_scale_contract_conflict.json` and
`artifacts/current-regional-physical-evidence.json`. Development screens do not
certify missing original RTL tiers. The PV03 calibration question remains pending
trusted review; no dispatch authority or accuracy fit is inferred.

| Visible program | Exact kernel replay calls | Complete model evidence |
|---|---:|---|
| ResNet50 model | 45/45, 24 forms | Full visible numerical gate passes; exact 1,000 outputs |
| TinyLlama prefill | 22/22, 8 forms | Full gate pending |
| TinyLlama decode | 22/22, 8 forms | Full gate pending |
| SmolVLA prefix_encode | 36/36, 14 forms | Full gate pending |
| SmolVLA flow_denoise | 34/34, 14 forms | Full gate pending |
| SmolVLA action_decode | 0 calls, 0 contraction forms | Full gate pending; no replay proof |

Each model receipt names its own source snapshot in
`artifacts/model-packed-final-*.json`; a newer source digest alone does not bind
an older replay. The full gate is separately pinned in `artifacts/final-evidence.json`.
ResNet's gate reports 314,288,496 measured cycles; its engine provenance is not
published, so this is neither a gSIM comparison nor a demonstrated model speedup.
Kernel replay and contraction census do not prove complete host/control execution
or all-operator placement. The other five full-program results remain pending.

`artifacts/whole-source-coverage-map.json` groups the six public source profiles
and all 68 selected model forms by shape, numerical boundary and static source
multiplicity. Host time, dynamic invocation counts and all-operator placement
remain UNKNOWN where absent. Vocabulary-projection and posted-load timing gaps
remain visible. Remaining work is supported timing for fresh narrow geometries,
connected-region reuse that improves complete physical latency, and missing
full-model/host attribution; the working incumbent is retained for each class.
