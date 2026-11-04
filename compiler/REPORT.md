# Fresh10 general Gemmini compiler

This is an authored target backend over the selected editable public Merlin compiler. It preserves source SSA axes, original source ownership, buffer effects, numerical boundaries and completion edges through planning, and emits ordinary explicit Gemmini commands through one LLVM terminal encoder. It does not use hardware loop descriptors or source-name dispatch.

## Evidence status

The selected inventory contains 101 original correctness cases, eight calibration cases, 40 scalability cases and three connected programs. Every original retains its required physical tier. Calibration and correctness permissions are separate. Sealed inputs and shapes remain the evaluator's responsibility; the results below concern visible inputs.

Milestone027 passed numerical execution on all 109 core cases, including the declared expected refusal, and all three connected programs. Its 101 individual original-case requests each passed the required gSIM qualification (`../artifacts/milestone027/physical_summary.json`). The simultaneous core response was captured before these individual certificates landed and reports only one certificate; it must not be read as a physical failure. The 40-case scalability refresh is still running; the earlier complete sweep passed all 40 numerically. The immutable source manifest and exact request records are in `../artifacts/incumbent027_source_manifest.json` and `../artifacts/milestone027/`. Current-source certificate reuse is checked separately after integration.

Native model030 built all 68 device kernels and matched all 159 replay calls exactly. It validates the integrated compatibility fix for both xDSL ICmp constraint representations and the packaged z3-solver dependency. The solver remains an ordinary host compiler dependency in `tool_dependencies/`, outside the editable Merlin namespace; its MIT license, version, source provenance and per-file hashes are retained. Exact native reports are in `../artifacts/native_compat028/{resnet50,tiny_llama,smolvla}_packaged.json`, source SHA256 `da82cf57f14731e25c505e8da86c2622ab448d214719f31c3a60bcddd293d06b`. Earlier native026 build refusals are resolved; they were not numerical mismatches.

| Model/program | Kernels built (030) | Exact replay calls (030) | Contracted operator census (030) |
| --- | ---: | ---: | ---: |
| ResNet50 / model | 24 | 45 | 54/54 device |
| TinyLlama / prefill | 8 | 22 | 3675/3675 device |
| TinyLlama / decode | 8 | 22 | 3675/3675 device |
| SmolVLA / prefix_encode | 14 | 36 | 3259/3259 device |
| SmolVLA / flow_denoise | 14 | 34 | 2036/2036 device |
| SmolVLA / action_decode | 0 | 0 | 0/0; empty device census |

The fixed-denominator model census covers the contract's selected operators. It does not establish eligibility or placement for every host operation. Current source profiles retain host operations and UNKNOWN numerical work and hardware eligibility. Kernel replay calls and functional-engine counts are not whole-chain elapsed cycles.

## Construction

`frontend.py`, `linalg_lowering.py`, `model_regions.py` and the source analysis modules construct source-derived command and connected-region plans. Original Merlin tensor preparation, CPU arithmetic, bufferization and runtime transport implement host regions. The target adapter does not substitute host arithmetic for required accelerator products. `backend.py` connects actual source products to checked dense, convolution, readout, residual, pooling and movement schedules; `isa.py` encodes all alternatives.

Dense planning consumes real indexed schedule IR, capacity-derived private row assignments, checked panel/block transforms and selected RTL completion facts. Convolution consumes source NHWC window row runs, real input retention and capacity-checked completion epochs. Per-column readout retains exact f32 scale words, RNE and source clamp/activation order. Bias is an accumulator seed; scalar and column exits preserve source integer behavior. A source-proven 1x1 view feeds the shared dense path without a packing conversion.

Explicit repeated resident handles now carry source lifetime proofs, including packing, eviction, actual reuse and intervening writes. Their required retention remains enabled when legal even when the optional performance preference is off. Identity residual readout uses a consumed doubled integer representation and half-scale readout, exhaustively checked on all signed-byte input pairs, with the source output unchanged. These are correctness choices with visible cost, not speedup claims.

Optional source preparation reuses actual immutable RHS layout cones before partitioning. On the complete attention source, explicit `prepack_weights` removes 52 CPU preparation operations and emits the changed program while preserving original ordinals, output writers and ABI. A representative retained/removed preparation pair matches bitwise over three native trials. The full attention program030 subsequently passed all numerical outputs through the broker; complete host-effect timing remains UNKNOWN, so no cycle reduction is established. Offline weight bundles refuse the absent manifest, payload and caller contract. The option remains off.

## Performance results

The following are complete-call or complete-region comparisons with unchanged checked outputs, matched timing provider/model/input/scope, and complete event accounting. Spike numbers are **provisional modeled cycles**, not measured accelerator occupancy. Exact identities, options, before/after source/IR/program hashes and decisions are retained under `../artifacts/experiments/`.

| Transformation and representative | Before cycles | After cycles | Evidence |
| --- | ---: | ---: | --- |
| Dense panels + epochs, PD ResNet m49 k512 n2048 | 5,246,646 | 660,403 | complete Spike model; training + five held-out shapes support bounded policy |
| Dense panels + epochs, TinyLlama down m16 k5632 n2048 | 15,869,778 | 3,325,389 | complete Spike model |
| Dense panels + epochs, ZA square dense | 7,905,363 | 1,165,087 | complete Spike model |
| Convolution retention + epochs, RN c512 h7 | 39,457,195 | 7,993,515 | complete Spike model; additional training and held-out convolution geometries |
| Held-out convolution, ZA c128 h28 batch4 | 29,694,343 | 12,700,560 | complete Spike model |
| Connected prepared ResNet region | 124,925,291 | 115,833,901 | complete Spike model including region host work; 7.28% reduction |
| Per-column producer/readout, RN stride2 1x1 | 106,598,054 | 13,473,245 | complete Spike model |
| Quantized-once conv-pool, large ResNet stem region | 302,932,816 | 51,469,086 | complete Spike model; all region outputs checked |
| Small conv-pool epochs, edge region | 329,111 | 126,650 | complete Spike model |
| Pointwise shared dense view, c12 n3 | 1,587 | 865 | complete Spike model; seven affected source geometries all improve |
| Pointwise shared dense view, aligned column readout | 1,304 | 1,093 | complete Spike model |
| Dense epochs required original witness | 47,752 | 14,514 | measured gSIM, matched witness only |
| Convolution epochs required original witness | 569,431 | 158,072 | measured gSIM, matched witness only |

The large pooled region's complete modeled compute floor is 460,992 cycles, so 51,469,086 elapsed cycles are 111.65 times this compute-only floor. Useful compute peak attainment is 0.896% over complete-call modeled elapsed time; useful/padded work is 18.75%. The complete roofline is partially attributed: this compute-only denominator is not a complete system bound. Model resource reservations overlap and are not summed into elapsed cycles. Host preparation/wait attribution and measured PE occupancy remain UNKNOWN.

Rejected or retained experimental alternatives remain explicit knobs. Materialized sibling scalar readout passed numerics and a required original gSIM check but increased the small complete model from 5,102 to 7,788 cycles; it stays off. Global optional RHS retention regressed against the selected panel incumbent; source-mandated retention is separate. Wider dense scheduling, shared RHS epochs and raw 1x1 convolution epochs have promising individual complete results but limited neighboring or held-out coverage, so remain opt-in. UNKNOWN large-case timing does not count as a regression or a win. Order-blind fast estimates do not decide transfer-width, overlap or issue-order changes.

## Limits and reproduction

Use the four commands in `manifest.yaml`. Example: `./gemmini-opt --convert-iface-to-gemmini --emit-target-artifact --emit-command-buffer=plan.json input.mlir`. `tuning_settings.json` selects the tested defaults; `TUNING_KNOBS.json` preserves alternatives. Actual receipts in the emitted command buffer identify the consumed schedule, numerical route, storage and source ownership.

Complete timing remains UNKNOWN for some wide and host-heavy connected programs. No whole-model elapsed-cycle improvement is inferred from isolated GEMMs, kernel replay counts or layout-operation removals. The 101-original physical campaign and six-program native refresh are complete for their pinned sources. The current-source official checkpoint and source-normalized scale-emission ladder remain separate obligations. The ladder currently returns an unavailable StageGateError without a diagnostic; bounded object compilation and original-emission diagnostics are isolating it. No private evaluator code, answers or reference performance was accessed.

The selected software declaration is SHA256 `3c1b8aa1c77775e1c0e590af871f05e038743d7000a99860fb12843f7053372f`; public compiler selection manifest is `1a5e1e0b125cec53a0b1027024db563a3157627e470af4c90ab0067c330d10fe`. Hardware facts, command ordering and every timing model/provider identity are packaged in source receipts and experiment records. Provisional modeled rankings are not a fitted accuracy certificate.
