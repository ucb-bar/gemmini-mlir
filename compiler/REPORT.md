| Model | Program | Compile/link evidence | Numerical evidence | Contraction census |
| --- | --- | --- | --- | --- |
| resnet50 | `model` | 24 device kernels built, 0 refused; full-program link UNKNOWN | 45 bit-exact Spike kernel calls | 54/54 source contractions |
| smolvla | `prefix_encode` | 14 device kernels built, 0 refused; full-program link UNKNOWN | 36 bit-exact Spike kernel calls | 3259/3259 source contractions |
| smolvla | `flow_denoise` | 14 device kernels built, 0 refused; full-program link UNKNOWN | 34 bit-exact Spike kernel calls | 2036/2036 source contractions |
| smolvla | `action_decode` | 0 device kernels built, 0 refused; full-program link UNKNOWN | No device kernels; full host execution UNKNOWN | 0/0 (no contractions) |
| tiny_llama | `prefill` | 8 device kernels built, 0 refused; full-program link UNKNOWN | 22 bit-exact Spike kernel calls | 3675/3675 source contractions |
| tiny_llama | `decode` | 8 device kernels built, 0 refused; full-program link UNKNOWN | 22 bit-exact Spike kernel calls | 3675/3675 source contractions |

These visible model checks describe immutable compiler tree `1c7ac1828c456e074ecd3b626183cf6a0b79a08a25f8027f02ed7833d2f855ab` and binding receipt `4389226998f53bffba5e134dd40ba3475d6546aacd29cb25ef18bee7921c484f`. They ran before subsequent dense, pool and head-row promotions. The model tool reports build, exact-kernel and model-reference gates as pass. The last gate is inferred by that tool from kernel exactness. It is not independent execution of the complete emitted host program. Each nonempty replay used one visible input and no shape variants. Sealed evidence is unavailable here.

The census denominator counts admitted source contractions. It does not count all graph operations or establish accelerator coverage of host arithmetic. The reported device MAC fraction is 1.0 for nonempty programs; action_decode has no contraction denominator. Full-program linking, complete-chain output comparison and complete-chain elapsed cycles remain UNKNOWN in these reports.

Required physical correctness: 74/101 distinct originals have observed passing gSIM results, comprising 75 executable certificates. These bind exact executable and input identities. Their reuse for current bytes requires an exact executable join. Pending, unavailable and unrun physical tiers are not passes. This ledger is separate from the model checks and from calibration; no fitted calibration was claimed.

The compiler resides in `submission/`. Its manifest exposes parse, interface lowering, command-buffer emission and LLVM lowering through `gemmini-opt`. The authored frontend handles typed interface programs and source linalg regions. Shared source, integer, layout, schedule and effect proofs feed target adapters; `encoding.py` is the common terminal instruction encoder. Dense and convolution alternatives preserve axes, storage effects and completion edges through their checked transformations. Host tensor preparation and CPU arithmetic use the editable Merlin library.

Selected performance evidence is provisional complete-call Spike modeling, including the reported host/device/completion window. It is not measured RTL performance or a model accuracy certificate. Each comparison below pins its own matched provider, input, domain and scope; unrelated rows are not additive.

| Selected region | Matched modeled cycles | Evidence |
| --- | ---: | --- |
| Dense M12544 K64 N256 | 2,640,760 → 1,387,560 | `../ab/dense-resident-readout/assessment.json` |
| Dense M12544 K256 N64 | 2,004,894 → 1,686,334 | same matched experiment |
| Dense M3136 K512 N128, heldout | 1,412,596 → 1,285,191 | same matched experiment |
| Narrow long-K frontier, K4096 / K4112 / heldout K8192 | 25,425 → 23,297 / 31,879 → 25,234 / 50,996 → 42,745 | `../ab/dense-frontier-narrow/promotion-assessment.json` |
| Large convolution/pool region | 63,293,786 → 43,391,833 | `../ab/pool-retain/promotion.json` |
| Same retained pool region, checked compute epochs | 43,391,833 → 27,085,200 | `../ab/pool-epochs/promotion.json` |
| Decode independent head products | 1,543,665 → 719,551 | `../ab/attention-head-rows/promotion_pin.json` |
| Multirow shared-RHS heads, checked ready readout | 1,214,906 → 1,141,199 | `../ab/attention-kv-ready/promotion_pin.json` |
| Source residual program, host elementwise fusion | 3,278,167 → 2,983,919 | `../ab/host-fusion/assessment.json` |
| Connected ResNet representative, host elementwise fusion | 115,148,971 → 97,066,568 | `../ab/resnet-host-fusion/assessment.json` |
| Attention QK, private zero-fill elimination | 43,238 → 23,053 | `../ab/zero-fill/experiment.json` |

The first dense source has actual N256 despite its public name ending n64. The head-row rule has one timed geometry; native ragged neighbors have UNKNOWN timing. Host fusion timing is explicitly untuned. Host bounded-RNE, and source preparation sharing retain optional status where complete/fast timing is missing. Guarded private zero-fill elimination is selected; the original splat route still lacks ordinary consumption evidence. Rejected rectangular panels, full-K wide transfers and compact convolution gathers remain available as alternatives rather than replacing tested incumbents.

Useful compute attainment uses useful MACs divided by selected array peak times complete-call cycles. Padding efficiency is useful MACs divided by issued padded MACs. Neither is measured processing-element occupancy. Detailed resource reservations are modeled and can overlap; they are never summed into elapsed time. Host wait attribution, unreported engine activity and missing queue/lifetime views remain UNKNOWN. Region-specific denominators and roofline gaps are retained in the experiment receipts.

The latest merged grade rows span older source versions. They still report unresolved residual/sibling chain broker failures and the distinct-column scale input-boundary failure. Independent native execution of all four published residual chains and both sibling chains matches public-source arithmetic, preserves every output and leaves inputs unchanged; this does not overwrite the broker failures. No fake nonidentity scale configuration or grading-contract edits were introduced. Exact public-source and executable identities are in `../ab/connected-native/` and `../ab/sibling-native/`.

Detailed evidence: `../ab/model-integrated-{resnet50,tiny_llama,smolvla}.json`, `../ab/model-program-evidence.json`, `../ab/physical-evidence.json`, `../OPTIMIZATION_RULES.json`, `docs/library_inventory.md`, and `../ITERATION_NOTES.md`. These files separate measured certificates, modeled cycle choices, numerical execution, structural diagnostics and missing evidence.
