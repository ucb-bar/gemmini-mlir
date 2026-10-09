# TinyLlama fresh shared compiler qualification and complete command cost

Recorded 2026-10-07. These are additive successor records; earlier source, timing,
failure and package seals retain their original bytes.

## Shared compiler delivery

Fresh installed Merlin `cdf117e3db27629d057885091e3886c3d298a92d` compiled the original
8-token, 22-layer capture through `spike_model.build`, the ordinary OOT catalog,
the public prepared-source callback, existing typed full-write/borrowed writer
contracts and the explicit host LLVM callback. The installed package and all
runtime resources were checked against their independent package identity.

All 155 source bindings, tensor types, regions, source ordinals and kernel symbols
match the existing delivery. The regenerated target kernel object is byte exact.
Every object leaf was newly built: model, device kernels, adapters/borrowed bridge,
runtime, original and hoisted weights, scalar observation table/guards, CRT and
harness. No historical object or ELF was linked. Final ELF:
`a29c1c76772d8e1292841f48550c2a46bc8b7294724bc29c0b667c6073245e7c`.

The native full original model and the target Spike execution both pass all
256,000 output values exactly against the original champion. The unchanged Torch
gate is `atol=0.03125`, `rtol=0.02`; maximum absolute difference is
`9.5367431640625e-06`. The canonical full output f32le digest is
`ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3`.
Final executable-section audit finds no FSM instructions and no unknown custom
instructions. Source and target memory/rank checks pass.

The fresh functional ROI reports 122,689,756 instructions. This is not a hardware
cycle measurement. Historical FireSim stock2085 remains 378,946,263 cycles on ELF
`0518189d448bfb8dd387397258bb263218121de832f557c02a57652a7ca4a3c2`.
That result does not transfer to the freshly built ELF. No FireSim job was submitted.

Delivery packet: `current_cdf_all_leaf_v4/delivery_packet.json`, SHA256
`b692728b685f481e9b79b61330823387782dcf49d30e33951e1c897a5a9fa77a`,
1,348 exact pins.

### Selection and API limits

The experiment still selects a retained typed expression recipe and two scalar
LLVM source ingredients. Every newly emitted arithmetic/use/observer tree is
structurally rebound with explicit effects; provenance identifiers do not select
production replacements. This qualifies reusable compiler mechanisms and a source
delivery, not automatic performance selection.

CDF's public model builder does not yet forward `MaskEffectContract` to its normal
lowering stage. The immutable CDF experiment uses the supported explicit late hook
to call installed `lower_to_llvm_ir` with the original feature set and typed effects.
The normal incoming/fresh public forward ABI is checked. Root is adding typed
contract forwarding to the shared model and both backend builders; the next
immutable successor can use that common API and remove this extra lowering.

Early failed successors are preserved: missing explicitly selected datatype facts,
an early unfused source view that could not prove a closed scalar expression, and
an experiment's incorrect 2GB memory selection. The current run explicitly pins
stock capability facts and the historical 16GB memory/256MB arena/16MB stack recipe.
These failures are not whole-model accuracy failures, and the memory selection was
an experiment error rather than a Merlin defect.

## Retained N command batching: complete cost

The normal catalog option derives legality from canonical i8×i8→i32 semantics,
dense layout, M/N/K geometry and declared device resources. The default module,
manifest, public bindings and target object remain byte exact. Unsupported
geometry or resource conditions retain the original implementation with an
explicit refusal. No model name, provenance identifier or golden selects a
production schedule. The option is still selected manually in the experiment.

Original M8/K2048/N5632 input data and complete config, all loads, command issue,
readback and fence were timed together. Four balanced ABBA arms pass every
180,224 i32 output word, complete read-only inputs and guards. Local normal GSIM
timing is:

| Variant | Arm cycles | Mean cycles |
| --- | --- | --- |
| Original | 2,017,932; 2,015,373 | 2,016,652.5 |
| Batch16 | 2,014,776; 2,009,572 | 2,012,174.0 |

Mean change is −0.22207594%, much smaller than the complete native retired
instruction change of −14.0806%. Batch16 reduces executed stack accesses by
84.79%; target command counts and DMA payload are unchanged. Individual-arm
spread is large relative to the small mean difference. This is weak local
evidence, not a strong promotion signal or a stock FireSim result.

Inference: reducing executed CPU work can be hidden by memory/accelerator service
and overlap. The cost model needs a measured service/overlap decomposition. It
cannot translate the instruction change to cycles or multiply this local delta
by 44 repeated model calls. Batch1's preserved +15.6256% instruction regression,
despite much fewer spills, also rules out a spill-only ranking model.

Complete cost packet: `ws_retained_n_catalog/complete_cost_packet.json`, SHA256
`6b5c818feef7a0aa4965283d8677d15a68b30dc3861e4dbacc815586cc97f4ed`,
599 exact pins. Engine completes 54,488,525 cycles in 3,956.53 seconds. Engine
identity and source lineage are pinned. No default or whole-model promotion.

## Common tooling suggested for the automatic phases

- Phase0: expose numeric/effect/ownership contracts, actual CPU and device CFG,
  command address order, resources, exact output checks and complete admission
  budget floors. Include before/after input validation in the whole capsule floor.
- Phase1: forward schedule and typed effect options through the existing ordinary
  catalog/model builders. Preserve original bindings and refuse unsupported
  layouts/resources rather than adding separate adapters per workload.
- Phase2: price executed postLLVM CPU instructions, stack accesses, target traffic,
  service and overlap together; use balanced complete timing and negative results
  to calibrate ranking before selecting whole successors.

The CCA's dispatch count and static loop-offload description do not express the
runtime CPU loop retention/batch choices measured here. Extend the existing
schedule/dispatch vocabulary and source→object→ELF-bound evidence. Generic costing,
budgeting, host algorithms and buffer contracts belong in Merlin. Device ISA,
resource legality and command emitters remain OOT.

Exact child token accounting is unavailable. Do not invent a token total from
transcript size or wall time.

## Scratch deduplication

Completed failed v1/v2/v3 owned payloads were hashed against immutable v4. All paths
and bytes are retained. This filesystem refuses CoW range deduplication with
errno95, so immutable payload hardlinks replaced only verified duplicates.
All historical drivers refuse an existing owner before writing. No active source,
proof or receipt was edited. Logical duplicate payload reclaimed is 5,439,863,024
bytes; observed free space rose to 12,361,170,944 bytes. Concurrent allocations can
affect the filesystem delta. Exact mappings and hashes are in
`inactive_cdf_payload_dedup_receipt.json`.

The correct complete producer/consumer command-pump run remains pending. Its
receipt must be terminal before reporting any measured win or regression.
