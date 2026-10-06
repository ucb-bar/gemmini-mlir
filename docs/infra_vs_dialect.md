# Infrastructure fixes and performance changes

## What has been fixed

The work falls into three groups. Portable host performance is compiler optimization in Merlin,
alongside the infrastructure that makes target optimization usable through a normal model build.
Gemmini instructions, device schedules and resource decisions remain in the OOT compiler.

| Group | Concrete changes | Current evidence |
| --- | --- | --- |
| Frontend and shared numerical correctness | BF16/f16 opmath, source reduction/order contracts, runtime helper ABI, explicit portable expf precision policy | Verified frontend fixes are upstream on model2MLIR main. Full original Smol now passes all1,600 native and actual target words exactly, retaining atol=.03125/rtol=.02. [Target](perf_records/smol_full_double_exp_target_exact.json) |
| Merlin build and runtime infrastructure | Generic loop outlining, fresh output ownership, linked-object identities, strict compiler flag precedence in compile and link | Corrected Smol outlined host compiles in111.254s after the monolithic900s timeout. Normal math-policy build reproduces the entire passing loaded target image except its diagnostic marker. A real compiled subnormal regression catches strict flags being overridden. [Normal build](perf_records/smol_normal_host_math_policy_equivalence.json) |
| Merlin planning and dispatch infrastructure | Complete CCA preservation, exact source-bound alternatives, normal build selection, distinct implementations for equal tensor shapes | Selected measured objects reach the real device catalog. Tiny full normal build retains all155 source calls, original256,000 words and the Torch gate. Whole-program search/cost optimization is still not wired. [Tiny qualification](perf_records/tiny_resident_a_prefetch_whole_spike.json) |
| Merlin host performance | Exact bounded quantization/RNE packets, pointwise scheduling, packing/readout/mean, ordered contraction scheduling and selected BF16 widening | Stock whole-model and complete capsule gains below; these are optimizations, not merely build fixes |
| OOT Gemmini performance | Resident/banked transfers, overlap and B slots, convolution stripes, primitive issue/store scheduling, device numeric and layout bindings | General shape/layout/resource predicates; every qualified final ELF passes the all-executable-section zero-FSM audit |

## Gains that can be attributed to a controlled change

Each row is a matched experiment with its own control. Separate rows must not be added to predict a
composition. Full model means stock FireSim forward cycles; GSIM capsule means the complete stated
fixture, not a whole-model speedup.

| Owner and change | Scope | Control → candidate cycles | Reduction |
| --- | --- | ---: | ---: |
| Merlin exact ResNet host quantization packets | Stock1853→1886, whole model |40,479,548→38,603,949|1,875,599;4.6334%|
| Merlin Tiny scalar pointwise packets | Stock1846→1880, whole model |569,151,067→531,072,370|38,078,697;6.6904%|
| Merlin exact early-saturation/eight-lane readout; OOT ABI binding | Stock1874→1900, ResNet whole model |39,201,279→37,946,541|1,254,738;3.20%|
| OOT resident/banked transfers | Stock1849→1853, ResNet whole model |42,269,808→40,479,548|1,790,260;4.235%|
| OOT banked residual added to transfer control | Stock1853→1874, ResNet whole model |40,479,548→39,201,279|1,278,269;3.158%|
| OOT full-reduction convolution stripes | Stock1853→1878, ResNet whole model |40,479,548→39,754,283|725,265;1.792%|
| OOT convolution stripes added to transfer/residual control | Stock1874→1897, ResNet whole model |39,201,279→38,468,933|732,346;1.868%|
| Merlin adjacent independently proved RNE packets | Stock1880→1902, Tiny whole model |531,072,370→527,255,504|3,816,866;0.719% single-run marginal|
| Merlin/OOT qualified host/readout/residual composition | Stock1886→1903, ResNet whole model |38,603,949→36,102,704|2,501,245;6.479%|
| OOT resident-A/B-prefetch alternative | Tiny-shaped synthetic int8/amplitude21 common-address GSIM capsule |111,885→76,456|35,429;31.67%|

Receipts: [ResNet host](perf_records/firesim1886_resnet_quant_packet_verified.json),
[Tiny host](perf_records/tiny_pointwise_packet_firesim.json),
[exact readout](perf_records/firesim1900_resnet_sat8_readout_verified.json),
[transfer](perf_records/resnet_transfer_command_policy_firesim.json),
[residual composition](perf_records/resnet_transfer_residual_composed_firesim.json),
[stripes](perf_records/resnet_resident_stripe_policy_firesim.json),
[stripe composition](perf_records/firesim1897_resnet_stripe_composition_verified.json),
[Tiny device capsule](perf_records/tiny_resident_a_prefetch_gsim.json),
[adjacent RNE](perf_records/firesim1902_tiny_adjacent_rne_verified.json),
[measured composition](perf_records/firesim1903_resnet_composed_verified.json).

The Tiny device alternative also passes the normal full-model original-output and zero-FSM gates.
Its first build selects one contraction. The expanded44-binding normal and controlled builds
also pass all original native/strict target output gates. Stock1911 measures the normal build
with its runtime compiler change;1912 isolates the device schedule with all1880 host/runtime
objects retained. Both hardware outcomes are pending. [Family](tiny-calibrated-family-20261005.md).

Rejected optimizations remain recorded. In particular, the complete Smol attention-head gamma
candidate takes3,077,601,494stockcycles versus2,618,580,085controlcycles(17.53% slower), despite
lower device/readback work. It is disabled. [Hardware](perf_records/firesim1895_original_attention_head_gamma_verified.json).

## Remaining gap and accounting

Current verified whole-model champions are ResNet36,102,704cycles and Tiny527,255,504cycles.
Smol has full original target correctness; stock1906baseline cycles are pending. These do not
meet the requested22M/5B goals. Matching Jack's permitted executable remains a device scheduling
oracle, with original source/numeric gates held fixed.

There is no defensible overall infrastructure-versus-dialect percentage of effort or tokens.
Requests and threads mix both kinds of work, and the controlled performance arms are not an
additive campaign attribution. The [journey](golden_optimization_journey.md) retains measured
campaign counters, per-experiment ownership, gains, negative results and remaining abstractions.

## How these changes generalize

Recent portable host optimizations prove ownership and representation before
changing traffic: private uniform buffers can fill their copy destinations
directly, and distinct fresh strided allocations can copy their longest common
contiguous suffix with ordinary runtime memcpy. Unknown aliases, shared roots,
dynamic extents or incompatible layouts are refused. These optional passes live
in Merlin and preserve normal pipeline coverage receipts. Their current whole
instruction reductions are0.465% and0.509%; stock gains remain unmeasured.

The exact i64 reconstruction helper also belongs in Merlin: a canonical numeric
range proof and disjoint scratch establish integer updates and one final exact
binary64 conversion independently of the producer target. Gemmini zero-tile
initialization, residency/coalescing and ISA/resource legality remain OOT. The
zero-tile/support composition loses1.7625% on the complete original bounded
capsule and is disabled; legal numeric proofs alone do not establish profitability.

Shared fast-screen fitting and grouped validation live in Merlin. Executed
Gemmini opcode/operand extraction and the optional isolated simulator hook live
in OOT. Provider pointers describe observed work; fitting/composition reuse the
existing shared calibration and resource abstractions. Missing physical traffic
or dependency/overlap facts are unknown. Independent coefficients must pass
section and whole known-reference checks plus within-workload ranking before
they can guide automatic phase1/2 selection. The current counter-only and
additive operand diagnostics both fail; neither is enabled as a cost provider.

Production transforms match typed operations, operand/result mappings, numeric contracts, layouts
and resource facts. They do not select an implementation by a model name. Source ordinals and hashes
bind a chosen implementation to the exact source; they are not profitability rules. This is an
explicit invariant in [AGENTS.md](../AGENTS.md).

- Host packet scheduling, ownership, compilation and dispatch fixes apply to any source satisfying
  their structural proof. Ordered-FMA and BF16 widening retain their source reduction contract.
- Device residency/prefetch/tiling applies when shape, scratchpad/accumulator capacity and live
  transfer intervals prove it legal. Independent nonbenchmark shapes and tails are tested.
- Numeric certificates are recomputed from each source's actual scales, rounding and value domains.
  The derivation generalizes; the resulting coefficient/table belongs to that particular operation.
- Explicit approximations require qualification against each model's unchanged accuracy criterion.
  One capture passing is not a universal accuracy claim.

Measured performance is narrower than applicability. A legal shared transform can lose on another
shape, blocking factor or composition. The normal calibrated selection path retains actual measured
costs and unknowns so those choices can be evaluated independently.
