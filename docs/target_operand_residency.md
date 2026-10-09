# Explicit target operand residency

These schedules retain operands within one primitive Gemmini invocation. They
change target placement and DMA requests. Callers must prove immutable dense
input storage for the complete invocation, output nonoverlap, and exclusive
accelerator use through the final fence. Dimension checks do not prove these
source/effect/lifetime obligations. Source packing, numerical permissions,
requantization, consumer proofs, host dispatch and cross-call lifetime management
remain in Merlin. Target admission uses the operation's dimensions and the
selected configuration's `DIM`, scratchpad rows and accumulator rows. No model
name, captured source ID or measured answer selects either schedule.

Resource admission establishes legality. It does not establish lower cycles:
bank service, code size, overlap, preparation and readback remain part of a
complete comparison. Every compiled device object and linked program requires a
zero-FSM executable audit. These options remain false by default.

## Complete convolution reduction weights

`GoldenFlatConv(..., cached_reduction_weights=True)` retains the original HWIO
weights in a disjoint upper scratchpad interval across every spatial band of its
existing 3x3 contraction. The original tap/channel order, activation placement,
accumulator work, output layout and readout policy are preserved. The complete
weight interval needs `9 * ceil(Cin / DIM) * ceil(Cout / DIM) * DIM` rows. This
schedule requires aligned `Cin` and refuses competing separate/ping-pong B
placements or activation/weight overlap. Channel and spatial output tails keep
their exact extents; padding follows the existing explicit halo/virtual-padding
contract. Nothing survives the final invocation fence.

The normal factory uses `select_kernel(..., cached_reduction_weights=True)`.
`select_cached_reduction_weights` retains every declared emission option and
ranks an admitted candidate by requested B bytes and load count. No reduction in
requested work preserves the control. Unsupported selected emitter families
also preserve the control, with an explicit refusal. The ordinary source-bound
bundle forwards the choice as `flat_cached_reduction_weights=True`; its CLI flag
is `--flat-cached-reduction-weights` and requires flat scheduling.

Source/primitive tests cover complete original tap/channel/input/output binding,
independent rectangular and stride cases, channel/spatial tails, capacity,
placement conflicts and ordinary spatial command loops. The normal flat factory
currently selects a separate B bank. Requesting cached weights preserves that
control and records the incompatible placement refusal; it does not silently
replace the bank choice. An explicitly supplied nonbank control can admit the
call-local schedule. The earlier 52-leaf preparation that changed one placement
used an option-overriding selector and remains historical evidence for that
sealed source. The repaired successor re-emits all 52 normal objects/adapters
exactly as the original control, with no cached application. New whole
numerical and hardware timing qualification remain pending.

## Complete product-sum operands

`GoldenProductSum(..., resident_operands=True)` retains only the referenced
signed-i8 planes for one ordered integer product sum. Its unchanged
`void(i8*, i8*, i32*)` ABI uses dense plane-major A and B. The caller supplies the
existing plane spans, magnitude domains and signed-i32 prefix bound from the
source proof. This target schedule adds no BF16 approximation, packing,
reconstruction or permission to reassociate floating products.

`operand_residency_plan` computes complete physical A/B tile storage from M/N/K,
referenced plane indices and `DIM`. It checks scratchpad and accumulator
capacity; the emitter preserves exact DMA tails, product pair order, K order,
all output writes and the final drain. Conflicting separate B placement refuses.
The resident path also refuses a batched ABI until that distinct input/plane
layout has its own admission. There is no cross-call prepared-RHS lease.

The encoded-zero provider has a different sparse panel placement. It explicitly
refuses resident operands: its sparse reloads would clobber retained A planes
needed by later dense blocks. Combining those schedules needs a separate typed
ownership and placement contract. Defaults remain unchanged for both providers.

The retained earlier native/Spike capsule covers five degree readouts, three
original geometries and three independent signed-i8 tail cases, all source
pairs and 16,570,700 checked degree words under multiple host rounding/flag
states. Current consolidation separately verifies exact emitted target and
lowered IR and objects against that original source. Input preparation,
BF16 reconstruction, consumer certification and whole performance are outside
that integer capsule's scope; fewer requested transfers earn no cycle claim.

## Topics deliberately kept separate

| Topic | Disposition |
|---|---|
| Cross-call operand residency | Pending immutable epochs, bank ownership and intervening clobber/invalidation proof in Merlin plus target lease implementation |
| Encoded-zero plus resident operands | Refused until separate placement and complete cost qualification |
| Provenance-preserving CSE and scoped FP epochs | Generic Merlin work; structural sharing does not supply provider effect or whole-accuracy admission |
| Polynomial BF16 buckets | Generic optional Merlin API; rejected coarse consumer policy stays disabled |
| Joint BF16 probability/denominator approximation | Rejected under the original whole elementwise gate; no target specialization copied |
| Component workflow, native image identity, performance policy | Generic Merlin topics reviewed by their source owners; no duplicate target infrastructure |
| Hardware normalization/softmax | No capability inferred or new instruction enabled from an unsealed target configuration |

The handwritten branch exposes reusable compiler mechanisms. Experiments and
old measurements retain their original source, binary and scope; integrating a
mechanism does not enable one automatically optimized configuration for all
workloads. Per-topic token allocation remains UNKNOWN.

## Consolidation qualification

The original isolated consolidation runs through independently installed Merlin
and the normal target compiler: 119 cached-weight checks, 70 product-sum checks,
20 default emissions, 60 integer-family target/lowered emissions and 12 fresh
maximal-degree objects. That frozen packet's 52-leaf candidate result used the
option-overriding selector; it is not evidence for the repaired normal factory.

The bank-preserving successor passes 123 affected source checks, including four
regression controls which fail on the old source. Its fresh normal 52-leaf
objects/adapters are exact to the original control; all cached requests retain
the incompatible selected bank/family instead of overriding it. Product-family
source is unchanged from the original sealed topic. These source, emission and
object results are functional compiler gates, not a new complete workload run.

The machine-readable source/diff/disposition record is retained under
`out/artifacts/audits/handwritten-consolidation-v1` in its qualification owner.
The repaired selector has a separate
`out/artifacts/audits/bank-preservation-successor-v1` packet and backup refs.
Historical native/Spike/GSIM receipts remain immutable and keep their original
configuration, binary, timer and consumer scope. Whole numerical qualification,
requested stock FireSim performance and automatic selection of either schedule
are still separate requirements.
