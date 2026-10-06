# Source-proven readout domains

The normal `captured_requant_bundle.build` path accepts the explicit, default-off
`readout_domain_policy="source_proven"` option (CLI `--readout-domain-policy`).
It requires exact integer readouts; it is not an unchecked-input switch.

The provider reopens the pinned typed source and verifies a signed-i8, from-zero
integer contraction, immutable zero bias, matching convolution geometry, ordered
source scales/ReLU, and the actual compiled i32 kernel and IR bytes. Merlin's
`IntegerSumProductsRange` derives every-prefix bounds including legal `-128`
operands and rejects signed-i32 overflow. The original readout domain must contain
that interval. Source IDs identify bindings only; they do not select strategies.

An admitted adapter keeps the original SAT8/threshold arithmetic. It omits the
redundant per-element domain trap after its producer, and checks scratch/output
byte-range disjointness once before the kernel call. The normal IR rewrite creates
separate fresh tensor destinations, including private i32 scratch. Original
alignment, shape and descriptor checks stay in place. Producer completion/fencing
and scratch lifetime remain part of the existing provider ABI.

Each selected route reports `readout_producer_domain` with the proof or refusal;
the bundle reports the policy and application count. Unsupported source geometry
or uncovered integer domains retain checked readout. Changed immutable source,
constant, numeric or executable bindings are hard errors, never a fallback.
Default policy emits the previous C bytes and no additional policy fields.

The interval proof and generic C arithmetic belong to Merlin. Source/kernel
binding, descriptor-width facts and adapter guards belong to this OOT provider.
This option does not change the whole-model numerical criterion, permit sampled
input bounds, or authorize treating earlier f32 tensor views as physical i8 data.
