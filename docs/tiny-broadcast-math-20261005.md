# Source broadcast math hoisting: current Tiny host evidence

## Hypothesis and ownership

The actual optimized host code repeats a row-invariant source reciprocal square
root across every channel. Proving that invariance before bufferization can
remove the repetitions while retaining the original scalar math contract.
The typed proof and host transformation belong in Merlin. Target capsules,
executable audits, object bindings and measured costs belong in OOT.

Merlin commit `564bbba77` adds the opt-in
`hoist_broadcast_source_rsqrt` feature after fusion and before bufferization.
Exact affine maps and scalar SSA dependencies prove each smaller domain.
Source casts, operation order, precision and live uses remain intact. External
call names never imply purity. Empty or dynamic consumers, unknown effects,
fast math and strict floating point scopes refuse. A fresh smaller tensor adds
allocation and traffic; complete measurements must include those costs.

All 45 actual source chains match. Twenty new compiled/refusal tests and
17 existing packet tests pass, including unequal rows, tails, permuted maps,
f64 intermediates, live initial outputs, NaN/Inf, empty domains and strict-scope
refusal. There is no workload-name strategy or automatic promotion.

## Actual control code and profile

The verified 1880 host model object was reproduced byte for byte:
`842037c4920947565207612e9eb02c51f3f08f9cba9835868180af1385a71279`.
Its selected source LLVM SHA is
`f4cce5abe900f28d10b7d8a8a2abea33bc4ccda52a3abe3599a2e4a15a200d07`.
Actual O3 forward-only code has 360 reciprocal-square-root-containing row
loops, each starting at zero, incrementing by one and stopping at 2048. The
45 normalizations over eight rows therefore execute 737,280 logical calls.
Each uses the original runtime's `1.0f / sqrtf(x)` implementation. The argument
is identical within its channel loop; the emitted external call blocks LICM.

The current 1901 conserved profile has 530,599,828 forward cycles:
179,768,482 inside device wrappers, including CPU adapters, and 350,831,346
outside. The two normalization boundary families total 127,531,447 cycles,
including surrounding residual and quantization work. The 22 pre-down
boundaries total 127,539,158 cycles. These costs do not isolate reciprocal
square roots or predict this optimization's whole-model gain.

The actual two-output pre-down LLVM bodies contain 16 FMA, 14 multiply,
two division, six addition, four i32-to-f32 and two f32-to-i32 conversions,
eight four-byte loads and two one-byte stores. Earlier informal counting of
18 multiplies was incorrect; the durable census checks all 22 matching bodies.
Logical input load bytes total 15,859,712. These are operand traffic rather
than measured physical DRAM traffic. Packet4's prior measured loss remains
rejected. The older 1837 profile is not used as current evidence.

## Complete source-bound capsule

A typed source dependency extraction retains the complete first 8 by 2048
normalization: original embeddings and weights, ordered sum of squares, divide,
epsilon, source rsqrt, weight/scaling chain, bounded RNE and output. Both arms
reuse the original 1880 runtime object. Native comparison passes all 16,384
i8 outputs. Strict Spike passes all five rounding modes with identical sticky
flags, full outputs and 128 dirty guards. All executable sections pass the
no-FSM audit. Spike reports 819,490 versus 328,202 retired instructions per
complete call; these are functional instruction counts, not hardware cycles.

The first GSIM harness spent approximately 15 minutes on ten untimed rounding
mode calls before reaching the timed interval. It was terminated without a
timing claim, retaining the complete strict qualification and immutable
arithmetic objects. A smaller timing-only harness links those same control,
candidate, runtime and data objects. It independently passes full outputs and
dirty guards, and runs four complete AB/BA calls. Its cold/warm context belongs
to that common harness, with all allocation, traffic and arithmetic included.
The actual paired GSIM result passes with these complete-call cycle counts:

| Order | Control | Source hoist |
| --- | ---: | ---: |
| AB | 2,351,622 | 838,846 |
| BA | 2,386,704 | 837,692 |

Mean cost falls from 2,369,163 to 838,269 cycles, a 64.6175% reduction. The
warm pair reduces cycles by 64.9017%. All four calls retain the complete
16,384-output comparison and 128 dirty guards. This is a complete first-norm
capsule result; whole-model costs remain unknown.

The common capsule allocator is a small 64-byte-aligned bump malloc/free,
while the exact original `mlir_rt.o` supplies sqrtf/rsqrt and the math ABI.
Allocation calls and intermediate traffic are timed. The frozen whole allocator
and address layout still require the controlled full-model gate.

The archived receipt is
[tiny_source_broadcast_math_complete_capsule.json](perf_records/tiny_source_broadcast_math_complete_capsule.json).

The committed pass reproduces both capsule LLVM variants byte-identically
after its additional strict-scope and location guards. Those fixes require no
duplicate numeric or timing run.

## Normal build and promotion gates

Normal Merlin `lower_model` on the frozen complete post-offload ABI source
reproduces the original raw LLVM (`0749b01c…74fcb`) byte for byte. Original
RNE processing and the immutable writer bridge also reproduce selected target
and native LLVM byte for byte. A preliminary direct `lower_toLLVM` invocation
omitted normal wide-C-interface inlining; its only nonliteral difference was
the normal `alwaysinline` attribute. That intermediate diagnostic is retained;
the normal build control closes it.

A positive complete capsule is required before whole compilation. The original
256,000-output and Torch gates, complete strict target execution and final
no-FSM audit are required before stock FireSim admission. The controlled whole
link changes only `model.o`; all original 1880 runtime, device, startup, weights
and other host objects stay frozen. There is no whole-cycle projection or
composition claim with adjacent-RNE, resident-A or constant-clamp candidates.

Token allocation per experiment is unavailable. The root records shared
campaign checkpoints; receipts mark `token_usage_available=false` and retain
source, LLVM, object and timing scopes.
