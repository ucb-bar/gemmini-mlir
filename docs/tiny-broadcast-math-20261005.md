# Source broadcast math hoisting: current Tiny host evidence

The verified1880 host model object was reproduced byte for byte: SHA
842037c4920947565207612e9eb02c51f3f08f9cba9835868180af1385a71279.
The exact selected source LLVM SHA is
f4cce5abe900f28d10b7d8a8a2abea33bc4ccda52a3abe3599a2e4a15a200d07.
Actual O3 forward-only code has360 reciprocal-square-root-containing row loops,
each starting at zero, incrementing by one and stopping at2048. This gives737,280
logical calls from45 normalizations over8rows. Each call uses the unchanged
runtime float reciprocal divided by newlib sqrtf. The small input value is
identical within its channel loop, but the emitted external call blocks LICM.

Merlin commit564bbba77 implements explicit typed source math hoisting after
fusion and before bufferization. Exact affine maps and scalar SSA dependencies
prove each smaller domain. Source casts, operation order, precision and live
uses remain intact; no external-call name implies purity. All45 actual source
chains match.20 new compiled/refusal tests and17 existing packet tests pass,
including unequal rows, tails, permuted maps, f64 intermediates, live initial
outputs, NaN/Inf, empty domains and strict-scope/unknown-effect refusal. This
mechanism and its numeric proof belong in Merlin; target qualification belongs
in OOT. There is no workload-name strategy or automatic promotion.

The current1901 conserved profile has530,599,828forward cycles:
179,768,482 inside device wrappers (including CPU adapters), and350,831,346
outside. The two normalization boundary families total127,531,447cycles;
surrounding residual/quantization work is included. The22 pre-down boundaries
total127,539,158cycles. These boundary costs do not isolate reciprocal square
roots, and cannot predict this optimization's whole-model gain.

The exact actual two-output pre-down LLVM bodies contain16FMA,14multiply,
2division,6addition,4i32-to-f32 and2f32-to-i32 conversions, eight4byte loads
and two1byte stores. Earlier informal18multiply counting was wrong; the durable
census checks all22 matching bodies. Logical input load bytes are15,859,712;
these are operand traffic, not physical DRAM traffic. Packet4's earlier measured
loss remains rejected. No older1837profile is used as current evidence.

A typed source dependency extraction retains the complete first8by2048
normalization: original embedding and weights, ordered sum of squares, divide,
epsilon, source rsqrt, weight/scaling chain, bounded RNE and output. Both arms
reuse the original1880 runtime object. Native all16,384i8 values pass. Strict
Spike passes all five rounding modes with identical sticky flags, full output
and128dirty guards; no FSM instructions occur in executable code. Actual GSIM
AB/BA is running with all allocation/traffic/arithmetic included. Spike's
instruction reduction is functional evidence only. The current committed pass
reproduces both capsule LLVM variants byte-identically after the additional
strict-scope/location guards; no duplicate numeric/timing rerun is needed.

Normal Merlin lower_model on frozen complete post-offload ABI source reproduces
original rawLLVM0749b01c…74fcb byte for byte. The original RNE and immutable
writer bridge also reproduce selected target and native LLVM byte for byte.
A preliminary direct lower_toLLVM call omitted normal wide-C-interface inlining;
its only nonliteral difference was that normal alwaysinline attribute. It is
retained as an intermediate diagnostic, and the normal control closes it.

The original256,000output/Torch gate, complete strict target and final no-FSM
audit are required before any whole candidate enters stock FireSim. Every1880
runtime/device/startup/weights boundary object stays frozen. No whole cycle
projection or composition with adjacentRNE/residentA candidates is claimed.
Token allocation per experiment is unavailable; the root records shared
campaign checkpoints. Receipts retain source/LLVM/object identity and scope.
