# Exact guarded quantized mean

This explicitly selected OOT lowering recognizes a canonical signed-i8 scalar
DQ, zero-initialized serial f32 mean over static trailing H/W axes, and symmetric
signed-i8 Q. It preserves the original reference and source arithmetic. The
current binder accepts unencoded static BCHW inputs, a positive reduction count
at most128, positive finite f32 scales and finite intermediate arithmetic.
Other arithmetic, axes, nonzero quantization offsets and noncanonical division
refuse. The physical input is contiguous; the ranked-memref adapter checks its
sizes and strides and returns the actual destination descriptor.

## Certificate

For finite binary32 round-to-nearest-even with gradual underflow,
`|fl(t)-t| <= 2^-24 |t| + 2^-150`. The derivation uses exact rational arithmetic
to bound each scaled input, every serial addition, division by the static count
and multiplication by the original rounded f32 quantization reciprocal.
Absolute error bounds remain valid through cancellation. Possible overflow
refuses. All signed-i8 arrays are covered; no captured value bounds are used.

For every reachable integer sum, the certificate encloses the source result in
a closed interval. Saturated round-to-even is monotone. If both endpoints yield
the same i8 value, that value is exact for every input sequence with this sum.
Otherwise the table contains a sentinel and the kernel replays the original
f32 operations in increasing reduction-index order. The emitter re-derives and
compares the entire certificate before producing code.

ResNet's original count49, input scale1.7433754205703735 and output
scale1.3525974750518799 yield12,496 entries and eight ambiguous totals:
`[-4581,-2300,-2262,-19,19,2262,2300,4581]`. The final error bound is
approximately0.000275341. The C fallback uses a volatile f32 product, and target
compilation uses `-ffp-contract=off`, to preserve separate product/add rounding.
The table and kernel have no mutable global state.

## Reproduction and compilation

Create an owned derived capture with unchanged inputs, weights and golden.
The bundle command rewrites only this derived model and its receipt:

```sh
python -m mlir_oot.guarded_mean_bundle \
  --derived-capture DERIVED_CAPTURE \
  --llvm-bin LLVM_BIN \
  --output NEW_BUNDLE_DIRECTORY
```

Wrap the existing exact provider callbacks with
`guarded_mean_bundle.merlin_callbacks(llvm_bin, bundle, base_callbacks)` and use
the ordinary Merlin build. The callbacks check call/declaration coverage,
source/proof metadata, access attributes, manifest, C source and object hashes.
The CPU adapter is partially linked with the existing device object and both
the intermediate object and final ELF undergo the zero-FSM audit. The native
oracle uses this same C adapter. The normal build marker includes its linked
bytes; final ELF SHA256 identifies the actual executable.

The full ResNet build retains the1812 device recipes, classifier schedule and
host transforms. Native and actual Gemmini Spike retain all1,000 original
f32 words. Spike retires10,816,839 instructions, versus1812's10,985,615.
This is a functional/proxy result; FireSim job1819 measures actual cycles.
See [whole-model gate](perf_records/resnet_guarded_quantized_mean_spike.json).

Seventeen focused tests execute all65,536 count2 arrays and all12,496 count49
sums in adversarial and reversed orders, including every fallback sentinel.
They check output guards, proof tampering, source arithmetic refusals and
serialized ranked-tensor writer access. The mathematical certificate supplies
coverage beyond these tested input sequences.

## Generalization

Count, scales, layout and the proof are derived from typed source IR. The
implementation contains no model-name check or captured-input constant. The
current pattern is intentionally narrower than arbitrary floating-point means.
A future automatic loop can select this exact bundle after proving its pattern
and comparing measured costs, retaining the float fallback and immutable
reference. Broader axes, dynamic shapes and encoded layouts require their own
binding proofs.

## Explicit packed NHWC variant

`--packed-nhwc` proves the existing named transpose has permutation[0,3,1,2]
from static NHWC to BCHW and consumes its physical input directly. It keeps
each logical channel's H/W index order in the ambiguous-sum replay. The current
packed implementation requires channels divisible by eight. Sizes/strides and
the original permutation are recorded in the bundle.

For aligned little-endian input, xor128 biases eight signed bytes to unsigned
values. Alternate bytes widen to four unsigned16-bit lanes in each of two
64-bit sums. Count at most128 bounds every lane by32640, so carries cannot cross
lanes. Subtracting count*128 recovers every exact signed integer sum. Unaligned
input uses the scalar strided path. Byte order is checked at compile time, and
word loads use a may_alias type without a restrict promise.

The complete model passes native and actualSpike with all1,000 original words
unchanged, retiring10,214,792 instructions (7.02% fewer than1812). Both fusion
environment flags and explicit host scheduling are pinned to the control. The
packed host scheduling true/false builds in fact have identical model objects
and finalELFs; no duplicate run was used. StockFireSim1824 measures cycles.
Twenty-three focused tests include packed/unpacked, every count2 pair and count49
sum, both reduction orders, aligned/unaligned data, count128, multiple batches,
guards and layout refusals. See
[packed gate](perf_records/resnet_packed_guarded_mean_spike.json).

The earlier contiguous guarded mean1819 measured46,680,853cycles,0.79% above
1812. It omitted explicit host scheduling and retained a final layout
transpose, so this timing alone does not isolate either factor's cost. It is
not the best recipe. The packed variant removes the transpose under a source
proof; hardware comparison still decides admission.
