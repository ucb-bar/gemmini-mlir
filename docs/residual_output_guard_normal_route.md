# Exact residual prediction with one output

The existing source-bound residual provider has an explicit
`single_output_guard=True` mode. It accepts a complete Merlin affine-pair
certificate. Selection matches the original source arithmetic and scales;
symbols serve only to bind and trace the selected calls.

```python
from mlir_oot.joint_residual_catalog import build

build(original_bundle, [pair_certificate], llvm_bin, selected_bundle,
      single_output_guard=True)
```

Pass `selected_bundle` through the existing
`residual_mixed_catalog.merlin_callbacks(..., joint_bundle=selected_bundle)`
route. Its preparation selects the actual declaration/call and its compilation
links the selected object into the normal catalog. This is an explicit option;
shared cost selection is not enabled.

The call retains three arguments: immutable signed-byte A/B and a fresh C0.
The Gemmini producer writes one scaled prediction and fences. Merlin's
complete-pair certificate derives which output byte values could be ambiguous;
the portable helper rejects words that contain no such byte, checks actual
source pairs where needed, and replays the original separate binary32
operations. The final contract is zero LSB error. The raw predictor's local
one-LSB error is recorded independently and never becomes the final gate.

The existing fresh-writer machinery owns C0's allocation/deallocation. No C1
allocation or readout is introduced. The ranked adapter checks shape, strides,
offset overflow and output/input nonoverlap. A/B stay alive and immutable
through correction; A/B may alias each other read-only.

The normal provider uses explicitly pinned Clang strict FP flags. A capsule
that includes the helper in a GCC-built harness is a distinct compilation
regime. Its timing does not establish the cost of the normal Clang adapter.

Qualification currently covers 41 focused checks: complete native pair-domain
correction, aligned and odd-offset output/dirty guards, actual O0/O2 upstream
lowering of an independent 48x64 shape with live inputs, one fresh allocation,
transactional source/effect/proof refusals, normal target object/catalog linkage
and no FSM. Pinned generated-C hashes preserve the old four-argument default.
The actual model binding validates all 16 original residual declarations and
selects one source relation with a three-argument call.

[The normal-route receipt](perf_records/residual_output_guard_normal_route.json)
records source, proof, compiler and object pins. Whole-model qualification and
stock cycle cost remain pending. Admission requires the unchanged original
all-1,000-word bitwise gate, strict target execution and final executable audit.
