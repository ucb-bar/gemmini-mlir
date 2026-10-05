# Exact pre-stem destination reuse and external alias limitation

Current control: ResNet FireSim1795, 47,020,321 forward cycles. Fresh profile1801
measures 5,238,180 cycles before the pooled stem. The control already fuses input
NCHW→NHWC conversion with exact clamp/RNE int8 quantization. It materializes an
interior int8 tensor and copies it into a zero-padded230×230×3 destination.

Default-off `reuse_tensor_destination` clones the unchanged pure pointwise
scalar body into a statically proved unit-stride view of a fresh filled pad.
Upstream owns bufferization legality. Native and actual Gemmini Spike match all
1000 original output bits; final ELF has zero FSM instructions. Spike instructions
fall from11,497,420 to11,001,021 (4.32%); this is not a hardware measurement.
The stock interface, device schedule, original golden and numerical policy stay
identical. Complete gate is `resnet_prestem_destination_spike.json`.

## Concrete remaining limitation

Upstream `mlir/lib/Dialect/Bufferization/Transforms/OneShotModuleBufferize.cpp`,
`aliasingFuncOpBBArgsAnalysis`, explicitly assumes every bodyless tensor result
can alias every tensor argument. `bufferization.access` read/write annotations
do not describe result identity. Analysis-only `print-conflicts` on this model
marks the pre-stem insert destination out-of-place because a later
`gemmini_exact_requant_15` reads weight argument125. This false alias propagates
through intervening bodyless device calls. Bufferization consequently allocates
a second pad and copies the first pad/interior. A local extract/generic/insert
fixture without these calls bufferizes into one pad allocation and a self-copy.

Artifacts are under the owned OOT worktree `out/prestem_destination_fused/`:
`manual_tensor.mlir`, `analysis.mlir`, `manual_bufferized.mlir`.

## Rejected wrapper experiment

A defined tensor wrapper that converts C with writable `bufferization.to_buffer`,
calls the memref adapter, and returns the original tensor C is WRONG. Although
analysis reports result=C, the writable conversion may allocate/copy C internally,
so the wrapper returns stale original data. A regression with a live original
input showed this exact counterexample, and the whole original-golden gate also
failed. No hardware job was submitted, and the experiment was removed from core.
Reproduction is retained only in owned scratch `out/rejected_destination_bridge/`
and `out/prestem_destination_bridge_v2/`; do not reuse it as an optimization.

A future interface must expose correct result-bearing destination semantics,
including copy-on-write and descriptor identity, through a real bufferization
model. Unsupported attributes, global noalias claims, and unchecked restrict
bridges are insufficient.
