# SmolVLA quantization audit

## Recorded recipe and actual coverage

The retained capture uses TorchAO 0.17.0 `int8_dyn_act_int8_weight` on
`lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
The original checkpoint parameters are mixed BF16/F32, not uniformly FP32.

Reapplying the original loader and quantization recipe on the same six retained
inputs reproduces **all 1,600 original quantized golden words exactly**.
All 303 registered `Linear` modules have actual stored i8 weights and
per-output-channel scales. The capture contains 302 linear call sites, all
integerized; module counts and captured call-site counts are different units.
The quantized graph still has 12 floating SDPA and 64 floating matmul call sites,
which become 88 floating bmm call sites in the prepared graph. The default
Linear filter does not quantize attention products.

`calibration_samples=1` is a capture argument here. This dynamic activation
recipe performs **no static calibration**. It does not establish dataset
representativeness or robot task quality.

## Quantization loss and compiler fidelity are separate

| Comparison on the retained fixture | Relative L2 error | Maximum absolute error | Values outside atol=.03125, rtol=.02 |
| --- | ---: | ---: | ---: |
| Recorded TorchAO output versus unquantized checkpoint | 2.9199% | 0.1773094 | 181 / 1,600 |
| Reproduced TorchAO output versus original quantized golden | 0 | 0 | 0 / 1,600 |
| Original ordered compiler control versus original quantized golden | 0 | 0 | 0 / 1,600 |

We had qualified lowering against the quantized golden. The separate
prequantization baseline comparison was missing before this audit. The measured
loss warrants a quality evaluation; it does not establish a TorchAO application
bug or explain additional compiler approximation failures against that golden.

The frozen single7, two-digit and wide64 attention experiments still fail the
unchanged compiler gate on 131, 106 and 144 values respectively. On this one
fixture the latter two have smaller error to the unquantized checkpoint than
the recorded TorchAO baseline. That diagnostic does not establish better robot
quality or authorize either implementation. Quantization and approximation
errors cannot be treated as additive scalar error budgets.

## Why attention needs its own source contract

The actual projection path includes integer accumulation, a rounded conversion
to BF16, separately rounded BF16 activation/channel scale multiplies and bias.
Attention extends the resulting BF16 values into ordered F32 FMA reductions.
Factoring scales through those operations requires a numerical proof.

Even unchanged operands with wider accumulation change 81 escaping i8 values
across 35 of 48 sections on independent original states. The first change is
in section 4, row 135, channel 689: **56 → 57**. No escaping scales change in
that diagnostic. More accumulator precision alone does not reproduce the
original reduction order or downstream quantization bins.

## Limits and follow-up

This audit covers one retained input fixture whose dataset attribution is
unknown. It measures neither held-out action error nor robot task success, and
does not constitute the same prequantization audit for ResNet or TinyLlama.
The legacy capture environment was deleted. Matching Torch/TorchAO versions
and the retained golden establish fixture reproduction; full historical
dependency closure remains unproved. An owned Diffusers import-only logger
initialization repair is recorded and verified against its original wheel
RECORD. Base environments, source captures and goldens are unchanged.

The generic model2MLIR follow-up is an opt-in capture quality report preserving
outputs before in-place quantization, with separate prequantization quality and
quantized-golden fidelity reports. It is currently being implemented on a fresh
main-based topic. The compiler acceptance gate remains unchanged.

Evidence: [root recipe review](perf_records/root_smol_torchao_recipe_quality_review_20261007.json),
[distinct oracle diagnostics](perf_records/root_smol_quantization_error_oracles_20261007.json),
[attention boundary review](perf_records/root_smol_attention_boundary_review_20261007.json).
