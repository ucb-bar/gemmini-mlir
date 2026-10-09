# Attention precision and cost

This comparison describes the retained captures. It does not establish the work
performed by Jack's unsupplied language-model benchmark recipes.

## Why the current captures differ

| Attention work | TinyLlama | SmolVLA vision encoder |
| --- | ---: | ---: |
| Sequence length | 8 | 1,024 |
| Layers | 22 | 12 |
| Rectangular score elements | 45,056 | 150,994,944 |
| Rectangular QK plus PV multiply-accumulates | 5,767,168 | 19,327,352,832 |

SmolVLA has about 3,351 times the rectangular attention work in these captures.
Parameter count is insufficient to predict attention cost. These tensor counts
exclude encoding, software certification and device scheduling overhead. Tiny's
separately proved triangular QK admission reduces its executed arithmetic; the
table does not claim equal masking or hardware cycle costs.

Precision also differs. Tiny's typed floating attention products use separate
binary32 multiplication and addition. SmolVLA's vision source stores Q/K/V as
BF16 and uses ordered binary32 FMAs. Its PV numerator consumes BF16-rounded
probabilities, while its denominator sums the original unrounded binary32
polynomial values in eight lanes. BF16 output then enters a dynamic quantizer
whose scale and integer codes are observable by the next projection.

Changing the denominator to the same BF16 probabilities used by PV failed the
original whole-model gate on **122 of 1,600 outputs**. The first source group
already changed 458 integer codes and nine escaping BF16 scales. Individual
final failures have not been causally assigned to those early differences.
The candidate is rejected; the existing atol 0.03125 / rtol 0.02 gate remains.

## What the selected hardware can execute

The current source chain for `FireSimGemminiRocketConfig` selects the default
integer Gemmini: INT8 operands and INT32 accumulation. Rocket's FPU formats are
IEEE FP16, FP32 and FP64; **BF16 is not an admitted native arithmetic format**.
IEEE FP16 and BF16 have different exponent/significand formats. A separate BF16
Gemmini generator exists, but this configuration does not select it.

BF16 storage can still be implemented through explicit bit conversion, binary32
arithmetic and correctly placed BF16 rounding. The current SmolVLA device path
instead encodes operands into three integer planes, performs nine pair products
and reconstructs/certifies the source observations on the host. This is costly:
four provider bodies account for 75.02 billion of the current executable's
117.64 billion retired instructions. Instruction counts are not cycle shares.

Current local source facts are pinned below. They are not a new correspondence
proof for the deployed FPGA bitstream, and the functional Spike run is not a
hardware performance measurement.

## Compiler work in progress

- Merlin: derive cheaper source-polynomial BF16 bucket certificates, retaining
  tight bounds for the original denominator and exact fallback when necessary.
- OOT dialect: retain complete operand planes within each existing degree
  callback. Legal source geometry predicts fewer requested bytes; emitted code
  and complete timings must verify the benefit.
- Both: preserve source precision boundaries and price preparation, conversion,
  packing, device execution, readout, certification and replay together.

[Source work and hardware facts](perf_records/attention_work_and_bf16_source_facts_20261008.json),
[rejected joint probability experiment](perf_records/smol_joint_probability_rejected_20261008.json),
[actual attention device audit](perf_records/smol_attention_device_cost_audit_20261008.json).
