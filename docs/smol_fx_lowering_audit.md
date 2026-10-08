# SmolVLA: matching PyTorch / FX attention and lowering audit

2026-10-07. The retained six-input capture is a full-checkpoint image/language/state
prefix followed by **one denoise step**, producing 1,600 values. A recurrent
multi-step action session is a different workload. No capture, reference, sequence
length or original `atol=0.03125, rtol=0.02` gate was changed.

## What the actual graphs contain

The checked `frontend-trace.json` matches the capture metadata SHA. The analysis
reads graph nodes, typed values, module stacks and operand edges; it checks product
dimensions and result shapes rather than inferring work from parameter count.

| Static captured work | SmolVLA | TinyLlama comparison capture |
| --- | ---: | ---: |
| Dense projection products | 302 already integer TorchAO products | 155 Q/DQ-wrapped dense products, before Merlin integer fusion |
| Scalar MACs in those products | 110,595,843,584 | 8,275,361,792 |
| Prepared floating batched matrix products | 88 | 45 |
| Scalar MACs in those batched products | 19,931,584,512 | 5,767,424 |

These are static FX work counts, excluding convolution and elementwise work;
they are **not cycles or a dynamic accelerator coverage percentage**. The original
Tiny capture has eight tokens and 22 layers. Smol's 512×512 image and 16×16 patch
embedding produce **1,024 vision tokens**. Its 12 vision attention layers each
have 12 heads of width 64: **150,994,944 scores** and **19,327,352,832 attention
MACs**. Smaller parameter storage does not imply less work for these captures.

The original Smol graph has 12 vision SDPA calls. Preparation exposes 24 vision
and 64 text/action floating BMMs, 44 softmaxes and 25 native layer norms. The
TorchAO `int8_dyn_act_int8_weight` recipe targets Linear modules and leaves these
attention products floating. It has not quantized every arithmetic operation.
The current explicit vision source lowering further partitions the 12 attentions
into 48 query groups, 96 QK and 288 PV contractions. Those tiled counts describe
the same vision work, with a different counting unit.

The text/action products account for **604,231,680 MACs**, or **3.03%** of the
prepared floating BMM work; vision accounts for **96.97%**. They still require
coverage: a scalar implementation can be expensive even when its arithmetic
share is small. Their six typed shape classes contain **40 f32 products and
24 BF16 products**. The f32 result dtype alone cannot determine whether the
operands originated from exact BF16 widening, so the next audit follows their
producer chains through the current typed IR and emitted calls. The earlier
executable inventory covered direct linalg multiply/add forms, excluding
SCF/FMA forms; that exclusion was an evidence gap, not proof of offload.

## Current executable coverage and cost

The independently checked current functional executable `b15444e0` contains
**62 live CPU attention product functions**, including scalar SCF/FMA forms:
15 text QK,15 text BF16 PV,8 action QK,8 action BF16 PV,8 cross QK and8 cross PV.
Their bodies contain scalar floating arithmetic and no device imports; actual
executed PC extents account for4,745,501,325 exclusive retired instructions.
The remaining graph-to-live-function correspondence for two of the64 prepared
nonvision products is UNKNOWN. Static function count is not dynamic call count.

The image patch convolution is also on the CPU. Its typed im2col contraction has
four interleaved K192 lanes and603,979,776 MACs. The current function retires
4,851,512,883 instructions, slightly more than all62 attention functions combined.
The attention functions account for4.03% of the117,636,499,028-instruction whole
executable trace. Vision provider, product reconstruction, frontier certification
and operand encoding together account for63.77% of that trace. These are exclusive
CPU function counts, including harness/startup/output in the denominator;
they are **not host/accelerator time percentages or FireSim cycle allocation**.

The exact optional scalar-accumulator pass initially changed named-copy fusion
and broke prepared immutable RHS physical identity. All48 groups then used their
original source fallback, despite exact final outputs. The shared Merlin fix
places scalarization after ordinary fusion/generalization and before bufferization.
It restores48 groups,12 preparations and23,040 integer product callbacks, with
all1,600 original words bitexact and zero fallback/owner errors. Combined clean
main `ea180dca9` passes400 source +400 independently installed tests.

A complete source-derived QK component, including private seed restoration and
result publication, improves104,063,475→91,996,770 retired instructions
(−11.5955%); an independent tail case improves4,369→3,865. Both retain exact
outputs and zeroFSM. This is a CPU component improvement, not additional offload,
whole-model speedup or hardware timing. Patch-convolution numerical/lowering
feasibility is a separate pending gate. An initial native symbol-rebinding seam
ran zero replacement calls and is invalid for policy coverage; it gives no
numerical candidate credit.

[Current executable/PC review](perf_records/smol_actual_current_executable_product_review_20261007.json).
[Scalar-stage source/coverage qualification](perf_records/smol_scalar_tensor_stage_source_topic_20261007.json).
[Current shared compiler composition](perf_records/shared_resource_scalar_tensor_stages_composition_20261007.json).
[Complete QK functional cost](perf_records/smol_scalar_qk_complete_functional_cost_20261007.json).

## Direct PyTorch source inspection

Current cached LeRobot 0.4.4 and Transformers 4.57.6 sources match the versions
in the previously qualified TorchAO recipe audit. Their bytes and package
metadata are pinned; historical complete dependency/source closure is still
unproven, and the old capture loader path has been removed during earlier cleanup.

- `SmolVLMVisionAttention` computes Q/K/V projections, reshapes into heads, and
  selects the configured attention interface. The matching original FX graph
  records SDPA. Switching to an eager attention backend changes the evaluated
  floating operation order and requires the original consumer/model gate.
- LeRobot's expert `eager_attention_forward` explicitly widens Q/K to f32 before
  QK, computes masked softmax, and casts probabilities to the value dtype before
  PV. An integer lowering must retain these observations or pass its separately
  selected approximate numerical policy under the unchanged whole gate.
- The current host path incurs packing, reconstruction, bounds, endpoint
  evaluation, denominator and quantizer costs around device integer products.
  Offloading an integer product does not establish efficient complete attention.

The local selected GSIM FIRRTL has a normalization passthrough, with its
mean/max/inverse outputs invalidated. This proves no active normalization compute
unit in that **local elaboration**. The historical stock FireSim bitstream's exact
normalization capability lineage remains separate. Do not infer compute capability
from command fields or current default Scala configuration alone.

## Decisions and next work

1. Prioritize a complete attention lowering and representation that removes
   repeated host reconstruction/certification work. Include all packing,
   matrix service, transfers, readbacks and original output observations.
2. Audit the remaining text/action BMM coverage in the actual current source and
   executable. Static FX counts alone do not prove that a given operation executes
   on the host in stock1906 or stock2113.
3. Keep representation error and floating accumulation order separate. The
   encoded K16 partial-dot experiment completed all48 groups but failed115/1600
   original outputs (max absolute0.1637932062). It is rejected, with no target
   provider or timing credit. The raw-source partial-order control also completes
   all48 groups but fails113/1,600 original outputs (max absolute0.15374207497),
   despite removing representation loss entirely. Its exact sparse fallback costs
   are paid in161.43 native diagnostic seconds, distinct from213.87 total native
   diagnostic wall time. K16 target work is abandoned: this accumulation order
   alone is insufficient under the original model gate.
4. Producer facts are an exact shared optimization, but their measured scope is
   modest: the packing-only complete GSIM component improves6,920,348 to
   6,901,740.5 cycles (−0.26888%). The later producer maximum/certificate topic
   saves about4.079% functional instructions on a complete head group; it has no
   cycle measurement, and these overlapping results cannot be added.

## Compiler ownership and automatic-loop requirements

| Area | Owner | Required improvement |
| --- | --- | --- |
| Quantization capture and original/prepared correspondence | model2MLIR | Retain exact policy/weights/numerical observations and expose remaining floating products with explicit correspondence gaps |
| Host packing, source/effect proofs, reconstruction and numerical policy | Merlin | Discover complete producer/use/epoch closure from current typed IR; preserve original fallback and full-model gates |
| Device arithmetic, legal formats, bank residency and commands | OOT dialect | Derive implementation from semantics and declared hardware capabilities; include complete transfer/readout costs |
| Phase0 | Generic infrastructure | Generate capsules from remaining semantic operations and mixed precision boundaries, including real shapes, tails and independent cases |
| Phase1 | Generic infrastructure | Require whole-graph coverage with explicit UNKNOWN/fallback reasons; close FX→IR→host/device binding and original model correctness |
| Phase2 | Generic infrastructure | Price complete host arithmetic, allocation, metadata, transfer, cache and device service; calibrate against paired complete measurements |

The retained FX trace explicitly reports incomplete original→quantized and
quantized→prepared correspondence. Automatic routing must resolve those gaps or
report UNKNOWN. Model names, provenance IDs, observed golden values and these
benchmark counts must never select a production strategy.

[Pinned static graph and source audit](perf_records/smol_matching_fx_pytorch_lowering_audit_20261007.json).
[Independent whole recipe/quantization quality audit](perf_records/root_smol_torchao_recipe_quality_review_20261007.json).
[Packing-only complete GSIM receipt](perf_records/smol_produced_packing_complete_gsim_20261007.json).
[Sealed rejected accumulation controls](perf_records/smol_ordered_partition_controls_seal_20261007.json).
[Independent root closure of their 184 original declared paths](perf_records/root_smol_ordered_partition_controls_reclosure_20261007.json).
