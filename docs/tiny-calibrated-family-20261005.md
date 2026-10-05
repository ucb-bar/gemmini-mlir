# Exact source-bound family calibration, 2026-10-05

This continues the qualified [single-source study](tiny-calibrated-resident-a-20261005.md).
The model-specific experiment driver enumerates equivalent contractions from
current source tensor types, dimensions, dense layouts and resource-legal
generated schedules. Source region labels and ordinals bind evidence after
selection; they do not choose the optimization policy.

## Shared physical implementation

Previously the operation exporter named device code by a digest that included
its source binding. Expanding one equivalent family would have emitted 44
identical device functions. The target emitter now separates the exact logical
alternative identity from the physical implementation identity. The latter is
the SHA-256 of the actual existing generated primitive MLIR, including its
types, layout/placement, constants and numeric policy. The bound emitter clones
that same target IR and verifies its fingerprint before compilation. It also
retains the source hash, exact ordinal/types and schedule mutation checks.

This uses existing target IR and shared planner/emission contracts. No second
scheduling IR or workload selector is introduced. Ordinary uncalibrated catalog
symbols are unchanged. Existing historical artifacts remain immutable; changing
implementation naming requires freshly bound exports and measurements.

OOT commits `a5cd11e` and `c047d0b` implement and test this identity rule. Tests
prove independent source labels keep distinct logical alternatives but compile
byte-identical physical code, different shape/schedule produces distinct code,
primitive mutation refuses, and two exact calibrated source selections link
one shared body through the ordinary model catalog. Prices in that regression
are explicitly unit-test mocks, not hardware evidence.

## Fresh measured pair and exact applicability

All 44 eligible current-source uses have individual control and candidate
exports and exact calibration packets. Every export independently verifies its
current source ordinal, tensor types and resource/legal schedule. Within each
arm, every generated primitive, final object, target IR and LLVM IR is identical.
The physical objects are:

| Arm | Kernel | Object SHA-256 |
| --- | --- | --- |
| Control | `gemmini_export_01ab3f78efa16052` | `6bbf9a886fa0ba7c0c50d8430330b28fdd40e82def36f62fbd71c5afa8de172b` |
| Resident A + B prefetch | `gemmini_export_13697184e4dab8ae` | `6db3803bcbcb92a8235a765ec00b689a8403e3de185ad9883198a2116469a41e` |

A fresh actual GSIM pair for these exact physical objects reproduces 111,885
versus 76,456 kernel ROI cycles (-31.6655%). Both ELF images contain both objects
in identical order, with identical static operand and output addresses. Every
one of 2,048 int32 outputs and 2,048 guard bytes is checked. Strict RV64GC Spike,
zero-FSM final executable audits and engine/ELF/input/object/UART pins close the
capsule. Initial diagnostic environment/metadata failures are retained; complete
receipts are separate and require successful measured execution.

The source/semantic/primitive/object equivalence is independently proved for
every binding. The prices are **one reused calibration assumption**, rather than
44 independently observed source-context durations. One synthetic immutable
amplitude-21 int8 operand fixture was timed, with common buffer addresses and
one reset harness context. It is not captured matmul_2 model operands. Each
source selection assumes that this physical implementation fixture cost ranks
its alternatives usefully; actual model operands, addresses, cache state, host
packing, dispatch and full-program context were not individually timed.

The normal solver rechecks each source, calibration, object and output contract
independently; those identity checks do not prove the reused latency assumption.
The fixture durations are never summed into a model prediction. Whole-model
cycles, compute/DMA/issue occupancy and physical floor remain UNKNOWN until
independently measured. The full FireSim comparison is the final aggregate
performance qualification. Five device bodies serve the complete 155-call
model, with the 111 unselected bindings preserved.

## Whole-model promotion gates

The family model uses the original eight-token, 22-layer source and original
packet2/scalar8/K2 host settings of verified stock job 1880. Adjacent-RNE1902
is a separate experiment. No accuracy/scale/tensor/quantization gate changes.
Promotion requires all 256,000 original compiled f32 words, original Torch
atol=0.03125/rtol=0.02, strict Spike rank0/DONE/exit0, exact source/object binding
and zero FSM words in every executable section. Only the fully qualified family
candidate is eligible for one stock FireSim comparison against 1880; the
intermediate single-source candidate remains held.

The fully normal family build passed native and strict target gates: all 256,000
original compiled output words, the original Torch gate, rank0/DONE/exit0 and
the final audit of every executable section. Its ELF SHA-256 is
`035b0458e0f899cd3e4406d525c950e7a72f72a63371c4786db0551b56ea673a`,
marker `907c25bd4349`. The strict functional simulator retired 166,799,960
instructions; this is not a hardware cycle measurement. Recovery independently
reclosed the standard collector adapter and admitted one stock job **1911**
against verified job 1880 (531,072,370 cycles). Performance promotion is pending.

The target and native host LLVM and target `model.o` are byte-identical to 1880.
Native reuses the prior native object only after full LLVM byte equality, then
rebuilds the current source-bound ABI shim and exact scalar integer stand-ins
and reexecutes the complete model. This source and reuse proof is retained.

There is a material attribution caveat: identical scalar runtime C is now
compiled with the lowered model's Clang ABI instead of GCC. Tiny calls both
`rsqrtf` and `memrefCopy` from this object, so job 1911 measures the complete
qualified compiler/runtime/device candidate. It does not isolate the device
schedule change. The default compiler consistency fix remains enabled.

A separate controlled link first reproduced the immutable 1880 ELF byte for byte
(`d0a2aaa939fdbe1ffd7d2931901cb49b37c6cbf0f00250d9ea7cc38c8c54de70`).
It retains every 1880 host, runtime, main, startup, weights and dispatch-shim
object and changes only the actual deduplicated family device aggregate. The
selected physical symbol is rebound losslessly to the old stable three-pointer
ABI symbol. Function instruction bytes and typed relative relocations prove
that the other four primitive implementations are unchanged. Device link layout
is included in this arm. An additional byte/relocation check proves the selected
function is exactly the original emitted calibrated primitive after this symbol
rebind. Fresh native and strict whole-model gates passed all 256,000 original
words, the original Torch gate and every executable-section no-FSM audit. The
controlled ELF SHA-256 is
`fa246138f914c225eb09cd2f83dc9e341d74b80757ec69dcb309e485bd143dbc`.
Recovery independently reclosed its standard adapter and admitted stock job
**1912** against 1880. It retired 166,760,642 instructions in strict Spike;
hardware cycles remain unknown. Its marker `37bdf9be0856` is inherited from the
frozen 1880 main object; exact ELF and component hashes identify the controlled
candidate, and this nonunique marker alone is not a new build identity.

Source preparation, 88 individual export compile durations, exact calibration
packets, shared plan receipts and complete artifact pins are retained under
`out/artifacts/probes/tiny-calibrated-resident-a-family-20261005` and in
[the GSIM/family receipt](perf_records/tiny_resident_a_prefetch_family_gsim.json)
and [the full normal strict receipt](perf_records/tiny_resident_a_prefetch_family_spike.json).
[The controlled experiment receipt](perf_records/tiny_resident_a_prefetch_family_controlled.json)
pins baseline reproduction, unchanged objects, lossless symbol rebind and
[the controlled full strict receipt](perf_records/tiny_resident_a_prefetch_family_controlled_spike.json).

## Ownership and accounting

Target primitive generation, code identity, scratchpad policy and ISA remain in
OOT. Merlin owns generic source/dispatch identity, shared planner, host build,
buffer ownership and runtime. The generic equal-shape dispatch correction is
`2221ce9ad`; default compatible identities retain prior call/declaration bytes.

`token_usage_available=false` for this subagent. Root attaches shared campaign
checkpoints. Exclusive per-optimization token use, billing and input/cache/output
allocation are unknown. The receipt records hypothesis, actual emitted change,
scope, numeric/ISA gates, retained negatives and promotion state.
