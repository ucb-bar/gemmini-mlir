# Calibrated resident A and B prefetch, 2026-10-05

## Hypothesis and emitted change

The prepared TinyLlama catalog has 44 contractions with M=8, N=256, K=2048.
Its existing wide-B tile schedule uses one N panel. The command-count policy
therefore declines cached A and B prefetch: caching does not reduce commands or
DMA bytes. Command count does not price overlapped issue, DMA and execution.

The explicit `dense_input_policy="resident_a_prefetch"` alternative retains
bm=1, bn=16, wide B, source dimensions and numeric policy. It makes the full A
operand resident and uses the existing increasing-K B prefetch implementation.
Legality derives from one row tile and the target scratchpad resource verifier.
Competing cached or pipelined placements and shapes that do not fit refuse.
No model, source label or golden output chooses this policy. It remains default
off; an exact source-bound calibrated alternative makes the selection.

Target placement and legality belong in OOT (`5eb6f9d`). Shared planning, normal
dispatch, host compilation and source/implementation binding belong in Merlin
(`2221ce9ad`). The existing GoldenGemm and shared global-plan IR are reused.

## Paired device measurement

Both linked capsules contain both original compiled kernel objects in identical
order. Only the call relocation changes. A, B and guarded output buffers have
identical addresses. Inputs and every expected output are immutable and have
equal SHA-256 hashes across the pair. The near full range int8 fixture has
amplitude 21; every one of 2,048 int32 outputs and 2,048 guard bytes is checked.

| Scope | Control | Resident A + B prefetch | Change |
| --- | ---: | ---: | ---: |
| Actual GSIM kernel ROI cycles, common addresses | 111,885 | 76,456 | -35,429 (-31.6655%) |

The ROI includes the device primitive transfers, compute, output readback and
completion fence. Fixture creation and whole-model host execution are outside
this ROI. This engine is pinned by SHA-256 and is not stock FireSim. Both final
ELFs pass strict RV64GC Spike and the audit of every executable section for
forbidden FSM instructions. An independent M=7, N=35, K=69 tail case checks
all 245 outputs and 2,048 guard bytes and passes strict Spike. Independent
resource/numeric-policy/refusal tests also pass.

Earlier separately linked probes measured 111,475 versus 76,205 cycles. They
moved physical buffer addresses, so the common-address pair supersedes that
screening result. Initial functional Spike counts rose 7,706 to 11,228 retired
instructions; those counts are not timing estimates.

Full pinned evidence: [receipt](perf_records/tiny_resident_a_prefetch_gsim.json).
The receipt retains exact export/object/ELF/engine/input/source/UART pins and
the actual shared planner result. Hardware occupancy and physical floor remain
UNKNOWN. This fixture price is not multiplied by the 44 recurring calls.

## Normal build and dispatch correction

`export_contraction` exposes the alternative in the ordinary source-bound
compiler API. `optimize_contraction` evaluates both pinned candidates with the
existing shared planning adapter and activity timeline, then recompiles the real
winning object and requires exact object equality. A model calibration packet
reaches `DeviceRouting.catalog_builder` through `merlin_builder`; it is applied
before ordinary source offload, shim generation and host compilation.

The first full build correctly refused source bytes printed by xDSL 0.72 instead
of the frozen 0.65 frontend. Pinning the original frontend import restored the
exact source SHA; source matching was not weakened. The next build correctly
refused an ambiguous source-bound kernel. Merlin had deduplicated equal shape
and precision into one callee even when their source bindings selected different
device implementations, and catalog validation required unique shapes.

The generic correction incorporates the bound implementation into callee
identity and resolves the device symbol from each exact source binding. It
checks current source ordinal, exact tensor types, dimensions, precision and
complete coverage. A single callee referring to different implementations
refuses. Equal implementations preserve the old default declaration/call bytes.
Tests cover selected/default/default calls with identical shapes, source ordinal
mutation, shape mutation and ambiguous callee mutation.

The normal Tiny build now retains all 155 calls and has six implementations.
One source contraction uses the measured winner; the other 154 catalog bindings
are byte-for-byte unchanged. The complete original prepared source SHA-256 is
`86cd4946421ba3d03062721604c62ef4591e62fd6e5ea04304a0102ea5cde454`.
Full native qualification passes all 256,000 original compiled output words and
the original elementwise Torch gate (atol=0.03125, rtol=0.02). Strict target and
hardware status are recorded separately in the receipt.

Full strict Spike also passes all 256,000 words, original Torch gate, zero
memref-rank mismatches, DONE and exit 0. The final ELF passes the audit of every
executable section with zero FSM instructions:
`9112dde7f14cd191c2bf1a5861025f08618d78c5d5a1172242b14f9b4469968e`,
marker `ae1d2385b858`. It retires 166,648,514 functional instructions, versus
166,605,674 for stock winner 1880; these are not hardware cycles. See the
[whole strict receipt](perf_records/tiny_resident_a_prefetch_whole_spike.json).
The candidate is held with a prepared standard reference adapter. No new stock
job has been admitted and there is no full-model timing improvement claim.

## Controls and remaining scope

The normal calibrated route accepts `large_n`, `prefetch_b`, the supported
`dense_input_policy` families (including this new explicit alternative), and
`dense_b_slot_policy`. The source binding selects an implementation, not an
optimization strategy based on a source name. Uncalibrated bindings retain the
ordinary catalog path and deduplicate only compatible identical implementations.

Manual tile sizes, transfer/layout flags and primitive probes remain diagnostic
paths unless exposed through a legal source policy and validated calibration.
This integration solves one exact contraction with an opaque invocation duration;
it is not a whole-model activity/resource/memory optimizer. Conv regions and
host epilogue/schedule decisions do not acquire full-model prices from this
capsule. No new duplicate scheduling IR, secondary DAG or workload selector is
introduced. The automatic loop can enumerate legal compiler alternatives,
produce paired source-bound evidence, then feed the same existing normal route.

## Tokens and promotion

Subagents do not receive a token usage counter: `token_usage_available=false`.
Root records shared campaign checkpoints. Exact exclusive optimization usage,
input/cache/output split and billing attribution are unknown. Each durable
receipt records hypothesis, owner, actual emitted change, metric scope, gates
and retained/refused/promotion state. No stock job is submitted until the full
original source/native/strict target/no-FSM qualification closes.
