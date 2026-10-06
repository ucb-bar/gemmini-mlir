# Coalesce DMA packets for a complete resident input

Owner: `reference_parity`. This is an opt-in Gemmini compiler schedule rule.
`resident_a_load_coalescing=False` preserves the default emitted objects. The
ordinary captured-source build and `--resident-a-load-coalescing` expose the rule.

## Legal layout and lifetime

`GoldenGemm.resident_a_load_tiles` groups one to four adjacent K tiles in an
input DMA packet. Grouping greater than one requires the already validated
complete cached A layout. It changes packet width, not scratchpad placement:
CONFIG_LD has a DIM-row block stride, so each extra DIM-column block lands at
the exact address used by the old individual packet. Every group stops at the
source K extent. Each M tail retains its exact row count. All input packets
remain before their consumers; B prefetch slots, B order, accumulator placement,
compute K order, source scales, rounding, activation and output stores are
unchanged. Input and B reserved extents retain the existing resource proof.

The general selector requires complete residency and multiple K tiles. It
reduces input DMA command count using the existing four-block primitive limit.
Its score is a command count, not a cycle prediction. Across the 24 selected source52 kernels, input packets decrease from 10,672 to 2,668; requested input bytes remain 2,533,888. An explicit remaining-B
slot selection retains this independent input grouping option.

The permitted q1013 ZIP supports this hypothesis: the owned static decode of
`fx_conv_45` contains 64-column input packets, source stride 512 and local block
stride 320. Its padding/channel layout differs from this dense cached-input
layout, and unresolved pointers and branches prevent an exact dynamic mapping
claim. The compiler rule comes from our typed dense layout and primitive DMA
semantics. Reference inputs, weights and numeric contracts are different.

## Matched evidence

The original 1897 M196/N256/K1024 i8 fixture was captured by a diagnostic native
wrapper. That whole model remained exact on all 1,000 original f32 words.
Control and candidate use identical operand/output addresses, all 50,176 output
bytes and 4,096 guard bytes pass, and both final ELFs contain no FSM commands.
The complete kernel measured 288,847 versus 277,410 GSIM cycles, a reduction of
11,437 cycles (3.96%). The old 832 input packets become 208. The decoded resident
input partition and every other command, including K order, are identical.
Requested operand bytes are unchanged; these are not measured DRAM bytes.

An independent M33/N73/K65 i32 contraction with amplitude 21 and explicit B slots
at rows 3072/4096 passed all 2,409 outputs and guards in GSIM and strict RV64GC
Spike. It exercises M/N/K tails and an independent B placement. Decoded-command
scalar arithmetic tests also exercise another explicit placement and K17.
Resource and typed-option refusals are tested. A fresh default object is byte
identical to the original 1897 M196 control (73fb5f25…2805e).

The focused suite passed 79 checks; the changed emission mutation guard and the
14 grouping checks then passed together. The emission guard binds the new
constructor option, so changing it after selection is refused. Ruff on the new
checks and `git diff --check` passed. The second original M784/N512/K256 i8 fixture also passed all 401,408 outputs and 4,096 guards, common addresses and strict Spike. Its complete kernel improved from 563,901 to 552,245 GSIM cycles (2.07%). The initial 3.5M cycle budget ended during output verification; those incomplete results remain retained. Both unchanged ELFs completed verification with a 6M budget. The normal source52 catalog qualification is recorded separately.

This evidence establishes local correctness and a matched simulator direction.
It does not establish whole-model FireSim savings or additive gains with other
schedules. The original zero-tolerance accuracy gate remains unchanged. The
shared global solver is not claimed to select this command-count opt-in rule.

## Durable receipts

- `docs/perf_records/resident_a_dma_coalescing_capsule.json`
- `docs/perf_records/resident_a_dma_coalescing_whole_qualification.json`
- Owned raw experiment: `out/resident_a_dma_coalescing/`

Whole-model accuracy, selected object closure and stock FireSim timing are
independent gates. Child token usage is unavailable; root retains cumulative
owned-thread counters without exclusive attribution to this experiment.
