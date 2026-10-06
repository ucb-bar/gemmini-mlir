# Retained resident reduction commands

The bounded dynamic-B PRELOAD extension is target owned. The row's declared
minimum, maximum, alignment and complete reserved extent are checked against
every value in the actual ordinary LLVM CFG before lowering mutates the module.
Dynamic B operands precede dynamic C operands. GARBAGE weight reuse remains a
separate static primitive. Shared fixed-width AND/OR/XOR tracing belongs in
Merlin; instruction packing remains here.

The optional resident reduction loop keeps the first K step explicit, then
retains the remaining K steps and later rows as ordinary CPU loops. Tap/K/N/row/X
order, overwritten versus accumulated outputs, stationary-weight reuse, all
DMA requests and every numerical operation match the existing command stream.
The current default and the previous retained-row kernel reproduce byte exactly.

| Complete GSIM kernel scope | Control | Retained K | Change |
| --- | ---: | ---: | ---: |
| Original H28/C128 boundary, 100,352 i8 outputs | 512,059 | 503,106 | −1.7484% |
| Independent 11×23/C48/N33, 8,349 i32 outputs | 34,368 | 29,761 | −13.404% |

Every output, 4,096 guard bytes and every immutable input byte pass on strict
RV64GC Spike and GSIM. Both pairs use identical operand/output addresses and
have a final executable zero-FSM audit. The independent case has N/X/stripe
tails; complete trace tests also cover odd K tile counts and resource refusals.
These are kernel measurements. Whole-model timing and the new whole-model
original 1,000-word gate remain pending.

For H28 the body shrinks from 132,308 to 16,996 bytes and the retired count drops
from 196,734 to 172,555. Integer loads fall from 35,705 to 1,211 and stores from
5,398 to 663, while conditional branches rise from 8 to 14,336. The earlier
retained-row arm has 60,000 bytes and 187,448 instructions but only a 0.1012%
cycle improvement. The independent K-loop arm improves despite retiring more
instructions (9,630 to 10,764). A global cycles-per-instruction or text-size
ranking cannot explain this evidence.

All three H28 arms issue the same 32,256 computes and preloads: 2,304 real B
preloads, 29,952 GARBAGE reuses, 247,808 requested source-load bytes, 100,352
requested store bytes and 14,848 local zero-fill payload bytes. Nominal padded
work is 516,096 DIM rows; this is not a hard cycle floor. Physical traffic,
cache misses, CPU issue width and DMA/execute overlap remain unknown.

The existing FP RAW-distance producer is reused, with empty FP edges for these
integer/custom command bodies. Integer-address dependencies remain unknown.
No rates are fitted here and no reference timing labels are used. Source and
machine-code features are in `out/resident_dynamic_b/feature_exports`; the
239-pin receipt and reclosure script retain source snapshots, exact generated
objects, original fixtures, strict logs, complete command fingerprints and
all paired observations. The preceding row-loop archive is unchanged.

The compiler extension is explicit and defaults off. Normal source binding and
full original-model qualification are the next gate; there is no whole-model
gain forecast or automatic profitability policy.
