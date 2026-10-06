# Capacity-proved complete weight residency

The permitted q1013 ZIP contains an executable strategy beyond the old cached-B
compile-size budget. Its `fx_conv_25` M784/N256/K512 body issues 128 B loads of
16×64 bytes and 392 A loads of 16×64 bytes per invocation. These cover each
weight and activation once. Its 25,088 computes reload the mesh B each time.
Other accumulator loads contribute 802,816 requested bytes; the complete
reference requested load payload is 1,335,296 bytes. Requested payload does not
establish physical DRAM traffic. This is evidence from our owned ZIP replay,
with Jack's own numeric/input contract.

`Shape.validate(cached_b_resource_capacity=True)` and the corresponding
`GoldenGemm` keyword expose the larger legal family explicitly. The old
128-tile budget remains the default. This option still proves the actual placed
B endpoint, including gaps between banks, every A slot, and accumulator bounds.
The ordinary source capture compiler accepts `dense_cached_b_capacity=True`
(CLI `--dense-cached-b-capacity`). Its admission retains existing cached A/B
families and only considers uncached full-weight matrices beyond the old budget.
A resource or layout refusal retains the previous generator. The manifest exposes
this actual capture-command selection owner; a shared solver does not choose it
and profitability is unknown until measured.

Aligned K uses the existing coalesced A loader and disjoint next-M scratch/ACC
slots. A K tail uses the existing exact one-panel A loader and retains all B.
Numeric type, scale, activation, unbiased seed and increasing source K are
preserved. The three-pointer ABI requires immutable A/B and fully written,
disjoint C for the whole call. No workload, source ID, or captured value selects
the strategy.

## Complete matched capsules

On original source-qualified M784/N256/K512 operands the pinned GSIM comparison
is 599,463 → 434,406 cycles (27.53%). The original M784/N128/K512 comparison is
318,381 → 216,041 cycles (32.14%). Both finish all byte outputs, exterior guards,
and unchanged inputs at common addresses, with actual strict RV64GC Gemmini
execution and zero FSM instructions. An independent non-square i32 case with
M37/N69/K513, including all three tails, measures 37,386 → 24,080 cycles and
finishes all 2,553 outputs. These are kernel observations in the pinned GSIM
memory regime, not FireSim or whole-model predictions.

The same independent M37/N69/K513 case with scaled byte output and ReLU measures
36,978 → 23,116 cycles. Its complete output, input preservation, and exterior
guards pass both GSIM and strict Spike.

A separate interpreter of the emitted static LLVM CFG follows every DMA source
and scratchpad cell, verifies each compute's original tensor indices and
increasing K, tracks accumulator lifetimes, and proves every output is stored
exactly once. N256 loads fall from 4,032 to 520 commands and requested payload
from 1,720,320 to 532,480 bytes; compute and store counts stay unchanged. Real B
mesh reloads increase from 3,584 to 25,088. Eight established default kernel
objects, including the actual original controls, reproduce byte-for-byte.

The normal source and whole-model qualification are separate gates. No whole
FireSim timing or automatic policy promotion follows from these capsules.
