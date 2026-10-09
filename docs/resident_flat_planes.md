# Complete resident input for flat spatial convolution

The explicit `flat_resident_planes` capture option and `--flat-resident-planes`
command option select a general layout after source-proven virtual zero padding.
Selection uses the existing typed convolution generator and declared resources;
it has no workload, region, symbol or observed-input predicate. The default is off.

For stride-one 3×3 convolution, each channel tile owns three separate planes of
`(H+2)*W` scratchpad rows. Plane `kw` contains the source cell
`(padded_y-1, x+kw-1)` or a statically proved zero. Each plane has only W columns,
so consecutive output pixels remain consecutive mesh rows across logical source
row boundaries. The A address for output pixel m at tap `(kh,kw)` is
`ki*3*(H+2)*W + kw*(H+2)*W + kh*W + m`.

Input DMA segments partition every allocated A cell exactly once. Complete A stays
immutable through all channel panels. Each output consumes the original increasing
HWIO reduction sequence. B and output ACC panels retain disjoint declared extents;
completed output tiles use the original typed store and numeric readout contract.
All three duplicated input planes are counted in the capacity check. Paired output
store ABIs are currently refused by this family.

The option preserves the previous generator when padding, stride, channel alignment,
input capacity, weight placement or complete accumulator capacity cannot be proved.
Declared admission is not a profitability decision. Requested DMA payload is distinct
from physical DRAM traffic, and smaller code or fewer loads do not establish cycles.

The experiment compares the original H7/C512 input boundary with its immutable
current i32 readout object, plus an independent non-square 3×5/C32/N67 case. The
reference ZIP's own H7 scope has the same compute count but a different i8 store
contract; its historical timing is used to identify a scheduling question, not as
an apples-to-apples capsule control. Whole-model timing remains unknown until a
qualified stock comparison completes.

## Matched capsule evidence

| Complete GSIM scope | Control | Selected | Result |
| --- | ---: | ---: | --- |
| Original H7/C512,25,088i32 outputs |764,114|718,722|−5.94%|
| Existing flat-loop-only alternative,same original inputs |765,872|718,722|−6.16%|
| Independent3×5/C32/N67,1,005i32 outputs |8,076|7,995|−1.00%|

All complete scopes pass strict RV64GC Spike, all output bytes,4,096guard bytes,
immutable input checks and a final zero-FSM audit. The initial3M-cycle original
attempts ended during post-ROI integrity checks; they remain incomplete receipts.
Byte-identical ELF replays at8M close qualification. These are kernel GSIM cycles,
not stock FireSim or whole-model savings.

The original pair retains36,864compute and128i32-store commands. Aloads change
2,016→328 and requested Abytes369,664→68,096; Bloads/requested payload are unchanged.
Retired instructions rise140,448→214,084 while touched instruction bytes fall
69,602→13,992. Neither statistic establishes cache misses or overlap. Four existing
default resident objects and the actual control and selected capsule objects
reproduce byte-for-byte.
