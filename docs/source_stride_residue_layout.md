# Group output rows through a source stride residue layout

The naive bounded channel panel was slower because it computed each output row
separately. A source-derived row permutation removes this extra mesh work.
Padded input rows are placed in slabs by their residue modulo spatial stride.
Rows that the source visits consecutively now follow each other in a slab.

For an even input pitch, mesh A stride 2 gives output pitch `input_pitch / 2`.
Two seven-wide output rows fit in one 15-row compute, including one ignored
gap. The 14×14/Cin512/Cout512 contraction therefore uses four row groups and
retains its original channel panel 16. Its 8,192 input rows, 256 weight rows
and 1,024 accumulator rows have complete disjoint resource bounds.

| Same original-input kernel | GSIM cycles | Correctness |
| --- | ---: | --- |
| Exact 1897 flat control | 774,520 | All 25,088 int8 outputs and 4,096 guards |
| Source stride residue layout | 712,682 | Same complete values and guards |

The measured saving is **61,838 cycles (7.98%)**. Actual target CFG tracing
retains 36,864 computes, reduces requested input bytes from 409,600 to 100,352,
and preserves all weight/output bytes. Both arms share operand/output addresses,
pass strict RV64GC Spike and contain zero FSM instructions. The new family
preserves the original increasing HWIO reduction and store arithmetic.

The row map is a bijection over allocated padded rows. Input pitch is rounded
to a multiple of source stride; added cells are zero. Every compute endpoint
fits in its channel/residue slab. The complete input stays live throughout
reduction, the weight panel occupies a disjoint reserved extent, and dummy
accumulator gaps are never stored. Independent decoded-command tests compare
source convolution arithmetic, K order and output coverage. A 9×5/Cin48/Cout67
case with three output rows per tile validates all 1,005 int8 values and 2,048
guards in GSIM and strict Spike, including height, width and channel tails.

`--source-stride-row-residue` requires `--source-stride-resident`. It adds a
general legal layout alternative to the target's issue/transfer ranking. The
ordinary source-bound bundle compiler consumes the selected generator. Existing
defaults retain their original objects. A geometry with no row-group reduction
keeps the previous source row order, preserving the queued 1914 arm's object.
The ranking has no hardware timing claim or model-name selector.

The owned reference's plane stride 320 and 16-row commands motivated the search.
The new residue mapping independently proves current source semantics; unresolved
reference addresses and its different numeric contract remain separate evidence.
No whole-model hardware gain is established yet. Exact receipts are in
[source_stride_residue_capsule.json](perf_records/source_stride_residue_capsule.json).
