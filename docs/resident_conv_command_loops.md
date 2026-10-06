# Retained resident convolution command loops

`GoldenResidentConv(..., compact_commands=True)` is an explicit, disabled-by-default target compiler choice. It retains bounded ordinary CPU K/spatial loops, including the existing alternating-bank B lookahead. Gemmini commands remain primitive; no FSM instruction is introduced.

The spatial rule derives an affine row step from declared shape, stride and residue layout. It checks every generated source row offset, retains the first real-weight preload, and handles the final short spatial group separately. Dynamic A and C rows carry complete reserved extents and maximum endpoints to the existing verifier. Increasing HWIO reduction order, accumulator initialization, output row layout and exact scaled readout are preserved.

For weight lookahead, the first panel is loaded once. A bounded odd/even K loop alternates the same disjoint bank slots. Static tails drain the last K panel and load the next spatial tap in the same order as the previous fully expanded implementation. No source or model identifier selects this rule.

## Evidence and admission

The 54 focused checks pass. Complete source-bound static CFG traces compare every encoded primitive and DMA pointer on nonsquare geometries, stride/residue layouts, odd/even K counts, output-channel tails, final row groups and both prefetch modes. Existing resource refusals remain active. Six previous/default IR cases are identical; the default H14 object byte-reproduces the actual current 1903 kernel.

The small independent 1,155-output i32 pair takes 4,496 GSIM cycles on both arms. A larger independent tail/prefetch pair is slower: 15,849 to 19,765 cycles. Both are complete, exact, guard/input checked and zero-FSM. This rejects blanket retention as a profitability rule.

The source-bound original H14/C256 candidate has 4,140 executable object bytes versus 153,052 for the exact current control. Its complete 65,328-command sequence and all DMA pointer values are identical. The original control takes 535,667 GSIM cycles; the candidate timing remains pending. Code size and instruction counts are witnesses, not cycle estimates. Hardware overlap and instruction-cache misses are unknown.

No normal source policy or whole-model object changes are enabled by this implementation. A positive complete original capsule and source/catalog/full original numeric qualification must precede a separate stock comparison. GSIM's memory regime differs from stock FireSim. Portable LLVM loop metadata comes from Merlin's existing `disable_loop_unroll` helper; target resource/address rules remain in OOT.
