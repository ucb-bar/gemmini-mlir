# Complete multirow input residency

`Shape.cache_a` can retain a complete M block across output-channel blocks.
Each row tile has its own contiguous K-panel range. Admission requires
`ceil(M / DIM) <= bm`; the usual accumulator and scratchpad limits still apply.
The compute address includes the row tile as well as the dynamic K tile.

For B prefetch, the **entire reserved A panel**, including unused reserved row
tiles, must fit the lower two scratchpad banks. B alternates in banks two and
three. Without prefetch, B begins after the complete reserved A panel. Bias,
source K accumulation order, output rounding and saturation remain unchanged.

Two explicit compiler search options use this schedule:

- `resident_a_command_cost` selects complete input residency when it reduces
  primitive command count and passes the resource checks.
- `transfer_command_cost` compares that candidate with the existing full-K
  banked candidate. Ties use DMA request volume and output-block count.

Both use source dtype, dimensions, epilogue fields and target resources. Model
names, source region IDs and golden values are absent from strategy selection.
These options are disabled by default. The costs are command counts and byte
estimates, not predictions of hardware cycles.

## Qualification

The current Merlin core and explicit OOT GSIM support provider compiled and ran
five capsules. Every output and both 1,024-byte guards passed, as did the final
linked executable-section no-FSM audit and strict RV64GC Gemmini Spike.
Independent M17/N73/K65 tails cover bias, both B-reuse modes, prefetch and a
reserved M block larger than the actual row count. Existing single-row cached
and uncached IR remains byte-identical in five independent cases.

For M49/N2048/K512 with a nontrivial source scale, complete A residency plus B
prefetch reduced fenced GSIM kernel cycles **367,749 to 284,299 (22.69%)**.
Total harness cycles increased **1,479,163 to 1,493,952**. The kernel result
justifies a whole-model experiment; it does not establish a whole-model gain.

See [qualification record](perf_records/multirow_resident_a_gsim.json) and
[independent review](perf_records/multirow_cache_a_independent_review.json).
