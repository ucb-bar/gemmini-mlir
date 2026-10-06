# Retained resident convolution command loops

`GoldenResidentConv(..., compact_commands=True)` is an explicit, disabled-by-default target compiler choice. It retains bounded ordinary CPU K/spatial loops, including the existing alternating-bank B lookahead. Gemmini commands remain primitive; no FSM instruction is introduced.

The spatial rule derives an affine row step from declared shape, stride and residue layout. It checks every generated source row offset, retains the first real-weight preload, and handles the final short spatial group separately. Dynamic A and C rows carry complete reserved extents and maximum endpoints to the existing verifier. Increasing HWIO reduction order, accumulator initialization, output row layout and exact scaled readout are preserved.

For weight lookahead, the first panel is loaded once. A bounded odd/even K loop alternates the same disjoint bank slots. Static tails drain the last K panel and load the next spatial tap in the same order as the previous fully expanded implementation. No source or model identifier selects this rule.

## Evidence and admission

The 54 focused checks pass. Complete source-bound static CFG traces compare every encoded primitive and DMA pointer on nonsquare geometries, stride/residue layouts, odd/even K counts, output-channel tails, final row groups and both prefetch modes. Existing resource refusals remain active. Six previous/default IR cases are identical; the default H14 object byte-reproduces the actual current 1903 kernel.

The small independent 1,155-output i32 pair takes 4,496 GSIM cycles on both arms. A larger independent tail/prefetch pair is slower: 15,849 to 19,765 cycles. Both are complete, exact, guard/input checked and zero-FSM. This rejects blanket retention as a profitability rule.

The source-bound original H14/C256 candidate has 4,140 executable object bytes versus 153,052 for the exact current control. Its complete 65,328-command sequence and all DMA pointer values are identical. The complete original pair takes **535,667 to 522,644 GSIM cycles**, saving 13,023 (2.43%). All 50,176 original i8 outputs, 4,096 dirty guard bytes and 640,000 immutable input bytes pass strict and GSIM. Hardware overlap and instruction-cache misses are unknown.

No normal source policy or whole-model object changes are enabled by this implementation. The large original body's local gain and the smaller body's loss remain separate profitability evidence. Normal source/catalog/full original numeric qualification must precede a separate stock comparison. GSIM's memory regime differs from stock FireSim. Portable LLVM loop metadata comes from Merlin's existing `disable_loop_unroll` helper; target resource/address rules remain in OOT.

The receipt pins 193 artifacts, including all three complete capsule pairs, emitted target modules, previous/default byte identity and exact sealed typed catalog/source-scale binding. Its recloser also reparses both actual target modules and compares the full encoded command and pointer sequence:

```sh
PYTHONPATH=/scratch/agustin/tmp/gemmini-resident-command-loops-20261006:/scratch/agustin/tmp/merlin-residual-output-word-20261005/src \
  /scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  tests/resident_retained_command_closure_probe.py \
  docs/perf_records/resident_retained_command_capsules.json
```

The historical capsule result field `prefetch_b` records the CLI flag. Actual implementation selection is bound by each fixture's `resident_options` and emitted module. The independent prefetched arms both use `prefetch_b=True`. Child token attribution is unavailable; root retains session accounting.

`captured_requant_bundle.build(..., compact_resident_commands=True)` and `--compact-resident-commands` now expose this explicit option on an already admitted resident convolution family. The default remains false. An opt-in does not infer profitability: unadmitted dense/flat families remain unchanged. The source route records the actual applied decision and requires the existing proved flat-spatial/virtual-padding route.

The normal source bundle preserves all 52 numeric/shape bindings, all adapter bytes, rewritten source MLIR and all 47 unselected kernel objects. Exactly five admitted resident objects change in the frozen source experiment. The 34 binding/primitive checks pass. Whole original numeric and final ELF qualification remain pending; no stock performance claim or automatic policy follows from this bundle closure. GSIM's memory regime differs from stock FireSim. Portable LLVM loop metadata comes from Merlin's existing `disable_loop_unroll` helper; target resource/address rules remain in OOT.

## Complete normal and frozen control qualification

[The whole qualification receipt](perf_records/resident_commands_normal_whole_qualification.json) closes 483 immutable artifact pins. The ordinary normal build and the controlled 1903 build each pass native and strict RV64GC Gemmini Spike against every original 1,000 output word, with zero tolerance and a fresh final zero-FSM audit. Strict Spike counters are retired instructions: 10,120,329 for the current normal build and 9,283,412 for the frozen host comparison; these are not target cycles.

The controlled arm byte-reproduces actual stock control 1903 (`7ee47f…30d97`) and changes exactly the five selected resident kernel leaves. All 52 adapters, 47 other requant kernels, residual arithmetic, two saturation adapters, stem, packed mean, host/runtime/weights and public ABI remain pinned. Its ELF is `eb003f…fb17e`. The current normal build uses the frozen committed Merlin `1c5b1d1dc` and is recorded separately; it is not used to claim device-only attribution.

Whole performance remains unknown until the sole queue owner independently closes and measures this isolated arm. The inherited control build marker is nonunique; the actual staged full ELF SHA is authoritative. Failed attempts from omitted RTL facts and duplicate dialect registration are retained as failed recipe evidence.
