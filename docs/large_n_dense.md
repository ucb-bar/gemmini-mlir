# Short-M, large-N dense GEMMs

The grouped `wide_b` schedule loads up to four adjacent 16-column weight tiles
per DMA command **inside each output-channel block**. It retains the complete
logical B row stride and includes the actual N-block offset. The final group
uses its exact column count, and the last K tile uses its exact row count.
Cached B uses the same grouping within each cached K panel. Existing 32/48/64
column wide loads retain their behavior.

Only DMA command grouping changes. Arithmetic order, signed i8 operands, i32
accumulation, output layout, and optional store epilogues stay the same. The
resource checks still apply. In particular, caching B requires a single output
N block, and caching A requires one M tile with sufficient scratchpad space.
No hardware-loop instruction or RVV instruction is introduced.

## Opt-in source-bound policy

The new policy is explicit:

- `choose_shape(dims, large_n=True)`;
- `golden_device_catalog --large-n` and `golden_contraction_upstream --large-n`;
- `merlin_builder(llvm_bin, large_n=True)` for complete-model compilation.

It enables grouped B loads for M≤16 and N>64. It also considers keeping A in
scratchpad across two or more N blocks, provided the resource checks pass and
the analytical primitive count decreases. The existing default policy remains
unchanged. Catalog manifests record `large_n_grouped_b_v1` and bind the selected
schedule into kernel symbols. The structural integer-GEMM matcher is unchanged;
nonzero accumulator initialization still refuses this route.

The estimator counts `ceil(channel_tiles_in_block / 4)` B commands per K tile,
rather than treating every output block as one wide transfer. Its padded panel
byte count remains an upper bound; grouping does not by itself reduce the
logical weight payload.

## Numeric and performance gates

Streaming grouped B, cached B, and cached A all pass mixed M/N/K-tail probes
(M23/N173/K37 and M8/N173/K37). The fixtures use signed operands with amplitude
17, independent integer oracles, complete output checks, and output guards.
All pass stock GSIM, strict Gemmini Spike, and final ELF no-FSM audits.

Representative TinyLlama projection, M8/N2048/K2048, i32, bm1/bn64, no bias:

| Schedule | GSIM kernel cycles | B DMA commands | A DMA commands |
|---|---:|---:|---:|
| Existing narrow B, uncached A |1,336,714|16,384|256|
| Grouped wide B, uncached A |979,233|4,096|256|
| Grouped wide B, cached A |927,138|4,096|128|

Caching A across only two N blocks gives another **5.32% reduction** over
wide B alone, supporting the opt-in reuse threshold.

The combined change is a **30.64% reduction** in the standalone GSIM measurement. Both check
all 16,384 outputs plus guards and pass actual Spike. Padded array issue floor
remains 262,144 cycles. FireSim A/B jobs 1763/1764 use the exact audited ELFs;
these measurements are pending and no complete-model speedup is inferred.

`tests/gsim_gemm_probe.py --static-inputs --embed-expected` embeds deterministic
input bytes and expected outputs so scalar fixture initialization does not
consume the cycle-accurate simulator budget. Data-only assembly emits the
inputs; all kernel instructions still come from typed xDSL/LLVM lowering. The
probe explicitly compiles for RV64GC.

## TinyLlama catalog analysis

On the pinned existing TinyLlama catalog, counted with source binding
multiplicities:

| Quantity | Existing | Opt-in candidate |
|---|---:|---:|
| B DMA commands |4,040,704|1,010,176|
| A DMA commands |38,144|24,768|
| Total primitive commands |12,186,896|9,142,992|
| Compute commands |4,040,704|4,040,704|
| Padded array issue floor |64,651,264|64,651,264|

These counts are not a calibrated cycle prediction. The model still has the
same arithmetic and nearly the same requested payload, and host attention,
nonlinearities, and data movement remain separate work.

The i32 store path retains its proven single-tile readout. Wide i32 readout
would need a separate target capability proof and hardware numeric gate.

Receipts: `docs/perf_records/large_n_gemm_gsim.json` and
`docs/perf_records/tiny_large_n_catalog_analysis.json`.
