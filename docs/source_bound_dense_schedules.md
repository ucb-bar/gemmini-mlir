# Source-bound dense schedule options

Captured requant bundles expose independent, default-off options:

- `--banked-prefetch`: i8 output, no bias/competing A cache, at least two M tiles,
  full K in one 32/48/64-column A transfer, and N fitting one full-tile wide store.
  Static A/B bank footprints and accumulator-slot capacity must fit. The selected
  schedule is bm1, cached B, two prefetched M slots in separate banks. Scale,
  activation, dimensions and external ABI remain bound to the original proof.
- `--grouped-b`: use up to 64-column weight transfers where validation succeeds
  and the exact primitive count decreases. This reuses the already validated
  arbitrary-N grouped-load implementation; compute order/readout are unchanged.

These options do not inspect model names, region names, or a particular scale.
The banked family's measured anchor is M3136/N64/K64, exact captured scale
0.0038317402359098196, no bias, ReLU: FireSim jobs1752/1753 measured
90,323 → 51,922 cycles. A distinct M47/N48/K32 signed-output tail probe passes
GSIM at983cycles and strict Spike. Other legal family members do not inherit
the anchor's measured performance claim.

Grouped B, exact late-layer shape M49/N2048/K512 and source scale
0.0023045858833938837 without bias/ReLU, improves GSIM493,598 →447,124cycles
(9.42%). All100,352 outputs and guards, strict Spike, and final no-FSM pass.
Its padded array issue floor is262,144cycles, leaving substantial scheduling
headroom. Across exact52,33dense calls lose64,224primitive commands; one call
selects banked prefetch, two dense signatures and all16direct schedules remain
unchanged. Whole-model numerical validation is a separate required gate.

The catalog's issue floors are7,805,952 direct cycles +8,893,440 dense unary
cycles. Adding approximately702,464 packed-stem,129,024 classifier and5.2M
wide-residual cycles brings the current padded-compute floor near22.73M,
before host, DMA and configuration overhead. A strict22M target therefore
requires algorithmic or padding-work reduction as well as scheduling.

The independent `--separate-b-bank` option places B at scratch row8192
(banks2/3), while A retains its existing lower-bank address scheme. It proves
that all reserved A slots/panels fit below that boundary and all B panels fit
above it; competing banked-M placement refuses. The source selector retains
its original legal schedule when this additional capacity proof fails, and
skips the redundant option when banked-M has already been selected. Cached-B
initial loads and all compute references use the same proven placement.
No compute ordering, command count, scale, activation or external ABI changes.

For M49/N2048/K512, grouped-B GSIM447,124 → separate-bank367,749cycles
(17.75% reduction), versus493,598 before either improvement. All100,352
outputs plus guards, strict RV64GC Spike and final zero-FSM audit pass.
The new result is1.403× the262,144 padded-compute floor. Mixed M/N/K-tail
i32 and cached-A signed-i8 probes also pass. This is simulator evidence;
other shapes and hardware require their own performance gates.

Evidence: `docs/perf_records/source_bound_dense_schedules.json`.

Full source-bound exact52 + virtual padding + wide16 residuals + NHWC layouts
+ static weight hoist passes all1000 fresh-golden outputs bitexact in native
and actual Gemmini Spike, with rank mismatches0 and final zero-FSM. There are
35dense separate-bank routes; the existing banked-prefetch route keeps its
own placement. Spike retired instructions19,141,081 →19,154,836 reflect
address-generation overhead, not memory-bank latency. Both full-model ELFs
remain hardware-unmeasured in these receipts. Evidence:
`docs/perf_records/whole_dense_virtual_wide16.json` and
`docs/perf_records/whole_dense_split_virtual_wide16.json`.
