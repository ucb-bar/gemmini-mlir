# Source-bound dense schedule options

Captured requant bundles expose two independent, default-off options:

- `--banked-prefetch`: i8 output, no bias/competing A cache, at least two M tiles,
  full K in one 32/48/64-column A transfer, and N fitting one full-tile wide store.
  Static A/B bank footprints and accumulator-slot capacity must fit. The selected
  schedule is bm1, cached B, two prefetched M slots in separate banks. Scale,
  activation, dimensions and external ABI remain bound to the original proof.
- `--grouped-b`: use up to 64-column weight transfers where validation succeeds
  and the exact primitive count decreases. This reuses the already validated
  arbitrary-N grouped-load implementation; compute order/readout are unchanged.

Neither option inspects model names, region names, or a particular scale.
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

Next controlled experiment: independent A/B scratchpad bank placement for
large K. The existing streaming schedule puts both operands in bank0; the
current `banked_m` capability requires K≤64 and cannot address those families.

Evidence: `docs/perf_records/source_bound_dense_schedules.json`.
