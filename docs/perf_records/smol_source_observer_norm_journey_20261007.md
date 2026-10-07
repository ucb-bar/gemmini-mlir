# Source observers and norm experiments — 2026-10-07

These experiments use the shared Merlin compiler and OOT backend architecture.
They use different explicit experimental revisions and frozen component leaves;
no single current automatic configuration has been qualified across ResNet50,
SmolVLA and TinyLlama. Source identities authenticate experiments and never
select a production optimization.

## Qualification and results

Every native experiment below preserves all 1,600 original SmolVLA output bits,
the fixed `atol=0.03125, rtol=0.02` gate, all 48 source calls and 12 preparations.
Complete source-group target screens retain the original 15 product kernels,
all readouts, workspace/ownership checks, source arithmetic, compiled consumer,
input/output guards and final drain. All final ELFs contain zero FSM operations.

The counts below are complete-group functional Spike retired instructions.
They are not modeled or measured FPGA cycles. Native timing is diagnostic only.

| Experiment | Complete ROI retired instructions | Change vs original | Decision |
| --- | ---: | ---: | --- |
| Original source-group control | 1,515,040,109 | — | Control |
| Exact five-output product family | 1,514,248,511 | -0.052249% | Explicit source topic; no automatic admission |
| Finite-point row observations | 1,472,334,273 | -2.818792% | Exact generic host topic sealed |
| Exact BF16 integer observation | 1,490,248,996 | -1.636334% | Exact generic host topic sealed |
| Composed point + integer observers | 1,468,831,808 | -3.049972% | Complete composition qualified |
| Cauchy source envelope | 1,558,265,540 | +2.853088% | Rejected for complete cost |
| Exact F32-square norm envelope | 1,536,334,803 | +1.405553% | Rejected for complete cost |
| Row-staged Cauchy factors | 1,528,765,395 | +0.905935% | Rejected for complete cost |
| Zero-inclusive exact radix rows | 1,555,407,882 | +2.664469% | Rejected for complete cost |

The three Cauchy variants reduce source replay and row refinement but lose after
preparation and certificate work is included. Their receipts and separate owners
are retained; no variant is selected or promoted. No further variants are queued.

The exact finite-point observer removes 37,503,064 endpoint `nearbyintf` calls by
proving equal finite endpoints have the same final integer observation. The exact
BF16 integer observer retains the original scale, inverse and BF16 product, then
decodes its exact nearest-even clamped integer observation. All 65,536 product
words, eight clamp plans, eight inverse values, refusal/effects and output guards
pass independent native UBSan checks. These two mechanisms overlap; their gains
must not be added. The separately owned complete composition preserves all 1,600 original bits,
consumer/guards and all eight statistics. Its incremental reduction after the
point observer is 0.237885%; these mechanisms substantially overlap.

## Source ownership

Generic proofs, host generation, observation semantics and private buffer/source
contracts belong in Merlin. Target shape binding, commands, accumulator resources,
strided family callbacks and device compilation facts belong in OOT.

Clean local Merlin topics, none pushed by this worker:

- Point observations: `e65ad5e2dc4454c2600ff883bf63b3b2e7bafc1e` in
  `/scratch/agustin/tmp/merlin-frontier-point-cells-main-20261007`.
- Formatted exact family: `953a73a31498adde310377a68fd1466fc59edd33` in
  `/scratch/agustin/tmp/merlin-integer-product-family-formatted-20261007`.
- Exact BF16 integer observer: `ac716135424a58db1f50a29c7e0d545bdb876890` in
  `/scratch/agustin/tmp/merlin-bf16-integer-observer-main-20261007`.

Receipts under `out/artifacts/probes/` remain immutable. Exact observer results
and complete group are in `bf16-integer-observer-v2-20261007/`. Norm negatives
are respectively `rms-cauchy-envelope-20261007/`,
`rms-f32-square-envelope-v2-20261007/` and
`rms-row-factor-envelope-20261007/`, each with a cost decision bound to its
qualification hash. The old five-output QK GSIM run exhausted its 50M complete
budget before the terminal protocol: partial rows remain unqualified, not a win.

## Compiler search implications

Phase 0 should retain typed source effects, every original quantization
observation, complete output/buffer ownership and independently proved callback
arithmetic. Pure MLIR `math.roundeven` permits a returned integer observation;
an arbitrary interposed C rounding call does not. Approximate RMS4 product
permission supplies no exact source-observation theorem.

Phase 1 needs explicit source producer/consumer composition and all live product
planes. Callback count is a poor objective: reducing 23,040 degree interfaces to
4,608 family interfaces saved only 0.052% complete-group retired work. Exact
semantic zero metadata belongs in mandatory producer packing, with complete
zero-plane writes and no extra full scan; device skip schedules remain OOT.

Phase 2 should price complete setup, source preparation, certification, replay,
all public output reads/writes and validation. GSIM admission budgets should be
derived from complete-main Spike retirement plus setup/validation, never ROI
alone. Better replay statistics cannot select a slower complete candidate.
Independent exact observers must be composed and repriced to account for overlap.

The next architectural hypothesis is a typed quantized-consumer writer that
publishes certified private i8 values plus the original source scale directly,
when complete source-use closure proves no other floating observation escapes.
The current certificate computes these observations, discards them, publishes
BF16 candidates, and the original consumer repeats scale/round/clamp/packing.
Such a rewrite needs complete tuple/buffer lifetimes and the unchanged original
whole-output gate. It is a proposal, not an implemented or admitted optimization.

## Token accounting

Exact per-change token billing is unavailable from this worker. No session-wide
aggregate meter is attributed to these topics. Receipts explicitly record that
absence; optimization steps, rejected hypotheses and what changed each measured
scope remain recorded here.

## Subsequent source audits

The zero-inclusive exact packing proof passes all BF16 words, three digit counts,
six row maxima, signed zero, subnormal/span refusal, strided complete guards and
UBSan. All original1,600 output bits and complete group consumer/guards remain
unchanged. Mandatory-scan instrumentation finds1,382,400valid rows,1,048,470old
fast rows and1,065,000new fast rows; only16,530zero-containing rows gain coverage.
The complete cost increases2.664469%, so the implementation remains an isolated
negative and is not promoted. Its exact-zero reconstruction is canonical+0,
including input-0; original source-sign policy is retained.

The actual current b154 linked source PC census prices the twelve quantized
attention consumers at1,258,292,004exclusive instructions across72BF16 helper
functions,1.069644%of117,636,499,028total executable dispatches. An unrelated
i8weight transpose35,463,228instructions is explicitly excluded. Shared callees
and source assembly copies remain unattributed; these are not predicted savings.
The direct i8+scale seam is therefore deprioritized against material provider/
evaluate/certify source work. Read-only source witnesses and actual linked prices
are in `out/artifacts/probes/quantized-consumer-inventory-20261007/`.
