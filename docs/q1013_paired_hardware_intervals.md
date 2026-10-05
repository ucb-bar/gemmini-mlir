# Reference q1013 and exact1849: measured hardware intervals

The owned reference diagnostic completed stock FireSim job1876 at **22,387,449
cycles**, matching the archive's whole-model measurement. Its54 layer records and
original1000-logit self-check passed. Its six replaced FSM words are dynamically
dead, with original/diagnostic executed-path identity documented separately.
This remains a modified reference diagnostic with its own input, weights and
numeric contract. No Jack directory was opened.

The comparison below pairs it with **job1850's profiled exact1849 control**. Both
receipts close against the same stock hardware/bitstream hashes. All53 convolution
geometries and the classifier geometry match. It does **not** infer a breakdown
for1853 or the newer39,201,279-cycle1874 result.
[Pinned54-layer pairing, objects, UART and source catalog](perf_records/q1013_1849_paired_hardware_intervals.json).

## Categories reconciled by geometry

Reference printed “Conv” includes three stride-2 **1×1** projections (layers15,
28 and47). Classifying by geometry moves1,607,026 cycles from that printed
category to pointwise. Comparing the printed totals directly with our compiler
categories would give the wrong priority.

| Interval | Reference1876 | Exact1849 profile1850 | Difference |
|---|---:|---:|---:|
| Pointwise |9,907,412|13,150,309|3,242,897|
| Direct3×3 |8,723,089|11,969,037|3,245,948|
| Residual |2,192,393|6,769,409|4,577,016|
| Stem and pool path |1,083,057|1,446,279|363,222|
| Classifier |429,781|479,009|49,228|
| Host intervals / other and uncounted |51,717|8,489,121|8,437,404|
| Total inside model |22,387,449|42,303,164|19,915,715|

The arithmetic reconciles exactly. Current device intervals total33,814,043;
host intervals total8,489,121. The current whole wrapper counter is42,303,592,
428 cycles beyond the inside-model counter. Profiling overhead against its
unprofiled1849 control is retained in the receipt. Reference “other”12,066 plus
uncounted39,651 is not the same instrumentation boundary as our host intervals.
These differences identify work to inspect; they are **not causal savings** from
substituting a reference implementation with different numerical semantics.

## Concrete intervals to optimize

The seven largest current host intervals sum **8,319,797 cycles:98.0% of its host
time**. Each interval includes every intervening CPU operation and timer overhead;
it cannot be attributed to a single operation without further instrumentation.

| Before current call | Cycles | Relevant source work to inspect |
|---|---:|---|
| Stem |4,017,289|Required f32 quantization, NCHW/layout packing, allocation and ABI preparation|
| matmul_26 |1,955,979|Exact readout after stride-2 convolution and ownership/layout materialization|
| matmul_49 |962,942|Exact readout after deep3×3 convolution and ownership/layout materialization|
| matmul_14 |615,099|Strided projection input slices, flattening and intervening operations|
| Classifier |344,540|Mean/quantization, scalar preparation and classifier boundary|
| matmul_27 |286,392|Projection input preparation and intervening operations|
| matmul_46 |137,556|Projection input preparation and intervening operations|

The prepared source shows two stride-2 tensor slices before each projection,
followed by collapse/expand to its dense operand. Their first projection interval
is615,099 cycles; the later intervals are286,392 and137,556. This is a concrete
candidate for a composed view/packing path, but timing the slices separately is
still required before assigning all those cycles to them.

Required entry quantization cannot simply disappear: our immutable capture
accepts f32 NCHW, whereas the reference starts with int8 data. Reusable Merlin
work should prove exact quantization/readout fusion, preserve caller ownership,
and propagate compatible layouts to avoid copies and repacking. A host-stage
profile and source-bound operation receipt must decide which of these dominates.
This work belongs in Merlin and applies independently of the accelerator.

The largest same-geometry device differences include:

| Ref layer / current binding | Geometry M/K/N | Reference | Current1850 |
|---|---|---:|---:|
|26 / matmul_25|196 /2304 /256|555,284|870,137|
|42 / matmul_41|196 /2304 /256|484,731|722,195|
|36 / matmul_35|196 /2304 /256|486,009|722,048|
|47 / matmul_46|49 /1024 /2048|588,801|819,456|
|49 / matmul_48|49 /4608 /512|652,750|882,818|
|44 / matmul_43|196 /1024 /512|472,798|694,562|

The full receipt includes every matching layer, source operation ordinal, current
schedule, device object and preceding host interval. Direct and pointwise scopes
may differ in their quantized epilogues; i32 readout can fall outside our device
timer. The source scales remain immutable. The residual comparison has only a
reference aggregate; **no individual reference residual timing is invented**.
Its324,576 versus22,208 executed computes reflect different source scale work.
The39-chunk first residual bound is specific to its existing arithmetic family.

**Schedule-version limit:**1850 timed1849's older spatial-flat H14 convolution
family. The later1853 transfer bundle already uses resident input channel planes
there. Its current per-layer times are unknown. The old3.25M direct-category
difference must not be treated as the remaining gap of that newer recipe. Profile
the best measured composition before choosing further deep-convolution work.

## Next general compiler changes

1. **Merlin host readout/layout fusion and ownership propagation.** Instrument
   the source-bound intervals above; remove a materialization only with exact
   arithmetic, layout and lifetime proof. Preserve required f32 entry/output work.
2. **OOT next-K operand transfer overlap and resource placement.** Pointwise
   compute counts agree555,840 each, yet timings differ. The general disjoint
   remaining-B-slot policy is already qualified locally and its isolated whole
   candidate is on stock1888. Explicit slot extents/bank hazards are part of its
   legality proof; command counts alone are not a timing model.
3. **OOT resident convolution stripes / channel blocking.** Existing wider
   stripes address the H56/H28 repeated transfers, with isolated stock1878
   complete at39,754,283 cycles,1.792% below1853. The composed stripe/transfer/
   residual candidate is released to hardware; gains remain unassumed.
   H14/C256 full weights exceed scratchpad capacity; the enabled channel-plane
   schedule already keeps bounded weight panels resident. A fresh best-recipe
   profile must justify any further channel stripes or exact readout/next-transfer
   overlap work.
4. **Shared selection reaching emitted code.** The real whole-model identity
   gate now exports graph/catalog/final-binary closure and compiler owners.
   It does not select a replacement IR or invoke the solver. The next narrow
   route now enumerates legal same-source contraction alternatives, accepts only
   source/object-bound measured prices, invokes existing `optimize_program`,
   and emits its selected implementation.
   [Actual selected-object qualification](perf_records/golden_calibrated_source_selection_qualification.json).
   Extending this to whole-model alternatives requires their complete explicit
   cost/transition coverage. Accelerator occupancy remains unknown.

The rejected complete-A dense stripes remain negative evidence: fewer real-B
reloads and fewer host instructions were50.6% slower in the paired full-output
screen. Do not assume resident data, fewer commands or local gains improve the
whole model. Stock original-source quality and timing gates decide admission.

## Reproduction and journey

The checked analysis driver reads only explicitly named owned artifacts:

```sh
python docs/perf_records/reconcile_q1013_hardware.py \
  --current-root /scratch/agustin/tmp/gemmini-golden-nofsm-20261004 \
  --reference-root /scratch/agustin/tmp/exo_q1013 \
  --output docs/perf_records/q1013_1849_paired_hardware_intervals.json
```

It validates actual ELF/UART hashes, stock pins, completed receipts, source
catalog/object identities, all54 records, exact geometry and aggregate cycle
conservation. This2026-10-05 reference-parity checkpoint changes measurement
interpretation, not compiler artifacts or model semantics. Parent-owned token
accounting supplies the journey counters; this child goal tracker is unavailable
and no exclusive per-optimization token attribution is claimed.
