# ResNet remaining gap: measured locations and unresolved attribution

## Current whole-model result

The original-source compiler model's best qualified stock result is job1903:
**36,102,704 cycles**, with all1,000 original output words exact and the final
ELF/staged ELF/stock bitstream pinned. The permitted ZIP diagnostic job1876
reproduces **22,387,449 cycles**. The measured whole-model difference is
**13,715,255 cycles**, or **61.26% above the reference**.
[Current result](perf_records/firesim1903_resnet_composed_verified.json),
[reference result](perf_records/q1013_diagnostic_reference_firesim.json).

The reference supplies useful scheduling evidence. Its numerical coefficients,
input representation and output epilogue differ from the immutable source model.
The compiler must still perform the source's required computation. These
differences do not establish that the current cost is necessary or optimal.

## Last measured section comparison

Job1899 profiles the unchanged job1874 implementation. Its conserved interior
counter is39,235,729 cycles; its unprofiled control is39,201,279. This comparison
is historical: it does not assign job1903's remaining13,715,255 cycles.

| Interval | Reference1876 cycles | Profile1899 cycles | Difference |
| --- | ---: | ---: | ---: |
| Outside timed calls / reference other and uncounted |51,717|8,521,815|8,470,098|
| Residual adds |2,192,393|5,471,537|3,279,144|
| Spatial convolutions |8,723,089|11,971,291|3,248,202|
| Pointwise convolutions, including stride-2 projections |9,907,412|11,344,741|1,437,329|
| Stem and pool |1,083,057|1,447,500|364,443|
| Classifier |429,781|478,845|49,064|
| Total inside model |22,387,449|39,235,729|16,848,280|

The rows reconcile exactly. All53 convolution geometries and the classifier
geometry are paired. Reference stride-2 1×1 convolutions are classified as
pointwise to avoid comparing incompatible printed categories. Reference
residual timing is aggregate; individual residual timings are unavailable.
[Complete source, shape and hardware alignment](perf_records/q1013_1874_current_profile_alignment.json).

Intervals inside device wrappers include adapter work. Host intervals contain
all intervening CPU work and timer overhead. Reference other/uncounted and our
host intervals have different boundaries. The differences locate costs;
they do not prove causal savings from a substitution.

In particular,51,717 is **not total reference CPU time**. The ZIP executable
places cycle timers around calls to its glue/layer functions; CPU address
calculation, command issue and any other CPU work inside a call are charged to
that layer. The reference comparison cannot establish an8.47M CPU-only gap.
The owned disassembly shows these timer/call pairs; no source file was present
in the ZIP and no Jack directory was inspected.

## Why matching the device schedule alone is insufficient

1. **Host quantization, packing, exact readout and materialization remain expensive.**
   In1899, the pre-stem interval is4,016,144 cycles. The intervals before the two
   pointwise consumers of integer convolution readouts are1,992,494 and962,283.
   Projection input preparation intervals are615,348,286,308 and137,288;
   the classifier preparation interval is341,419. These seven intervals account
   for8,351,288 cycles,98.00% of the measured host time. They require further
   instrumentation before assigning a cost to one operation. The reference
   begins with int8 input; our source accepts f32 NCHW.
2. **Residual numerical work differs.** The first source residual needs39
   chunks in the currently implemented positive-integer coefficient/single-f32
   scaled-store family. The reference's identity-weight residual uses much less
   work. The exhaustive certificate proves minimality only within that family.
   A four-chunk exact threshold alternative exists, but its optimistic scalar
   readout cost exceeds the old measured device interval. This remains a search
   for a better exact algorithm, not permission to change source scales.
   [Family certificate and alternative](residual_coefficient_minimality.md).
3. **Transfer and execute scheduling still differ.** Matching pointwise compute
   counts did not match elapsed cycles. Spatial schedules trade tail computes
   against activation traffic and weight reuse. Residency or fewer commands
   alone did not predict performance: several measured variants regressed.
   [Executed schedule comparison](q1013_current_strategy_parity.md).

## Changes since this profile and next measurements

Job1903 combines exact host quantization packets, banked residual scheduling and
exact early-saturation/eight-lane readout. It measures3,098,575 fewer cycles than
the unprofiled1874 control. This whole difference cannot be redistributed into
the historical section rows without a current profile. Job1903 does not contain
the independently measured1897 convolution-stripe arm.

Source-stride resident convolution and the residue-grouped input layout have
independent original-input capsule evidence:20.59% and7.98% respectively. Their
whole-model timing is pending; those gains are not predictions for job1903.

The hardware owner qualified a70-boundary profile of the actual1903 objects,
with a byte-identical uninstrumented control, all1,000 original output bits,
call conservation, all-executable-section zero-FSM audit and stock hardware
identity. It is queued as **job1919**, with hardware timing pending.
[Qualification](perf_records/resnet1903_profile_qualification.json),
[admission](perf_records/firesim1919_admission.json). Named CPU-stage attribution
will be recorded only when the instrumentation separates that stage. Remaining
unknowns are transfer/execute overlap inside a device interval and the exact
split of combined host intervals.

Reusable host-stage instrumentation, layout/readout fusion, packing, ownership
and numeric algorithms belong in Merlin. Instruction scheduling, target layouts,
bank/resource legality and target timing providers belong in OOT. Production
rules remain semantic and resource based, independent of workload names.
