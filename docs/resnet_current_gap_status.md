# ResNet remaining gap: measured locations and unresolved attribution

## Current whole-model result

The original-source compiler model's best qualified stock result is job1982:
**34,792,010 cycles**, with all1,000 original output words exact and the final
ELF/staged ELF/stock bitstream pinned. The permitted ZIP diagnostic job1876
reproduces **22,387,449 cycles**. The measured whole-model difference is
**12,404,561 cycles**, or **55.41% above the reference**.
[Current result](perf_records/stock1982_paired_flat_terminal.json),
[reference result](perf_records/q1013_diagnostic_reference_firesim.json).

Current1982 section attribution is **UNKNOWN**. The latest completed stock
profile1919 binds1903; diagnostic1990 now queues a conserved70-boundary profile
of1974's resident-weight arm with all1000originalwords/noFSM closed. It does not
profile1982. The historical comparisons below retain their actual object and
timer scopes; their differences cannot be reassigned to the current champion.
[1974 profile qualification](perf_records/root_resnet1974_leaf_profile_qualification.json).

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


## Current exact profile in a separate RTL regime

The actual1903 profile ELF completed in GSIM with all1,000 original words,
70events and complete conservation. Stock1919 subsequently completed with
36,138,975forward cycles,30,678,196device-wrapper and5,460,779host-gap cycles.
[Completed stock1919](perf_records/resnet1903_stock1919_profile.json).
These GSIM costs have a different memory regime from stock FireSim and cannot
be subtracted from reference1876stock costs to assign causal savings.

| Current GSIM interval | Cycles |
| --- | ---: |
| Pointwise wrappers |11,107,859|
| Spatial wrappers |10,989,417|
| Residual wrappers |5,512,622|
| Stem/pool wrapper |1,344,602|
| Classifier wrapper |467,751|
| All device wrappers |29,422,251|
| Intervening host gaps, including final73,136tail |4,517,213|
| Complete forward |33,939,464|

Host gaps include entry1,776,646, after-readout1,060,273/530,513,
projection420,292/188,349/87,734 and classifier preparation329,339.
These boundaries include other CPU work; they do not isolate individual copies,
quantization or arithmetic. The paired original projection capsule qualifies a
borrowed-input alternative, and both readout families have complete range-bound
paired cost evidence. Compact ordinary CPU command loops have an independent
full-output timing result and original schedule geometry object-size evidence.
Its complete current stride2geometry with independent inputs improves1.7326%
in GSIM. The normal source build and device-only frozen1903 arm both pass all
original1,000words; stock1928 is queued. The borrowed-input arm is queued as1927.
The complete two-readout epilogue with compiler-copy decoding improves71.41%
and70.40% in its two ROIs. Normal preparation now allocates actual i8 scratch;
normal and frozen1903 all-output/native/strict/noFSM gates pass. Its typed
allocation candidate is queued as1930. Stem CPU loops also pass the original
complete capsule with3.0401%lower cycles and full normal/frozen output gates;
stock1929 is queued. The87.8%stem text reduction is not a cycle reduction.
Each still needs a controlled whole stock result before assigning gap reduction.
[Profile](perf_records/resnet1903_gsim_conserved_profile.json),
[projection](perf_records/segmented_input_original_projection_gsim.json),
[readout](perf_records/resnet_bound_readout_paired_gsim.json),
[paired typed whole](perf_records/resnet_paired_readout_typed_whole_qualification.json),
[paired complete cost](perf_records/exact_pair_readout_builtin_matched_gsim.json),
[stem](perf_records/stem_spatial_command_loop_capsules.json),
[loops](perf_records/flat_spatial_command_loop_qualification.json).
