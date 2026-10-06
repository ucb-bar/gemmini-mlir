# ResNet remaining gap: measured locations and unresolved attribution

## Current whole-model result

The original-source compiler model's best qualified stock result is job2052:
**30,650,056 cycles**, with all1,000 original output words exact and the final
ELF/staged ELF/stock bitstream pinned. The permitted ZIP diagnostic job1876
reproduces **22,387,449 cycles**. The measured whole-model difference is
**8,262,607 cycles**, or **36.91% above the reference**. Retaining ordinary loops
in two flat kernels improves65,762cycles/0.2141% versus2039 in one observation
per arm. Primitive work features tied before timing; CPU issue/footprint effects
remain unpriced.
[Current result](perf_records/root_resnet_stock2052_flat_loops_terminal_review_20261006.json),
[reference result](perf_records/q1013_diagnostic_reference_firesim.json).

The2039 object-preserving profile2046 covers all71 active primitive
calls and conserves30,756,710 forward cycles. Callback windows measure
27,791,061 cycles and outside intervals2,965,649 cycles. The40,892 difference
from the uninstrumented control includes wrappers/layout; it is not an isolated
overhead rate. Source callbacks include host issue, transfers and waits. These
section measurements are not reassigned to2052.

| Current2046 role | Reference1876 cycles | Current2046 cycles | Difference |
| --- | ---: | ---: | ---: |
| Residual callbacks |2,192,393|5,471,731|3,279,338|
| Outside timers/callbacks |51,717|2,965,649|2,913,932|
| Pointwise callbacks |9,907,412|10,920,222|1,012,810|
| Spatial callbacks |8,723,089|9,457,058|733,969|
| Stem and pool |1,083,057|1,445,170|362,113|
| Classifier plus integer mean callback |429,781|496,880|67,099|

The pre-stem interval is2,195,679 cycles. Flat kernels11/43 currently measure
755,946/906,911 cycles. Residual and outside intervals comprise about74% of
the8,369,261 diagnostic difference. This is role/geometry accounting, not
numerically equivalent substitutions. Reference classifier timing includes
average; current CPU mean finishing remains in gaps. Fresh current geometry
checks cover52 convolutions; stem/classifier role bindings inherit previously
qualified symbols, with classifier geometry absent from the new manifest.
[Current terminal review](perf_records/root_resnet2039_stock2046_profile_terminal_review_20261006.json),
[current reference alignment](perf_records/q1013_current2039_stock2046_profile_alignment.json).

## Earlier diagnostic history

The following diagnostic
profiles actual2013, before its mean/full-K composition. This profile is
qualified from the actual winning frozen recipe: the unprofiled ELF reproduces
byte for byte, all1,000 original words remain exact,70 active primitive
callbacks occur once each and intervals conserve. Three obsolete original
primitive objects remain linked and audited, but are not active boundaries.
Stock2021 completed with32,600,719forward cycles:29,338,102callbacks and
3,262,617host gaps. Its47,080cycle instrumentation delta is not a speed claim. The profiler
reconstructs original link leaves from their hash-bound manifest separately
from selected semantic routes, fixing a catalog interpretation error.
[Current profile release](perf_records/root_resnet_current2013_profile_release_20261006.json),
[actual current alignment](perf_records/q1013_current2013_stock2021_profile_alignment.json).

| Current2021 interval | Reference1876 cycles | Current2021 cycles | Difference |
| --- | ---: | ---: | ---: |
| Outside timers/callbacks |51,717|3,262,617|3,210,900|
| Residual callbacks |2,192,393|5,471,507|3,279,114|
| Spatial convolution callbacks |8,723,089|10,553,446|1,830,357|
| Pointwise convolution callbacks |9,907,412|11,390,982|1,483,570|
| Stem and pool callback |1,083,057|1,443,413|360,356|
| Classifier callback |429,781|478,754|48,973|
| Total |22,387,449|32,600,719|10,213,270|

The largest current host intervals are before stem2,197,344, matmul26
361,018, classifier343,752 and matmul49 177,084cycles. These are measured
combined preparation/readout intervals; no isolated stage cost is inferred.
Reference outside timers is not totalCPU time. Callback intervals include CPU
issue, transfers and synchronization; these differences locate work and do
not prove a causal saving from any substitution.

Independent integer mean2018 measures33,230,295cycles and full-K weight
residency2020 measures33,026,561cycles, each against frozen1992
33,500,256. They save269,961 and473,695cycles respectively, but do not replace
segmented2013. Their qualified composition with2013 now measures31,808,394cycles in2023,
745,245below2013. Source-stride2022 independently measures32,444,367; it is
not included in2023. Their actual composition2025 retains the frozen2023
host/runtime and source-stride24 leaf from2022, measuring31,697,615cycles.
Only route24 changes; the independent root probe recloses29,611pins.

Six additional resident spatial stripes now close over actual2025, with37,015
root-rehashed pins. Original source capsules at common addresses pass all903,168
outputs per arm and improve15.8%/19.1% in GSIM. Stock2026 measures30,977,892,
719,723cycles/2.2706%below2025. The unchanged46kernels,52adapters, source/numeric proofs,
host/runtime/weights and current winning strategies are retained.
[Six-stripe release](perf_records/root_resnet_current2025_resident_stripes_release_20261006.json).

The new model calibration study pairs the identical1903profile ELF across
GSIM and stock1919 and reparses both retained consoles. Complete geometry
holdouts predict68of70callback intervals with3.88%median/11.85%maximum error.
Host correction remains inaccurate; this does not qualify a whole-model estimate
or independently demonstrate variant ranking. A dedicated primitive benchmark
is now being built. Jack1876timing is excluded from all fitting.
[Model diagnostics](perf_records/paired_engine_geometry_screen_diagnostic_20261006.json).

Stock1999 profiles1988's
nonflat paired/packet implementation with its uninstrumented control reproduced
byte for byte and all1,000 outputs exact. It conserves33,671,936interior cycles
as29,365,948primitive callbacks+4,305,988outside intervals. Its outer33,672,428
is39,319above the uninstrumented1988 control. The older1990 diagnostic binds1974;
its costs cannot be reassigned to1988 or1992.
[Exact1988 terminal](perf_records/stock1999_exact1988_profile_terminal.json),
[Source/geometry alignment](perf_records/q1013_exact1988_stock1999_profile_alignment.json).

| Interval | Reference1876 cycles | Exact1988 profile1999 cycles | Difference |
| --- | ---: | ---: | ---: |
| Outside layer timers / outside primitive callbacks |51,717|4,305,988|4,254,271|
| Residual callbacks |2,192,393|5,466,355|3,273,962|
| Spatial convolution callbacks |8,723,089|10,676,947|1,953,858|
| Pointwise convolution callbacks |9,907,412|11,298,231|1,390,819|
| Stem and pool callback |1,083,057|1,444,192|361,135|
| Classifier callback |429,781|480,223|50,442|
| Total inside model |22,387,449|33,671,936|11,284,487|

All70newsource-bound events and54reference geometries reconcile. The largest
current outside intervals precede stem2,193,677, matmul14 614,595, matmul26
377,397, classifier339,189, matmul27 285,467, matmul49 185,169 and matmul46
137,414cycles. These are combined preparation/readout/glue intervals, not
isolated operation measurements. Callback times include CPU command issue.
Reference51,717outside layer timers is **not total referenceCPU time**. Numeric,
input and timing boundaries differ; these locations are not isolated causal
savings claims. Latest1992 needs its own profile before exact section attribution.

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
the historical section rows without a matching profile. Job1903 does not contain
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
