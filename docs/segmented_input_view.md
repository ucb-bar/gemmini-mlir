# Borrowed row-segmented matrix inputs

The shared Merlin `SegmentedRows` contract keeps element width, source allocation
extent, source origin and the exact row map. Its structural proof follows static
nested slices and whole linear reshapes without editing source IR. A consumer
must close dense physical owner storage, read-only access, source lifetime,
output nonoverlap and ABI acceptance before removing a copy.

The OOT `GoldenGemm(input_view=...)` option accepts physical i8 storage and complete
cached A residency. It splits each input DMA panel at physical row discontinuities,
uses the proved byte row stride, and writes the established local A cells. Scratchpad
and accumulator bounds, source increasing K order, B reuse/prefetch and exact store
semantics remain unchanged. Uniform physical pitch merges logical boundaries.
Unsupported residency, width, shape, field range and batch ABI refuse. Default
model policies remain unchanged.

## Qualified independent evidence

`perf_records/segmented_input_independent_qualification.json` pins strict RV64GC
Spike and the existing GSIM engine for M33/N73/K65, source segments of seven rows,
nonzero origin, unequal physical strides, wide K packets and partial M/N/K tails.
All 2,409 i32 outputs, 4,096 guard bytes and 14,745 input bytes pass. Final ELF
contains zero FSM instructions. The GSIM kernel interval is 3,741 cycles; this is
a correctness capsule, with no stock or whole-model performance claim.

Thirteen shared structural checks and 39 focused target checks cover nested
offset/stride composition, integer/float width preservation, decoded source/local
addresses, exact resident-cell partition, unchanged compute/store streams,
resource/tail refusals and composition with existing A grouping/B slot transforms.
`segmented_input_default_identity.json` proves all 36 actual dense catalog target
modules byte-identical against pre-edit commit `663b15f` with the option absent.

## Current source boundary

`segmented_input_source_view_proofs.json` pins current post-provider typed source,
the selected catalog and the physical host LLVM. Three noncontiguous source views
have a legal current cached-A consumer. Twenty-one contiguous views require no
segmented loader; twelve other source views lack this consumer's residency proof.
Names are source binding identities, never strategy selectors.

The selected residual providers already write dense i8 allocations. Their derived
projection maps avoid 351,232 logical copy bytes, or 702,464 read/write payload bytes.
Physical DRAM traffic and latency remain unknown. The measured current1903 GSIM
preceding-host intervals total 696,375 cycles, but include work beyond these copies.
That total is neither expected savings nor a stock measurement.

Normal call/declaration and native-oracle binding is pending. Host copies remain
enabled. The original model source, inputs, weights, numeric proofs and all-1,000-word
exact accuracy gate are retained. Next admission requires a matched original-input
capsule, explicit borrowed owner ABI and full native/strict target qualification.
## Matched original projection screen

The pinned GSIM capsule checked every original consumed input and all 401,408
output bytes plus 4,096 guard bytes. Both strict RV64GC Spike replays passed and
both final ELFs contain no FSM instructions. The copied dense arm took 866,791
cycles; the segmented borrowed arm took 538,815, a reduction of 327,976 cycles
(37.838%). Both ELFs contain the same code and storage at the same addresses;
their only differing byte selects the arm.

This timer includes a generated portable CPU copy followed by the current dense
device implementation. It does not establish identity with the whole-model
host LLVM copy. All consumed values are the original capture, while unread
source allocation cells are synthetic zeros. GSIM has a different memory
system from stock FireSim. The result admits the typed consumer binding work;
it does not establish a whole-model or stock speedup. See
`perf_records/segmented_input_original_projection_gsim.json`.
