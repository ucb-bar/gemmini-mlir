# Source-bound BF16 attention: native whole policy controls

The normal source route now binds all 48 complete attention groups, covering
384 ordered BF16 contractions and 19,327,352,832 source f32 FMAs. Every group
matches the complete normalized source scalar DAG, including masks, maximum,
polynomial exponential, intermediate BF16 probability, denominator reduction,
source alpha, separately rounded PV partial additions and final BF16 conversion.
Existing integer projections keep their original bindings. The same shared
borrowed writer ABI uses a fresh fully written output; chronological group IDs
exist only in experimental diagnostics.

## Paired original-model gate

| Registered native policy | Original output gate | Replayed source FMAs | Decision |
| --- | --- | --- | --- |
| Fixed maximum one adjacent BF16 endpoint bin | 113 / 1,600 failures; max absolute error 0.1629244 | 429,497,664 (2.2222%) | Reject whole-model enablement |
| Exact endpoint control, identical host image and ABI | All 1,600 original words bit exact | 5,507,717,696 (28.4970%) | Accept source closure and native ABI control |

The whole criterion remains atol=0.03125 and rtol=0.02. All 48 exact-control
endpoints were independently compared with the actual compiled original source
group on their current live operands; every one of 196,608 BF16 words per group
matched. Both arms made 48 callbacks and zero whole-group exception fallbacks.
The first bounded group changed 53 endpoint words; its 11 input hashes matched
the accepted original source tap. Small local endpoint errors still accumulated
through later model operations beyond the original whole acceptance gate.

Merlin owns source closure, host ABI, fresh ownership, numeric certificate and
source replay. OOT owns target integer-product instructions and resources. This
screen uses native exact-integer NumPy products as a functional device stand-in.
It proves no actual target dispatch or hardware performance. Native runtime and
replay counts cannot be substituted for stock FireSim cycles. The actual target
provider is independently under qualification; neither losing policy nor this
control enables ordinary production dispatch.

The following single authorized screen keeps the same three signed-digit
products and complete source finalization DAG, rounds reconstructed QK/PV dots
once to f32, and omits interval/replay. It has no local endpoint error guarantee;
the unchanged whole output gate decides admission. No precision sweep or target
admission follows from these control receipts.

Receipts and complete per-group descriptor/input/output hashes are archived in
perf_records/smol_source_group_{bounded,exact}_*.json; the journey records source,
numeric and emitted object pins. Token allocation per optimization is unavailable;
the parent campaign ledger owns shared usage snapshots.
