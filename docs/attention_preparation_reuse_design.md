# Explicit attention operand preparation reuse

Status: design derived from the accepted source and current executor; no routing,
ABI, allocation, or performance promotion. The current normal provider continues
with its qualified single private workspace.

## Measured reason to investigate

The complete classification-provider Spike diagnostic attributes 293,443,169
instructions to radix preparation and 106,878,984 to source gathers. Nested dot
bounds spend 73,638,528 on right-operand metadata, 339,835,904 on left-operand
metadata, and 588,779,350 on output bounds. Instrumentation changes generated
code. These are instruction scopes, not hardware cycles or additive savings.
Gamma/environment admission already runs once per product call (96 per group).

## Two distinct reuse opportunities

1. Within a group, the immutable Q rows are encoded for both QK key tiles.
   Prepare the Q signed planes, reconstructed values, row scales, and admitted
   source/error norms once per head and retain them across those two calls.
   This does not apply automatically to dynamic probability intervals in PV.
2. Across four adjacent query-quarter calls, typed source analysis proves 96
   groups of equivalent K/V views: 432 requests reduce to 144 logical views
   (48 Q, 24 K, 72 V). This is a logical census, not a physical preparation
   contract. Both the current provider's formats and ownership must be explicit
   before sharing.

## Required physical representation

For each head independently, the current callback consumes A planes as
`[plane,row,k]` and B planes as `[plane,k,column]`, with three signed 7-bit
planes and source-derived power-of-two row scales. Preparing all heads must
therefore distinguish `[head,plane,...]` from `[plane,head,...]`; they are not
interchangeable. Q/K logical vectors use the last axis; V vectors use the
reduction axis and require a transposed logical row view.

A prepared format must bind these axes, signed range, radix, digits, padding,
source/reconstructed scalar types, exact encoding policy, and norm proof version.
Retain original widened values needed for ordered source replay. A format key
cannot be only a shape or pointer. The scalar encoder and norm algorithms remain
Merlin; target panel packing and primitive admission remain OOT.

## Ownership and admission

Use compiler-visible preparation results with explicit immutable consumers.
The source proof must establish equal root SSA value, static slice coordinates,
layout, and element type, plus no intervening writes and owner lifetime through
all consumers. An opaque runtime pointer cache is insufficient. The preparation
must dominate its uses; allocate owned storage once in the caller and release it
after the last synchronous consumer. Baremetal arena free is a no-op, so a
compiler-owned reusable region must also establish bounded total allocation.

Stable RNE, gradual underflow, nontrapping execution, and unobserved exception
flags are required for moving norm preparation. Unsupported inputs/plans retain
the original source fallback. Preparation failure may not publish an admitted
handle. A later consumer may not reinterpret a differently oriented handle.

## Qualification before changing the current provider

- Independent multi-head and non-square fixtures distinguish every physical
  axis; include unequal strides, tails, signed extremes, and rejected layouts.
- Compare every plane, scale, reconstructed value, and norm with the current
  checked preparation, including error-zero and nonzero-error rows.
- Exercise live aliases and intervening writes: refuse sharing unless the typed
  lifetime/effect proof holds. Dirty reused buffers must be fully initialized.
- Preserve actual source reductions, masks, denominator and quantization
  observations, original whole1,600-output gate, and source fallback ownership.
- Measure a complete original group with preparation/storage costs included.
  Do not project the logical census or instruction counts to a whole-model gain.

## Local encoding screen outcome

The local Q encoding-only implementation is not enabled. Three complete original
12-head target capsules passed all 196,608 accepted carrier words and guards,
with unchanged product/replay counters. Against 5,034,507,191 control retired
instructions, the first implementation used 5,152,843,018, forced inlining used
5,397,217,199, and the final shared evaluator used 5,053,569,315 (+0.379%).
The outlining hypothesis did not explain the outcome. No FPGA arm was admitted.
See `perf_records/smol_local_query_preparation_negative.json` for immutable pins.

This screen caches encoding, not admitted row norms. Q-only norm reuse is a
separate possibility, but source work limits its scope: each query row visits
2 × 64 QK reduction elements and 2 × (192 + 192 + 128) PV elements. Eliminating
one repeated QK norm preparation removes 64 of 1,152 left-row element visits
(5.56%). This is an operation count, not a timing fraction: QK exact inputs and
PV uncertain probability intervals have different work. The measured 339.8M
left-metadata scope cannot be claimed as its potential saving. Equal PV segment
lengths alone do not authorize reuse because the source slices differ.

Cross-group K/V preparation remains an unimplemented typed format/lifetime
contract. Further work should price its complete storage and preparation costs
against the current qualified numerical capabilities, rather than carrying
forward the packing-only performance hypothesis.

## Concrete cross-group physical handle proposal

The existing `TensorPreparationOpportunity` witness is the semantic source
identity, not a handle. Revalidate it immediately before any rewrite. For each
opportunity, derive batch/head count, row axis and reduction axis from the typed
contraction maps. The source K row is `[head,key,depth]`; the source V row for
encoding is `[head,channel,key]`, although its retained replay values remain in
original `[head,key,channel]` order. This distinction is part of the format.

A first conservative owned allocation contains disjoint aligned spans:

| Span | Scalar | Layout | Required content |
| --- | --- | --- | --- |
| Source replay | f32 | Original logical source axes | Exact BF16 widening |
| Reconstruction | f32 | `[head,row,k]` | Current encoder's returned values |
| Reconstruction | f64 | `[head,row,k]` | Exact widening of preceding span |
| Signed planes | i8 | `[head,plane,k,row]` | Three current signed-7 radix planes |
| Scales | f64 | `[head,row]` | Exact widening of current f32 row step |
| Source norm | three f64 | `[head,row]` | Admitted L1, maximum, L2 |
| Error norm | three f64 | `[head,row]` | Admitted representation-error norms |

Each element consumes 19 bytes across the four element spans; each row consumes
56 bytes for its step and two admitted norms, before alignment. The f32
reconstruction could later be temporary, but the first implementation keeps it
so every existing preparation result can be compared directly. The format
identity must hash all axes, scalar widths, row/reduction extents, radix/sign
bounds, alignment, current encoder and norm semantics, rounding environment and
numeric capability contract. The old logical-census format hashes are never
accepted as these physical format hashes.

For the current source's one quartet, two K and six V handles contain 1,572,864
elements and 16,896 logical rows. At these conservative sizes the spans total
30,830,592 bytes (plus alignment). This is live storage for one quartet, not an
allocation per call. A normal compiler lifetime plan may recycle this region
only after the fourth synchronous consumer returns; all regions must be
initialized again for the next quartet. No pointer-keyed cache is permitted.

The preparing operation accepts the exact source descriptor and one owned byte
region. It validates every source value and produces a successful immutable
handle only after every plane, step and norm is complete. Failure invokes the
retained original source function; it cannot publish a partially valid handle.
The handle binds the allocation owner/generation, typed source witness and
format, offsets/capacities, and successful numeric admission. Consumers receive
borrowed read-only spans. Device callbacks may read planes and write only their
separate i32 readout; they may not mutate the prepared allocation. Original
semantic BF16 inputs remain available for complete source fallback.

Normal preparation should emit one explicit operation dominating the four
consumers, append the handle only to private borrowed provider interfaces, and
keep the public model ABI unchanged. The complete consumer set and last use must
be validated before mutation, alongside no intervening writes, unknown escapes
or asynchronous uses. A shape match or reusable pointer is insufficient. This
proposal does not yet install a physical prepared ABI or remove any copying.

Qualification should compare actual K and V planes/scales/reconstruction/norm
bits to independently repeated preparation, then poison the region and exercise
another quartet. Independent two-head, non-square, transposed and strided cases
must distinguish head/plane ordering and V orientation. Refuse stale generation,
wrong format, partial failure and overlapping mutable readout. Only after these
checks should a complete original group and ordinary full48 route be measured;
logical reuse counts cannot establish a whole-model speedup.

## Current composed-provider repetition and cost census

The qualified radius/L1/certified-row source has **no RHS preparation inside
refinement**. All 96 RHS encodes and metadata preparations occur during the
initial two QK and six PV calls per head. A proposed cache across refinement
would therefore remove no such work.

An explicit diagnostic split preserves the original compiled consumer outputs,
scales, guards and all eight provider counters. Nested machine-counter scopes
measure 92,951,781 RHS encoding and 68,195,232 RHS metadata instruction ticks:
5.574% of its 2,890,952,429-tick provider ROI. The diagnostic changes codegen and
adds counters; its ROI is 0.403% above the uninstrumented composed provider.
These figures bound the scope worth investigating and are not projected savings.
The first unavailable `rdinstret` probe is retained; the successful fresh probe
uses the existing machine-cycle counter under Spike's instruction-tick model.

Four-query typed K/V sharing remains a real source opportunity. Its proposed
30,830,592-byte retained region, generation/lifetime checks, additional pointer
loads and fallback/preparation costs must be measured before enabling it. Most
current provider instructions lie elsewhere. See
`perf_records/attention_rhs_preparation_cost_census.json` for immutable evidence.
