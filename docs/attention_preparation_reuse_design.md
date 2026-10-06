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
