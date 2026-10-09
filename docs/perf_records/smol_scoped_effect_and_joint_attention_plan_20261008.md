# Scoped CSE effects and a distinct joint attention candidate

Status: bounded source/object evidence and design, 2026-10-08. No model build,
new numerical permission, provider execution, output accuracy or cycle claim.
Merlin owns the source contracts, closure, CSE epochs and representation policy.
OOT owns target callback implementations, their effect facts, integer product
commands and physical resources.

## Current source and provider compatibility

The pinned current `smol-exact-math-normal` prepared source has 407 calls. The
sealed nested CSE algorithm preserves every callee and its order, all operand
and result types, every semantic call attribute, and each operand's original
source identity. This covers 1,850 operation-result operands and 727 block
arguments. The 73 function declarations retain their semantic attributes.
The original file and native module remain unchanged. The transformed module
was not emitted. This is typed compatibility; bufferization, normal provider
execution, all 48 groups/23,040 callbacks and the original whole gate remain
separate obligations.

The current first three BF16-to-i8 quantizers share an immutable tensor input.
Between adjacent quantizers, the only call is `merlin_dev_gemmini_0`. Each uses
the same integer `1024x768` by `768x768` contraction ABI. These names and the
source IDs identify this audit, and are not proposed compiler selectors.

The actual linked-input wrapper object maps this callback to
`gemmini_golden_91bac0fa78353415`. The wrapper has 124 decoded instructions;
the target function has 1,019. Neither has host floating-point instructions or
CSR instructions. The wrapper's sole call relocation is that target function;
the target function has no call relocation. Its source-matched inline assembly
forms are `fence` and custom3 commands with integer destination x0. The wrapper
contains an `unimp` guard for invalid descriptors: nontrapping permission cannot
be granted merely from absence of FP instructions. Valid descriptor, disjoint
output and accepted-shape proofs must discharge that guard.

These instruction facts do not by themselves prove the target custom commands
preserve host FCSR. The provider must supply that effect fact tied to the actual
source, object, ISA/target fact bundle and transitive callback closure. No
permission is inferred from an RNE check. The numeric attention provider calls
`fegetround`; the native outward-conversion provider temporarily sets and
restores rounding. Those providers require separate summaries and are not
covered by the integer dense callback census.

## Generic scoped CSE effect design

Keep the existing global six-permission API and strict/nested features exact.
A separate future scoped API should admit only source-derived effect epochs:

1. A typed provider summary identifies exact source and object hashes, complete
   ABI, complete transitive calls, target fact identity, valid-input predicate,
   memory reads/writes and alias/fresh-result behavior. Its host effect fields
   explicitly describe FRM preservation, exception-flag reads/writes,
   nontrapping valid-domain behavior and interposition/errno observations. A
   callback identity or a return-value RNE guard is insufficient.
2. The source owner supplies local observation permissions. FP exception flags,
   traps and interposition must be unobserved in the selected source region.
   A provider summary is not allowed to manufacture these source permissions.
3. In each admitted single-block region, begin an FP epoch. An unknown callback,
   unresolved indirect call, source rounding mutation, flag observation or
   incomplete summary starts a distinct epoch. Only a fully matched summary
   preserving the relevant host state allows the epoch to continue. Refuse
   ambiguous control-flow joins; initially refuse loops with an unknown effect
   in their scalar region instead of pretending one static epoch covers every
   iteration.
4. Attach private comparable epoch constraints to FP-sensitive operations and
   their enclosing scalar regions. Constants alone do not convey source FENV
   safety. Equal pure parents can share only when their complete semantic
   attributes, operand identities, local epoch and nested epoch constraints
   agree. Native MLIR remains the sole equivalence/effect authority. Preserve
   every original nested result join and remove all private constraints before
   publication. Refusal leaves the source unchanged.
5. Meaningful controls must include an unknown callback between equal
   quantizers, a known FRM mutator, a rounding-preserving integer callback,
   an exception-flag reader, an incomplete transitive call summary, mismatched
   object hash/ABI, invalid descriptors, nested loops and branch joins. The
   known integer callback case must merge parents; the mutator and unresolved
   cases must retain independent computations. Existing default/strict/nested
   global emissions must remain unchanged.

For the actual first Q/K/V triple, this design can avoid assuming all 48
attention callbacks are globally inert: the two intervening integer dense
calls have a narrow source/object closure. Actual OOT custom-command host-state
facts and valid-descriptor/source-observation admission are still pending.

## Existing attention semantics and permissions

The current source performs ordered f32 QK FMAs, scales scores, computes an
online maximum, and evaluates the original polynomial. PV consumes BF16-rounded
probabilities, while the denominator consumes the **unrounded f32** polynomial
values using eight left-fold lanes, the original lane tree and alpha FMA.
Six ordered PV partials are then combined, divided and BF16-rounded before the
complete cross-head quantizer. The current exact observation route separately
certifies the i8 words and BF16 scales.

The RMS4 policy is an approximate source-roundoff estimate. Its contract
explicitly preserves representation uncertainty, prefix safety, nonlinear/order
rules and complete source fallback. It does not authorize replacing max,
probability, denominator or final observer semantics. `word_budget=0`, BF16
probability equality and exact consumer observations are proof obligations of
this route. They are not an extra user accuracy threshold. The user's final
criterion remains all 1,600 outputs with atol=0.03125 and rtol=0.02.

Simply decoding full-K integer dots once and removing all certificates is
already a rejected joint source-DAG experiment: 121/1,600 failed. The adjacent
BF16 policy failed 113; separate PV and QK approximations failed 92 and 168;
the K16 encoded/raw partition controls failed 115 and 113. The once-real patch
control failed 82. These negatives must remain available; no K sweep follows.

## One distinct numerical hypothesis for review

Propose an explicit, default-off **joint normalized-probability policy** over a
typed closed attention-plus-quantizer region. Its defining change is to use the
same BF16 probability values for **both** integer PV and the source-ordered
eight-lane denominator. It preserves the lane/tree/alpha operation ordering,
mask, cutoff, polynomial expression and original output quantizer, but changes
the denominator's representative values. This deliberate intermediate drift
requires new permission; it is not RMS4 or an exact observation theorem.

The hypothesis is that consistent numerator/denominator representatives may
avoid some accumulated normalization drift seen in the rejected original-DAG
center route. This is unproved and may also fail. It is sufficiently distinct
to isolate before lowering operand precision. The first native control would
keep current three-digit packing/products and change only this joint semantic
policy. If the unchanged original whole gate fails, stop and retain the result;
do not sweep digit counts. If it passes, one separately costed two-digit product
candidate can follow under the same explicit policy and final gate.

The closed-source analysis must prove complete uses of score/max/P/den/PV,
the four query partitions, all heads, escaping i8/scale observations, source
effects, input immutability, owner epochs and fallback. No workload name, source
ID or captured-value admission is permitted. Nonfinite inputs, unsafe integer
prefix/accumulator ranges, invalid/zero denominator cases, unsupported layouts
or effects execute the complete original source path. The original zero-den
convention remains part of source finalization. Public output is written only
after the candidate completes; no exact-observation binder can certify this
approximation without its separate permission and whole qualification.

## Complete work comparison, not a cycle forecast

Current vision work has 1,152 QK services, 2,304 PV192 services and 1,152 PV128
services: 4,608 base dot services and 19,327,352,832 source MACs. Three signed
seven-bit planes entail nine integer products per base service, grouped into
five degree readouts: 41,472 integer products/23,040 public readouts. Two planes
would entail four products and three degree readouts: 18,432 products/13,824
readouts. A one-plane control would have 4,608 products/readouts. These are
static work counts only; new target ABI/resource/cost qualification is required.

The proposed first control still pays all current packing and nine products.
It would remove dot-bound/norm/dual-endpoint probability processing, source
score/bin/partial certification, endpoint interval construction and multi-pass
quantizer refinement from the admitted joint region. It still pays one scalar
polynomial evaluation per score, original source-ordered denominator operations,
all product transfers/readouts, reconstruction, partial combination, final
quantization, memory allocation and original fallback on refused inputs.

The current b154 whole functional census contains 37.54B provider, 17.94B
evaluate-products, 9.89B certify-frontier and 9.65B encode-operand dispatches.
These are exclusive function categories in that census; finer source-phase
figures overlap these categories and must not be added again. Removing a source
routine is not an equal cycle reduction, and the remaining host polynomial and
packing cost may still exceed the 5B objective. All setup/service/fallback work
must be inside a complete source cost comparison before any GSIM or whole
FireSim claim.

The selected target contract explicitly exposes integer contraction/movement
and fused readout capabilities, with no normalization/softmax capability. It
warns that header normalization encodings exist while the selected source
configuration has `has_normalizations=false`. Exact stock normalization
capability is not independently reclosed here. The candidate therefore keeps
max, polynomial and denominator on the host; it invents no target feature.

## Next review gates

First review the provider-bound epoch design separately from this numerical
policy. For the joint policy, authorize one native control only after the typed
closure/effect/fallback interface is explicit. Independent masked/zero/tail,
cancellation, overflow, signed-zero/nonfinite and denominator cases precede
the unchanged all-1,600 gate. No target bundle follows a failed native gate.
If the native gate passes, price complete source preparation, service and
quantization before device lowering. Preserve old receipts, default behavior
and every rejected numerical variant.

Per-topic token use is unavailable; no token estimate is assigned.
