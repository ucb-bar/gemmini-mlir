# Optimization evidence and independent capsule coverage

This is an investigator's design record. It connects measured changes to reusable
compiler tests; it is **not an experimental author's input corpus**. Model names,
captured shapes, weights, outputs and performance labels in the linked history
must stay outside an independent Phase1/2 author view. Generate stimuli and
private answers from separately reviewed operation and hardware contracts.

## What the measurements establish

The journey identifies useful rules and counterexamples. It does not isolate the
cycle contribution of every combined change. The scopes below distinguish a
matched section, a whole model, a composed change, a native measurement and a
structural result. Overlapping gains must not be summed. A functional transform
is not automatically a profitable policy.

For every family, generate three separate obligations:

1. **Applicability:** the input has the required semantics, layout, users, effects,
   numerical policy and resource proof; nearby illegal inputs refuse or fall back.
2. **Profitability:** compare legal alternatives on the same generated work and
   output contract, including all setup, packing, proof, transfers, reconstruction,
   replay, allocation, dispatch and publication inside the declared cost boundary.
3. **Generalization:** withhold independent rectangles, tails, capacity regimes,
   reuse counts and compositions. A training-size win cannot certify these cases.

### Sizes come from contracts

Use independently declared logical extents, array axes, operand/accumulator widths,
transfer limits, reserved intervals and physical banks. For this backend's current
square array, let `D` be its declared edge, `mt=ceil(M/D)`, `nt=ceil(N/D)` and
`kt=ceil(K/D)`. Complete A occupies `mt*kt*D` rows; complete B occupies
`kt*nt*D` rows; an output block occupies `bm*bn*D` accumulator rows, doubled
when the selected schedule actually keeps two blocks live. These equations are
backend facts, not constants to put in Merlin's shared rules.

Test the **sum of simultaneously live allocations**, including reservations,
against each relevant interval. Derive the last fitting and first overflowing
sizes, plus neighboring aligned and tail cases. Test bank placement and packet
limits independently of nominal total capacity. Host cache lines, associativity
and capacity require selected, pinned host facts; absent facts remain UNKNOWN.
The local simulator's cache parameters are not stock Rocket facts.

An exact equality to capacity may be unreachable after alignment or reservation.
Record that fact and the actual inequalities rather than relabeling the last
fitting point as an exact-capacity case. Bound enumeration explicitly and refuse
an excessive request; never silently truncate the sweep.

## Optimization families

### 1. Complete A residency, output blocking and coalesced transfers

- **Pattern:** immutable A reused across independent output-channel blocks;
  contiguous or proved segmented source; increasing-K reduction unchanged.
- **Cases:** joint A/B footprints around scratchpad boundaries; output blocks
  around accumulator boundaries; N around the operand transfer width; independent
  M/K/N tails and segment crossings. Vary rectangles and reuse, not only square GEMMs.
- **Cost/feedback:** preparation, gathers, configuration, every A/B request and
  B reload, output stores, decoder and drain; effective emitted options, interval
  allocation, spills and actual complete timer.
- **Evidence:** composed compact case `685,067.5 -> 470,524.5` GSIM cycles;
  independent full-resource case **+21.376%**. These are different cases, not a
  universal rule. [Matched complete cases](perf_records/resnet_compact_family_complete_20261008.json).

### 2. Complete cached B / reduction weights

- **Pattern:** immutable weights reused over multiple M/spatial blocks within one
  call. No implicit device lease across calls.
- **Cases:** complete B plus all live input and ping-pong rows at capacity/bank
  boundaries; accumulator blocks; K aligned and +/-1; reuse once, twice and many.
- **Cost/feedback:** initial preload, every input transfer and output, final drain;
  actual preload multiplicity, allocation and exact reduction/readout ABI. A
  byte reduction has no cycle price without calibration.
- **Evidence:** complete primitive gains **6.8661% i8 / 6.4599% i32**; adapter
  costs excluded from that primitive remain separate.
  [Complete weights](perf_records/resnet_full_weights_complete_gsim_20261007.json).

### 3. Virtual padding, source strides and row-residue convolution

- **Pattern:** affine source slice/reshape reaches contraction with proved virtual
  zeros and source pitch; no escaping materialized im2col use.
- **Cases:** independently declared windows, strides and pads; H/W/C/output tails,
  row residues, resident planes/stripes around capacity; incompatible stride and
  width controls. Include live users that prevent removal of materialization.
- **Cost/feedback:** source preparation/transposition/zeroing, gather, transfers,
  contraction, decoder and publication; typed affine/layout proof, emitted entry,
  source-to-object binding. Requested bytes and physical traffic remain distinct.
- **Evidence:** section `818,566 -> 707,799`; separate whole model
  `29,514,240 -> 29,402,206` stock cycles.
  [Section](perf_records/root_resnet_source_residue_pair_stock2088_2089_terminal.json),
  [whole](perf_records/root_resnet_source_residue_whole_stock2090_terminal.json).

### 4. Retained ordinary command loops

- **Pattern:** identical bounded primitive commands, pointers and reduction order
  expressed as CPU loops. Hardware FSM/LOOP instructions remain prohibited.
- **Cases:** iterations once, twice and many; nested tails, segment transitions;
  linked code, stack and spill regimes. Storage/order proofs stay unchanged.
- **Cost/feedback:** full callback, CPU branches and dependencies, configuration,
  transfers and decoder. Record dynamic and linked-code features, not code size alone.
- **Evidence:** stock `1,001,893 -> 898,799` (**10.2899%**), but GSIM
  **1.7874%**; an independent tail case regresses **0.0872%**.
  [Matched engine comparison](perf_records/root_resnet_source_stride_stock2083_2084_stock2083_2084_terminal.json).
  Another loop changes instructions/code much more than cycles:
  `512,059 -> 511,541` (**0.1012%**).
  [Neutral loop](perf_records/root_resident_inner_row_model_review_20261006.json).

### 5. Stationary B: short tile before the last full tile

- **Pattern:** independent M tiles share B; actual preload/transposer/garbage-D
  legality proves a short tile can precede the final full tile.
- **Cases:** `M=qD+r`, independently varying q and `r=0,1,D-1`; real D, bias and
  epilogue alternatives; incompatible placements refuse.
- **Cost/feedback:** setup, source/zero transfers, both stores/fences and exact
  decoder; wave state and actual primitive order. Nominal padded MACs are not
  necessarily a mandatory hardware floor.
- **Evidence:** matched section `890,568 -> 809,631` stock cycles; whole saving
  is not inferred. [Tail ordering](perf_records/root_resnet_stationary_B_tail_stock2093_2094_terminal.json).

### 6. B prefetch into slots beside complete resident A

- **Pattern:** immutable complete A and current output coexist with next B in
  disjoint, explicitly allocated banks.
- **Cases:** zero/one/two free slots after **all** resident storage, bank crossings,
  output widths, segmented source and tails.
- **Cost/feedback:** complete callback including coalescing, configuration,
  arbitration, issue, stores and drain; ABI-forwarded layout witness and measured
  overlap. Overlap remains unpriced when unobserved.
- **Evidence:** `527,421 -> 507,686.5` GSIM cycles despite **+3.4108%** retired
  instructions; independent gains **6.2005% / 4.1844%**.
  [Remaining slots](perf_records/resnet_remaining_slots_complete_callback_gsim_20261007.json).

### 7. Exact quantized residual networks and panel/key batching

- **Pattern:** two byte inputs reach ordered scale/add/round/clamp; a source-derived
  complete-domain fibre/correction proof preserves the original observation.
- **Cases:** independent coefficient/sign/zero/clamp/tie laws; collision refusal;
  source-type domains and i32 prefix bound; one/small/many panels around capacity
  and transfer limits, with activation order controls.
- **Cost/feedback:** certificate preparation at its legal invariant scope, all
  products/corrections/transfers, lookup/decoder scans, fences and publication;
  actual chunk/product/request/fence counts and proof/scan costs.
- **Evidence:** four-panel section **17.496%** but separate whole only
  `30,169,093 -> 29,891,965` (**0.9186%**).
  [Four panels](perf_records/root_resnet_stock2068_four_panel_terminal_review_20261007.json).
  Key batching changes one whole route:
  `28,728,702 -> 28,649,233` (**0.2766%**).
  [Key batches](perf_records/root_resnet_key_batch16_stock2109_qualification.json).

### 8. Checked packed readout and decoding

- **Pattern:** complete returned byte objects or paired i32 products feed an exact
  lookup/finisher without other live reads.
- **Cases:** every word-alignment residue; lengths zero, one, word-1, word, word+1;
  pitch/segment tails, alias and endian controls. Word width comes from the ABI.
- **Cost/feedback:** both device stores, fence, scan/decode, guarded tail and
  publication; checked coverage and actual memory operations.
- **Evidence:** `849,365 -> 593,115` GSIM cycles; independent
  `18,570 -> 11,843`. No whole timing credit.
  [Checked readout](perf_records/resnet_checked_pair_scan_complete_current_whole_20261007_qualification.json).

### 9. Independent scalar quantizer scheduling

- **Pattern:** lane-independent ordered multiply/clamp/RNE/conversion with fixed
  source effects and rounding.
- **Cases:** lane choices including nonpowers; row lengths around each lane width;
  several rows and tails, counting `sum(ceil(row_length/lane_width))`;
  half/double independent extents and linked FP/GPR dependency/spill regimes.
- **Cost/feedback:** complete preparation and finishing/stores; ordered FP
  distances, dependencies, spill/code footprint. Equal opcode counts do not tie schedules.
- **Evidence:** unchanged arithmetic counts but stock gains **31.253%, 31.019%,
  30.033%** over three extents.
  [Spacing](perf_records/root_source_host_quant_spacing_stock2063_terminal_review_20261007.json).
  Separate packet scheduling **2.879% retired** is not a cycle gain.
  [Packet instructions](perf_records/resnet_current_runtime_packet_batch_20261008.json).

### 10. Exact final-integer observer bins

- **Pattern:** closed scalar source expression reaches the original RNE/clamp/
  conversion; an exact preimage or sufficient zero bin retains the continuation.
- **Cases:** ties/clamps/zero boundaries, signed zero, subnormal/nonfinite controls,
  FRM/effects, contraction lengths and reuse. Do not require hidden intermediate
  equality when the source exposes only the integer observation.
- **Cost/feedback:** producer, proof/table preparation, observer, continuation and
  scans; admission/fallback counts and their complete prices.
- **Evidence:** section `3,712,239.5 -> 3,483,695` stock cycles despite
  **+1.2994%** retired instructions;
  [section](perf_records/root_tiny_rne_zero_stock2082_stock2082_terminal.json).
  Separate whole `380,396,343 -> 378,946,263` (**0.3812%**).
  [Whole](perf_records/root_tiny_rne_zero_stock2085_terminal.json).

### 11. Producer-exact packing and immutable polynomial context

- **Pattern:** a finite typed producer feeds exact widening/radix packing;
  source polynomial/floor/exponent/cutoff law has immutable context. An observable
  F32 denominator remains an F32 observation. Exact intermediate agreement is
  one selected proof route. A separately admitted approximation may change
  intermediates while retaining the unchanged final gate; local permission does
  not authorize replacing whole attention.
- **Cases:** independently generated finite-word/range/sign/zero/subnormal classes,
  branch/floor/tie boundaries, row tails, context once/many and epoch mutation.
- **Cost/feedback:** metadata/proof validation, packing, both endpoints, denominator,
  every product/readout/refinement, quantizer and fallback; replay counts and intervals.
- **Evidence:** exact-row section `3,936,970,420 -> 3,857,281,394`;
  [row proof](perf_records/root_smol_exact_row_stock2091_2092_terminal.json).
  Theorem-plus-context composition `3,857,281,394 -> 3,611,264,318`, not
  constants-alone attribution.
  [Composition](perf_records/root_smol_polynomial_constants_stock2098_terminal.json).
  Coarse buckets regress **10.1796% complete native slice runtime**, with
  replayed denominator headrows **2 -> 157**; neither target nor whole gate was
  run. [Native negative](perf_records/smol_polynomial_coarse_native_negative_20261008.json).

### 12. Exact multi-product degree sums and resident planes

- **Pattern:** several signed integer products produce public exact degree planes;
  output stride and prefix bounds are proved. No inferred FP reassociation.
- **Cases:** one/multiple products, shared operands, referenced-plane counts,
  signed-prefix overflow boundaries, all output planes/tails; joint referenced A/B
  capacity and accumulator output blocks, alias and epoch refusal.
- **Cost/feedback:** all packing/products/transfers, readout and reconstruction,
  proof and fallback. Callback count is only one feature.
- **Evidence:** fused readout section `4,256,146,700 -> 4,027,126,711` stock
  cycles. [Fused readout](perf_records/root_smol_fused_radix_stock2069_terminal_review_20261007.json).
  A partial alternative cuts callbacks/bytes but reaches **5.349B** retired
  versus **2.165B** exact, and is rejected.
  [Partial negative](perf_records/root_digit_norm_partial_complete_negative_review_20261007.json).
  New full operand residency is functional evidence only, without inherited cycle gain.

### 13. Source-derived scalar carriers and interval tables

- **Pattern:** two integer products feed a closed scalar expression and final
  integer observer; exact and locally permitted approximate policies remain separate.
- **Cases:** independent source DAGs/coefficients; table sizes at selected host
  cache boundaries, locality/fanout, cold/warm setup and lifetime; ambiguous/tie
  inputs and original fallback.
- **Cost/feedback:** both producers, DMA/readout, table/proof creation, observer,
  continuation, publication and allocation/free; actual working set/reuse/spills.
- **Evidence:** complete affine case `8,589,793 -> 8,288,352` GSIM cycles;
  independent case `27,123 -> 39,103` (**+44.169%**).
  [Affine cases](perf_records/tiny_affine18_complete_timing_20261008.json).
  Earlier 8MiB table section improves **6.531%**, but separate whole
  `422,018,733 -> 424,921,379` regresses **0.6878%**.
  [Section](perf_records/tiny_source_interval_table_complete_gsim.json),
  [whole](perf_records/root_tiny_source_interval_terminal_review_20261006.json).

### 14. Closed all-use masked contraction skipping

- **Pattern:** typed mask proves complete reduction tiles discarded by every
  observer; reduction order and nontrapping/unobserved effects remain fixed.
- **Cases:** independently generated triangular/banded/rectangular masks,
  full/partial/empty tiles, tails, escaping users and FCSR refusals. Partial tiles
  retain original arithmetic; producer precision is a separate obligation.
- **Cost/feedback:** full producer/consumer compound, mask, packing, observer,
  fallback, configuration and readout; source use-closure and observed arithmetic.
- **Evidence:** whole `412,672,134 -> 410,147,055` (**0.612%**).
  [Whole](perf_records/root_tiny_stock2062_masked_whole_terminal_review_20261007.json).
  Separate compound **16.38%** does not transfer proportionally.
  [Compound](perf_records/root_tiny_masked_whole_stock_release_20261007.json).

### 15. Dependency-local endpoint certificate reuse

- **Pattern:** private prepared DAG and producer epochs allow interval narrowing
  to retain facts only inside the old bound; mutation invalidates dependencies and
  denominator-wide observations refresh correctly.
- **Cases:** narrower/equal/wider/replaced bounds, context/epoch mutation,
  repeated/one-shot consumers, shared columns, zero/tail/nonfinite/refusal and
  allocation limits. Exercise the actual multi-pass protocol, not an invented cache lookup.
- **Cost/feedback:** all preparation, products/emulation, metadata allocation/free,
  snapshots, observer passes, replay/membership, quantizer and publication;
  actual dirty-cell/pass/replay counts.
- **Evidence:** native complete case `242.300191 -> 241.463285 ms`
  (**0.3454%**). No target/whole cycle credit.
  [Complete protocol](perf_records/smol_real_frontier_narrowing_20261008.json).
  Separate point cells improve retired instructions only.
  [Instruction result](perf_records/smol_frontier_point_cells_group_20261007.json).

## What exists in the automatic-loop infrastructure

Merlin already has independent component generation, tile-relative extents and
encodings, selected-address-space residency ladders, independent goldens, written
program admission, explicit zero-MAC objectives, frozen corpora, reviewed edit
surfaces, component feedback/CCA and calibrated analytical provider primitives.
Those mechanisms are real; they do not constitute a complete convergence run.

The joint-footprint boundary extension is published on Merlin main and runs
through the existing generator and writer. It passes 20 source and 20 physically
outside-installed checks at two synthetic hardware geometries. It supplies size derivation,
not a proof that a backend selects or profits from residency. General semantic motifs, user/effect/epoch scenarios,
numeric stress generation, matched ablation plans and a bound coverage-plan
receipt still need implementation. An audit also found the legacy writer could
consult a capture census while generating independent components. A separate
published fix threads trusted independent mode to bypass lookup, stat and read.
It passes 29 source and 29 physically outside-installed checks, including a
populated hostile census and byte-identical legacy annotations. Synthetic
fixtures without a census alone do not establish that isolation boundary.
Ordinary fresh component authoring currently
refuses because transport and required isolation are not qualified. Provenance-
preserving QKV CSE and scoped floating epochs remain prototypes/unqualified.

Do not infer functional Phase1 from this matrix. Phase1 needs independent
functional/fallback composition guards over the declared supported domain.
Phase2 needs complete-cost alternatives and hidden transfer cases, plus calibrated
feedback. Final evaluation must freeze the compiler first and use the unchanged
whole-output budgets on held-out workloads.

## Prioritization and feedback contract

Use the selected corpus's **own** measured or bounded work shares. Rank proposals
by complete critical-path opportunity, confidence and evaluation cost. Required
support work, scalar observations and copies can be zero-MAC bottlenecks. A device
MAC roofline alone cannot choose these transformations.

Each result should identify the source pattern and refusal reason; effective
transform options and emitted entry; source/object/ELF identity; resource and
layout proof; full cost boundary; correctness and original fallback; measured
counter units; prediction interval and calibration regime; and independent
transfer results. Preserve regressions and ties. If a constituent contribution
cannot be isolated, expose that uncertainty and schedule a matched ablation.

The unchanged functional gates, all outputs and final executable instruction
audit remain mandatory. No phase may buy a speedup by weakening the budget or
substituting a model-specific route.
