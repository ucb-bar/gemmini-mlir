# Checked completion composition

The production compiler exposes `completion_placement=drained|ordered_k` in
`TUNING_KNOBS.json`. Both paths use the original schedule dialect, original
family construction, and the same target data encoder and full completion
callback. The only emitted change on the selected development sources is
removing redundant full `fence` operations inside an output-block K stream.

## Ownership and composition

`completion_adapter.apply_completion(original, context)` is the production
source-bound entry named by `COMPLETION_ADAPTER.json`. Context contains the
problem, actual block, resources, original storage, actual encoder/projector,
selected admission, and a plain premises dictionary. Storage is the original planning ownership view; the consumer replays the original row-base and window-offset decisions to validate the actual selected view. Only literal `True` supplies
a hardware/program premise. Missing, false, or unknown stream premises keep the
original boundaries. The optional `completion_plan` is an actual typed
`schedule.plan` with `schedule.bind` and `schedule.completion_epochs`.

The general consumer accepts checked panel and checked segmented results.
An author wrapper may expose `completion_source()`, returning its actual checked
owner; both objects must revalidate and have exactly the same kernel. Fresh6's
retained-LHS wrapper uses this interface. Its route stays
`retained_lhs_segmented_k`; no conversion to two-window construction occurs.
A segmented admission retains its original combined law, and its effective
issued limits are the minimum of the overlap and reduction limits.

The registered pipeline is `completion_source` (stage 3),
`completion_issued_scope` (4), `completion_ordered_k` (4), and
`completion_emission_input` (5). The unconditional drained alternative uses
`completion_drained` at stage 4. The emitted artifact is the checked result
returned by that pipeline. The ordinary emitter consumes its actual kernel;
completion annotations are lowered through the original full `b.fence` callback.

## Proof decomposition

1. Revalidate the original owner and rerun its route-specific full source,
   storage, integer-domain, and epoch checker on the current context. The K=1
   retained construction also checks its actual full-LHS/two-RHS assignment. The original plan is reread to recover its selected row bases and window offsets within that ownership view.
2. If explicitly requested by the production caller, bind a separate issued
   scope. Count actual dynamic calls in each store-free original output-block
   K stream and check its compute bound against `block.m * block.n * K_tiles`.
   The new compiler issued bound is the maximum counted stream. Every decoded compute output coordinate is proved to stay in one original output block, with outer coordinates symbolic. Physical
   pending-compute, pending-load, pending-store, configuration and execution
   premises are unchanged. `apply_completion` itself never raises a budget.
3. Consume `schedule.completion_epochs`, bound to the exact original command
   and storage context. Only `drained` and `ordered_k` are legal. The scope is
   an output block and preserving readout boundaries is mandatory.
4. Reuse `dense_panel_epochs` to propose completion deletion and prove that
   every load at a removed boundary is read by the next original epoch.
   Selected RAW and WAR laws order a later rewrite through that reader;
   ordered accumulator writes preserve reduction commitment. The original
   full-LHS initializer is protected by its actual effects.
5. Check exact symbolic data-call/tree/order equality, integer bounds, static
   command budget, and the actual resulting completion counts. Revalidation
   binds the original kernel, transformed kernel, context, law, and source owners.
6. Emit that returned checked kernel. No reconstruction of loads, addresses,
   loops, segmentation, or arithmetic is allowed in the completion consumer.

Every original readout boundary, output-block completion, and closing completion
is retained. The optional readout-merging capability of the supplied epoch core
is explicitly disabled for this study. Retained and longer-segment routes use
their original legality primitives rather than pretending to be two-window.

## Outcomes and limits

An unchanged schedule with unknown law or insufficient merged issued budget is a
safe fallback. Invalid ownership, storage, original completions, or plan bindings
raise a precise proof refusal. An unsupported dialect request is malformed IR.
The production caller preserves its original checked result if the new consumer
refuses; the refusal remains in `realization['completion_refusal']`.
These outcomes do not imply numerical execution or physical timing success.

All three public development sources pass exact L2 numerical execution. Same-domain
provisional timing-model reductions are 15.82%, 16.23%, and 17.91%; no physical
speedup is established. Full receipts, actual product comparisons, and contract
checks are recorded outside the package in `EXPERIMENTS.md` and `RESULT.md`.
