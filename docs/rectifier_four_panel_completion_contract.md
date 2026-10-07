# Four-panel exact rectifier batch

This explicit, default-off target schedule preserves every certified
prediction/correction product. It groups up to four independent sixteen-row,
64-column panels, retaining completion fences at both external-memory phase
boundaries. No reservation-station DDR alias inference is made.

## Storage and publication

The batch has four private 64-row accumulator slots, `[0,256)`, versus the
1024-row capacity. Each of the six moving SPAD planes reserves 256 rows:
lhs `[0,256)`, temporary `[256,512)`, prediction `[512,768)`, rhs
`[4096,4352)`, indicator0 `[4352,4608)`, and indicator1 `[4608,4864)`.
Existing readonly weights and constant seeds retain their separate bank2 and
bank3 intervals. All full reserved extents are checked for overlap/capacity.
Partial final batches use one, two or three of the same reserved slots.

For each batch and logical column group:

1. Load each panel's immutable A/B inputs, compute its complete predictor in
   its distinct ACC slot, and store to its distinct private C tile.
2. Execute a mandatory completion fence after **all** predictor stores.
   Only then reload those C tiles into their distinct prediction SPAD slots.
3. Execute the unchanged exact rectifier products for each panel, using its
   distinct input/temporary/indicator slots and shared readonly seeds. The
   existing pinned same-tile SPAD chain proof applies separately to each panel.
4. Write each final result from its distinct ACC slot. Execute a mandatory
   completion fence after **all** final stores, before next-batch reuse.

The output remains a fresh private owner disjoint from input/constant storage;
the caller cannot observe intermediate prediction bytes. The ranked adapter
publishes its result descriptor only after the kernel's final completion fence.
Per-output coefficient chunk order, exact integer stages, readout rounding and
the complete finite source certificate remain unchanged. Reordering panels
does not reorder any output's reduction.

## Pinned ordering witness

`rectifier_panel_batch.PanelBatch.require` binds the existing execute/mesh/local
ordering contract plus LoadController, StoreController, Scratchpad, Controller,
DMACommandTracker and RocketCore source identities.

RocketCore lines412–419 stalls a fence while the RoCC is busy. Controller line443
includes reservation-station and scratchpad busy. DMACommandTracker retains
the command until every requested return is accounted for. StoreController
lines188–204 reports completion from that tracker. Scratchpad line494 includes
the reader, writer and pending write queues. Thus each retained boundary drains
the predictor writes or final writes before the following phase.

Within the reload/correction phase, ReservationStation lines344–351 holds an
execute reader behind overlapping pending SPAD loads; lines358–378 holds a
store reader behind its pending ACC producer. The reload therefore completes
before its real-D consumer, and the final ACC output cannot publish earlier.
No earlier predictor ACC slot is overwritten until the retained phase fence.
All store-scale changes occur after that fence, with constant scale within
each phase. No CPU access, DMA reload, unknown side effect or external-memory
ordering boundary is removed by the internal SPAD fence pass.

## Prospective performance scope

The four-panel hypothesis is lower completion/issue overhead and additional
legal overlap between independent panels. Array products, requested load/store
bytes and real/retained B preload counts are unchanged. The frozen row/B cost
model therefore ties these alternatives; it cannot price their fence or
cross-panel scheduling change. CPU instruction count alone cannot establish a
winner. Complete original-input ranked producer cost is required, including
tables, seed loads, stores, reloads, all retained fences and descriptor checks.

Default factor1 target IR and object bytes are checked against the immutable
old source for four independent geometries, both internal-fence policies.
Resource/refusal tests cover all partial batch counts. Exact target checks bind
the entire 65,536 source pair domain and an independent M80×N192 case that
executes full and partial batches with multiple column groups. Whole-model and
stock performance remain unknown until separately qualified and measured.
