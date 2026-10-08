# Explicit target operand residency

These schedules retain operands within one primitive Gemmini invocation. They
change target placement and DMA requests. Callers must prove immutable dense
input storage for the complete invocation, output nonoverlap, and exclusive
accelerator use through the final fence. Dimension checks do not prove these
source/effect/lifetime obligations. Source packing, numerical permissions,
requantization, consumer proofs, host dispatch and cross-call lifetime management
remain in Merlin. Target admission uses the operation's dimensions and the
selected configuration's `DIM`, scratchpad rows and accumulator rows. No model
name, captured source ID or measured answer selects either schedule.

Resource admission establishes legality. It does not establish lower cycles:
bank service, code size, overlap, preparation and readback remain part of a
complete comparison. Every compiled device object and linked program requires a
zero-FSM executable audit. These options remain false by default.

## Complete convolution reduction weights

`GoldenFlatConv(..., cached_reduction_weights=True)` retains the original HWIO
weights in a disjoint upper scratchpad interval across every spatial band of its
existing 3x3 contraction. The original tap/channel order, activation placement,
accumulator work, output layout and readout policy are preserved. The complete
weight interval needs `9 * ceil(Cin / DIM) * ceil(Cout / DIM) * DIM` rows. This
schedule requires aligned `Cin` and refuses competing separate/ping-pong B
placements or activation/weight overlap. Channel and spatial output tails keep
their exact extents; padding follows the existing explicit halo/virtual-padding
contract. Nothing survives the final invocation fence.

The normal factory uses `select_kernel(..., cached_reduction_weights=True)`.
`select_cached_reduction_weights` retains every declared emission option and
ranks an admitted candidate by requested B bytes and load count. No reduction in
requested work preserves the control. Unsupported selected emitter families
also preserve the control, with an explicit refusal. The ordinary source-bound
bundle forwards the choice as `flat_cached_reduction_weights=True`; its CLI flag
is `--flat-cached-reduction-weights` and requires flat scheduling.

Source/primitive tests cover complete original tap/channel/input/output binding,
independent rectangular and stride cases, channel/spatial tails, capacity,
placement conflicts and ordinary spatial command loops. The normal flat factory
currently selects a separate B bank. Requesting cached weights preserves that
control and records the incompatible placement refusal; it does not silently
replace the bank choice. An explicitly supplied nonbank control can admit the
call-local schedule. The earlier 52-leaf preparation that changed one placement
used an option-overriding selector and remains historical evidence for that
sealed source. The repaired successor re-emits all 52 normal objects/adapters
exactly as the original control, with no cached application. New whole
numerical and hardware timing qualification remain pending.
