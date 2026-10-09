# Dense stationary-B short tile placement

This opt-in schedule requires the typed complete cached-A and stationary-B reuse
family, at least two full DIM output-row tiles, one short tile, and no bias seed.
Only independent output PRELOAD/COMPUTE pairs move. The first tile still loads real
B, every output retains increasing K, and all configurations, requested transfers,
ACC destinations, stores and fences retain their exact operands/order. Existing
segmented i8 owner views remain explicit; no source storage is reinterpreted.

The complete captured M49/K2048/N512 i8 producer recompiled byte-identically to its
actual current control before modifying the row order. On pinned GSIM it measures
290,215 to 239,164 cycles with every 25,088 output, 4,096 guard and 1,148,928 immutable
operand byte checked. An independent M37/K33/N35 i32 producer measures 2,008 to
1,930 cycles and checks all 1,295 outputs. Both arms have common operand addresses,
production strict Spike passes and final executable no-FSM audits.

These are primitive complete-finish timers including configurations, input/weight/
zero transfers, computations, stores and final completion fence. Ranked adapter
work and whole-model timing are not measured here. The first fixture was captured
read-only from the accepted current-2090 native execution, preserving all original
1,000 outputs. Its dense source/numeric/object bytes are unchanged in current2095,
which changed only the separately qualified flat spatial producers.

The observed difference is not a pure array-service cost. The conditional hardware
feed rule can be resolved for a short stationary compute followed by a garbage-B
preload independently of single-mul/mul-pre pairing; standalone preloads, dispatch,
CPU issue, memory and overlap are not priced. Nominal command/array counts alone
predict a tie. No whole gain or other shape's timing is inferred from this pair.

Default compilation and all unsupported source/resource families retain existing
behavior. Any later normal model choice must use actual selected catalog variants;
old stored leaves overwritten by segmented consumers are not the executed selection.
A separate original M196/K512 borrowed-view capsule is the next cost qualification.
