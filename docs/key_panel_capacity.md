# Capacity-derived residual panel batches

The key-indicator residual lowering now accepts positive integral panel batches and
checks every live SPAD extent and ACC reservation against target resources.
Completion-order evidence is required for any batch greater than one. The
existing default is four; non-key implementations retain their existing limits.
No model or provenance identifier selects a strategy in the production compiler.

Independent shapes, tails, reduced capacities and all 65,536 source byte pairs
pass. The complete original ranked residual checks 802,816 output bytes,
1,605,632 immutable input bytes, descriptors, 4,096 guard bytes and flags.
Fresh batch-four compilation reproduces the current ResNet2101 objects. Both
release ELFs contain the same implementation/data, differing in one selector byte.

Stock FireSim2105/2106: **1,262,667 → 1,188,667 cycles**, saving74,000
(**5.86061%**). Retired instructions increase167,086→188,323, showing why
command retirement alone cannot price completion drains. Timings include private
scratch, setup, commands, DDR readback/reload and the final fence. Batch16 needs
16KiB private scratch versus4KiB, and larger code/register pressure.

Whole upstream integration and whole-model timing are separate pending gates.
The fixed default remains unchanged. These section savings are not added to a
whole-model estimate. See the immutable root stock qualification and source packet
in `docs/perf_records`.
