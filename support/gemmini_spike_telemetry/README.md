# Isolated Gemmini operand telemetry

This optional hook observes successfully executed primitive instructions at the
existing `gemmini_custom` seam. It receives the exact guest PC and register
operands and reads the functional extension's current configuration. It writes
no guest registers, memory or Gemmini functional state.

The builder copies Spike and the existing Gemmini extension sources into a new
directory, adds the observational hook and builds a separate shared library. It
never installs into or overwrites the production toolchain. `--extlib` selects the
copied library explicitly. Build receipts pin the copied source, hook, builder,
Spike executable and extension binary. A new directory is required for each
build. Unknown insertion seams refuse.

Telemetry is disabled unless `MERLIN_GEMMINI_TELEMETRY` specifies an output path.
When enabled, it aggregates primitive counts, actual packet extents, configuration
strides and requested payload bytes. Zero-fill loads request no external bytes.
Accumulator loads/stores use their configured element width. Pool stores use
configured output extents. Unsupported normalization or unconfigured DMA remain
unknown. Configuration validity is tracked independently, avoiding reads from
fields the original functional extension has not initialized.

`MERLIN_GEMMINI_TELEMETRY_SCOPES` optionally names a file containing `id begin end
entry_pc` rows, with disjoint function ranges and an explicit primitive entry PC.
The hook records actual entry-event order and invocation ordinals. The source
qualification must prove entry multiplicity and match all source call names and
primitive classes to the exact executed histogram. It must not divide a shared
function's counts by an assumed number of calls.

`MERLIN_GEMMINI_TELEMETRY_AGGREGATE_PC=1` groups within a scope/invocation by packet
geometry. Each group retains its first/last target-command event IDs; ordered
entry records retain their exact PC. This compact summary describes command
multiplicities and call order, without claiming a full chronological command log.
The default enabled path aggregates by exact PC and geometry.

`mlir_oot.operand_features` validates count/geometry/entry conservation and emits
existing shared provider pointers for commands, nominal padded array work and
requested DMA payload. It fits no cycles. DIM cubed per compute is nominal padded
work, rather than nonzero MACs, measured activity or a cycle floor. Physical DRAM
traffic, dependencies, bank conflicts and overlap remain unknown.

## Qualified probe

`docs/perf_records/spike_operand_telemetry_qualification.json` closes original
production versus copied-observer stdout and complete PC-histogram byte identity:

| Probe | Numeric contract | Observer wall time | Summary bytes |
| --- | --- | ---: | ---: |
| Independent M33/N73/K65 | All 2,409 i32 outputs and guards | 0.024 s | 76,366 |
| Reference diagnostic | All 1,000 original reference outputs | 27.52 s | 332,185 |
| Current control 1874 profile | Original 1,000 f32 words and full SHA | 27.24 s | 335,509 |

The full command classes match the actual executable histogram. The 54 reference
contraction calls and 16 residual calls, and all 70 current catalog calls, match
actual ordered observer entries. Reference hardware timing labels remain held out
from calibration. Reference Zicntr and production strict RV64GC contracts retain
their separate receipts. No hardware jobs or production schedules were changed.

The independent tail probe requests 6,890 load bytes (33×65 + 65×73) and 9,636
store bytes (33×73×4), with zero unknown DMA. These values count requested payload;
they do not measure bus bursts, physical DRAM transactions or reuse.

Re-export the owned qualified artifacts using:

```sh
python tests/operand_telemetry_probe.py --artifact-root /path/to/owned/artifacts
```

Focused tests independently compile the observer and check width/zero-fill/pool
semantics, unknown normalization, ordered entries, compact grouping and malformed
scope refusal. Transport tests reject FSM/unknown commands, inconsistent padded
work/counts, missing invocation markers and changed engine geometry.
