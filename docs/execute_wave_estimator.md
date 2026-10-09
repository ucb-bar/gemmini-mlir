# Static execute wave geometry

`mlir_oot.execute_wave_estimator` derives the supported controller row rule
from the actual `ExecuteController.scala` bytes. An explicit Scala tokenizer
ignores strings and nested comments. The rule, its guard, inactive port row
counts, minimum row count, transpose mapping and single multiply D inactivity
must all be recognized exactly; changed or ambiguous expressions refuse.
The source digest is part of every hardware-fact record. The caller supplies
the dimension from the target capability facts.

For the currently pinned source, an untransposed weight-stationary wave with
effective D garbage uses `max(active A rows, active B rows, 4)`. Inactive A/B
each contribute one row. Other resolved modes use the dimension. Unknown
dataflow, port activity, row extents or transpose state retain the dimension
with an explicit uncertainty classification. These are mesh feed rows, not
measured CPU, DMA or whole-model cycles.

## Structural tracing and actual instruction configuration

The generic Merlin `static_llvm_cfg.trace_static_function` follows verified
ordinary LLVM CFGs. Declared-width integer wrap, signed/unsigned comparisons,
branch arguments and symbolic one-dimensional addresses are evaluated exactly.
The backend supplies opaque primitive operations and a pointer index width;
Merlin never interprets target commands. Loads, unsupported layouts, poison
flags or unresolved control flow refuse a completed trace. Symbolic pointers
do not establish aliasing permissions. An interrupted iterator is partial.

`commands_from_function` decodes target primitive operations. It checks the
actual normal lowerer's configuration encoding against explicit encoder calls
to derive dataflow and transpose settings. Dynamic scratchpad A addresses are
resolved by the core iterator and checked against their declared range. It
does not independently assume default configuration from absent attributes.

```python
facts = derive_facts(controller_source, dimension=target_facts.DIM)
commands = commands_from_function(function, pointer_index_bits=pointer_index_bits)
report = estimate_commands(commands, facts, retain_records=False)
```

The report retains every primitive command count and completed-trace status,
padded compute geometry, a state/classification histogram and the minimum and
conservative rows among retained per-compute dispatch alternatives. Detailed
records additionally preserve each possible mode, effective A/B/D port state
and logical resident weight tile.

## Dispatch limits

The logical stationary B weight tile is traced through preload and compute
flip/stay commands. It is distinct from the effective D port. A single multiply
has inactive D. A compute merged with the following real preload has active D
and therefore uses the full dimension even when its A tile has only one row.
For a following garbage preload, both alternatives can use the shorter wave.

Source command order does not reveal `cmd.valid`, hazard state or reservation
station dispatch. Both single and combined alternatives are retained; unknown
operand state stays conservative. The totals exclude standalone preload waves,
memory latency, CPU issue, queueing, stalls and start/drain. Their minimum is
the minimum among supported retained scenarios, not a universal timing floor.
No executable, production schedule or numeric policy is changed by this API.

## Qualification

Focused tests cover a derived minimum of seven rows, changed guards and row
expressions, comments, D activity, transpose/OS/unknown refusal or fallback,
logical weight reuse independent of D, flip/stay behavior, dynamic scratchpad
addresses through a real CFG and incomplete input refusal. The generic core
tests cover integer wrap, comparisons, pointer offsets, branches and poison or
dynamic-value refusals. All owner gates pass for the generic core addition.

The current 70 immutable device modules were independently traversed. Primitive
counts match the earlier source-pinned audit exactly: padded dimension geometry
22,989,952, minimum retained compute rows 21,682,816 and conservative retained
compute rows 22,690,134. The classifier's resident weights remain known, while
8,063 of its 8,064 computes can be either four-row single waves or sixteen-row
waves merged with a subsequent real preload. This is an analytical distinction,
not a measured speedup. Current Scala source provenance is pinned; this audit
does not establish the simulator bitstream's original build inputs.
