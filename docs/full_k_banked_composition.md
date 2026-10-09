# Full-K banked GEMM compiler policy and experiments

## General command-cost policy

`captured_requant_bundle --dense-input-policy banked_command_cost` is an explicit
compiler search option. It considers each source-proven dense contraction using
its dtype, dimensions, epilogue and target resources. An aligned unbiased i8
contraction with multiple M tiles can use a full-K A panel, cached B, alternating
physical A/accumulator banks and ordinary CPU M prefetch. The existing resource
validator admits the candidate; no model name, source region ID or output value
participates in choosing it.

The policy selects the candidate only when its primitive command count is lower
than the legal current schedule. Equal cost, unsupported geometry and capacity
refusals preserve that schedule and record the reason. This cost is a command
count, not a FireSim cycle prediction. The policy preserves all dimensions,
scale, activation, output type and bias semantics. It remains default off and
cannot be mixed with explicit full-K source selections.

Independent actual kernel checks cover176x256x64 and an odd95-row/48-column/K80
tail, comparing every output and2048 guard bytes. On the first case GSIM measures
21,008 current-schedule versus14,915 banked kernel cycles (29.00% lower). Both
strictRV64GC Spike executions also pass. These are capsule results; whole-model
accuracy, final linked-ELF audits and stock FireSim measurements are separate.

## Historical source-selected experiments

`captured_requant_bundle --full-k-banked-region REGION` selects an explicit
source binding for a full-K A panel, cached B, alternating physical A and
accumulator banks, and M prefetch. The option is repeatable and empty by
default. It preserves each binding's complete source transition proof,
scale, activation, dimensions, and integer bias policy. Missing, refused,
or non-dense requested regions fail closed.

Admission requires aligned unbiased i8 GEMM, multiple M tiles, full A/B
panels within reserved scratch banks, and the output panel within each
accumulator bank. The generator's existing resource validator is the
single authority. Legality does not imply a performance win.

The first composition selects only `matmul_11`: M3136/N128/K256, the exact
shape of the positive screen in `resnet_matmul11_banked_m_gsim.json`.
There are no other bindings with this measured signature in the current
ResNet catalog. Matmul5/8 have K256 but N64; those schedules are retained
because the new full-K bank pattern has not been measured there.

The composition retains the original blocked-channel mean from the fastest
1812 control, together with its exact readouts, residuals, pooled stem,
virtual padding, classifier and host transforms. The exact guarded-mean
1819 arm regressed in hardware and is kept as a separate negative receipt. Its actual linked device bytes enter normal build
identity. Full original native and Spike output validation and a final
zero-FSM ELF audit are required before any hardware submission. The local
GSIM improvement is not a whole-model or hardware performance claim.
