# Source-selected full-K banked GEMM

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

The composition uses the same guarded mean, exact readouts, residuals,
pooled stem, virtual padding, classifier, and host transforms as the
qualified mean control. Its actual linked device bytes enter normal build
identity. Full original native and Spike output validation and a final
zero-FSM ELF audit are required before any hardware submission. The local
GSIM improvement is not a whole-model or hardware performance claim.
