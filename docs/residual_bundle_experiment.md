# Source-bound residual numerical-policy experiment

`captured_residual_bundle` defaults to `--max-output-lsb=0`, which requires
exact source behavior for all 65,536 signed-i8 operand pairs. Explicit value 1
permits at most one output integer step for each matched residual. This local
bound is not a whole-model quality guarantee and does not change any default.

The binder matches scalar DQ → f32 sum → optional ReLU → Q, verifies identical
static tensor shapes and symmetric signed-i8 domains, and flattens only extents
exactly divisible by 64. Calls retain read/read/write bufferization access and
typed source, capture receipt, qparams, proof hash, and numerical-policy attrs.
Each adapter owns its constant 16×16 identity and aligned 1,024-byte scratch;
execution is sequential and the kernel fences before returning. Runtime
memref shape/stride checks reject incompatible layouts.

The native standin performs two independent f32 scaled-load rounds and clamps,
then integer addition and the f32 readout round. It deliberately reproduces
the primitive arithmetic; the original capture golden is never replaced.

The whole-model helper opts in using `--residual-add
--residual-max-output-lsb=1 --allow-bounded-output --atol .03125 --rtol .02`.
It records maximum absolute error, relative L2, allclose and bit equality to the
unchanged original golden. Actual Gemmini Spike must separately match the
same candidate's native output bits exactly. A failed whole-model quality gate
retains receipts and blocks subsequent hardware promotion.

Validation: 15 matcher/binder/quality tests pass, including default rejection
of a one-step residual, typed declaration round trips, contiguous reshape
verification, and all-pair native oracle validation. The fresh closed capture
compiles 16 explicit bound-1 routes and passes the linked zero-FSM audit.
Whole-model quality and target execution remain independent required gates.

Nonunit primitive simulation subsequently passed every signed-i8 operand pair
and the output guard using the first captured load scales0.9669194221496582
and0.8079284429550171, readout1:54,310 GSIM kernel cycles. The final ELF
passed the zero-FSM audit. Receipt: `docs/perf_records/residual_nonunit_fullpair_gsim.json`.

Composition also supports an explicitly empty remaining-direct component after
earlier fusions consume every direct convolution. Standalone no-match refusal
is unchanged; composed empty artifacts contain an uncalled ordinary CPU anchor
so strict executable-section auditing is preserved.

Adapters with owned constant/data tables must use `-mcmodel=medany` on this
bare-metal target: default medlow produced an R_RISCV_HI20 overflow for an
identity table above0x80000000, caught at final link. The adapter compiler now
records its exact command and compiler/source/object hashes. A regression
compiles a table-owning adapter and links its text/data at0x80000000; the
resulting ELF passes the ordinary no-FSM audit. Object audit alone cannot
establish that the complete device/host memory model links correctly.

## Whole-candidate outcome: rejected

The closed-recipe candidate combined52 unary routes (46exact plus6 explicitly
bounded), one pooled stem,16 bounded residuals and the remaining classifier.
Its final ELF passes the zero-FSM audit and actual Gemmini Spike matches all
1,000 same-candidate native output words exactly, with zero rank mismatches.
However it fails the unchanged original-golden allclose gate at explicit
atol0.03125/rtol0.02: maximum absolute error1.29150390625 and relative
L2=0.014636863999840584. The candidate is **not queued or promoted**.
Spike reports400,297,663 retired instructions; this is not hardware timing.
Receipt: `docs/perf_records/resnet_combined_bound1_rejected.json`.
