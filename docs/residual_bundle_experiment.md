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
