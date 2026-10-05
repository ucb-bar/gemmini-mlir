# Stock full-width accumulator execute reads are unsupported

The proposed radix residual schedule requires exact `C = C + C` inside the
accelerator: preload zero weights, stream zero A, read old C as the real D
operand, and accumulate the result into C. Large coefficients could then use
base-256 digits and eight exact doublings. This schedule is refused for stock
`FireSimGemminiRocketConfig`.

## Hardware contract

The stock configuration uses i8 input/weight lanes and i32 accumulator memory.
`ExecuteController` forces accumulator execute reads to `full=false`, takes the
scaled narrow `data` response rather than `full_data`, and casts A/B/D to the
input operand type. `Scratchpad` independently forces execute reads to
`full=false`. Its DMA read path uses the full-row address bit. The mesh D lane
is the weight type. These facts apply to compute A/BD and preload BD, regardless
of which operand supplies D in the chosen dataflow.

Setting `ACC_FULL_ROW_BIT` on an execute source cannot request an exact i32 D
read. For example, narrowing old C=2609 to an i8 operand cannot supply 2609 for
the required doubling. Accumulator MVIN has no arbitrary scale multiplier in
this stock configuration either.

The OOT verifier now rejects an explicit full-row accumulator source for
`ComputeOp.a`, `ComputeOp.bd`, and `PreloadOp.bd`. The garbage sentinel is
exempt. Scratchpad and narrow accumulator source encodings remain legal, and
full-width accumulator DMA readout remains legal. Dynamic A retains its
existing bounded scratchpad contract. No reusable host compiler change is
needed for this target capability refusal.

## Retained experiments

Two tiny xDSL capability probes initialize arbitrary signed i32 accumulator
values, issue zero A/weights with accumulator D, and request complete output
and guard verification. One uses the same C address; the other seeds a distinct
destination with the original value to remove the address overlap. Both final
ELFs pass the no-FSM audit. Both stall on the pinned GSIM engine's reservation
station assertion and do not complete numeric verification. Strict Spike
rejects the accumulator D address with `out_of_range`.

These failures are retained as failed experiments. They do not establish the
numeric narrowing result or identify a unique cause of the stall. The refusal
rests on the explicit stock RTL decode/data-width contract. The receipt pins
the current clean Gemmini source/config chain and generated ELF/IR bytes. The
GSIM receipt retains its engine/derivation hashes; its original raw FIRRTL path
is no longer present, so the current Scala checkout is not claimed byte
identical to the engine's original inputs.

The focused verifier and existing cached-A, prefetch, wide-residual and source
proof tests pass: 38 tests plus 12 positive subtests. No unsupported kernel is
enabled or submitted for hardware, and no accuracy gate is changed.

Receipt: [full_width_execute_read_refusal.json](perf_records/full_width_execute_read_refusal.json).

## Remaining schedule constraint

Within the existing direct diagonal signed-i8 coefficient family, a positive
coefficient p needs at least `ceil(p/127)` products. Two independent source
coefficients p2609/q2180 therefore need at least 21+18=39 chunks. This bound
applies to that family. A better legal implementation needs a different exact
data representation or supported feedback path, its own full source-domain
proof, and measured hardware support.
