# Next-K prefetch in remaining scratchpad rows

`GoldenGemm` accepts an explicit two-slot B placement as a separate target
schedule fact. `Shape.validate` proves complete reserved A and B extents,
DIM alignment, non-overlap, scratchpad bounds and containment of each B panel
within one bank. The default retains banks 2 and 3 and identical emitted IR.
Tensor dimensions, source reduction order, bias, scale, saturation and ReLU
retain their existing meaning.

The general `select_remaining_b_slots` selector derives slots after the full
cached A extent. It skips a bank boundary when a panel would straddle it,
refuses every resource or competing schedule failure, and retains existing
B-prefetch choices. The explicit `remaining_rows` policy is available through
the source-bound bundle builder and CLI. It is off by default. No model name,
source region or measured cycle value selects a placement.

The initial two panels are loaded, and each following K panel loads into its
free alternating slot before the current panel computes. Increasing source K
order and the exact final K/N/M tails are preserved. Two slots can share a bank;
the address proof admits them, and actual bank arbitration decides their speed.

## Measured source-scale capsule

M196/N512/K1024 with BM13/BN4 retains 13,312 scratchpad rows for A. Two 64-row
B slots at rows 13,312 and 13,376 fit in bank 3. The existing schedule takes
540,046 fenced GSIM kernel cycles; prefetch takes 511,096, saving 28,950
(5.36%). Fixtures, source scale and ReLU are identical. Both pass every one of
100,352 outputs, all 2,048 guards, strict RV64GC Spike and final zero FSM.

Complete static traversal of both emitted LLVM CFGs proves the same 832 A
loads, 512 B loads, 26,624 compute/preload pairs and 104 stores. Every ordered
source A/B contraction and accumulator update agrees, despite the changed
transfer order and local addresses. These are exact static command traces,
separate from the measured hardware cycles.

An independent i32 M123/N73/K1041 fixture covers multiple N blocks and all
three independent tails. All 8,979 outputs and guards pass in GSIM and strict
RV64GC Spike with zero FSM. Fifty-five focused tests cover complete resource
proofs, refusable placements, idempotence, source field preservation, default
IR identity, upstream lowering and existing dense/requant contracts.
[Local qualification and complete trace](perf_records/remaining_b_slots_prefetch_gsim.json).

## Compiler infrastructure lesson

Phase 1 and 2 schedule tooling needs complete operand lifetimes, resource
intervals and source/SSA equivalence alongside measurements of the actual
command order. Lower retired instruction counts or fewer mesh weight loads
alone did not select the faster dense stripe schedule. Bank layout and primitive
commands belong in the OOT backend; generic dependency, proof and recording
infrastructure belongs in Merlin.

## Full source and target gate

The immutable 52-route source bundle retains every numeric proof, exact readout
contract and rewritten IR byte. Seven pointwise objects change, selected only
by the general slot policy. All 70 fresh writer contracts remain covered.
Native and actual strict RV64GC Gemmini Spike match all 1,000 original f32
output bits exactly, and the final linked ELF passes zero FSM.

All host/runtime/weights/shim/classifier bytes match the 1853 control through
the pinned normal builder. ELF `127d5adb...ea2b74`, marker `bf4c0c49c6e1` is
ready for stock FireSim. Its 10,195,247 Spike instructions are functional
evidence, not hardware cycles. [Full gate](perf_records/resnet_remaining_b_slots_policy_spike.json).
The [optimization journey](perf_records/remaining_b_slots_optimization_journey.json)
records the resource refusals, local measured win, whole correctness gates,
ownership and metric scopes. Stock whole-model timing remains the admission
gate; capsule gains are not assumed additive.
