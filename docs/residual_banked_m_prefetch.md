# Exact banked residual M prefetch

The wide residual backend has an explicit, default-off input-panel prefetch
schedule. `golden_wide_resadd.build(..., prefetch_m=True,
banked_accumulators=True)` alternates two input and accumulator slots. The
source-bound bundle builder exposes the same options. Selection uses the
existing static contiguous four-tile-wide residual contract and hardware
capacity; model names and provenance IDs do not select the strategy.

## Resource and arithmetic contract

Both input operands of a panel share one scratchpad bank. The two slots use
banks 0 and 2; the three resident diagonal coefficient tables use bank 1.
Each input slot occupies `2 * 4 * DIM` rows, and the tables occupy `3 * DIM`
rows. Optional accumulator slots start at zero and `ACC_BANK_ROWS`. The plan
checks bank counts, capacities, non-overlapping extents and signed-i32 bounds
against the pinned target facts. Resource attributes are retained on the
module and function. All instruction mapping remains in this OOT backend.

The first panel is loaded before the CPU loop. A paired ordinary LLVM loop
loads the next panel into the other slot before computing the current panel.
The final one or two panels are drained separately. No load crosses the
tensor extent. A single-panel input uses the original serial emission.

Each output uses the same increasing chunk order, first-write/accumulate
commands, coefficient values and one final scaled/RNE/clipped store as the
serial kernel. The first source residual still uses 39 chunks. No full-width
accumulator execute read, new arithmetic approximation or aliasing permission
is introduced. The original tensor-value input/full-write output ABI applies.

## Qualification

The complete 65,536 input-pair capsule uses the immutable original source
table, independently checked by the previous native certificate. Both output
guards pass, along with final no-FSM audits and strict RV64GC Spike execution.

| Schedule | GSIM kernel cycles | Change from serial |
| --- | ---: | ---: |
| Serial | 174,255 | control |
| Input prefetch, original accumulator slot | 176,722 | 1.42% slower |
| Input prefetch, alternating accumulator banks | 164,532 | 5.58% fewer |

The input-only negative result is retained. Alternating accumulator banks
allows the next panel to compute without reusing the previous store's
accumulator rows. These measurements establish the combined schedule's
capsule benefit; they do not isolate each hardware stall or predict full
FireSim performance.

Independent numeric cases cover an odd three-panel drain with p127/q128 and
signed output, an even two-panel drain at maximum p/q, and the single-panel
fallback. Executed-command tests cover M16/32/48/64/80/1024 and five coefficient
pairs, resolve input/output pointer bounds, and compare every arithmetic
command with the serial schedule. The source-bound independent fixture checks
identical rewrite, proof, adapter object and native oracle. The focused gate
passes 110 tests plus 18 subtests, with no skips. The default object is byte
identical to the retained pre-change source.

## Isolated whole ResNet arm

The normal builder recompiles the 16 residual device schedules using the
current exact coefficient contracts. Adapter objects, numeric proofs, source
rewrites and native oracle remain identical. Against measured control job
1849, `model.o`, selected/native host LLVM, shim, weights and model I/O header
are byte identical. The original 1,000 outputs pass bit for bit in native and
actual Spike with zero tolerances, DONE and zero descriptor rank mismatches.

The candidate ELF is `aa762ab861c1033b69515383f4b6856addba597b3939c85f2fbc79198f285955`,
build marker `efca9cdab000`. Spike retires 10,040,333 instructions versus
10,013,297 for control; these are functional simulation instructions. Recovery
independently verified the closure and queued stock FireSim job 1854. Full
hardware performance remains pending. The 22M goal is not achieved by this
qualification.

Receipt: [residual_banked_m_prefetch_qualification.json](perf_records/residual_banked_m_prefetch_qualification.json).
