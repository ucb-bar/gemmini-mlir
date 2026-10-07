# Explicit existing exact-math feature selection

The original whole source recipe omitted Merlin's existing
`lower_exact_math_inline` feature. This normal build adds it to current main
3430c2ca9 while retaining the original numerical provider, source plan and
full 1,600-output .03125/.02 gate. Native output words are bitwise exact and
the final target ELF has zero FSM instructions. The target model object has
no undefined `__truncsfbf2` reference. The full target run now passes all 1,600
original output words bitwise exactly, zero rank mismatches and final zero-FSM.
The functional counter is 117,635,197,150; stock cycles remain unmeasured.
[Completed target qualification](../../docs/perf_records/root_smol_exact_math_normal_whole_target_20261007_qualification.json).

These archived scripts retain explicit original source/provider and artifact
paths under the owned scratch trees. The original executable collectors remain
in `/scratch/agustin/tmp/merlin-latest-20261004/out/artifacts/probes/smol-exact-math-normal-20261007`.
Run those originals with their pinned inputs. Archive location does not change
the script's selected output root or establish portable execution. Preserve
completed native and strict receipts; `seal_strict.py` creates a distinct
whole-target successor and refuses overwriting it. Native products are exact integer
stand-ins; the original full target run establishes physical execution.

The feature is generic host lowering in Merlin; this experiment adds no new
workload-specific production matcher or default numerical policy. Comparison
against the earlier f90 executable does not isolate hardware cost of one pass.
