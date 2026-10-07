# Explicit existing exact-math feature selection

The original whole source recipe omitted Merlin's existing
`lower_exact_math_inline` feature. This normal build adds it to current main
3430c2ca9 while retaining the original numerical provider, source plan and
full 1,600-output .03125/.02 gate. Native output words are bitwise exact and
the final target ELF has zero FSM instructions. The target model object has
no undefined `__truncsfbf2` reference. These are source/native/build results;
full target execution and stock cycles remain pending.

These archived scripts retain explicit original source/provider and artifact
paths under the owned scratch trees. The original executable collectors remain
in `/scratch/agustin/tmp/merlin-latest-20261004/out/artifacts/probes/smol-exact-math-normal-20261007`.
Run those originals with their pinned inputs. Archive location does not change
the script's selected output root or establish portable execution. Never start
a duplicate of its active full strict run. Native products are exact integer
stand-ins; the original full target run establishes physical execution.

The feature is generic host lowering in Merlin; this experiment adds no new
workload-specific production matcher or default numerical policy. Comparison
against the earlier f90 executable does not isolate hardware cost of one pass.
