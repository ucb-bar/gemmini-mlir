# cca-exo (seeded FULL-NOREF + own source originals + CCA bank, 4 h)

This branch preserves the exact compiler submission used by experiment `CCA-EXO`. It is an archival experiment version, with its original source under `compiler/` and its setup and results under `experiment/`.

| Item | Record |
|---|---|
| Experiment branch | `paper/cca/exo/final-8h-total` |
| Original run | `/work/merlin/tasks/el4-grader-throughput-20261024/merlin/out/runs/gemmini/phase1/cca-coverage-20261029-cca-exo` |
| Source digest | `f2c963582800acbbfe533eacb2eb3a9d2065f5bba7a05405ffe56b866659fe8c` |
| Source files | 1236 |
| Paper revision | `24181ce016544dac2742825dbc401230ebb85100` |

Start with [the manifest](experiment/manifest.json), [experimental setup](experiment/setup.json), and [recorded metrics](experiment/metrics.json). The [measurement records](experiment/measurement-records.csv) retain image and cache-profile identities. [Selected master rows](experiment/selected-master-records.csv) preserve the original panel selection where it exists. Rows from different images are not silently pooled.

Verify the compiler bytes without executing it:

```sh
python3 experiment/verify_snapshot.py
```

The original `compiler/manifest.yaml` declares the compiler entrypoints. Shared imported code shipped within the frozen compiler is retained because it is part of the measured source identity. Additional captured generic-library dependencies, when available, are under `dependencies/`. The manifest pins the remaining evaluator and toolchain records. Reconstruction of a full execution environment remains part of the artifact package.

The archived measurements are historical observations. No compiler, agent, simulator or FPGA experiment was rerun to prepare this branch. Construction-suite completion and the frozen 89-case kernel panel have different scopes. Read the setup and per-record statuses before interpreting an aggregate.
