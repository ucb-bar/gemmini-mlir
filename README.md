# full-ref-cca (twelve_hour_campaign_final; 12 h scratch campaign, 23 raw expert files, no map, CCA declared off)

This branch preserves the exact compiler submission used by experiment `FULL-REF`. It is an archival experiment version, with its original source under `compiler/` and its setup and results under `experiment/`.

| Item | Record |
|---|---|
| Experiment branch | `paper/construction/full-ref/final-12h` |
| Original run | `/work/merlin/tasks/el4-grader-throughput-20261024/merlin/out/runs/gemmini/phase1/next-batch-20261029-full-ref-cca` |
| Source digest | `a854ff36c7821d4678e7fdfd24d41b257023cd6a3345c55547ea0390417d131e` |
| Source files | 1247 |
| Paper revision | `24181ce016544dac2742825dbc401230ebb85100` |

Start with [the manifest](experiment/manifest.json), [experimental setup](experiment/setup.json), and [recorded metrics](experiment/metrics.json). The [measurement records](experiment/measurement-records.csv) retain image and cache-profile identities. [Selected master rows](experiment/selected-master-records.csv) preserve the original panel selection where it exists. Rows from different images are not silently pooled.

Verify the compiler bytes without executing it:

```sh
python3 experiment/verify_snapshot.py
```

The original `compiler/manifest.yaml` declares the compiler entrypoints. Shared imported code shipped within the frozen compiler is retained because it is part of the measured source identity. Additional captured generic-library dependencies, when available, are under `dependencies/`. The manifest pins the remaining evaluator and toolchain records. Reconstruction of a full execution environment remains part of the artifact package.

The archived measurements are historical observations. No compiler, agent, simulator or FPGA experiment was rerun to prepare this branch. Construction-suite completion and the frozen 89-case kernel panel have different scopes. Read the setup and per-record statuses before interpreting an aggregate.
