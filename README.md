# E2: C++ interface support


This branch preserves the exact compiler submission used by experiment `EL2-r5`. It is an archival experiment version, with its original source under `compiler/` and its setup and results under `experiment/`.

| Item | Record |
|---|---|
| Experiment branch | `paper/scaffolding/e2-cpp-support-4h` |
| Original run | `/work/merlin/tasks/el4-grader-throughput-20261024/merlin/out/runs/gemmini/phase1/xs-el123-rawrtl-20261029-r5-el2` |
| Source digest | `7aa2b5bf48bf20e0ddb8811d29beb8613db67f3bab33fe37391f51b2d674bc0c` |
| Source files | 30 |
| Paper revision | `24181ce016544dac2742825dbc401230ebb85100` |

Start with [the manifest](experiment/manifest.json), [experimental setup](experiment/setup.json), and [recorded metrics](experiment/metrics.json). The [measurement records](experiment/measurement-records.csv) retain image and cache-profile identities. [Selected master rows](experiment/selected-master-records.csv) preserve the original panel selection where it exists. Rows from different images are not silently pooled.

Verify the compiler bytes without executing it:

```sh
python3 experiment/verify_snapshot.py
```

The original `compiler/manifest.yaml` declares the compiler entrypoints. Shared imported code shipped within the frozen compiler is retained because it is part of the measured source identity. Additional captured generic-library dependencies, when available, are under `dependencies/`. The manifest pins the remaining evaluator and toolchain records. Reconstruction of a full execution environment remains part of the artifact package.

The archived measurements are historical observations. No compiler, agent, simulator or FPGA experiment was rerun to prepare this branch. Construction-suite completion and the frozen 89-case kernel panel have different scopes. Read the setup and per-record statuses before interpreting an aggregate.

## Run identity

Original result ID: `EL2-r5`. Permanent measured-snapshot tag: `mlsys-paper-v1/scaffolding/e2-cpp-support-4h`. This branch adds navigation metadata to that snapshot. Compiler and dependency trees and all saved measurements are unchanged. The tag retains the original complete commit.

Names containing `4plus4h` mean four hours of seed construction followed by four additional hours of refinement. A `30min` suffix is the local study budget after its existing seed, not the cost of constructing the seed.

See [the experiment index](https://github.com/ucb-bar/gemmini-mlir/tree/paper/index) for all original run IDs, paths, commits and source digests.
