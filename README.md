# scheduling-completion-authors: control

This branch preserves the final compiler for the control arm of `scheduling-completion-authors`. The source under `compiler/` matches the native final-stop checkpoint `501f436fd54232a61445f3124eac80b6b05d9f9f4690582b9230d7f62a5d1161`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.
