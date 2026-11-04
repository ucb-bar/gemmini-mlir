# followup-cca-h1d-p1r: control

This branch preserves the final compiler for the control arm of `followup-cca-h1d-p1r`. The source under `compiler/` matches the native final-stop checkpoint `d15f8cd55e3405c7d21bd3b54f13641d9899978f95615f4ad9489b220613df06`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.
