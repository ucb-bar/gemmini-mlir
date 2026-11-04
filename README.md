# followup-cca-h1d-p2r: control

This branch preserves the final compiler for the control arm of `followup-cca-h1d-p2r`. The source under `compiler/` matches the native final-stop checkpoint `24e849ab6e03b6236c8c2d411d30fa18db4eac50cb225d687409cb80f5193fbc`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.
