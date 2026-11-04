# followup-cca-h1d-p2r: treatment

This branch preserves the final compiler for the treatment arm of `followup-cca-h1d-p2r`. The source under `compiler/` matches the native final-stop checkpoint `8926cc6f2f5ea4daa5ca06fb00e10f71c2c1cc9d0bd2c5f1bbf2a06001364dcd`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.
