# followup-library-oracle-h2conv-30min-v3: control

This branch preserves the final compiler for the control arm of `followup-library-oracle-h2conv-30min-v3`. The source under `compiler/` matches the native final-stop checkpoint `d178b14ca6fc62f143171816f7b3e71e6870d52ebfd732be9740e8e0ae45ed3d`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.
