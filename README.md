# followup-scheduling-extension-v1: treatment

This branch preserves the final compiler for the treatment arm of `followup-scheduling-extension-v1`. The source under `compiler/` matches the native final-stop checkpoint `09efb4b77fd23db9559f6543614a5b8746abd66751f2afe91250d888c9af81a2`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.
