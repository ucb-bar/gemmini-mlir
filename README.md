# followup-scheduling-extension-v1: control

This branch preserves the final compiler for the control arm of `followup-scheduling-extension-v1`. The source under `compiler/` matches the native final-stop checkpoint `5fc8073033888faa912e245033801b95b774b00d192478304ba5b3ae95683b5e`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.
