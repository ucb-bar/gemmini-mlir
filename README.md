# followup-scheduling-h3a-30min-v1: treatment

This branch preserves the final compiler for the treatment arm of `followup-scheduling-h3a-30min-v1`. The source under `compiler/` matches the native final-stop checkpoint `9dcb1bdc6adc5d30e514eed879d63eb75f21485d22635cc9c56bf61e114c543d`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.
