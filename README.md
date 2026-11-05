# Convolution transformation-library study: extended library


This branch preserves the final compiler for the treatment arm of `followup-library-oracle-h2conv-30min-v3`. The source under `compiler/` matches the native final-stop checkpoint `9eee76b39dbf47cad735a5edc66a2f0aeafa2f5a2690a668043a0f23b18992e2`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.

## Run identity

Original result ID: `followup-library-oracle-h2conv-30min-v3-treatment`. Permanent measured-snapshot tag: `mlsys-paper-v1/transformations/convolution-30min/extended-library`. This branch adds navigation metadata to that snapshot. Compiler and dependency trees and all saved measurements are unchanged. The tag retains the original complete commit.

Names containing `4plus4h` mean four hours of seed construction followed by four additional hours of refinement. A `30min` suffix is the local study budget after its existing seed, not the cost of constructing the seed.

See [the experiment index](https://github.com/ucb-bar/gemmini-mlir/tree/paper/index) for all original run IDs, paths, commits and source digests.
