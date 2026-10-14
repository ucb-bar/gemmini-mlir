# Generated Gemmini Phase 1 backend

This directory preserves the target-specific compiler bundle authored in Merlin
EL4 run `el4-proper-20261014-r48`. It is separate from the handwritten
`mlir_oot/` implementation in this repository. The standalone
`manifest.yaml` names `mlir_oot/gemmini-opt` as its tool entrypoint.

The generated backend contains its own frontend, target lowering, device
codegen, instruction definitions, hardware facts, and pinned Python
dependencies. Reusable host lowering, packing, runtime, dispatch, and
experiment grading remain in Merlin. The copied target files are byte-identical
to the R48 submission; the full submission digest is
`3a8ee6fd85b99934246d178786a03eef43697f71ced852e4e69518791d7fc20d`.
The qualification record is in `QUALIFICATION.md`.

For a public interface input, run:

```sh
./phase1_generated/mlir_oot/gemmini-opt --verify-diagnostics path/to/input.mlir
./phase1_generated/mlir_oot/gemmini-opt --convert-iface-to-gemmini --emit-target-artifact path/to/input.mlir
```

The vendored dependency notices are in `third_party_licenses/`.
