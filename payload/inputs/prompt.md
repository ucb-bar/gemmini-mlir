# Generation prompt (Merlin-assisted entry)

This package, `agent_spec_v0_mlir_oot`, is the **Merlin-assisted** entry of the conference
comparison. It is graded through the **same** external CLI/file boundary as the raw baseline
(`merlin.targetgen.oot_runner` over `bench_contract` v0.1), but it was *authored* with the Merlin
target-generation tooling (structured spec/plan synthesis, an xDSL prototype plane, documented
RoCC codegen patterns, structured failure provenance, and the AET recording layer).

## Goal

Produce a self-contained, **non-exempt** out-of-tree MLIR-level Gemmini target backend that:
- defines a real `gemmini` target dialect (res_pack/matmul/commit/evict; resident_tensor/accumulator),
- lowers `merlin_iface` → `gemmini` → command-buffer JSON → LLVM/RoCC MLIR,
- certifies the supported rungs (G0–G3) through the shared command-buffer / reference / oracle ladder,
- passes the integrity scan (no runtime import of Merlin internals).

## Claim discipline

Narrow claim only. No general RTL→target generation. No full Gemmini coverage. No performance.
RTL participates only as the Verilator certification oracle (see `rtl_facts.yaml`).

## Fairness

Every RoCC encoding fact reused from Merlin's certified native path must be tagged in
`source_manifest.yaml` as PUBLIC (also available to the baseline) or a Merlin TOOLING-ADVANTAGE,
and the advantage facts are reported as such in `REPORT.md`.

## Provenance / profiling references (mirrored by runtime/oracle_runner.py)

- https://github.com/ucb-bar/mlirAgent/blob/main/src/mlirAgent/tools/trace_provenance.py
- https://github.com/ucb-bar/mlirAgent/blob/main/src/mlirAgent/tools/provenance.py
- https://github.com/ucb-bar/mlirAgent/blob/main/src/mlirAgent/tools/_provenance_common.py
- https://github.com/ucb-bar/mlirAgent/blob/main/tests/test_mlir_bindings.py  (MLIR python bindings)
