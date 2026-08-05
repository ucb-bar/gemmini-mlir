# gemmini-mlir

The out-of-tree **target home** for **gemmini** (Berkeley systolic-array accelerator,
`chipyard/generators/gemmini`), modeled by Merlin as a `tensor_resident` target.

This repo is the single per-target home under the `<target>-mlir` convention (`gemmini-mlir`,
`rvv-mlir`, `atlas-mlir`, `radiance-mlir`, …). Merlin hooks up to it natively: it fetches this repo
into its generated-target home (`out/build/generated/gemmini/`) and discovers the target with zero
extra configuration — the reference backend below self-registers via the contract's `plugin.backend`.

## Layout

- `contracts/target_contract.yaml` — the Merlin capability manifest (intent + the ABI encoding
  surface RTL facts cannot ground) and the `plugin.backend: backend` declaration.
- `contracts/residual.yaml` — the human-authored residual the capability deriver reproduces.
- `backend/` — the **reference backend**, a Python package evicted from Merlin core's
  `runtime/backends/`. Importing it self-registers the `gemmini` backend in Merlin's registry:
  - `backend/gemmini.py` — compile + run on the oracle, parse output, gate.
  - `backend/gemmini_codegen.py` — command buffer → bare-metal C via low-level `libgemmini` intrinsics.
  - `backend/gemmini_codegen_mlir.py` — command buffer → LLVM-dialect RoCC MLIR (`.insn` on stock LLVM).
  - `backend/__init__.py` — registers under the package name and exposes the codegen submodules.
- `evidence_concepts.yaml` — per-target concept vocabulary for the evidence pass.
- `AGENT.md` — the target's agent-facing brief.

The reference backend imports Merlin internals (`merlin.runtime.*`); it is loaded by Merlin's
out-of-tree backend loader (`merlin.runtime.backends.base`) by file path, so it is not a standalone
distribution — it travels with an installed Merlin.

## Branch model — one branch per champion experiment

- **`main`** is the target *base*: the contract + the reference backend + metadata. This is what
  Merlin fetches by default, and what every champion experiment forks from.
- Each **champion experiment** (a certified/hand codegen package) is its **own branch**, forked from
  `main` so it inherits the reference backend, and adding its buildable codegen payload
  (`payload/`, `.merlin/` provenance, CMake, `tools/<target>-opt/`). Example: `baseline` carries the
  `hand_v0` codegen champion (tagged `v0-hand_v0`).

The history of each branch is that champion's promotion trail; the tags mark promoted versions.

## How Merlin fetches this repo

```sh
merlin-target-fetch gemmini                 # fetch main (base + reference backend)
merlin-target-fetch gemmini --champion baseline   # fetch a champion-experiment branch
```

This clones the repo into `out/build/generated/gemmini/`, where Merlin's target registry resolves it
with zero env and loads the `plugin.backend`.
