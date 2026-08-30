# gemmini-mlir

Standalone, buildable out-of-tree Merlin codegen backend for **gemmini** (family `unknown`).

This repository is **generated** by Merlin's `merlin-target-publish` bridge: it is the certified champion codegen package for the target, exported as its own repo. The buildable tree at the repo root *is* the content; the package manifest + provenance ride along under `.merlin/`.

## What

- Champion package: `agent_spec_v1_mlir_oot`
- Family: `unknown`
- Recorded status: `unknown`
- Merlin git sha (this export): `9b8a684`

## How to build

```sh
git clone <this-repo> gemmini-mlir
cd gemmini-mlir
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/gemmini-opt --version
```

The codegen payload (schedule/knobs for rvv; dialect/lowering/contracts for gemmini) lives under `payload/`.

## Provenance

- Certification: `pass`
- Certified by run: `vcert_g3_acc_scale_relu_20260828T210542Z`
- Certified against: cycle-accurate RTL (`rtl_verilator`)
- Fingerprint: `3c12a6e5c824e8111139712e06156b56d12442c003f93d87e2569bf0ed5fad41`

See `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the full lineage. Each commit on this repo is one promotion; the history is the provenance trail.
