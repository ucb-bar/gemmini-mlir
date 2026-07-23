# gemmini-mlir

Standalone, buildable out-of-tree Merlin codegen backend for **gemmini** (family `tensor_resident`).

This repository is **generated** by Merlin's `merlin-target-publish` bridge: it is the certified champion codegen package for the target, exported as its own repo. The buildable tree at the repo root *is* the content; the package manifest + provenance ride along under `.merlin/`.

## What

- Champion package: `hand_v0`
- Family: `tensor_resident`
- Recorded status: `rtl_certified`
- Merlin git sha (this export): `3f9cf97`

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

- Certification: `recorded:rtl_certified`
- Certified by run: `n/a`
- Fingerprint: `n/a`

See `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the full lineage. Each commit on this repo is one promotion; the history is the provenance trail.
