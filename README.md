# gemmini-mlir

Standalone, buildable out-of-tree Merlin codegen backend for **gemmini** (family `unknown`).

This repository is **generated** by Merlin's `merlin-target-publish` bridge: it is the certified champion codegen package for the target, exported as its own repo. The buildable tree at the repo root *is* the content; the package manifest + provenance ride along under `.merlin/`.

## What

- Champion package: `gemmini_xdsl_rtl_v0`
- Family: `unknown`
- Recorded status: `unknown`
- Merlin git sha (this export): `68ae8bb`

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
- Certified by run: `cert_A2_verilator`
- Certified against: cycle-accurate RTL (`rtl_verilator`)
- Fingerprint: `78fa4933da15dde238797bc696f5a3901e047547981766aab808eabcc1b7d47a`

See `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the full lineage. Each commit on this repo is one promotion; the history is the provenance trail.
