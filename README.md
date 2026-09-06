# gemmini-mlir

Standalone, buildable out-of-tree Merlin codegen backend for **gemmini** (family `tensor_resident`).

> ## ⚠ NOT CERTIFIED — published with `--no-gate`
>
> This package did **not** pass Merlin's publication certification gate, and was exported anyway with `--no-gate`. It is **not a champion** and it is **not the baseline**.
>
> Gate refusal, verbatim: `mlir_oot gate: status='capsule_graded_l3_partial' certification='not_certified' (need rtl_certified or oot_runner.certify pass)`
>
> Recorded status: `capsule_graded_l3_partial`. Whatever this package earned is recorded under `.merlin/certification.yaml` and in the `grading:` block of `manifest.yaml` — read those before citing any number from it. A certification gate is not a formality here: a functional pass, a graded pass and a cycle-accurate RTL certification are three different claims.

This repository is **generated** by Merlin's `merlin-target-publish` bridge. The buildable tree at the repo root *is* the content; the package manifest + provenance ride along under `.merlin/`.

## What

- Package: `gemmini_xdsl_oot_v0`
- Family: `tensor_resident`
- Recorded status: `capsule_graded_l3_partial`
- Merlin git sha (this export): `6ca662e`

## How to run it

No build step: `gemmini-opt` is a script and the tree it imports ships beside it.

```sh
git clone <this-repo> gemmini-mlir
cd gemmini-mlir
./gemmini-opt --help
```

`manifest.yaml` declares the entrypoint and the argv of every command the experiment ABI expects; run those, not a build.

## Provenance

- Certification: `not_certified`
- Graded by run: `NOT_CERTIFIED_graded_by_merlincirct_g4p1_20260905`
- Oracle behind that tier: cycle-accurate RTL (`rtl_gsim`)
- Fingerprint: `n/a`

See `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the full lineage. Each commit on this repo is one promotion; the history is the provenance trail.
