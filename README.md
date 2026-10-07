# gemmini-mlir

## Handwritten implementation

The `handwritten-implementation` branch contains the handwritten xDSL Gemmini
backend, source-bound upstream lowering, primitive device schedules, and the
recorded optimization experiments. The golden device path emits configuration,
DMA, preload, compute and fence instructions. Ordinary RISC-V loops repeat the
schedule; Gemmini FSM loop instructions have no lowering in this path. Every
device object and final linked ELF must pass the executable-section
[zero-FSM audit](mlir_oot/no_fsm_audit.py).

Start with [the compiler export API](mlir_oot/golden_compiler_export.py),
[device lowering](mlir_oot/golden_device_lower.py), and
[device compilation](mlir_oot/golden_device_compile.py). Target implementations
live here; reusable proofs, host compilation, packing, runtime and dispatch live
in [Merlin](https://github.com/ucb-bar/merlin). Frontend capture fixes live in
[model2MLIR](https://github.com/ucb-bar/model2MLIR). Production decisions use input
semantics, shapes, numeric contracts and hardware capabilities. Model selections
and measured recipes remain explicit experiments.

Use Python with Merlin installed and the compatible xDSL dependencies, plus an
LLVM installation containing `mlir-translate` and a RISC-V-capable `clang`:

```sh
python -m mlir_oot.golden_device_compile \
  --kernel gemm --m 17 --n 73 --k 65 --output-dtype i32 \
  --llvm-bin "$LLVM_BIN" --workdir out/handwritten-gemm
```

This produces typed Gemmini/LLVM IR, a RISC-V object, compiler hashes and its
instruction audit. Upstream contraction and capture export commands are declared
in [manifest.yaml](manifest.yaml). The compatible dependency revisions and
publication checks are recorded in [the branch publication notes](docs/handwritten_implementation.md).

The latest verified whole-model observations are ResNet50 **29,698,347**,
TinyLlama **394,765,577**, and SmolVLA **258,621,872,969** stock FireSim cycles.
SmolVLA group 2072 takes **3,918,275,805** cycles; that is a section result.
The 22M/300M/5B whole-model goals remain unmet. Read the
[performance evidence](docs/golden_progress.md),
[optimization journey](docs/golden_optimization_journey.md), and
[fused encoder reproduction recipe](experiments/fused_encoder_radix/README.md).

## Published parent provenance

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

## Experimental no-FSM golden branch

Implementation ownership is mandatory: target-specific dialects, instructions, kernels,
schedules and ABI glue live here; reusable host code generation, packing, requantization,
global optimization, dispatch and runtime infrastructure live in Merlin. See [AGENTS.md](AGENTS.md).
The backend must generalize: production passes select from input semantics, shapes, layouts,
numeric contracts and hardware capabilities; workload-specific selections stay in experiments.

The `golden/nofsm-wholemodels` working branch adds an **uncertified candidate** beside the published capsule backend. See [docs/golden_progress.md](docs/golden_progress.md) for the exact Jack ZIP reference, reproducible commands, measured probes, upstream lowering results, and remaining model work. The [optimization log](docs/golden_optimization_log.md) records each measured schedule change and the compiler or infrastructure abstraction needed to automate it. The historical `manifest.yaml` grading above applies to the published parent, not this branch.
