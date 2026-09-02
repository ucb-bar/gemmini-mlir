# gemmini-mlir

Merlin's published codegen packages for the **gemmini** target.

This repository is **generated** by Merlin's `merlin-target-publish` bridge. It uses **branch-per-version** publishing, so *this* branch is only a directory — the packages themselves live on the branches below. Check one out to get a standalone, buildable out-of-tree tree plus its provenance under `.merlin/`.

## Published packages

| branch | package | dtype | status | what it is |
|---|---|---|---|---|
| `baseline` | `hand_v0` | `fp32` | `rtl_certified` | frozen unoptimized control (the before/after reference) |
| `stable/agent_spec_v1_mlir_oot` | `agent_spec_v1_mlir_oot` | `fp32` | `certified (cycle-accurate RTL, 3 rungs, rtl_verilator)` | certified champion |
| `stable/gemmini_xdsl_rtl_v0` | `gemmini_xdsl_rtl_v0` | `fp32` | `certified (cycle-accurate RTL, 5 rungs, rtl_verilator)` | certified champion |

## Using a package

```sh
git clone -b <branch> <this-repo> gemmini-mlir
cd gemmini-mlir
```

## Compiling a model with it

This repository is the **backend**: the target's codegen payload plus its capability contract. The thing that compiles a model is Merlin, which consumes this repo. You need both, and the loop is three commands.

```sh
# 1. Merlin itself (the driver, the frontend, the runtime)
git clone https://github.com/ucb-bar/merlin.git && cd merlin
cp .env.example .env          # then point MERLIN_* at your toolchain / simulators

# 2. Fetch THIS repo as the target's out-of-tree backend
merlin-target-fetch gemmini --champion <branch from the table above>

# 3. Compile a workload onto it
merlin-compile --workload <workload> --target gemmini --verify
```

`merlin-target-fetch` clones the chosen branch into `out/build/generated/gemmini/`, and the target registry then resolves the capability contract and this codegen payload together — so which champion you compile against is the branch you fetched, recorded rather than implied.

`merlin-compile` takes `--run {none,host,spike,verilator,zephyr,k1}` and `--verify`. Start with `--run host` to check the lowering is numerically right, then move up the oracle ladder; `--verify` gates the answer against the workload's golden rather than reporting that something merely ran.

**What you need beyond this repo**: an LLVM/MLIR install matching the `llvm:` block of the package manifest (the out-of-tree C++ API moves between versions), a RISC-V toolchain, and whichever simulator your chosen `--run` needs. Merlin's `docs/guides/getting_started.md` is the base install; `docs/guides/adding_a_target.md` explains the contract this repo carries.

## Provenance

Each commit on a package branch is one promotion, and its message embeds the champion package id, the internal run id, the Merlin git sha and the certification summary. History is the provenance trail; the branch tip is the current champion.

Generated from Merlin `02454ab`.
