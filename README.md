# Gemmini compiler architecture refactor

The `refactor/architecture` branch adds an installed provider that consumes the
selected Merlin shared compiler library. The paper branch and permanent tag
below remain the exact archival baseline. Existing compiler and experiment
files have not been rewritten by this packaging step.

Install the matching Merlin refactor checkout and this package in a Python 3.12
or newer Linux environment:

```sh
python -m pip install -e /path/to/merlin[verify,targetgen]
python -m pip install -e .
python examples/matmul.py --output out/build/matmul
# The same Python service is exposed by the installed command.
gemmini-opt examples/matmul.mlir --output out/build/another-matmul
```

The example compiles explicit signed int8 multiplication with int32 accumulation
into LLVM dialect and a command plan. It does not link an ELF or execute hardware.
Its `gemmini.json` binds source and artifact hashes and explicitly records
that execution and qualification have not occurred. Unsupported source fails;
an existing output directory is never overwritten. Mixed host lowering still
requires the selected native MLIR/LLVM environment.

The package's `gemmini.compile_source` is the active backend service. It imports
shared implementations from the installed Merlin SDK and includes no vendored
`merlin` package. The historical `compiler/compiler.py` remains an archival
entrypoint selecting its captured library. The active service does not reuse
that entrypoint's cache, whose identity assumed an embedded SDK. Cross-repository
compatibility is being checked against the coordinated refactor checkouts;
the package version alone is not a backend or hardware qualification.

The same installed provider is registered with Merlin's `merlin.backends`
entry-point group. The shared `merlin.api.compile` service can select it with
`backend="gemmini"`; its output is a region artifact, so platform and hart
requests are rejected until a complete image builder is selected.

### Native region artifacts with explicit tools

The shared service can also produce a RISC-V object and a candidate C transport
wrapper for a static device-only region. Select this artifact route explicitly:

```python
from pathlib import Path
from merlin.api import CompileOptions, compile

result = compile(
    Path("examples/matmul.mlir"), backend="gemmini", output=Path("out/build/callable"),
    options=CompileOptions(parameters={
        "gemmini.emit": "native-object",
        "gemmini.mlir_translate": "/selected/llvm/bin/mlir-translate",
        "gemmini.clang": "/selected/llvm/bin/clang",
        "gemmini.march": "rv64gc",
        "gemmini.mabi": "lp64d",
    }),
)
```

The ISA and psABI in this example must agree with the selected system. Paths
are explicit executable selections; missing tools refuse before compilation.
The tools must accept this backend's LLVM dialect, with the historical LLVM
pin documented in `compiler/backend/data/abi.yaml`. Tool hashes, versions,
commands and emitted source hashes are recorded in `callable.json`.

`compiler/callable.py` reconciles source tensor order with the actual emitted
logical pointer ABI, preserves the target instruction/fence sequence, and gives
the device function a distinct symbol. `call.c` invokes that exact device
function; `device.o` must be emitted by the selected tools. These are concrete
link inputs, pending admission to Merlin's shared serial control emitter.
Command JSON remains an explanatory plan; it is not
misreported as an executed binary payload. Tensor pointers require the selected
logical ABI's alignment. Caller-owned output storage must be disjoint from
inputs and other outputs, as required by the shared C call contract.

`gemmini.emit=c-call` currently refuses before compilation: the selected-platform
accelerator completion and CPU visibility contract is not yet established.
`native-object` produces inputs for that future integration and reports the
unavailable admission explicitly; it does not satisfy a callable-output request.

This initial route refuses mixed host tasks, physical storage encodings, extra
host workspace, module globals and external function calls. Ordinary region-only
emission retains its broader existing behavior. The callable product is still
not an ELF image: final linking, system/runtime admission, independent numerical
verification and observed accelerator execution are separate obligations.
Native object generation and hardware execution have not been qualified by the
local wrapper transport tests. More fundamentally, no independently admitted
selected-platform contract currently establishes that the emitted fence drains
accelerator operations and makes their stores CPU-visible. The provider therefore
retains native products with `c_call_admission.status=unavailable` and publishes
no `merlin.c_call.v1` metadata. `callable.json` records the required C ABI separately
from this missing physical completion admission. There is no boolean override.
A compiler memory clobber and fence spelling establish emitted structure only.

## Archived refinement with Exo kernels


This branch preserves the exact compiler submission used by experiment `CCA-EXO-R3`. It is an archival experiment version, with its original source under `compiler/` and its setup and results under `experiment/`.

| Item | Record |
|---|---|
| Experiment branch | `paper/refinement/exo-kernels-4plus4h` |
| Original run | `/work/merlin/tasks/el4-grader-throughput-20261024/merlin/out/runs/gemmini/phase1/cca-coverage-20261029-cca-exo-r3` |
| Source digest | `4994f7b478047decf473e8263f6246b493f5189f0fbfef70b921ba8f99b012f7` |
| Source files | 1251 |
| Paper revision | `24181ce016544dac2742825dbc401230ebb85100` |

Start with [the manifest](experiment/manifest.json), [experimental setup](experiment/setup.json), and [recorded metrics](experiment/metrics.json). The [measurement records](experiment/measurement-records.csv) retain image and cache-profile identities. [Selected master rows](experiment/selected-master-records.csv) preserve the original panel selection where it exists. Rows from different images are not silently pooled.

Verify the compiler bytes without executing it:

```sh
python3 experiment/verify_snapshot.py
```

The original `compiler/manifest.yaml` declares the compiler entrypoints. Shared imported code shipped within the frozen compiler is retained because it is part of the measured source identity. Additional captured generic-library dependencies, when available, are under `dependencies/`. The manifest pins the remaining evaluator and toolchain records. Reconstruction of a full execution environment remains part of the artifact package.

The archived measurements are historical observations. No compiler, agent, simulator or FPGA experiment was rerun to prepare this branch. Construction-suite completion and the frozen 89-case kernel panel have different scopes. Read the setup and per-record statuses before interpreting an aggregate.

## Run identity

Original result ID: `CCA-EXO-R3`. Permanent measured-snapshot tag: `mlsys-paper-v1/refinement/exo-kernels-4plus4h`. This branch adds navigation metadata to that snapshot. Compiler and dependency trees and all saved measurements are unchanged. The tag retains the original complete commit.

Names containing `4plus4h` mean four hours of seed construction followed by four additional hours of refinement. A `30min` suffix is the local study budget after its existing seed, not the cost of constructing the seed.

See [the experiment index](https://github.com/ucb-bar/gemmini-mlir/tree/paper/index) for all original run IDs, paths, commits and source digests.
