# Handwritten Gemmini implementation

The public working branch is `handwritten-implementation` in `ucb-bar/gemmini-mlir`.
It contains the xDSL compiler implementation, source-bound experiment drivers,
optimization journey, numerical refusals, measured results and token ledgers.

## Dependencies and compilation

The integrated Merlin source revision is
`c0f40f8f8d100b841c26fe8bbd6c09e79b6de17c`.
The model2MLIR revision is
`7915e23475c6db446a3c404847b11e8bc72c8a27`.
The qualified Python environment uses xDSL 0.68.0, NumPy 2.4.6 and PyYAML 6.0.3.
Device compilation uses LLVM's `mlir-translate` and RISC-V-capable `clang`.
Upstream model lowering also requires Merlin's configured compiler Python with
torch-mlir; ordinary Python and the device compiler are separate dependencies.

`golden_compiler_export.py` binds actual upstream contractions and captures.
`golden_device_lower.py` lowers verified primitive target operations to
side-effecting RoCC instructions and ordinary LLVM control flow.
`golden_device_compile.py` records generated IR, objects, compiler binaries and
command hashes. Unsupported operations refuse before device emission. The
complete linked ELF must pass `no_fsm_audit.py`, including library code outside
the measured region.

For explicit commands and resources, read [manifest.yaml](../manifest.yaml).
The original generated package's certification metadata remains historical;
it does not certify the combined handwritten branch.

The earlier`95e8142d9` wheel preserves all995 compiler Python modules,28 runtime headers
and five C templates byte for byte across source, wheel and installation outside
the checkout. The installed critical gate passes 1,085 checks with no skips or
failures, covering interval publication, mask/softmax flags, probability/fused
preparation and exact residual certificates.

Later generic topics are independently qualified before direct-main delivery:
the audit baseline`3a1e24c77`, finite guards`b1108c60d` (95source/95installed),
predictor-key corrections`b1b6d1379` (45/45), and explicit rounded-polynomial
theorem consumption`0a9c14552` (104/104), and exact bounded RNE observer cells
`c0f40f8f8` (107/107). The last topic verifies both implementation modules and
all30 installed header payloads with typed source/effect refusals. These focused checks do not
relabel an earlier whole-package suite as rerun at the latest head.

The latest qualified whole-model cycles are ResNet2090 **29,402,206**,
Tiny2085 **378,946,263**, and Smol1906 **258,621,872,969**. Original output gates
and final zero-FSM instruction audits remain mandatory. All requested whole
targets are still unmet; explicit experiment recipes do not install a default
numerical policy or imply that rejected private prototypes reached main.

## Integrated target and shared mechanisms

Target code includes resident/tiled GEMM and convolution schedules, physical
resource checks, exact finite residual rectifiers, bounded private panel batches,
source-bound operand domains and restricted internal SPAD fence coalescing.
The latest ResNet residual source modules were integrated individually; newer
compiler code from the working branch was retained.

The existing six resident K/row-loop winners' implementation and normal API
were restored as a separate source delivery topic, including bounded dynamic B
rows, with seven modules byte-identical to qualified173ab44. Explicit source-stride
command retention composes with that option and leaves other families intact.
The combined source/resource/key gate passes109 checks; its narrow option
integration passes47. Fresh source generation and actual whole performance
retain separate qualification records; frozen winning objects alone do not
establish reproducibility from the published compiler.

The source-stride whole2086 admission independently closes1,959 pins, all52
fresh default kernels/adapters, the one changed candidate kernel, both whole
numeric gates, actual entry execution and complete relocation-aware linked-body
identity. Its seven compiler source modules match this published implementation.
Whole stock2086 later closes29,514,240 cycles with all original outputs exact.
[Whole terminal](perf_records/root_resnet_source_stride_stock2086_stock2086_terminal.json).

Merlin owns the generic interval tables, typed observer closure, exact cold
source continuation placement, direct certified integer observation publication,
integer contraction/domain proofs, probability point preparation, fused radix
finishing, host rounding schedules, immutable helper bases and cost diagnostics.
These remain explicit mechanisms with their original legality and effect gates.
No model name or golden output selects a production optimization.

The source-wide interval table has an explicit partition and allocation budget.
The TinyLlama source re-emission check reproduced all 22 typed observer closures,
44 LLVM routes, helper bodies, the 512 KiB table, integer lookup and retained
source continuation from the measured 2070 build. This establishes compiler
output identity before code generation; its whole-model measurements retain
their original compiler and executable provenance.

The [fused encoder recipe](../experiments/fused_encoder_radix/README.md) packages
the SmolVLA group 2072 adaptation. Its fresh controlled reproduction matches
provider sources, objects, native libraries and complete group ELFs byte for
byte, with the actual 124,061,504-byte workspace ABI and final zero-FSM audit.
Its baseline, captured inputs, compiled assets and qualification are authenticated
external experiment inputs.

The [Tiny observer recipe](../experiments/tiny_closed_observer/README.md)
regenerates those same 22 proofs and 44 routes from published Merlin, then
recompiles authenticated prepared LLVM and replays the controlled links. Both
model objects and complete control/candidate ELFs reproduce byte for byte, and
both final executable audits pass. Its four refusal tests pass. Capture, full
upstream lowering, whole execution and stock timing retain their original gates;
this reproduction does not rerun them.

## Publication checks and measured scope

Earlier combined publication source checks passed (before the later focused
main topics recorded above):

- 1,358 Merlin compiler/runtime/diagnostic checks, with seven skipped cases.
- 61 additional generic source interval/continuation/integer observer checks.
- Merlin structure, derived-document freshness and incremental format gates.
- 61 OOT compiler/export/instruction checks, including six passing subtests.
- 64 integrated target residual/domain/fence/panel checks.
- Four fused encoder reproduction admission/identity checks.
- The README's 17-by-73-by-65 integer GEMM command produces a verified RISC-V
  object with no forbidden or unknown instructions in its executable section.

Publication does not establish a new whole-model timing measurement. The latest
verified stock FireSim whole observations remain:

| Workload | Cycles | Original target |
| --- | ---: | ---: |
| ResNet50 (2090) | 29,402,206 | about 22M |
| TinyLlama (2085) | 378,946,263 | about 300M |
| SmolVLA (1906) | 258,621,872,969 | about 5B |

SmolVLA group 2072 is 3,918,275,805 cycles, using its explicitly recorded
experimental numeric policy. Keep group cycles separate from whole-model cycles.
Original accuracy gates and source fallback obligations remain unchanged.

Read [the evidence](golden_progress.md), [the journey](golden_optimization_journey.md)
and [infrastructure ownership notes](infra_vs_dialect.md) for phase 0, 1 and 2
tooling requirements, negative results and the remaining performance work.

## Publication workflow

The user authorized this named branch and clean direct-main integration for
Merlin/model2MLIR. New PRs require explicit user approval in every repository.
Reviewed topics reach main with one commit per topic. Existing PR records are
closed and archived after content preservation; already merged PR records cannot
be archived through GitHub's supported mutation. Published history is retained.

The completed scoped cleanup covers 42 authored records: 34 are archived and
return HTTP 404 without authentication, eight already merged records remain
visible, and no authored PR remains open in either repository. Thirty-one exact
closed Merlin topic refs and four merged model2MLIR topic refs were removed after
local preservation and dependency checks.
