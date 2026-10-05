# Golden compiler inventory and emission wiring

## Actual scope

The development package now exposes three explicit commands through `gemmini-opt`.
The existing interface commands retain their original component membership. This
does not alter an already sealed package, certify the backend, select whole-model
schedules through a shared solver, or change qualified model artifacts.

| Command | Executed compiler route | Checked result |
| --- | --- | --- |
| `--emit-golden-inventory PATH` | Merlin `inspect_compiler_package`, `build_compiler_edit_contract`, `phase2_edit_contract.load` and `validate_against_package` | Six exact AST decision/codegen owners, real command membership, canonical edit-contract identity |
| `--export-golden-contraction --region ID --llvm-bin PATH --workdir PATH SOURCE` | Existing exact upstream integer contraction matcher, target schedule selection, primitive xDSL generation, ordinary LLVM/RoCC compilation | Source SHA, operation identity, selected kernel symbol, object SHA, shared `GlobalPlanEmission` boundary/accounting checks and object no-FSM audit |
| `--export-golden-capture --llvm-bin PATH --workdir PATH model.mlir` | Existing pinned capture epilogue binder and compiler | The binder's original exact source/numeric/parameter receipts and actual selected routes |

The contraction export covers one unbatched `i8 × i8 → i32` contraction from
zero. Surrounding model operations, whole-model allocations and whole-model
quality remain outside this operation export. A region ID selects the operation
to bind; source semantics, dimensions and target resources select its schedule.
Unsupported semantics or batched ABI are refused.

The selected singleton `GlobalPlan` controls compilation: the emitter lowers its
bound generator, emits the plan's implementation symbol, and returns the existing
Merlin executable accounting receipt. Changed source, logical dispatch, plan or
generator schedule is refused. This demonstrates the concrete emission seam;
it does not invoke the shared cycle-ranking solver. Costs and occupancy are
explicitly unknown by default.

Python callers may attach an existing Merlin `ActivityTimeline` with the exact
emitted object SHA. Every event requires finite durations and explicit provenance;
the shared scheduler must reproduce the supplied timeline. Event provenance and
occupancy remain caller-supplied evidence, with no hardware calibration inferred.

## Edit surface ownership

`manifest.yaml` declares shape selection, dense transfer-family selection, remaining
B slot placement, dense output command emission, convolution family selection and
resident convolution stripe emission. Each declaration resolves to an existing
symbol in an actual new compilation command component. Numeric proof functions,
instruction encodings, hardware identities and verification are not granted as
optimization surfaces.

Inventory export describes candidate-owned declarations. The experiment host
must review, freeze and enforce its edit contract through the existing Merlin
machinery before accepting compiler edits. Export itself grants no edit authority.

## Verification and next connection

`tests/test_golden_compiler_export.py` checks the actual package schema, command
ownership, AST owners and canonical contracts; executes manifest commands; compiles
independent tail shapes; proves a selected schedule changes the emitted object;
rejects source/plan/schedule mutations and inert CLI flags; and checks explicit
artifact-bound timelines. Capture export delegates the already qualified binder.

The first operation export also runs on strict RV64GC Spike: `M123/N73/K1041`,
resident A, `BN5`, B slots `8448/8528`, amplitude 21. All 8,979 outputs and
2,048 guard bytes pass; the linked ELF has zero forbidden or unknown Gemmini
instructions. Its counter is functional execution evidence, not hardware
timing. The source, selected symbol, object, plan emission and final ELF pins
are retained in [the qualification receipt](perf_records/golden_compiler_export_qualification.json).

## Enabled model path

`gemmini-model-build` invokes the normal captured-model build and accuracy driver.
Its `DeviceRouting` is now wrapped by `bind_device_routing` by default. Before
catalog compilation, the wrapper outlines the complete prepared source and emits
an identity cover through `OutlinedGlobalPlanEmitter`. Expanded before/after
outlined emission computations must be structurally equivalent. Original prepared
bytes are passed unchanged to the established compiler after this gate. Strict
source-to-outlined structural equivalence is unknown because the existing outliner
can clone initializers; this adapter does not substitute outlined IR for source.

The wrapper permits explicitly declared external catalog functions during
outlining, then requires their actual compiled definitions. It derives C-interface
entry symbols from `llvm.emit_c_interface`, closes all actual catalog calls and
primitive symbols, and matches every remaining exact integer contraction to the
compiler's selected dense binding. The final hook joins the ordinary Merlin
catalog/shim ABI receipt, linked image symbols, source snapshots, host/link object
hashes and no-FSM audit. The existing normal native/original-golden and strict
target accuracy gates still run.

The resulting `device_prepared/global_plan/compiler_plan.json` carries complete
prepared-IR preservation and artifact bindings for Phase 2 review alongside the
existing package inventory/edit contract. Unsupported external symbols or
incomplete catalog/binary bindings are refused. The identity plan preserves the
existing compiler decisions. It claims no optimization or shared solver use.

The unchanged qualified 1853 artifact was replayed through these same hooks:
3,434 source graph nodes, 70 external catalog routes and the exact classifier
binding close against the actual composite object and linked ELF. This is a
read-only accounting replay of the existing artifact; it is not a newly compiled
or newly timed model. The receipt records original/prepared source identities and
the downstream limits explicitly.

Whole-model selection still needs the existing routing callbacks to enumerate all
legal region/transition alternatives, provide calibrated full-shape costs and
occupancy, pass them to the shared exact-cover selector, and compile the selected
target implementations. Ordinary host lowering also needs complete source-task
to machine CFG accounting before that stronger proof can be claimed. This adapter
labels that accounting, original rewrite-chain equivalence and costs as unproven
or unknown; it does not manufacture `mixed_program_plan_v1` evidence.
