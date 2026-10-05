# Golden compiler inventory and emission wiring

## Actual scope

The development package now exposes four compiler export/selection commands through `gemmini-opt`.
The existing interface commands retain their original component membership. This
does not alter an already sealed package, certify the backend, select whole-model
schedules through a shared solver, or change qualified model artifacts.

| Command | Executed compiler route | Checked result |
| --- | --- | --- |
| `--emit-golden-inventory PATH` | Merlin `inspect_compiler_package`, `build_compiler_edit_contract`, `phase2_edit_contract.load` and `validate_against_package` | Seven exact AST decision/codegen owners, real command membership, canonical edit-contract identity |
| `--export-golden-contraction --region ID --llvm-bin PATH --workdir PATH SOURCE` | Existing exact upstream integer contraction matcher, target schedule selection, primitive xDSL generation, ordinary LLVM/RoCC compilation | Source SHA, operation identity, selected kernel symbol, object SHA, shared `GlobalPlanEmission` boundary/accounting checks and object no-FSM audit |
| `--optimize-golden-contraction --region ID --llvm-bin PATH --workdir PATH --calibrations PATH SOURCE` | Existing shared planner ranks pinned full-fixture measurements and compiles its selected generator | Actual selected object matches calibration; source/input/engine/ABI hashes closed; complete ranking with unresolved physical floor retained |
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
resident convolution stripe emission, plus calibrated selection reaching actual emission. Each declaration resolves to an existing
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
incomplete catalog/binary bindings are refused. The identity plan is an admission
check for the existing compiler decisions. Its checked outlined IR is discarded;
the selected plan does not control compiled IR or target schedules. It claims no
optimization or shared solver use.

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

## Measured shared selection that reaches emitted code

`gemmini-opt --optimize-golden-contraction --region ID --calibration FILE
--llvm-bin PATH --workdir PATH SOURCE` and the manifest command
`optimize_golden_contraction` use existing Merlin `ExplicitPlanningAdapter`,
`ActivityTimeline`, `optimize_program` and `GlobalPlanEmission`. The source
contraction is the complete declared optimization unit. Surrounding model work
is outside this route.

First compile legal alternatives through `--export-golden-contraction`. Its
receipt now carries the explicit compiler options. Measure their actual objects
with the existing full-output GSIM probe and pinned static inputs. A calibration
manifest names at least two exports and measurements, each by path and SHA256,
plus the exact GSIM engine pin:

```json
{
  "schema": "golden_contraction_calibrations_v1",
  "gsim_engine": {"path": "/path/to/emulator", "sha256": "..."},
  "candidates": [
    {"export": {"path": "a/golden_export.json", "sha256": "..."},
     "timing": {"path": "a_probe/result.json", "sha256": "..."}},
    {"export": {"path": "b/golden_export.json", "sha256": "..."},
     "timing": {"path": "b_probe/result.json", "sha256": "..."}}
  ]
}
```

The compiler re-derives every candidate from the current source semantics and
explicit options. It checks exact source/operation geometry, schedules/slots,
compiled object and IR identities, engine/linked-ELF/no-FSM closure, raw duration
and numeric UART markers, and identical static input/expected-output hashes.
Unpriced, duplicate, altered or incompatible candidates refuse. The shared
search evaluates every admitted alternative; `SourceContractionEmitter` compiles
the winner and requires its emitted object hash to equal the calibrated object.
The plan's selected implementation therefore changes actual target code.

Only an opaque measured invocation is priced. Its coarse timeline cannot reveal
compute/DMA occupancy or overlap inside the accelerator. Physical/legal roofline
floors remain unknown, so the shared result retains `status: refused` for that
stronger attainment gate while returning its complete best measured plan.
The receipt explicitly distinguishes complete calibrated enumeration from
resolved roofline attainment. No fixture duration is reused for another source,
input fixture, kernel shape or full model.

Actual source `M17/N73/K65` with amplitude21 and independent tails checked every
1,241 output plus2,048 guard bytes. Control measured3,234 GSIM kernel cycles;
cached A/wide B/next-K prefetch measured2,357 (27.1% lower). The manifest command
selected and emitted the latter actual object, and strict RV64GC Spike with the
explicit Gemmini extension and final no-FSM audit passed.
[Measured source-selection qualification](perf_records/golden_calibrated_source_selection_qualification.json).
This is a working scoped selection/emission foundation; the normal whole-model
binding remains an identity admission gate until its complete legal alternatives,
host transitions, costs and downstream instruction accounting are supplied.
