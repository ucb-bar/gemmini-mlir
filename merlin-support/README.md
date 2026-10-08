# Merlin support package

Portfolio orchestration selects the lazy backend capabilities `host_witness_abi`
and `primitive_probe`, not target-derived module names. These expose the existing
Gemmini implementations unchanged and retain their source-file provenance.
Importing a capability does not derive an ABI, build a probe or certify hardware.
`portfolio_capabilities_migration.json` records the export change.
The lazy `completion_contract` capability exposes existing completion-ABI derivation;
`completion_capability_migration.json` records its export. Unavailable or malformed
declared evidence must not silently become a valid completion contract.

The backend's `runtime_environment(binaries=..., gsim_max_cycles=..., environment=...)`
returns a copied child environment with both explicitly selected engines and the
requested cycle cap. It does not mutate process state or execute tools.
`pinned_runtime` applies that same policy under its existing restoration lock.
Callers must independently verify certificate binary identities; configuration
alone is not engine or hardware qualification. `runtime_environment_migration.json`
records this evolution without modifying historical migration inventories.

Schema status: **unqualified**. The exact canonical source contract lacks the required `legality`
field. The migration check asserts this recorded diagnostic remains unchanged; a passing migration
test is not a schema-validity claim. Resolve that source contract defect in a separate reviewed change.

This directory contains the target definition and, where present, reference support plugins.
Select it explicitly with `MERLIN_TARGET_PATH=/path/to/this/repo/merlin-support`.
It is not an evaluated compiler candidate and this metadata grants no trust or certification.

The files recorded in `provenance.json` are byte-identical migration snapshots. Existing contracts
retain their prototype, derived-fact, and requires-human-review qualifications. No hardware was
executed for that initial support snapshot, which removed no canonical Merlin source. The later
conformance relocation removes its old canonical owners and records that separate move in
`conformance_migration.json`. Ignored/generated source
artifacts have content hashes but no invented source commit.

The historical candidate/schedule payload and its certificates remain unchanged outside this tree.
Those certificates describe their original revisions, not the current branch with added support.
Do not grade or publish this entire repository as a candidate: export the original candidate or
schedule payload without `merlin-support/`. Existing recursive integrity checks intentionally
remain unchanged and may reject harness-importing support code inside a candidate tree.

The OOT `backend/gemmini_sched.py` now derives non-curated single-instruction
schedule bindings from the selected header, RTL legal-funct evidence and declared
semantic classes. `schedule_vocabulary_migration.json` records the exact support
file transition from the earlier local companion revision and the incoming Merlin
implementation. Its pure header test does not qualify RTL evidence or hardware.

Reference backend/tool plugins are experimenter-side answer-bearing code. Their presence does not
make them public agent inputs; only reviewed contract grants may cross that boundary.

Run the lightweight checks with an installed Merlin core:

```sh
MERLIN_TARGET_PATH="$PWD/merlin-support" PYTHONPATH="$PWD/merlin-support" \
  python -m pytest merlin-support/tests
```

## Conformance and kernel authoring

`gemmini_conformance.model_slices` owns the historical C0–C6 Gemmini model-slice
recipes and capsule exporter, including its Gemmini-default interface adapter.
It reuses Merlin's target-independent matmul interface emitter and golden primitives.
This is host-private support, never a compiler candidate grant. Relocation provenance
is recorded in `model_slices_migration.json`; offline interface tests do not qualify
golden export, compilation, or hardware execution.

`gemmini_conformance/` owns the Gemmini rung definitions, AET recording, resumable sweep,
and agentic kernel slot. It requires the Merlin experiments distribution and this explicitly
selected support provider; a same-named native backend is not a fallback. Inspect the CLI with:

```sh
MERLIN_TARGET_PATH="$PWD/merlin-support" PYTHONPATH="$PWD/merlin-support" \
  python -m gemmini_conformance --help
```

These modules are host-private experiment machinery, not compiler candidate inputs or grants.
The kernel authoring loop calls a paid agent when executed; lightweight tests never do so.
`build_support/` is the backend's pure target-specific program-build dependency, recorded in
`build_support_migration.json`; `conformance_migration.json` records the relocated policy sources.
Neither inventory is a scientific qualification or a replacement for existing Merlin seals.

The support snapshot has **no RTL facts artifact**. MLIR emission and ISA prompt assembly refuse
missing selected-provider facts; they do not borrow Merlin's same-name cache. An operator may
explicitly select existing evidence with `MERLIN_RTL_FACTS=/absolute/path/to/facts.json`, preserving
that artifact's source commitments and qualification status. This does not repair the contract's
missing `legality` or certify a compiler. Recording retains its historical behavior of recording
available artifacts when emission fails; a passing injected oracle result alone does not prove
that an MLIR artifact exists. Lightweight ledger tests use an explicitly synthetic compiler output
to test artifact attribution, and separate refusal tests cover unavailable real facts.

## Provider resources

`resources/gemmini-rocc-tests/` owns the curated int8 header/build-input closure, including upstream
licenses. Default runtime and calibration selection use these bytes, never Merlin's native tree
or an implicit Chipyard fallback. `MERLIN_GEMMINI_HARNESS_DIR` remains an explicit operator override;
an absent override is not replaced with another configuration. Native-convolution evidence uses
the selected provider facts API (or explicit `MERLIN_RTL_FACTS`) and retains the stale hardware hash
check. No facts artifact is bundled.

`cost_model/` owns calibration policy, existing coefficients/vocabulary and the two consumed
validation inputs. Its target is explicitly Gemmini, not the arbitrary provider directory name.
The coefficients retain their original screening-only qualifications; relocation does not refit
or validate them. `resource_migration.json` records source/destination hashes, including the two
backend resource-binding edits. The original `provenance.json` remains the historical snapshot;
its content test permits only explicitly hash-linked replacements in the new inventory.

`backend/gemmini_program_build.py` discovers active installed Merlin owners and the complete
provider pure-build Python membership, using the existing build request's dependency records.
Its optional, request-hashed import-root mapping supports installed/split source layouts;
older requests retain their original checkout-only decoding, not newly minted commitments.
Installed numeric format data use the existing paired schema relation and canonical grant
destinations. Recipe headers, support sources and linker inputs retain their byte pins.
`build_source_identity_migration.json` links the exact implementation predecessor and replacement.
Tests use a relocated provider with installed wheels and reject changed files or pure-package
membership. Worker validation is exercised only up to the compiler boundary. This is not a
complete transitive tool/runtime/header closure, a sandbox grant, or hardware qualification.

The example is `examples/conformance/run.py`; its unchanged run/ledger defaults resolve against
the installed Merlin workspace, not the caller's current directory. Set `--runs-root` and `--ledger`
explicitly to choose another output location. Historical native findings remain archived in Merlin.

`tests/integration/` contains the relocated historical compiler/oracle tests. Collection requires
`GEMMINI_RUN_INTEGRATION=1`; these checks may run installed simulators, so do not opt in for a
lightweight check. For tests needing historical fixtures, explicitly set
`GEMMINI_LEGACY_FIXTURE_ROOT=/absolute/path/to/merlin-checkout`. This temporary fixture dependency
covers shared `merlin/contract` schema/examples, curated header bytes under
`merlin/experiments/capsule_bench`, original agent output under `merlin/tests/data/gemmini_cert`,
and archived candidate manifests under `out/artifacts/targets/gemmini`. These artifacts were not
rewritten or moved; several have unrelated remaining consumers. Backend implementation comes
only from this support provider, not that fixture checkout. Missing real facts remain a refusal,
not a synthetic replacement or a passing hardware test.
