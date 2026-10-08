# Independent component execution features

Select this full provider explicitly with `MERLIN_TARGET_PATH=<OOT>/merlin-support`.
The feature decoder belongs to the pinned sibling `<OOT>/mlir_oot` package. A
standalone support copy without that companion refuses; it never imports an
ambient package or copies the target compiler into a reference provider.
Merlin core and experiments must contain the component cost/execution owners,
strict scoped package executor and actual invocation recorder.

## Closed selected service

The ordinary component loader calls:

```python
backend.prepare_component_execution_service(
    baseline=baseline, qualification=qualification, view=view, runtime=runtime,
    corpus=corpus, target_experiment=target_experiment, contract_root=contract_root,
    source_root=source_root, scope=scope, output=private_output,
)
backend.prepare_component_feature_provider(execution_service=service)
```

The prepared feature provider is a bound method on a frozen owner. Its selected
service and live source/runtime pins are verified before generic cache reuse;
callback code identity alone does not bind a mutable execution context.

These are typed frozen host inputs. No JSON module import, callback name or pass
flag can create executor authority. The target descriptor requires exactly this
selection section (the paths and full digests must name actual selected files):

```yaml
component_feedback:
  schema: gemmini.component_feedback.v1
  certificate: {path: /absolute/independent_certificate.json, sha256: '<full SHA256>'}
  rtl_facts: {path: /absolute/selected_facts.json, sha256: '<full SHA256>'}
  machine_configuration: {path: /absolute/machine.json, sha256: '<full SHA256>'}
```

The separately pinned machine JSON contains exactly `schema` (value
`gemmini.component_machine.v1`), `engine` (absolute path and SHA256), `isa`,
`harts`, `memory` (explicit base/extent or null), `domain_sha256`, and a nonempty
`runtime_dependencies` array of absolute file pins. The selected actual Spike
extension libraries must occur in that closure. The selected ISA/harts/memory
must agree with declarations in the executed ELF. No guessed machine default
or another same-named engine is accepted.

Merlin's closed `prepare_component_normal_execution` binds the existing
`prepare_development_feedback` / `development_executor` owners, the independent
functional qualification and frozen public library/runtime, exact contract,
corpus members and selected target. Each exact independent workload must be
admitted by its own engine-equivalence certificate; validation-model certificate
coverage is not transferred. The selected native `_build_service_for` source and
renderer code are verified separately. The actual contract grader retains its
ordinary native build and original output comparisons; verifying the build
service alone does not claim that an unrelated pipeline used that service.

Historical rate seeds and run directories are not supplied. The ordinary
candidate entrypoints remain under the strict component package executor even
when called through the historical campaign wrapper. Declared untrusted package
build scripts refuse without an independently boxed build service. The inner
executor must use one worker until scoped context propagation is qualified.
The normal broker provides process-tree deadlines and shared resource leases;
this route preserves the 600-second development simulation ceiling.

## Actual observations and unresolved facets

The actual ordinary `run_elf` owner can observe a selected cold Spike run with
`-g`. It retains the actual command, engine/ELF/runtime pins, unfiltered console,
full histogram and invocation receipt. These bytes are decoded by the existing
ELF census and final no-FSM audit. An unordered histogram supplies multiplicities,
not ordered FP/GPR dependencies, distance, replay/fallback scope, fences/drains,
live allocations, cache footprint or hardware cycles. Those features remain
explicitly UNKNOWN. Operand telemetry requires an independently qualified
producer; a label or file hash alone supplies no producer qualification.

The current normal contract executor has no independently qualified warm PC
scope. It returns a typed unavailable warm regime, writes a private reason and
preserves all eleven warm stages as UNKNOWN. It never copies the cold histogram
or calls another cold run warm. Both compiler arms must use identical admitted
input bytes. Complete costs cover preparation, allocation, packing, proof,
device, readout, reconstruction, replay, dispatch, publication and cleanup; no
stage is zero-filled, and this adapter refuses supplied region prices without
an evaluated stage timeline.

`evaluate_component_execution_stages` independently reopens actual source,
conversion, command-buffer, artifact, object/link and engine invocation records.
The optional typed `ComponentSourceApplicability` is independently rerun and
joined to the exact observed source path, digest and frontend. Its private facts
can establish source-only applicability of pure closed tensor SSA; physical
runtime obligations remain UNKNOWN. Dictionaries, pass flags and observations
from another source path cannot supply this authority.

It issues a typed observed assessment with per-facet missing authorities. This
source inspection runs before completed functional qualification and therefore
does not bootstrap qualification from itself. `verify_component_execution_witness`
conditionally promotes only a complete assessment whose actual records still
replay and whose typed stage parents/output/effect rosters pass Merlin's verifier.
Currently preparation/partition semantic lift, full emitted host/device
correspondence, FX/import where applicable and complete runtime effects are
unavailable. Their UNKNOWN results refuse functional qualification. A successful
numerical result or `PASS` field cannot fill those semantic gaps.

This implementation supplies concrete selected service wiring and observed
subsets. It does not establish qualified hardware, warm scope, a calibrated whole
predictor or completed campaign performance. Predictor activation still requires
independent mechanism training and held groups, complete legality/function gates,
and the existing preregistered ranking/error/interval-coverage screen.
