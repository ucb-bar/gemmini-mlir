# gemmini-mlir

> **UNSEALED LEGACY LINEAGE** — this champion predates the sealed phase 0. It has no `corpus_seal_digest` or `phase0_evidence_digest`; an explicit legacy lineage stands in. Do not read it as a sealed result.

Out-of-tree Merlin codegen export for **gemmini** (family `unknown`).

> **Phase-2 champion.** Merlin's champion export checked this package against its phase-2 evidence profile, and it passed:
>
> - whole-model GSIM certification: `pass`
> - FireSim: 38097435 cycles on `gemmini_rocket_u250_firesim_bitstream`, header `3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924`, vendor control in the same batch (ratio 1.010797)
> - exactness contract `82474ee9954c1280d72bd5875bbf4ec49c3c962c8c6b5f56e05393ff88431e24`: bounded(<=1 LSB) x8, bounded(<=2 LSB) x7, bounded(<=4 LSB) x1, exact x55
> - whole-ELF scan: `clean` against 14 prohibited instructions (roles loop_descriptor)
>
> The evidence is in `.merlin/provenance.json`, `certification.json`, `measurements.json` and `isa_prohibition.json`.

This repository is **generated** by Merlin's champion export through the `merlin-target-publish` bridge. The package payload at the root is byte-identical to the bytes the evidence above was taken on; Merlin adds only this page and `.merlin/`. External dependency closure is **not-attested**: the evidence is about these bytes in the environment its records name.

## What

- Champion package: `ba34625e224f`
- Family: `unknown`
- Recorded status: `unknown`
- Merlin git sha (this export): `ad999e0`

## How to run it

No declared build step. The manifest names `gemmini-opt`; use its declared command argv and required interpreter/environment, not an invented help flag.

```sh
git clone <this-repo> gemmini-mlir
cd gemmini-mlir
```

`manifest.yaml` declares the entrypoint and the argv of every command the experiment ABI expects; run those, not a build.

```yaml
commands:
  emit_analysis_bundle:
    argv:
    - '{tool}'
    - --convert-iface-to-gemmini
    - --emit-command-buffer={output_json}
    - --emit-target-artifact
    - '{input_mlir}'
  emit_command_buffer:
    argv:
    - '{tool}'
    - --convert-iface-to-gemmini
    - --emit-command-buffer={output_json}
    - '{input_mlir}'
  lower_interface_to_target:
    argv:
    - '{tool}'
    - --convert-iface-to-gemmini
    - '{input_mlir}'
  lower_target_to_llvm:
    argv:
    - '{tool}'
    - --convert-iface-to-gemmini
    - --emit-target-artifact
    - '{input_mlir}'
  parse:
    argv:
    - '{tool}'
    - --verify-diagnostics
    - '{input_mlir}'
```

## Provenance

- Champion evidence profile: `phase2`, passed (the evidence is listed above)
- Package-payload certification by `oot_runner.certify` rungs (a separate gate): `not recorded`
- External dependency closure: `not-attested`
- Fingerprint: `n/a`

See `.merlin/provenance.json` for the full lineage and `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the publish layer's own records.

## UNSEALED LEGACY LINEAGE

- Reason: the lineage was graded and measured on the input bundle merlin_assisted_rtlchecks_public_v0 before phase-0 corpora were sealed: no corpus seal and no phase-0 evidence digest exist for it, and no lookalike digest is substituted
- Predates: sealed phase 0
- Stands in for: `corpus_seal_digest`, `phase0_evidence_digest`
- Input bundle: `merlin_assisted_rtlchecks_public_v0` (manifest sha256 `5d540284de06193b2af67ffdb03d35892913141ede6ffdcfefe3514afa5a6f1b`)
- Run: `runs/gemmini/capsule-bench/merlin_assisted/merlincirct_p1froz`
- Run: `runs/gemmini/perf-bench-wholemodel/20260928T201055Z_phase2_measured_nofsm_seed000_58cccdf`
- Run: `runs/gemmini/perf-bench-wholemodel/20261001T020811Z_phase2_cell_g1_stem_seed000_e24cc4c`
- Run: `runs/gemmini/perf-bench-wholemodel/20261001T054423Z_phase2_cell_g1_stem_seed000_c60ef7a`
- Run: `runs/gemmini/perf-bench-wholemodel/20261001T060212Z_phase2_cell_conv3x3_seed000_c60ef7a`
- Run: `runs/gemmini/perf-bench-wholemodel/20261001T055748Z_phase2_cell_mm1x1_seed000_c60ef7a`
- Run: `artifacts/perf-studies/group-capsules/gemmini/composed_801eec3f_20261001`
- Run: `runs/gemmini/perf-bench-wholemodel/20261001T205301Z_phase2_measured_nofsm_seed000_ca8153e`
- phase-1 run started: 2026-09-24T04:46:41Z
- cand_07 graded (round 07): 2026-09-24T14:05:24Z
- 079a9cb8 authored: 2026-09-28T20:50:55Z
- 801eec3f composed: 2026-10-01T10:58Z
- lean board, job 1436: 2026-10-01T19:50:58Z
- stock board, job 1939: 2026-10-06T06:26:26Z
- whole-model GSIM certification: 2026-10-07T06:50:57Z
- earliest sealed phase-0 release found: phase0-20261005T213658Z-c0764ae

| Hop | Driver | Model | Effort |
|---|---|---|---|
| 1ba5615f cand_07 | claudecode | claude-opus-5 | high |
| 079a9cb8 FSM then no-FSM line (about 40 runs) | claudecode | claude-opus-5 | high |
| efaed4dc g1 cell winner | codex | gpt-6-sol | high |
| 27901e2e g1 stem winner | codex | gpt-6-sol | high |
| 4184392d conv3x3 winner | codex | gpt-6-sol | high |
| 036e1d96 mm1x1 winner | codex | gpt-6-sol | high |
| 801eec3f composition | operator-side Claude Code subagent, mechanical git merge-file | claude-opus-5-5 | not an authoring session |

## Composition

This champion was composed by three-way merge of cell winners g1_stem 27901e2e, conv3x3 4184392d, mm1x1 036e1d96 onto efaed4dc.

- Base: `efaed4dcd64a195423c6b79bdbe4a2c558c769fc79fbadf0698c8b6ff778e5ed`
- g1_stem: `27901e2e16babdbe04d507d1adc294fa2c23fe34a9c62ad88442e25ec4b314e1`
- conv3x3: `4184392da7a2a93add3b6d91d3c328664fea2cbd4ae397f0e1b53ecbc57224d5`
- mm1x1: `036e1d96d82677f590b8dfa14404475807f1b3965a35e0b09f62d03b9f90f2f6`
- method: three-way merge
- tool: git merge-file (a file changed by one winner is copied from it; a file changed by two is merged)
- conflicts: 0
- hand_edits: False
- composed_by: an operator-side Claude Code subagent (claude-opus-5-5) on the coordinator's instruction; no agent session authored the merge
- reproduced: re-running git merge-file on the three winners reproduces 801eec3f's files byte for byte
- board_check: job 1436 (lean): 37,455,758 against the best single winner's 37,751,010 in the same batch

## Reconstructed history

`reconstructed: true` — no harness `oot/` repository was kept for this lineage, so its history was rebuilt from stored package bytes, each hop digest-checked. Reason: the lineage predates the harness oot/ repository: its packages survive as measurement-store entries and round submissions, not as commits

- `544d298ed2c2` phase-1 seed 806a1f32, gem_p1_seeded_0923_0628 frozen submission (`806a1f3258783755feca1de8a359ee30594871e5d1ea8254ba63ae3cf3720cc8`)
- `08239bcfedc3` phase-1 cand_07, merlincirct_p1froz round 07 (`1ba5615f0d32c54b162703871cea3a2b7faf7ced393ee3d787802899d0e41adc`)
- `52a858925814` no-FSM line 079a9cb8, nested re-rolling (`079a9cb8a547f1136e0059aaaadceddc317af990cf00010a385dd5f30b8ed236`)
- `ba5c9d5f4f1e` g1 cell winner efaed4dc, collateral-checked (merge base) (`efaed4dcd64a195423c6b79bdbe4a2c558c769fc79fbadf0698c8b6ff778e5ed`)
- `ba34625e224f` composed 801eec3f, three-way merge of three cell winners onto efaed4dc (`801eec3ffeb4a25ac7b3d73fee824f2ea1c1efdbab7dd093ab206dfa1891bbca`)
