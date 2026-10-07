# gemmini-mlir

> **UNSEALED LEGACY LINEAGE** — this champion predates the sealed phase 0. It has no `corpus_seal_digest` or `phase0_evidence_digest`; an explicit legacy lineage stands in. Do not read it as a sealed result.

Out-of-tree Merlin codegen export for **gemmini** (family `unknown`).

> **Phase-1 champion.** Merlin's champion export checked this package against its phase-1 evidence profile, and it passed:
>
> - public capsules at L3: 121/180 (121 executed now, 0 carried)
> - hidden capsules at L3: 14/14 (14 executed now, 0 carried)
> - L3 engine `gsim` (binary `1a3de02a19ddcb4704443e2b565fd6eb0299a940afdcc7c037e2604bbcbb52ea`)
> - grader commit `7013aff4b72f846cf114254a025c7ee3566fbc5f`
> - whole-ELF scan over 175 ELFs: `clean` against 14 prohibited instructions (roles loop_descriptor)
>
> The evidence is in `.merlin/provenance.json`, `certification.json`, `measurements.json` and `isa_prohibition.json`.

This repository is **generated** by Merlin's champion export through the `merlin-target-publish` bridge. The package payload at the root is byte-identical to the bytes the evidence above was taken on; Merlin adds only this page and `.merlin/`. External dependency closure is **not-attested**: the evidence is about these bytes in the environment its records name.

## What

- Champion package: `08239bcfedc3`
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

- Champion evidence profile: `phase1`, passed (the evidence is listed above)
- Package-payload certification by `oot_runner.certify` rungs (a separate gate): `not recorded`
- External dependency closure: `not-attested`
- Fingerprint: `n/a`

See `.merlin/provenance.json` for the full lineage and `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the publish layer's own records.

## UNSEALED LEGACY LINEAGE

- Reason: graded on the input bundle merlin_assisted_rtlchecks_public_v0 before phase-0 corpora were sealed: no corpus seal and no phase-0 evidence digest exist, and no lookalike digest is substituted
- Predates: sealed phase 0
- Stands in for: `corpus_seal_digest`, `phase0_evidence_digest`
- Input bundle: `merlin_assisted_rtlchecks_public_v0` (manifest sha256 `5d540284de06193b2af67ffdb03d35892913141ede6ffdcfefe3514afa5a6f1b`)
- Run: `runs/gemmini/capsule-bench/merlin_assisted/gem_p1_seeded_0923_0628`
- Run: `runs/gemmini/capsule-bench/merlin_assisted/merlincirct_p1froz`
- seed frozen (806a1f32): 2026-09-23T09:27:30Z
- phase-1 run started: 2026-09-24T04:46:41Z
- cand_07 graded (round 07): 2026-09-24T14:05:24Z
- L3 certification (grade_l3_fresh): 2026-10-06T07:08:46Z to 09:03:18Z
- earliest sealed phase-0 release found: phase0-20261005T213658Z-c0764ae

| Hop | Driver | Model | Effort |
|---|---|---|---|
| 806a1f32 seed (gem_p1_seeded_0923_0628) | claudecode | claude-opus-5 | high |
| 1ba5615f cand_07 (merlincirct_p1froz round 07) | claudecode | claude-opus-5 | high |

## Reconstructed history

`reconstructed: true` — no harness `oot/` repository was kept for this lineage, so its history was rebuilt from stored package bytes, each hop digest-checked. Reason: the lineage predates the harness oot/ repository: its packages survive as measurement-store entries and round submissions, not as commits

- `544d298ed2c2` phase-1 seed 806a1f32, gem_p1_seeded_0923_0628 frozen submission (`806a1f3258783755feca1de8a359ee30594871e5d1ea8254ba63ae3cf3720cc8`)
- `08239bcfedc3` phase-1 cand_07, merlincirct_p1froz round 07 (`1ba5615f0d32c54b162703871cea3a2b7faf7ced393ee3d787802899d0e41adc`)
