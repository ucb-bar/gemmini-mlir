# Merlin provenance — `merlin_assisted_rtlchecks` run (rounds 1–3)

Written from what actually happened. "Did not help" appears where that is the truth.

## 1. Merlin tools used

| Tool (path) | Used? | What you used it for |
|---|---|---|
| `targetgen/rtl/facts.py` (`load_facts`) | **yes, every round** | The single source for mesh `DIM` (16), scratchpad (262144 B) / accumulator (65536 B) capacity, the i8-operand / i32-accumulator datapaths, and the legal-funct decode table with its names. Every one of those numbers reaches the emitted code through `mlir_oot/target/facts.py` rather than a literal. |
| `targetgen/rtl_backend.py` (`target_profile`, `derived_levers`) | **yes** | The lever list this hardware implies. Round 1 returned three axes; **this round it returns seven**, which is how the four new axes were noticed at all. `manifest.yaml`'s `optimization_surfaces` is written against that list. |
| `kernels/cca_contract.py` (`check_bijection`) | yes | Confirmed `clean: true`, no `orphan_fields`, no `orphan_routes` — i.e. no leverable axis left unrouted and no phantom route. Run each round as a regression check on the manifest map. |
| `kernels/action_catalog.py` (`escalation_ladder`) | yes | Walked for `spatial.dataflow` (round 1) and for all four new axes this round. Every ladder pointed at `<oot_package>/lowering/` — the tile-program emitter — which is where `KernelBuilder.config_ld`, `_already_resident` and `transposed_b` were then declared as the real seams. It is also what told me `dispatch.loop_offloaded` has no backing lever here, so it is deliberately left undeclared. |
| `targetgen/generate/` (scaffold gen) | **yes** | `target_repo.generate_skeleton('gemmini')` invoked before the first submission edit; returned 15 relpaths (`README.md`, `AGENT.md`, `pyproject.toml`, `xdsl/`, `lib/`, `tools/`, `runtime/`, `llvm/`, `tests/` …). Used as the **shape** for the package layout (a separated dialect / passes / tools / tables split). Its contents were not copied into `submission/`. |
| `xdsl_dialects/` (dialect patterns) | yes, read | The IRDL idiom for declaring ops with verifiers — `irdl_op_definition`, `var_operand_def`, `Dialect(...)`, raising `VerifyException` from a verifier — which is the pattern `mlir_oot/frontend/iface_dialect.py` and `mlir_oot/target/dialect.py` follow. |
| `targetgen/contract/interface_emit.py` | **yes, decisive this round** | `_NAMED_OP_OPERAND_KEYS` is the authority for each whole-op's POSITIONAL OPERAND NAMES. This is what gave `rmsnorm -> (src, gamma)`, `rope/softmax -> (src,)`, `residual_add -> (lhs, rhs)`, `bias_add -> (src, bias)`. Before this, `residual_add` was extracting `in0/in1`, so its command buffer could not name what the runner reads. Its comments also state that `bias_add`'s operands are in the accumulator's dtype and that `residual_add` rounds once — both later confirmed against `command_buffer_abi.yaml`. |
| `runtime/commandbuffer.py` (`materialize_inputs`) | **yes, decisive this round** | Reproduces a capsule's declared leaf tensors deterministically by name. This is the tool that made the truncating-cast finding possible (see §3): it let me recompute candidate op definitions on the *same* inputs the grader used, with no access to any golden. |
| `targetgen/synthesize/` | no | Read only. The synthesis entry points target a different authoring flow than the hand-driven xDSL pass pipeline this package ended up being; nothing from it is in the submission. |
| `isa_tools.py` (derived asm/disasm/lint) | yes, every edit | `lint` on every emitted artifact (191/191 report 0 UNKNOWN); `disasm` to reconcile decoded operand fields against what the command buffer declares — that is how the CONFIG_LD f32 scale field was confirmed to carry `lhs_scale`/`rhs_scale` this round. |

## 2. Files generated with Merlin tooling

| submission file | origin | notes |
|---|---|---|
| package layout (`mlir_oot/{frontend,lowering,target,codegen,tables,ir}/`) | mixed | Split follows `target_repo.generate_skeleton`'s returned shape; the files themselves are authored. |
| `mlir_oot/frontend/iface_dialect.py` | mixed | IRDL idiom from `xdsl_dialects/`; the op set, attribute requirements and verifiers are authored against `interface_dialect_contract.yaml` + `interface_emit.py`. |
| `mlir_oot/frontend/extract.py` (`_OPERAND_KEYS`) | mixed | The operand-name table is transcribed from `interface_emit._NAMED_OP_OPERAND_KEYS`'s vocabulary. |
| `mlir_oot/target/facts.py`, `mlir_oot/tables/funct_table.py` | mixed | Values derived from `rtl.facts.load_facts('gemmini')`; `tables/gen_funct_table.py` regenerates the table from those facts. |
| everything else | hand | Lowering passes, contraction emitter, im2col address stream, scalar lane, LLVM emitter, re-roller, devtools. |

## 3. Failures encountered, and which Merlin tooling diagnosed

| round | capsule | failure plane / violations | fix | Merlin tool that helped (or "none") |
|---|---|---|---|---|
| 1b | 87 capsules | `trace does not open with a FENCE` | issue the scalar fence first | none (verdict text was explicit) |
| 1b | every non-aligned multi-row operand | RTL oracle disagreed while cheap tiers passed | DRAM row pitch = trailing extent rounded UP to a tile edge | none — `mlir_oot_backend_contract.yaml`'s `pointee_layout` |
| 1b | multi-block matmuls | wrong accumulate bit | key the bit by (m-tile, n-tile, row), not (m-tile, row) | none; found by the package's own `devtools/simulate.py` |
| 1c | 9 capsules | `host_compute`: N host ops per element | `codegen/reroll.py` rolls arithmetic-progression command windows into loops | none |
| 1d/1e | 10 + 4 capsules | "never wrote output Y0"; `lanes` plane | host lane must COMPILE the region and name tensors by the ABI's role vocabulary | `kernels/roles.py` / `linalg_iface.py` for the name vocabulary |
| 3 | 18 capsules (`softmax`/`rmsnorm`/`rope`) | **parse ERROR** — op not registered | registered all three as IRDL ops + structural extraction | **`interface_emit.py`** supplied the operand names; `xdsl_dialects/` the op idiom |
| 3 | `OC_bias_add_*` (4) | declined: "operands are in the ACCUMULATOR's dtype" | derive the container width and use the load path's `shrunk` bit | `interface_emit.py`'s own comment + `command_buffer_abi.yaml` |
| 3 | `OC_residual_add_*`, `GR0/GR1/GR2` (7) | declined: no lowering | `_acc_add`; per-operand multipliers on the CONFIG_LD scale field | `isa_tools.py disasm` confirmed the field actually carries 1.0 / 0.5 |
| 3 | `OC_rmsnorm_*` (4) | `numeric fail`, `max_abs_diff: 1`, `mismatch_count: 50/256` | **the cast truncates, it does not round** | **`runtime/commandbuffer.materialize_inputs`** — reproducing the declared inputs turned a blind guess into a solved constraint: the reference is the candidate whose disagreement-with-my-output set equals the verdict's `mismatch_indices`. Also `AccumulatorScale.scala` + `CustomConfigs.scala` (RTL, not a Merlin tool) established that no normalizer is instantiated, which is why the stage belongs on the scalar lane at all. |
| 3 | `OC_rmsnorm_qkv_*`, `OC_rope_qkv_*`, `OC_attention_mx_*` (14) | trace: required classes missing; numeric `max_abs_diff` 13–15 on the lane route | **declined by name** (`MixedLaneProgram`) rather than shipped wrong | none — `materialize_inputs` was used to search the chained-precision space and did **not** find the reference, so the honest outcome was a stated coverage gap |

## 4. Files changed per iteration

| round | files changed | result |
|---|---|---|
| 1 | whole package created | 6 → 78 pass |
| 1b | `target/layout.py`, `lowering/contraction.py`, `lowering/conv.py`, `devtools/simulate.py` | 78 pass |
| 1c | `codegen/reroll.py`, `lowering/lanes.py` | 87 pass |
| 1d | `codegen/host_lane.py`, `codegen/fmath.py`, `frontend/linalg_model.py` | 97 pass |
| 1e | `lowering/lanes.py`, `codegen/scalar_lane.py` | 101 / 103 |
| 2 | `lowering/quantize.py`, `codegen/quant_lane.py` (route Q) | (scope changed before regrade) |
| 3 | `frontend/iface_dialect.py`, `frontend/extract.py`, `lowering/iface_to_gemmini.py`, `lowering/lanes.py`, `codegen/scalar_lane.py`, `cmdbuf.py`, `driver.py`, `manifest.yaml`, `devtools/simulate.py` | 0 ERROR (was 18); 15 capsules newly exact on L2; 14 honest declines |

## 5. Final-artifact integrity (self-attestation — the grader verifies independently)

- Imports Merlin runtime code? **no.** `grep -rn 'import merlin\|from merlin\|reference_outputs\|pipeline.execute\|runtime.simulator\|runtime.reference' submission/` returns nothing, including `submission/devtools/`. `materialize_inputs` was used only in throwaway analysis scripts run from the workspace root; no such script is in `submission/`.
- Self-contained (graded only through its CLI entrypoints)? **yes.** `submission/gemmini-opt` execs `mlir_oot.driver`; the only third-party import is xDSL.
- Merlin authoring artifacts left in `submission/` by accident? **none.** The scaffold's returned files were used as a layout reference and never copied in.

## 6. One-line summary

Merlin tooling helped materially in two specific, attributable places this round — `interface_emit.py`
gave the ABI's operand names for three ops that had been failing to parse at all, and
`runtime/commandbuffer.materialize_inputs` converted an undocumented exact-integer op definition from a
blind guess into a solved constraint (the truncating cast) using only the redacted verdict's mismatch
indices — while `rtl/facts.py` + `rtl_backend.derived_levers` kept every ISA/mesh number and the lever
map RTL-derived rather than invented; it did **not** help with the hybrid mesh-plus-lane kernel that
the 14 remaining declines need.

---

# Round 4 addendum

## 1a. Merlin tools used THIS round

| Tool (path) | Used? | What you used it for |
|---|---|---|
| `targetgen/rtl/facts.py` (`load_facts`) | yes | Re-run at the start of the round. Unchanged: mesh 16x16 (256 `Tile` instances, `corroborated: true`), scratchpad 262144 B / depth 4096, accumulator 65536 B / depth 512. The i8-operand fact is what makes route W (the wide-operand digit split) necessary rather than optional. |
| `targetgen/rtl_backend.py` (`derived_levers`) | yes | Returned the same seven axes as round 3; no new axis appeared, so `optimization_surfaces` was not extended. |
| `kernels/cca_contract.py` (`check_bijection`) | yes | `orphan_fields: []`, `orphan_routes: []`, `unclassified: []`, `ladder_errors: []` — clean, so the round's work went into correctness rather than into wiring a missing axis. |
| `kernels/action_catalog.py` (`escalation_ladder`) | yes | Walked `spatial.dataflow`. Its single row (`HEURISTIC`, `<oot_package>/lowering/`, `needs_new_code: True`) is the seam the three new passes were written into. |
| `targetgen/generate/` (`target_repo.generate_skeleton`) | yes | Re-invoked; 15 relpaths returned. The package layout already follows it; nothing new taken this round. |
| **`runtime/tensor.py` (`Tensor.deterministic`)** | **yes — decisive, twice** | Reproduces a capsule's declared operands exactly (backed by `merlin.common.stimulus`; default fill `lo=0, hi=3` where the capsule declares no `stimulus_range`). This is what turned a redacted `(mismatch_count, max_abs_diff)` pair into an identified op definition, with no golden read. See §3a. |
| `capsule.pytorch.py` of `model/M0_small_llama_gemmini` | yes — decisive | The corpus's OWN `rope` definition, in source. Fixed the half-split convention and the `theta**(-(j/half))` frequency without a single guess. (An allowed capsule input, not a Merlin tool.) |
| `isa_tools.py` (`lint` / `disasm`) | yes, every edit | `disasm` is what exposed the `llvm.fptosi` defect: every operand of every command decoded as `kind: unknown` and CONFIG_* as `UNKNOWN`. That is a structural-parse failure in the consumer, and nothing in the numeric planes would have shown it. |
| `targetgen/synthesize/` | no | Read only; nothing from it is in the submission. |

## 3a. Failures this round, and what diagnosed them

| capsule(s) | failure plane / signal | fix | what diagnosed it |
|---|---|---|---|
| `GR2_resadd_seam_i8` | `elaborated_rtl`, `mismatch_count 8`, L2 pass / L3 fail | general DRAM dependency fence in `iface_to_gemmini.run` | `isa_tools disasm` — MVOUT of the staged intermediate at #14, MVIN of the same argument at #18, no fence between |
| 14 `OC_*_qkv` / `OC_attention_mx_*` | `backend_declined` | routes X / W / rowscale / softmaxfuse | the capsule interfaces themselves; `lanes.MixedLaneProgram` was an honest refusal, not a bug |
| all hybrid capsules (latent, would have hit hidden grading too) | every decoded operand `kind: unknown` | `codegen/fpcast.py` — narrowing float cast built from registered ops only | `isa_tools disasm` |
| `OC_rmsnorm_qkv_*` | `mismatch_count 250/256, max_abs_diff 13` | `lowering/rowscale.py` | `Tensor.deterministic` + the redacted verdict: computed both candidate definitions on the capsule's real operands and compared them TO EACH OTHER — `max_abs 13, 250 differing` reproduced the verdict exactly, identifying the reference without reading one |

## 5a. Final-artifact integrity (re-checked at the end of round 4)

- Imports any Merlin runtime code? **no** — `grep -rn 'import merlin\|from merlin\|reference_outputs\|pipeline.execute\|runtime.reference\|runtime.simulator' submission/` returns nothing.
- Self-contained, graded only through its CLI entrypoints? **yes.**
- Regular expressions anywhere in the package? **none** — `grep` for `import re` / `re.match|search|sub|findall|compile` over `submission/mlir_oot/` returns nothing; all IR is parsed structurally through xDSL.
- Merlin authoring artifacts left in `submission/`? **none.** `Tensor.deterministic` and the capsule pytorch file were used to DERIVE definitions during authoring; neither is imported, and no value read from either is baked into the package — the emitted trig table is computed from the capsule's own declared `theta` and extents.

## 6a. One-line summary (round 4)

Merlin tooling was decisive twice this round and in the same way both times: `Tensor.deterministic`
let me reconstruct the capsule's real operands, which converted a redacted `(mismatch_count,
max_abs_diff)` pair into a *proof* of which op definition the reference uses — that is what found
"the fused operation is defined on the unrounded intermediate", worth 10 capsules. The RTL facts made
the wide-operand problem unambiguous rather than a guess, and `isa_tools disasm` caught an
unregistered-op defect that no numeric plane could have reported.

## 5b. Round 5 — Merlin tooling re-run, and what it was worth

All four required derivation calls were re-run before touching the package, and their returned
evidence is recorded here rather than paraphrased:

| call | returned |
|---|---|
| `cca_contract.check_bijection('gemmini')` | `orphan_fields: []`, `orphan_routes: []`, `unclassified: []`, `ladder_errors: []` — every leverable axis routed, no phantom lever. Unchanged from round 4, which is the point of running it: a regression check on `manifest.yaml`'s `optimization_surfaces`. |
| `rtl_backend.derived_levers(target_profile('gemmini'))` | 7 axes: `spatial.dataflow`, `spatial.accumulator_resident`, `memory.capacity_fit`, `dispatch.descriptor_reuse`, `dispatch.dma_overlap`, `dispatch.loop_offloaded`, `layout.operand_major`. Same seven as round 4; the 11 declared surfaces still cover them. |
| `rtl.facts.load_facts('gemmini')` | mesh 16x16 (256 `Tile` instances, `mac_idiom` 1 mul / 7 adds / 3 regs), scratchpad 262144 B / depth 4096, accumulator 65536 B / depth 512, datapaths i8 input (`scratchpad smem UInt<8>`) and i32 accumulator (`AccumulatorMem SInt<32>`), and the 26-entry legal-funct table with names. Source pinned to `chipyard.harness.TestHarness.GemminiRocketConfig.fir`. |
| `action_catalog.escalation_ladder(axis,'gemmini')` | Walked for `spatial.dataflow`, `dispatch.dma_overlap`, `memory.capacity_fit`. Each returns a single `HEURISTIC` rung at `<oot_package>/lowering/`, `forkable_now: false`, `needs_new_code: true` — i.e. no stronger stock rung exists to escalate to; the lever has to be built in our own emitter, which is where all three already are. |
| `targetgen.generate.target_repo.generate_skeleton('gemmini')` | 15 relpaths, unchanged. Re-run as the generation witness; nothing regenerated, since the package layout already follows it. |

**Honest assessment of this round specifically: Merlin tooling did not move the score, because
nothing could.** The 12 open capsules are harness-side (9 crash computing the golden from the
capsule's own declared fused op; 3 have no weights asset in this snapshot), and the derivation calls
correctly reported that the compiler side has no unrouted axis left. Where the tooling *did* earn its
place was as a negative check: `check_bijection` returning four empty lists is what justified NOT
adding a lever this round, and the `escalation_ladder` rows saying `forkable_now: false` is what
confirmed there is no cheaper rung available than the heuristics already written. That is a real
result — it is the difference between "converged" and "out of ideas".

The one lever the tooling did *not* settle is standalone `rmsnorm` on the mesh (`X @ diag(G)`); that
was decided against on measured grounds recorded in `REPORT.md` and `iteration_notes.md`, not on
anything a Merlin call reported.

## 5c. Final-artifact integrity (re-checked at the end of round 5)

- `import merlin` / `from merlin` / `reference_outputs` / `pipeline.execute` / `runtime.reference`
  anywhere under `submission/`: **none** (re-run this round).
- Regular expressions anywhere in the package: **none** (`import re`, `re.match|search|sub|findall|compile`).
- `manifest.yaml`: `integrity_exempt: false`, `language: python`, no build block; all 5 declared
  commands have `components:` entries and every declared path exists; all 11
  `optimization_surfaces` symbols resolve to a real Python AST name in the file they name (verified
  by AST walk, not by grep).
- Grader-reported `integrity_status`: **clean**.

## 5d. Round 6 — Merlin tooling re-run, and what it was worth

All required derivation calls were run **before** the round's one submission edit, and returned
non-empty results:

| call | returned |
|---|---|
| `cca_contract.check_bijection('gemmini')` | `orphan_fields: []`, `orphan_routes: []` — unchanged; no leverable axis unrouted, no phantom route |
| `rtl_backend.derived_levers(target_profile('gemmini'))` | the same 7 axes as rounds 4–5 |
| `rtl.facts.load_facts('gemmini')` | mesh 16x16 / 256 `Tile`, scratchpad 262144 B depth 4096, accumulator 65536 B depth 512, i8 operand + i32 accumulator datapaths, 26-entry legal-funct table |
| `generate.target_repo.generate_skeleton('gemmini')` | 15 relpaths (generation witness; layout already follows it) |
| `action_catalog.escalation_ladder(axis,'gemmini')` | walked for **all 7** axes this round. Every one returns a single rung at `<oot_package>/lowering/` with `forkable_now: false` — no stronger stock rung to escalate to |

**Where Merlin tooling earned its place this round: grounding a cost advisory that would otherwise
have been guesswork.** The grader's trace check reported that our memory-load transfers use one
16-column tile where the target admits four. Rather than trust or dismiss it, the claim was checked
against the shipped `gemmini_params.h` (`MAX_BYTES 64`, `MAX_BLOCK_LEN = MAX_BYTES/(DIM*1) = 4`)
**and** independently corroborated by `load_facts`' own geometry — `BANK_NUM*BANK_ROWS*DIM*1 =
262144` B and `ACC_ROWS*DIM*4 = 65536` B reproduce the reported capacities exactly. That two-source
agreement is what turned "an advisory we could rationalise away" into a real, documented lever. It
also showed the advisory is *not* the same kind of checker artefact as the round-5 rejects, which is
the distinction that matters when deciding what to believe.

It did **not** move the score, because nothing could: `check_bijection` returning empty lists and
every `escalation_ladder` rung reporting `forkable_now: false` is the machine-checked statement that
the compiler side has no unbuilt lever left, and the 12 open capsules are harness-side.

Honest note on the round's one code change: the softmax exponential fusion came from reading our own
`scalar_lane.py` against the measured `tier_cycles` distribution in the redacted verdict — no Merlin
call suggested it.

## 5e. Final-artifact integrity (re-checked at the end of round 6)

- `import merlin` / `from merlin` / `reference_outputs` / `pipeline.execute` / `runtime.reference`
  anywhere under `submission/`: **none**.
- Regular expressions anywhere in the package: **none**.
- `manifest.yaml`: `integrity_exempt: false`, `language: python`, no build block. All 5 declared
  commands have `components:` entries and **every declared path exists**; all 12
  `optimization_surfaces` resolve — every `path` present and every `symbol` found by AST walk of the
  file it names (re-verified this round, not carried forward).
- Merlin authoring artifacts accidentally left in `submission/`: **none**.

## 5f. Round 7 — Merlin tooling re-run, and the one defect it framed

All four mandated calls were made again before the first submission edit, and all returned non-empty:

| call | returned |
|---|---|
| `cca_contract.check_bijection('gemmini')` | `orphan_fields: []`, `orphan_routes: []`, `unclassified: []`, `ladder_errors: []` — still clean |
| `rtl_backend.derived_levers(target_profile('gemmini'))` | the same 7 axes as round 5/6 (`spatial.dataflow`, `spatial.accumulator_resident`, `memory.capacity_fit`, `dispatch.descriptor_reuse`, `dispatch.dma_overlap`, `dispatch.loop_offloaded`, `layout.operand_major`) |
| `rtl.facts.load_facts('gemmini')` | mesh/memory/interface facts incl. the `funct_decode_table` with its 26 legal functs and names |
| `generate.target_repo.generate_skeleton('gemmini')` | 15 relpaths (unchanged; the package layout already follows this shape) |

**`interface_emit.py` was the round's most useful tool, in a new way.** Its
`parse_interface_mlir` / `emit_interface_mlir` round-trip was used to build a GENERALITY PROBE rather
than to author anything: 608 extent-perturbed variants of the corpus, re-emitted and pushed through
the package's own emit path. That probe is what surfaced the round's one real defect. It also has a
documented limit worth recording — `emit_interface_mlir` raises
`unsupported opcode 'MOVEMENT' for interface grammar v0.1`, so it cannot round-trip every shipped
capsule, and the perturbation had to rewrite tensor type spans directly for those.

The defect itself was NOT found by a Merlin call. It came from reading the shipped ISA header
(`gemmini.h:317`, `gemmini_extended2_config_st`) and noticing that `pool_size`/`pool_stride` occupy
**2-bit** fields, then checking what the package does with `pool_size = 4`: `emit_command_buffer`
declared an `on_mesh` COMMIT while `emit_target_artifact` died with
`EncodingError: pool_size=4 does not fit in 2 bits`. The header is the authority that made the bound
derivable rather than guessable; `isa_tools disasm` then confirmed the fix, showing `CONFIG_ST`
decoding `pool_size`/`pool_stride` exactly as the command buffer declares them for the sizes that do
fit, and no instructions at all for the sizes that now decline.

`rtl_checks` in the redacted verdict was read again and produced **nothing actionable**: its 41
`encoded_field_intent` warns are refuted by the oracle (all 41 `mismatch_count: 0`, 36 of them
`L3: pass`), which is a stronger signal than the static checker.

## 5g. Final-artifact integrity (re-checked at the end of round 7)

- `import merlin` / `from merlin` / `reference_outputs` / `pipeline.execute` / `runtime.reference`
  anywhere under `submission/`: **none** (re-scanned this round).
- Regular expressions (`import re` / regex matching) anywhere in the package: **none**.
- The generality probe and the perturbed capsules live in `/tmp`, NOT in `submission/`; the only
  files this round changed are `mlir_oot/target/isa.py`, `mlir_oot/lowering/epilogue.py`,
  `mlir_oot/driver.py` and the three docs.
- Merlin authoring artifacts accidentally left in `submission/`: **none**.

## 6b. One-line summary (round 7)

Merlin tooling did not move the score — nothing could, since the 12 open capsules are harness-side —
but `interface_emit.py`'s round-trip turned the corpus into a 608-variant generality probe that
found a real crash-instead-of-decline defect, and the shipped ISA header supplied the field width
that made the fix derived rather than guessed.

---

# Round 8

## 1. Merlin tools actually invoked (before the first submission edit)

| call | what it returned | what it was used for |
|---|---|---|
| `merlin.kernels.cca_contract.check_bijection('gemmini')` | `orphan_fields: []`, `orphan_routes: []`, `unclassified: []`, `ladder_errors: []` | confirmed the lever set is complete and no phantom route was declared |
| `merlin.targetgen.rtl_backend.derived_levers(target_profile('gemmini'))` | 7 axes incl. `dispatch.dma_overlap` | named the axis this round's work lands on |
| `merlin.targetgen.rtl.facts.load_facts('gemmini')` | mesh 16x16, scratchpad 262144 B / depth 4096, accumulator 65536 B / depth 512, i8 input / i32 accumulator datapaths | the capacity and element-width numbers the block-length derivation rests on |
| `merlin.targetgen.generate.target_repo.generate_skeleton('gemmini')` | 15 relpaths | re-confirmed the package layout against the generated scaffold |
| `merlin.kernels.action_catalog.escalation_ladder('dispatch.dma_overlap','gemmini')` | HEURISTIC rung, seam `<oot_package>/lowering/`, `forkable_now: False`, `needs_new_code: True` | this pointed directly at the seam the round's change was written into |

The ladder call is the one that paid off this round: it named `lowering/` as the seam for
`dispatch.dma_overlap` and said the lever needed new code, which is exactly what
`mlir_oot/lowering/coalesce.py` is.

## 2. What Merlin tooling diagnosed

Nothing in the score — the 12 open capsules are harness-side and no emission can move them (the
`tiers: {}` on every `runner_internal` row proves no tier ran). The useful contribution was
directional: `derived_levers` and the `dispatch.dma_overlap` ladder identified the DMA block
transfer as a real, unbuilt lever on this hardware, and `load_facts` supplied the element widths
that turn `MAX_BYTES 64` into `MAX_BLOCK_LEN 4` for the scratchpad and `1` for the accumulator. The
decisive encoding details — that `block_stride` is a CONFIG_LD field rather than implicitly DIM, and
that `has_acc_bitwidth` gates the accumulator's narrower block bound — came from reading the shipped
RTL (`LoadController.scala`, `DMA.scala`), not from a Merlin call.

## 3. Final-artifact integrity (re-checked at the end of round 8)

- `import merlin` / `from merlin` / `reference_outputs` / `pipeline.execute` / `runtime.reference`
  anywhere under `submission/`: **none** (re-scanned this round).
- Regular expressions (`import re` / regex matching) anywhere in the package: **none**.
- Files changed this round: `mlir_oot/lowering/coalesce.py` (new), `mlir_oot/lowering/contraction.py`,
  `mlir_oot/lowering/kernel.py`, `mlir_oot/lowering/iface_to_gemmini.py`,
  `devtools/simulate.py` (a development aid, not a declared command), `manifest.yaml` and the docs.
- The 455 generality-probe variants live in `/tmp`, NOT in `submission/`.
- Merlin authoring artifacts left in `submission/`: **none**.

## 4. One-line summary (round 8)

Merlin's `derived_levers` + `escalation_ladder` named `dispatch.dma_overlap` and the exact seam to
build it in; the RTL itself supplied the encoding. The result was a 37.6 % cut in emitted load
transfers with zero numeric regressions across 161 passing capsules and 455 synthetic shapes.

---

## Round 9 addendum — what the Merlin tooling did this round

All four mandated calls were run before the first submission edit, and their returned values printed:

| call | returned |
|---|---|
| `merlin.kernels.cca_contract.check_bijection('gemmini')` | `BijectionReport(backend='gemmini', orphan_fields=[], orphan_routes=[], unclassified=[], ladder_errors=[])` — clean, so no leverable axis was left unwired and no phantom route was declared. |
| `merlin.kernels.action_catalog.escalation_ladder('dispatch.loop_offloaded','gemmini')` | one **PASS** rung, `seam_file: <oot_package>/lowering/`, `needs_new_code: True`. |
| `merlin.targetgen.rtl_backend.derived_levers(target_profile('gemmini'))` | `['spatial.dataflow','spatial.accumulator_resident','memory.capacity_fit','dispatch.descriptor_reuse','dispatch.dma_overlap','dispatch.loop_offloaded','layout.operand_major']` |
| `merlin.targetgen.rtl.facts.load_facts('gemmini')` | mesh `DIM=16`; `scratchpad 262144 B / depth 4096`; `accumulator 65536 B / depth 512`; datapaths `input i8 (scratchpad smem UInt<8>)` and `accumulator i32 (AccumulatorMem SInt<32>)`. |
| `merlin.targetgen.generate.target_repo.generate_skeleton('gemmini')` | 15 relpaths (`README.md`, `AGENT.md`, `pyproject.toml`, `CMakeLists.txt`, `docs/`, `contracts/`, `xdsl/`, `include/`, `lib/`, `tools/`, `runtime/`, `zephyr/`, `llvm/`, `examples/`, `tests/`). |

### The one that actually changed the code this round

**`dispatch.loop_offloaded` was the lever this arm's tooling named and this round finally wired.**
Earlier rounds recorded it as an axis `derived_levers` reports but for which this package had *no
backing lever*, so it was deliberately left out of `optimization_surfaces`. That was the honest
statement at the time, and it was also the pointer: the ladder's single **PASS** rung for that axis
names `<oot_package>/lowering/` and `needs_new_code: True` — i.e. "decide whether a loop is offloaded
to the unit, and you will have to write the pass." Round 9's `lowering/sumsq.py` is exactly that pass,
and `optimization_surfaces` now declares the axis against a real symbol (`sumsq.plan`) instead of
omitting it.

**The `rtl_checks` block was the diagnosis.** The four `OC_rmsnorm_*` capsules were *passing* the
numeric, trace, L2 and L3 planes, so no pass/fail signal pointed at them. The only thing that did was
this arm's advisory `rtl_checks`, which reported them as the sole `reject` rows carrying
`T0.decode_clean` — *"empty instruction trace (no RoCC instructions decoded) … backend emitted no
custom-3 .insn"*. Cross-checked against `--offload-census` (`host_only`, `routed_macs: 0`) and the
capsules' own `semantic.must_accelerate: True`, that identified a real hidden-capsule exposure inside
an all-green family. **This is the clearest case so far of the arm's RTL-grounded advisory finding a
defect the functional planes could not report**, and it is the reason the arm exists.

`load_facts` supplied the two numbers the new pass is built on: the i8-operand / i32-accumulator
datapath pair (which is what makes the reduction exact on the mesh) and `DIM = 16` (which is the band
edge the reduction is emitted at). Neither is a literal in `sumsq.py`; both arrive via
`mlir_oot/target/facts.py`.

`isa_tools` `lint` + `disasm` were used on every emitted artifact as before: 191/191 report 0 UNKNOWN,
and `disasm` is how the new MVIN/MVIN2 band offsets were confirmed to decode as
`kind: argbase, arg_index: 0` rather than as baked DRAM addresses.

### Self-containment (unchanged, re-verified)

`submission/` imports no Merlin code. The scan below is over the shipped tree:

- no `import merlin` / `from merlin`
- no `merlin.runtime.reference`, no `simulator`, no `reference_outputs`, no `pipeline.execute`
- no `import re` / regex text-matching anywhere; the interface MLIR is parsed structurally with xDSL

Merlin remained an **authoring** aid only.

## Round 10 addendum — what the Merlin tooling did this round

All mandated calls were run before the first submission edit of the round, and their returned values
printed:

| call | returned |
|---|---|
| `merlin.kernels.cca_contract.check_bijection('gemmini')` | `BijectionReport(backend='gemmini', orphan_fields=[], orphan_routes=[], unclassified=[], ladder_errors=[])` — still clean: no leverable axis unwired, no phantom route declared. |
| `merlin.kernels.action_catalog.escalation_ladder(...)` for `spatial.dataflow`, `dispatch.dma_overlap`, `layout.operand_major`, `dispatch.loop_offloaded` | one rung each — HEURISTIC, HEURISTIC, KNOB, PASS — every one naming `<oot_package>/lowering/` as the seam with `needs_new_code: True`. |
| `merlin.targetgen.rtl_backend.derived_levers(target_profile('gemmini'))` | `['spatial.dataflow','spatial.accumulator_resident','memory.capacity_fit','dispatch.descriptor_reuse','dispatch.dma_overlap','dispatch.loop_offloaded','layout.operand_major']` |
| `merlin.targetgen.rtl.facts.load_facts('gemmini')` | mesh `DIM=16` (256 `Tile` instances, `corroborated: true`); `scratchpad 262144 B / depth 4096`; `accumulator 65536 B / depth 512`; datapaths `input i8 (scratchpad smem UInt<8>)`, `accumulator i32 (AccumulatorMem SInt<32>)`. |
| `merlin.targetgen.generate.target_repo.generate_skeleton('gemmini')` | the same 15 relpaths; the shipped package still follows that shape. |

### What the tooling was worth this round

**`load_facts` is what settled both of the round's changes**, and in both cases it was the *pair* of
declared containers that mattered rather than any single number:

- the `residual_add` factorisation puts integer multipliers on the load units and one rounding
  multiplier on the store path. That this is EXACT rests on `datapaths` declaring the accumulator as
  `SInt<32>` — an integer container wide enough to hold the scaled sum before the single store-path
  rounding. `mlir_oot/target/facts.py:ACC_BYTES_PER_ELEM` carries it; the new
  `_factor_scales` reads its exactness bound from the declared operand containers, not from a
  constant.
- the `movement` fix is entirely a `load_facts` argument: the RTL declares **two** on-chip containers
  (`scratchpad smem UInt<8>` and `AccumulatorMem SInt<32>`), and the old lowering asked only about
  the first. An i32 source is the second one's own container, so the trip is legal in the wide
  container and was being refused to the host for no hardware reason.

**`rtl_checks` was read first, as it has been every round since it appeared.** It went from 18
`reject` rows to 14, confirming that round 9's `OC_rmsnorm_*` work cleared its `T0.decode_clean`
findings. The one remaining `severity: error` is `T0.decode_clean` on `SY_host_only_attention`, whose
own capsule declares `must_accelerate: false`, `lanes.forbid: [on_mesh]` and
`expected.instruction_classes: []` — an empty trace is the required answer, so that row is correctly
left alone. Unlike round 9, `rtl_checks` surfaced no new actionable defect this round; the round's two
findings came from the package's own synthesized attribute sweep instead. Recording that honestly:
the advisory's value this round was as a REGRESSION signal (it showed the previous round's fix had
landed and nothing new had broken), not as a diagnosis.

`isa_tools` `lint` + `disasm` were used on the changed artifacts: 0 UNKNOWN, and `disasm` is how the
scale factorisation was reconciled field by field against the command buffer (`CONFIG_LD` scales
2.0 / 1.0, `CONFIG_ST acc_scale 0.5`, `acc_act` / `relu` carried correctly, row pitch matching the
declared tensors).

### Self-containment (unchanged, re-verified)

`submission/` imports no Merlin code: no `import merlin` / `from merlin`, no
`merlin.runtime.reference`, no `simulator`, no `reference_outputs`, no `pipeline.execute`, and no
`import re` / regex text-matching anywhere — the interface MLIR is parsed structurally with xDSL.
Merlin remained an **authoring** aid only.

### Round 10 postscript — what the advisory confirmed

The round's two code changes were screened locally and then **confirmed by the arm's own signals** on
the next official grade: `GR0_resadd_i8` and `GR1_resadd_relu_i8` both reached `L3: pass` with
`mismatch_count: 0`, so the elaborated RTL agrees with the scale factorisation that the local
zero-tolerance datapath model could only suggest.

`rtl_checks` moved 52 -> 54 `warn` (rejects unchanged at 14). The two added rows are
`T0.encoded_field_intent / config_scale` on exactly those two capsules, and they are a checker
artefact rather than a defect: the check reads the emitted `CONFIG_ST acc_scale` against the
declaration the command buffer makes for that command, and a `residual_add` buffer declares
`lhs_scale` / `rhs_scale` / `bound_lsb` with no field for a store-path factor. Both capsules pass the
hardware plane, so the buffer was NOT given an invented field to quiet the advisory. Recorded because
"the advisory got slightly worse while the hardware got stricter agreement" is exactly the kind of
result worth writing down rather than smoothing over.
