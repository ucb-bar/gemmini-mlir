# Pointwise mesh mode: weak result, stationary default retained

The permitted reference executable uses mesh FLIP for every compute in a comparable M784/N512/K128 pointwise contraction. Our current resident-A implementation uses one FLIP then stationary STAY across M. The experiment changes only the existing `reuse_b` option, preserving reduction order, arithmetic/scale, tile geometry, cached A, B prefetch, DMA and output layout.

| Complete common-address GSIM capsule | Stationary control | All FLIP | Change |
|---|---:|---:|---:|
| Original M784/N512/K128, 401,408 i8 outputs | 298,600 | 297,401 | −0.402% |
| Independent M257/N33/K65, 8,481 i32 outputs and tails | 13,394 | 13,434 | +0.299% |

Every output, 4,096 guard bytes and both immutable operands pass strict Spike and GSIM. Final executables pass zero-FSM audits. The complete original native capture also retains all 1,000 original binary32 words with zero tolerance. This small single-pair change does not establish a material general scheduling win. Keep the stationary default; no whole build or stock submission follows this hypothesis.

The complete source-bound static CFG counts change from 256 FLIP and 12,288 STAY to 12,544 FLIP. Both arms retain 648 mvin, 12,544 preload/compute pairs and 1,568 mvout. Counts alone do not explain timing; physical traffic and CPU/accelerator overlap remain unproved.

## Correct binding and preserved failed first attempt

The first driver guessed a symbol ordinal: `matmul_13` was treated as `requant13`. The sealed catalog actually binds it to `requant12`; `requant13` has a different K extent and scale. Strict execution failed immediately (`i0 got-2 want4`), so that attempt has no qualified GSIM timer. Its source, fixture and failed output are retained in the receipt.

The corrected capture resolves the actual sealed catalog row, verifies all four declared tensor dimensions and element types, and matches the numeric proof and schedule to the target manifest and current 1903 profile. Its control is the byte-identical current `gemmini_exact_requant_12_kernel` object (`cb1b4785…522909`). Explicit `.text` and relocation section hashes also agree. No ordinal, source-region name or observed operand value determines a strategy. The normal compiler is unchanged.

The 150-pin receipt preserves both complete pairs and the initial refusal. Reclose it with:

```sh
PYTHONPATH=/scratch/agustin/tmp/gemmini-mesh-flip-screen-20261006:/scratch/agustin/tmp/merlin-residual-output-word-20261005/src \
  /scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  tests/captured_schedule_closure_probe.py \
  docs/perf_records/mesh_flip_source_bound_capsules.json
```

GSIM and stock FireSim have different memory regimes. Reference operands/numeric contracts and timing wrappers also differ; reference times are hypothesis evidence, not matched-control times. Child token attribution is unavailable; root owns accounting.
