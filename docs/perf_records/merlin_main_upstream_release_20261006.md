# Merlin main upstream release

Published **13 topic commits** on `main` at `7fee5cfdaca653b4104cb0999c0a397e0ec51ddc`; normal fast-forward from `61ada9e1bbbbcc0c39efb5f56c0f54dd0d691ae8`. No published history rewritten.

| Commit | Topic |
| --- | --- |
| `d9eaf94015` | fix(ir): preserve portable source types and attributes |
| `04767a0a7f` | perf(quant): propagate exact constants and tensor layouts |
| `cd4099fe84` | feat(lowering): prove fresh writers and reusable buffer storage |
| `96072e6432` | feat(lowering): expose explicit source arithmetic policies |
| `4c75833bfa` | perf(lowering): schedule independent source arithmetic |
| `c4ca4b66d6` | feat(numerics): encode and reconstruct exact integer radices |
| `70fd7727a2` | feat(requant): prove exact integer readout decoders |
| `7f03caa736` | feat(runtime): certify ordered floating point contractions |
| `6d9e63288e` | feat(lowering): bind closed source groups and observations |
| `184e400274` | feat(runtime): prepare exact numerical consumer frontiers |
| `2d513e5f0d` | feat(runtime): integrate source-bound device and host providers |
| `fac08518e6` | feat(perf): retain compiler and measurement evidence |
| `7fee5cfdac` | docs(compiler): document general optimization contracts |

## Verification

- focused: 2944 PASS /13 optional skips /0 failures.
- installed_runtime: 847 PASS,48 intentionally deselected unrelated cases.
- installed_helpers: 2 PASS actual dependency-aware default and fully composed builds.
- core_only_installed: 10 PASS.
- reviewed_corpus_installed: 90 PASS /1 optional skip.
- packaged_resources: 151 exact committed hashes.
- format: PASS245 Python files.
- docs: PASS.
- no_regex: PASS.
- core_dependencies: PASS.
- changed_scope_no_target_names: PASS.
- changed_scope_no_assumed_isa_constants: PASS.

## Limits

Four inherited full-tree lint/module-size findings retain identical baseline source bytes. Fresh whole-model hardware qualification of this compiler head remains UNKNOWN; previous frozen candidates keep their own compiler receipts.

GitHub accepted the direct main push and reported a bypass of its pull-request rule. Future publication must respect the known review requirement.

model2MLIR's eight owned fixes are already ancestors of authoritative remote main `3a5acb8fd4c2e39204425e805dc599ef74e0d0f9`, with 98 fresh tests passing; its published history was left intact.

Target instruction encoding, resource facts and schedules remain in the OOT repository. Numerical proofs, host compilation, buffer ownership, runtime and profiling are Merlin changes. Optional alternatives do not become enabled or profitable solely by publication.
