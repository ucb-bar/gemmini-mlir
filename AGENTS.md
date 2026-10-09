# Compiler implementation ownership

This OOT MLIR dialect repository owns **all target-specific code**: Gemmini dialect
operations and verifiers, instruction encodings, device kernels and schedules, hardware
layout/resource facts, target ABI glue, target simulation and execution support.

**Merlin owns reusable infrastructure:** host code generation, packing, requantization,
graph/global optimizations, dispatch, buffer ownership, device compilation orchestration,
and runtime. An optimization selectable independently of the accelerator belongs in Merlin,
even when this target provided its first performance result. Split mixed work at an explicit
contract: generic algorithm and semantic checks in Merlin; hardware facts and instruction
implementation here. Promote generic prototypes into Merlin and delegate from OOT; do not
maintain duplicate shared implementations here or add target-specific branches to core.

## General compiler behavior

The OOT MLIR dialect is a **compiler backend, not a workload-specific kernel generator**.
Production transforms and lowering derive choices from operation semantics, shapes,
layouts, numeric contracts and declared hardware capabilities. Do not select behavior by
model name, captured provenance ID, golden output or benchmark constants. Constant/shape
specialization must derive from the current input IR with legality and resource checks.
Source-bound model selections belong in experiment drivers. Promote winning strategies
into general passes and cost models; qualify independent shapes, spatial/channel tails,
numeric policies and fallback/refusal cases. Performance targets motivate optimization;
they do not justify replacing model computation or baking in the benchmark.
Provenance IDs remain valid for traceability and exact source-to-device binding; they
must not select the optimization strategy.

Preserve explicit numeric policy selection, immutable original model accuracy gates, exact
source/catalog bindings, and final-ELF instruction audits when moving an implementation.
Keep measured optimization notes and negative evidence; simulator instruction counts,
analytical floors, kernel cycles and full-model FireSim cycles are different quantities.

## Publication authorization

Do not open a pull request in any repository without explicit user approval.
An instruction to upstream or push code does not authorize creating a PR.
Preserve reviewed changes on the user-authorized branch and follow the requested
direct-main integration workflow for Merlin and model2MLIR.

## Integrated worktree cleanup

After useful changes are reviewed, integrated and verified on the requested
remote branch, remove obsolete delivery/review worktrees that have no active
agent, process, experiment or artifact dependency. Preserve every working-tree
file and link, including ignored and untracked outputs, in a verified archive
with file hashes and explicit original-path-to-archive-member mappings. Retain
shared Git objects, recovery refs, primary Git stores and active source/runtime
owners. Immutable historical receipts keep their original bytes; removed paths
need recorded archive successors. Report logical savings separately from
filesystem free space. Do not remove an active tree merely because its branch
has been merged.
