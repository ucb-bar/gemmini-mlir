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

Preserve explicit numeric policy selection, immutable original model accuracy gates, exact
source/catalog bindings, and final-ELF instruction audits when moving an implementation.
Keep measured optimization notes and negative evidence; simulator instruction counts,
analytical floors, kernel cycles and full-model FireSim cycles are different quantities.
