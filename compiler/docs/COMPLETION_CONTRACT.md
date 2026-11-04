# Backend completion composition contract

Production: `dense.plan_matmul` -> existing scheduler/retained helper -> `completion_adapter.finish_completion` -> `apply_completion` -> returned result's original index/effect emitter. The adapter manifest is `COMPLETION_ADAPTER.json`.

`apply_completion(original, context)` accepts a mapping or an attribute context. Required fields are `problem`, `block`, `resources`, `storage`, `adapter`, `admission`, and `premises`; `premises` is a plain dictionary of Boolean laws. Optional `completion_plan` is a typed xDSL scheduling module (a parsed module is accepted). When omitted, the entry creates the same explicit decision from the supplied context and consumes it. An explicitly selected `original` policy stays unchanged. Extra or ignored decisions, a foreign binding and stale context refuse.

The admission retains the original typed hardware/execution/descriptor/configuration/pending fields. Its compiler issued bounds must cover each new completion scope. Production `make_completion_context` computes and records exact finite counts from the existing program, then admits those bounds. The public entry never silently increases an insufficient caller budget. No physical queue gains capacity.

The new `schedule.completion_placement` operation has `target`, `policy`, and three i64 `issued_bounds` (compute pairs, loads, readouts). Its binding includes the original actual Kernel digest and problem/block/resource/storage/adapter/premise identities. It is a post-construction decision: the preceding dense schedule's family, blocking, retention and readout decisions remain intact. Actual output Kernel, not the decision text, is emitted. The receipt includes consumed decision text, original/actual/data digests, original and retained dynamic drains, protected final-K scopes, issued counts and the load-reuse proof result.

The returned optimized `DenseCompletionResult` owns the actual checked Kernel. `revalidate`, `materialize_scf`, `emit_index_program`, and `emit_llvm` retain its custody. Revalidation checks the original owner, exact transformation, selected decision/context, pass source and boundary-analysis source. Diagnostic receipt dictionaries do not authorize emission.

Fallback preserves the original actual Kernel and its emitter. `completion_placement.status` is `unchanged`, with a categorized reason. This includes unsupported owners such as cross-call resident handles; no valid original route is discarded to satisfy a completion consumer. A malformed input is not represented as a newly proved program.

The local interval/ordering proof is conditional on the common selected premises. It supplies no hardware certificate, cycle spacing, token, external kernel or performance target. All readout, output-final-K and final boundaries are preserved.
