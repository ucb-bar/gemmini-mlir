# Pure build support

Target-owned C caller formatting and storage-word representation only. No candidate
imports, backend discovery, golden/reference execution, simulator, tool discovery or
ambient hardware policy. Generic build services load this package by its exact pinned
path. Legacy backend wrappers inject their existing layout/measurement callbacks.

Do not duplicate harness semantics here: the legacy caller delegates to this one
renderer, with generated-byte parity tests. Explicit storage preserves host prepack
authorization checks and does not authorize arithmetic or change tensor values.

An explicit trusted full-value readback policy may change only the whole-program
output transport, leaving the command buffer and default rendered source
unchanged. The generic codec comes from Merlin; this target-owned adapter
supplies its existing buffer layout and storage-word types. It must refuse
non-whole-program callers and may not infer a policy from capsule params.
The binary alternative delegates packing to Merlin's separate codec and uses
the provider's explicit-length coherent SYS_write callback; the B64 path and
default caller stay unchanged. A receipt and full output-frame check are still
required before numerical comparison.
