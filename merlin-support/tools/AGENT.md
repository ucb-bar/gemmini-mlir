# Gemmini reference-program tools

`baremetalc_corroborate.py` owns the reference-program corroboration interface
declared by this provider's `plugin.reference_programs`. It builds upstream
bareMetalC programs, executes explicitly configured simulators, and compares
outputs against independent Tensor goldens.

These are host-owned reference oracles, never evaluated compiler payloads.
Select this support provider with `MERLIN_TARGET_PATH` before use. Shared Merlin
loads the declared tool by its plugin key; it must not name this implementation.
Imports remain absolute or provider-relative without ambient `sys.path` changes.
Source relocation and import tests do not authorize simulator execution or certify
hardware results. Historical migration hashes are immutable.
