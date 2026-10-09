# Current whole-model profiling recipes

These are frozen root experiment recipes for owned artifacts at the absolute
paths recorded in their manifests. They are not installed compiler commands.
The checked result is stock2080, a diagnostic of the accepted Tiny2076 objects.

The first build incorrectly expected `func.call` in the catalog snapshot. That
snapshot contains typed `linalg.generic` contractions, so it refused before
profile linking. The retained baseline had already reproduced byte exactly.
The corrected recipe binds source operation ordinals, complete tensor types,
contraction semantics and catalog entries, then wraps the five actual external
primitive entries in the target shim. The host `merlin_dev_*` functions have a
different expanded ABI and cannot be wrapped as three-pointer primitives.

All 155 execution IDs/order, source objects, original output words, final
instruction audit, queue terminal and pre-teardown staged identities close.
Source provenance labels only explain measured locations; they select no
production transform. Costs include profiling overhead. Neither callback nor
gap time is pure accelerator or CPU utilization.

Phase 0 should record exact runtime/toolchain identity and terminal protocol.
Phase 1 should expose typed source-to-selected-entry relations and refuse
snapshot/ABI ambiguity. Phase 2 should carry those relations through object
ownership and actual link selection, conserve complete intervals, and retain
an uninstrumented paired measurement. Analytical estimates must expose unknown
observer, replay, preparation and transfer costs rather than subtract old
callback measurements from a new model.
