"""Opt-in source private storage planner; bindings remain compiler-owned."""
from merlin.llvmlower.source_storage_lifetimes import (
    StorageRequest, EmittedStorageRequest, plan_source_storage, plan_emitted_storage,
)

ENABLED = False


def plan(module, **contracts):
    """Return no aliases by default; enabled planning requires explicit effects."""
    if not ENABLED:
        return None
    return plan_source_storage(module, **contracts)


def plan_emitted(function, **contracts):
    """Use actual emission liveness; source order is attribution only."""
    if not ENABLED:
        return None
    return plan_emitted_storage(function, **contracts)
