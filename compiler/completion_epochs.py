"""Selected-fact adapter for completion placement in checked dense K streams.

This consumer runs after the incumbent's layout/retention/readout decisions.
It changes only explicit completion calls. Hardware order is conditional on
pinned selected premises; the library separately proves the program's reader
chain and unchanged data projection. No physical qualification is asserted.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from merlin.compile.scheduling.dense_panel_epochs import (
    DensePanelEpochLaw, schedule_dense_panel_epochs,
)
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN, DensePanelPipelineResult
from merlin.compile.scheduling.block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.sched.ir import Call, Loop

COMPLETION = Call(DRAIN, ())

# Pins are the shared public inputs, not inferred properties of a candidate.
PREMISES_SHA256 = "f59ff7d4dcca5ca65414e7373b9814bef641be8c2a1a36d4a52a1d10bff9ebfb"
ORDERING_SHA256 = "f941f5e2199b59f1d6649739682b66081958cbb0e3b35dabc839913c86a80970"
FACTS_SHA256 = "b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af"
HARDWARE_STREAM_FLAGS = (
    "local_read_after_write", "local_write_after_read",
    "ordered_accumulator_writes", "ordered_configuration_capture", "finite_queue_progress",
)


@lru_cache(maxsize=1)
def selected_premises():
    """Unknown, missing, stale or false selected premises never remove a drain."""
    root = Path(__file__).resolve().parent
    try:
        raw = (root / "completion_evidence/COMPLETION_PREMISES.json").read_bytes()
        ordering_raw = (root / "completion_evidence/COMMAND_ORDERING_SELECTION.json").read_bytes()
        facts_raw = (root / "rtl_facts/facts.json").read_bytes()
        if (sha256(raw).hexdigest() != PREMISES_SHA256
                or sha256(ordering_raw).hexdigest() != ORDERING_SHA256
                or sha256(facts_raw).hexdigest() != FACTS_SHA256):
            return None
        selected, ordering = json.loads(raw), json.loads(ordering_raw)
        if (selected.get("schema") != "merlin.public_completion_premises.v1"
                or selected["inputs"]["facts"]["sha256"] != FACTS_SHA256
                or selected["inputs"]["ordering"]["sha256"] != ORDERING_SHA256
                or selected["ordering_identity"] != ordering["ordering_sha256"]):
            return None
        premises = selected["premises"]
        if any(premises[flag].get("status") != "fact"
               or premises[flag].get("value") is not True for flag in HARDWARE_STREAM_FLAGS):
            return None
        facts = json.loads(facts_raw)["facts"]
        mesh = next(a for a in facts["arrays"] if a["name"] == "mesh")
        memories = {a["name"]: a for a in facts["memories"]}
        return {**selected, "adapter_geometry": {
            "tile_m": mesh["rows"], "tile_k": mesh["rows"], "tile_n": mesh["cols"],
            "lhs_bytes": memories["scratchpad"]["elem_bits"] // 8,
            "rhs_bytes": memories["scratchpad"]["elem_bits"] // 8,
            "accumulator_bytes": memories["accumulator"]["elem_bits"] // 8,
        }}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def stream_scope(problem, block, resources, storage, epoch_scope="output_block"):
    """Bound issued work of ONE output block, and its simultaneously live rows.

    K count multiplies issued commands, never live allocation or pending queues.
    The entire checked stream has K*(BM*BN) pairs and K*(BM+BN) operand loads.
    Initial load epochs may need the incumbent's two-panel load budget instead.
    Tails issue no more work and occupy no more rows than these padded bounds.
    """
    selected = selected_premises()
    if selected is None or block.k != 1 or problem.tiles[1] < 2:
        return None
    if epoch_scope not in ("paired_steps", "output_block"):
        return None
    geometry = selected["adapter_geometry"]
    if any(getattr(problem, name) != value for name, value in geometry.items()):
        return None
    lhs_rows = 2 * block.m * problem.tile_m
    rhs_rows = 2 * block.n * problem.tile_k
    acc_rows = block.m * block.n * problem.tile_m
    if (lhs_rows > storage.lhs.capacity or rhs_rows > storage.rhs.capacity
            or acc_rows > storage.accumulator.capacity
            or (lhs_rows * storage.lhs.unit_bytes + rhs_rows * storage.rhs.unit_bytes
                > resources.operand_bytes)
            or acc_rows * storage.accumulator.unit_bytes > resources.accumulator_bytes
            or not storage.admits(problem, block, resources)):
        return None
    views = (storage.lhs, storage.rhs, storage.accumulator)
    for i, left in enumerate(views):
        for right in views[i+1:]:
            if (left.backing == right.backing
                    and left.base*left.unit_bytes < (right.base+right.capacity)*right.unit_bytes
                    and right.base*right.unit_bytes < (left.base+left.capacity)*left.unit_bytes):
                return None
    # The selected adapter uses distinct canonical operand/accumulator stores.
    if storage.accumulator.backing in (storage.lhs.backing, storage.rhs.backing):
        return None
    kt = problem.tiles[1]
    issued_kt = min(kt, 2) if epoch_scope == "paired_steps" else kt
    return dict(scope=("two adjacent original K steps" if epoch_scope == "paired_steps"
                       else "one original output block, complete original K stream"),
                completion_epoch_scope=epoch_scope, issued_k_tiles=issued_kt,
                k_tiles=kt, block_m_tiles=block.m, block_n_tiles=block.n,
                max_issued_compute_pairs=issued_kt*block.m*block.n,
                max_issued_loads=max(2,issued_kt)*(block.m+block.n),
                live_lhs_rows=lhs_rows, live_rhs_rows=rhs_rows, live_accumulator_rows=acc_rows,
                live_operand_bytes=lhs_rows*storage.lhs.unit_bytes+rhs_rows*storage.rhs.unit_bytes,
                live_accumulator_bytes=acc_rows*storage.accumulator.unit_bytes)


def verify_completion_policy(original, actual, adapter):
    """Prove the task's boundary policy on the actual counted command trees.

    Retain original scopes and every data operand, allow only original drain
    deletions, protect all readout/initialization epochs and every pure-compute
    epoch (including final K before readout), and require a closing full drain.
    This is program evidence, independent of selected hardware order and time.
    """
    old_completions, new_completions, protected = set(), set(), set()
    kinds_at_completion = {}

    def effects(body):
        for statement in body:
            if type(statement) is Loop:
                yield from effects(statement.body)
            elif statement != COMPLETION:
                yield adapter.project(statement)

    def collect(body, path, completions, protect=False):
        for statement in body:
            if type(statement) is Loop:
                child_path = (*path, statement.var)
                if statement.body and statement.body[-1] == COMPLETION:
                    kinds = {type(op) for op in effects(statement.body)}
                    if protect:
                        kinds_at_completion[child_path] = kinds
                        if (kinds & {Store, InitializeAccumulator}
                                or kinds and kinds <= {Preload, Compute}):
                            protected.add(child_path)
                collect(statement.body, child_path, completions, protect)
            elif statement == COMPLETION:
                completions.add(path)

    def same_data(left, right):
        left = tuple(s for s in left if s != COMPLETION)
        right = tuple(s for s in right if s != COMPLETION)
        if len(left) != len(right):
            return False
        for before, after in zip(left, right):
            if type(before) is not type(after):
                return False
            if type(before) is Loop:
                if ((before.var, before.extent) != (after.var, after.extent)
                        or not same_data(before.body, after.body)):
                    return False
            elif before != after:
                return False
        return True

    def last_call(body):
        if not body:
            return None
        statement = body[-1]
        if type(statement) is Loop:
            return last_call(statement.body) if statement.extent > 0 else None
        return statement

    collect(original.body, (), old_completions, True)
    collect(actual.body, (), new_completions)
    if (not same_data(original.body, actual.body)
            or not new_completions <= old_completions
            or not protected <= new_completions
            or last_call(original.body) != COMPLETION
            or last_call(actual.body) != COMPLETION):
        raise ValueError("completion placement changed data, a required boundary or the closing drain")
    removed = old_completions - new_completions
    if any(not kinds_at_completion.get(path)
           or not kinds_at_completion[path] <= {Load, Preload, Compute} for path in removed):
        raise ValueError("completion placement removed a boundary outside a pure operand/K stream")
    return dict(data_calls_operands_scopes_and_order_equal=True,
                only_original_completions_removed=True,
                original_completion_sites=len(old_completions),
                actual_completion_sites=len(new_completions),
                protected_sites=[list(path) for path in sorted(protected)],
                removed_sites=[list(path) for path in sorted(removed)],
                all_readout_initialization_and_pure_compute_boundaries_kept=True,
                closing_full_completion_present=True)


def merge_panel_completions(original, problem, block, resources, storage, adapter, epoch_scope="output_block"):
    """Keep incumbent on unsupported construction or any unproved obligation.

    Program obligations: this backend emits the entire closed kernel inline,
    without interleaving another device task; the original checked views own all
    local effects until its output-block drain. Entry and final fences bound the
    invocation. The SAME Builder.fence callback emits the real RISC-V `fence`
    used by the drained route and selected header's gemmini_fence. Configuration
    is set once before the stream and is never changed by these data calls.
    """
    scope = stream_scope(problem, block, resources, storage, epoch_scope)
    if (scope is None or type(original) is not DensePanelPipelineResult
            or original.bias_source is not None
            or (original.realization.get("schedule_ir") or {}).get("pipeline_kind") != "two_window"):
        return original
    if (original.admission.max_issued_compute_pairs < scope["max_issued_compute_pairs"]
            or original.admission.max_issued_loads < scope["max_issued_loads"]):
        return original
    selected = selected_premises()
    law = DensePanelEpochLaw(
        adapter.configuration_identity, adapter.numerical_policy_identity,
        selected["ordering_identity"],
        **{name: True for name in HARDWARE_STREAM_FLAGS},
        complete_device_drain=True, exclusive_private_state=True,
        # Hardware visibility is true in the supplied table, but this study
        # explicitly preserves every original readout boundary. Do not use it.
        accumulator_result_visibility=False,
    )
    try:
        result = schedule_dense_panel_epochs(original, problem, block, resources,
            storage, adapter, original.admission, law,
            static_budget=128, timeout_ms=15000, max_read_instances=4096)
        policy = verify_completion_policy(original.kernel, result.kernel, adapter)
    except ValueError:
        # A checker refusal is an unchanged route, never a proof by larger bound.
        return original
    result.realization["completion_adapter"] = dict(
        scope, program_boundary_proof=policy, selected_premises_sha256=PREMISES_SHA256,
        ordering_sha256=ORDERING_SHA256, facts_sha256=FACTS_SHA256,
        hardware_accumulator_visibility=selected["premises"]["accumulator_result_visibility"],
        readout_merge_policy=False,
        program_drain="same Builder.fence callback as incumbent; original readout/block/final drains retained",
        program_ownership="closed inline kernel; disjoint checked private views; no interleaved device effects",
        caveats=selected["coverage_caveats"],
        qualification="selected ordering semantics; RTL certification not established")
    return result
