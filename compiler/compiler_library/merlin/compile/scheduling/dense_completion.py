"""Completion placement over an already checked dense instruction Kernel.

This pass changes only drain calls inside counted reduction streams.  It does
not rebuild a dense schedule or enlarge the old disjoint-epoch admission.  The
new checker uses a separate ordered-access law and the actual issued scope.
Target facts, a full completion implementation, and private ownership are
independent inputs; capacity or an instruction count cannot supply them.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.sched.ir import Call, Loop, Kernel, check_structure, evaluate
from merlin.sched.ir.expr import free_vars
from merlin.transforms.pipeline_state import PassSpec
from .block_schedule import Compute, Load, Preload, Store
from .dense_panel_pipeline import DRAIN


COMPLETION_PASS = PassSpec("ordered_dense_k_completion", 60,
    requires=("checked_dense_source", "selected_ordering", "private_full_drain"),
    produces=("checked_completion_placement",), ir_level="dense_kernel")


class CompletionNotApplicable(ValueError):
    """Unknown or unproved premises require the original drained program."""


@dataclass(frozen=True)
class DenseCompletionLaw:
    hardware_identity: str
    ordering_identity: str
    numerical_policy_identity: str
    local_read_after_write: bool | None
    local_write_after_read: bool | None
    ordered_accumulator_writes: bool | None
    ordered_configuration_capture: bool | None
    finite_queue_progress: bool | None
    pending_loads: int
    pending_compute_pairs: int
    operand_row_bytes: int
    coverage_caveats: tuple[str, ...] = ()

    def require(self, adapter):
        if (self.hardware_identity != adapter.configuration_identity
                or self.numerical_policy_identity != adapter.numerical_policy_identity
                or not self.ordering_identity
                or any(x is not True for x in (self.local_read_after_write,
                    self.local_write_after_read, self.ordered_accumulator_writes,
                    self.ordered_configuration_capture, self.finite_queue_progress))
                or type(self.pending_loads) is not int or self.pending_loads < 1
                or type(self.pending_compute_pairs) is not int or self.pending_compute_pairs < 1
                or type(self.operand_row_bytes) is not int or self.operand_row_bytes < 1
                or adapter.exact_accumulator_continuation is not True):
            raise CompletionNotApplicable("selected ordered-access conjunction is incomplete")


@dataclass(frozen=True)
class DenseCompletionOwnership:
    """Program owner obligations, checked separately from selected hardware."""
    source_kernel_sha256: str
    exclusive_storage: bool
    full_drain_implementation: str
    serialized_invocation: bool

    def require(self, kernel):
        if (self.source_kernel_sha256 != kernel.digest()
                or self.exclusive_storage is not True
                or self.serialized_invocation is not True
                or not self.full_drain_implementation):
            raise CompletionNotApplicable("private storage or real closing completion is unproved")


def _calls(body):
    for s in body:
        if type(s) is Loop:
            yield from _calls(s.body)
        else:
            yield s


def _erase_drains(body):
    return tuple(replace(s, body=_erase_drains(s.body)) if type(s) is Loop else s
                 for s in body if type(s) is Loop or s.instr != DRAIN)


def _dynamic_calls(body):
    return sum(s.extent * _dynamic_calls(s.body) if type(s) is Loop else 1 for s in body)


def _require_plain_loops(body):
    for s in body:
        if type(s) is Loop:
            if s.iter_args or s.yields or s.results:
                raise CompletionNotApplicable("carried events require their original completion consumer")
            _require_plain_loops(s.body)


def _instances(body, env=None):
    env = {} if env is None else env
    for s in body:
        if type(s) is Loop:
            for i in range(s.extent):
                yield from _instances(s.body, {**env, s.var: i})
        else:
            yield replace(s, args=tuple((k, evaluate(v, env)) for k, v in s.args))


def _propose_inner(kernel, adapter):
    """Recognize reduction loops by projected operands, never their names."""
    changes = []

    def visit(body, path=(), multiplicity=1):
        result = []
        for s in body:
            if type(s) is not Loop:
                result.append(s)
                continue
            calls = tuple(_calls(s.body))
            try:
                effects = [adapter.project(c) for c in calls if c.instr != DRAIN]
            except (ValueError, KeyError) as error:
                raise CompletionNotApplicable("unknown projected effect in completion stream") from error
            computes = [op for op in effects if type(op) is Compute]
            preloads = [op for op in effects if type(op) is Preload]
            drains = sum(c.instr == DRAIN for c in calls)
            # Only repeated K streams with no stores, initialization, config,
            # callbacks or externally visible effects.  Initial staging and
            # the later readout/block closure remain in their original places.
            if (s.extent > 1 and drains and computes
                    and all(type(op) in (Load, Preload, Compute) for op in effects)
                    and all(s.var in free_vars(op.input_block[1])
                        and s.var not in free_vars(op.input_block[0]) for op in computes)
                    and all(s.var not in free_vars(op.weight_block[1])
                        and s.var not in free_vars(op.accumulator_row) for op in preloads)):
                actual = replace(s, body=_erase_drains(s.body))
                removed = _dynamic_calls(s.body) - _dynamic_calls(actual.body)
                changes.append(dict(path=(*path, s.var), extent=s.extent,
                    removed_dynamic_drains=multiplicity*s.extent*removed))
                result.append(actual)
            else:
                result.append(replace(s, body=visit(s.body, (*path, s.var), multiplicity*s.extent)))
        return tuple(result)

    return replace(kernel, body=visit(kernel.body)), changes


def _propose(kernel, adapter, mode):
    inner, inner_changes = _propose_inner(kernel, adapter)
    if mode == "inner_stream" or not inner_changes:
        return inner, inner_changes
    if mode != "ordered_stream":
        raise ValueError("unknown completion proposal mode")
    changes = []

    def pure(body, allowed):
        try:
            calls = tuple(_calls(body))
            return bool(calls) and all(c.instr == DRAIN or type(adapter.project(c)) in allowed for c in calls)
        except (ValueError, KeyError):
            return False

    def visit(body, path=(), multiplicity=1):
        result = []
        i = 0
        while i < len(body):
            start = i
            while i < len(body) and type(body[i]) is Loop and pure(body[i].body, (Load, Preload, Compute)):
                i += 1
            group = body[start:i]
            # A complete pure K stream ends at an original drained readout.
            # Its last compute epoch, every readout and the block closure stay.
            followed_by_readout = (i < len(body) and type(body[i]) is Loop
                and pure(body[i].body, (Store,)))
            eligible = (len(group) > 1 and followed_by_readout
                and any(change["path"][:len(path)+1] == (*path, op.var)
                    for op in group for change in inner_changes)
                and any(c.instr == DRAIN for c in _calls(group[-1].body)))
            if eligible:
                for op in group[:-1]:
                    actual = replace(op, body=_erase_drains(op.body))
                    removed = _dynamic_calls(op.body) - _dynamic_calls(actual.body)
                    if removed:
                        changes.append(dict(path=(*path, op.var), extent=op.extent,
                            removed_dynamic_drains=multiplicity*op.extent*removed))
                    result.append(actual)
                result.append(group[-1])
            else:
                for op in group:
                    result.append(replace(op, body=visit(op.body, (*path, op.var), multiplicity*op.extent)))
            if i < len(body):
                op = body[i]
                result.append(replace(op, body=visit(op.body, (*path, op.var), multiplicity*op.extent))
                    if type(op) is Loop else op)
                i += 1
        return tuple(result)

    return replace(kernel, body=visit(kernel.body)), changes


def _protected_drain_positions(kernel, adapter):
    """Original readout/block/final completions, counted against data position."""
    protected = Counter()
    position = 0
    pending = 0
    previous_store = False
    for call in _instances(kernel.body):
        if call.instr == DRAIN:
            pending += 1
            continue
        store = type(adapter.project(call)) is Store
        if previous_store or store:
            protected[position] += pending
        pending = 0
        position += 1
        previous_store = store
    protected[position] += pending
    return protected


def verify_ordered_dense_completion(original, actual, storage, adapter, law, ownership,
                                    *, max_dynamic_calls=1_000_000):
    """Verify every issued instance under the separate selected ordering law.

    A load-load WAW is NOT assumed ordered.  Each rewritten local row must have
    been read since its previous write: older load -> consuming read (RAW) ->
    younger load (WAR).  These row-wise transitive edges cover reuse even when
    load completion is out of order.  ACC commits use the separate in-order
    execution premise.  Readouts require an original real drain before them.
    The finite expansion is a compiler proof budget; larger programs fall back.
    """
    law.require(adapter)
    ownership.require(original)
    if check_structure(original) or check_structure(actual):
        raise CompletionNotApplicable("invalid dense Kernel")
    _require_plain_loops(original.body)
    if replace(actual, body=_erase_drains(actual.body)) != replace(original, body=_erase_drains(original.body)):
        raise CompletionNotApplicable("completion placement changed non-completion structure or operands")
    total = _dynamic_calls(actual.body)
    if max(total, _dynamic_calls(original.body)) > max_dynamic_calls:
        raise CompletionNotApplicable("ordered completion instance proof budget exceeded")
    views = {"lhs": storage.lhs, "weight": storage.rhs, "acc": storage.accumulator}
    if (len(original.args) != 3 or tuple(a.dtype for a in original.args) != ("i8", "i8", "i32")
            or any(v is None for v in views.values())
            or storage.lhs.backing != storage.rhs.backing
            or storage.accumulator.backing == storage.lhs.backing
            or storage.lhs.unit_bytes != law.operand_row_bytes
            or storage.rhs.unit_bytes != law.operand_row_bytes
            or storage.accumulator.unit_bytes != 4 * law.operand_row_bytes
            or not (storage.lhs.base + storage.lhs.capacity <= storage.rhs.base
                    or storage.rhs.base + storage.rhs.capacity <= storage.lhs.base)):
        raise CompletionNotApplicable("operand assignments are not separate private windows")
    # Each row denotes its full hardware row, conservatively ignoring columns.
    written, consumed = set(), set()
    actual_drain_positions = Counter()
    data_position = 0
    high_water = {"lhs": 0, "weight": 0, "acc": 0}
    issued = dict(loads=0, compute_pairs=0, readouts=0)
    maxima = dict(issued)
    epochs = []
    pending = None
    has_uncompleted_compute = False
    dynamic_drains = 0
    row_reuse_edges = 0
    previous = None

    def primitive(*dimensions):
        if any(type(v) is not int or not 0 < v <= law.operand_row_bytes for v in dimensions):
            raise CompletionNotApplicable("wide or unknown primitive transfer geometry")

    def rows(role, start, count):
        view = views[role]
        if (type(start) is not int or type(count) is not int or count <= 0
                or start < view.base or start + count > view.base + view.capacity):
            raise CompletionNotApplicable("actual local access escapes original live storage")
        high_water[role] = max(high_water[role], start + count - view.base)
        return {(view.backing, i) for i in range(start, start + count)}

    for call in _instances(actual.body):
        if any((call.unit, call.produces, call.awaits, call.stages, call.sets, call.assumes)):
            raise CompletionNotApplicable("annotated effects need a separate completion consumer")
        if call.instr == DRAIN:
            if call.args or pending is not None:
                raise CompletionNotApplicable("invalid full completion or split descriptor")
            dynamic_drains += 1
            for key in maxima:
                maxima[key] = max(maxima[key], issued[key])
            # Compact runs: don't retain a record for every output block.
            if not epochs or epochs[-1]["issued"] != issued:
                epochs.append(dict(issued=dict(issued), occurrences=1))
            else:
                epochs[-1]["occurrences"] += 1
            issued = dict(loads=0, compute_pairs=0, readouts=0)
            written.clear(); consumed.clear()
            actual_drain_positions[data_position] += 1
            has_uncompleted_compute = False
            previous = call
            continue
        op = adapter.project(call)
        data_position += 1
        if issued["readouts"] and type(op) is not Store:
            raise CompletionNotApplicable("output-block completion was removed")
        if type(op) is Load:
            primitive(op.rows, op.cols)
            if pending is not None or op.role not in ("lhs", "weight"):
                raise CompletionNotApplicable("load splits descriptor or has unknown ownership")
            affected = rows(op.role, op.row, op.rows)
            overwritten = written & affected
            if not overwritten <= consumed:
                raise CompletionNotApplicable("unconsumed load-load overwrite has no ordering proof")
            row_reuse_edges += len(overwritten)
            written.update(affected); consumed.difference_update(affected)
            issued["loads"] += 1
        elif type(op) is Preload:
            primitive(op.rows, op.cols, op.weight_rows)
            if pending is not None:
                raise CompletionNotApplicable("preload overwrites an unconsumed descriptor")
            consumed.update(rows("weight", op.weight_row, op.weight_rows))
            rows("acc", op.accumulator_row, op.rows)
            pending = op
        elif type(op) is Compute:
            primitive(op.rows, op.cols)
            if pending is None:
                raise CompletionNotApplicable("compute has no adjacent preload")
            consumed.update(rows("lhs", op.input_row, op.rows))
            pending = None
            issued["compute_pairs"] += 1
            has_uncompleted_compute = True
        elif type(op) is Store:
            primitive(op.rows, op.cols)
            if pending is not None or has_uncompleted_compute:
                raise CompletionNotApplicable("required pre-readout completion was removed")
            rows("acc", op.accumulator_row, op.rows)
            issued["readouts"] += 1
        else:
            raise CompletionNotApplicable("effect is outside ordered dense completion law")
        previous = call
    if previous is None or previous.instr != DRAIN or pending is not None or any(issued.values()):
        raise CompletionNotApplicable("the actual program has no closing full drain")
    protected = _protected_drain_positions(original, adapter)
    if protected - actual_drain_positions:
        raise CompletionNotApplicable("an original readout/block/final boundary was removed")
    return dict(schema="merlin.ordered_dense_completion.v1",
        original_kernel_sha256=original.digest(), actual_kernel_sha256=actual.digest(),
        dynamic_drains=dynamic_drains, dynamic_data_calls=total-dynamic_drains,
        max_issued_between_actual_drains=maxima,
        pending_physical_bounds=dict(loads=law.pending_loads, compute_pairs=law.pending_compute_pairs),
        actual_address_high_water_rows=high_water, load_reuse_row_edges=row_reuse_edges,
        unchanged_assigned_storage=asdict(storage),
        protected_original_dynamic_drains=sum(protected.values()),
        epochs=epochs, ordering_identity=law.ordering_identity,
        full_drain_implementation=ownership.full_drain_implementation,
        coverage_caveats=law.coverage_caveats,
        preserved="all non-drain statements, loops, operands, readout and block/final boundaries",
        target_execution="not_established", timing="UNKNOWN")


@dataclass(frozen=True)
class DenseCompletionResult:
    original: object
    kernel: Kernel
    realization: dict
    checked_kernel_sha256: str
    library_source_sha256: str
    selected_sources: tuple[tuple[str, str], ...] = ()

    def revalidate(self):
        self.original.revalidate()
        if (self.kernel.digest() != self.checked_kernel_sha256
                or sha256(Path(__file__).read_bytes()).hexdigest() != self.library_source_sha256
                or any(sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in self.selected_sources)):

            raise ValueError("completion result or its proof implementation changed")

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from .llvm_program import emit_index_effect_program
        self.revalidate()
        receipt = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return receipt


def coalesce_dense_k_completions(original, storage, adapter, law, ownership, *, mode="ordered_stream"):
    original.revalidate()
    actual, changes = _propose(original.kernel, adapter, mode)
    if not changes:
        raise CompletionNotApplicable("no repeated pure K completion stream")
    report = verify_ordered_dense_completion(original.kernel, actual, storage, adapter, law, ownership)
    report["changed_streams"] = changes
    report["placement_mode"] = mode
    report["unchanged_original_padded_high_water_bytes"] = original.realization.get("source", {}).get("padded_high_water_bytes")
    realization = dict(schema="merlin.dense_completion_pipeline.v1",
        kernel_sha256=actual.digest(), source=original.realization.get("source"),
        input_schedule=original.realization, completion_placement=report,
        dynamic_epoch_drains=report["dynamic_drains"], epochs=report["epochs"],
        completion_law=asdict(law))
    return DenseCompletionResult(original, actual, realization, actual.digest(),
        sha256(Path(__file__).read_bytes()).hexdigest())
