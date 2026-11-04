"""Checked full-K RHS lookahead with bounded load-first K bursts.

Loop proposals grant no execution law. This consumer reuses the immutable-load
configuration admission and checks the complete rewritten source and epochs.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.sched.ir import Call, Loop, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.sched.primitives import _bound_vars, _loop_at, _splice
from .block_schedule import Compute, Load, Preload
from .compact_dense import _digest
from .dense_panel_pipeline import DRAIN, _calls, _verify_dense_epochs
from .dense_rhs_panel_pipeline import DenseRhsPanelResult
from .library_sources import shared_library_sources
from .loop_schedule import LoopTransform, propose_loop_transforms


@dataclass(frozen=True)
class DenseRhsBurstResult:
    kernel: object
    realization: dict
    source_identity: str
    admission: object
    original: DenseRhsPanelResult
    checked_kernel_sha256: str
    checked_admission_sha256: str
    checked_owner_sha256: str
    checked_library_sources: tuple

    def revalidate(self):
        self.original.revalidate()
        self.admission.__post_init__()
        if (self.source_identity != self.original.source_identity
            or self.admission != self.original.admission
            or self.kernel.digest() != self.checked_kernel_sha256
            or _digest(asdict(self.admission)) != self.checked_admission_sha256
            or sha256(Path(__file__).read_bytes()).hexdigest() != self.checked_owner_sha256
            or tuple(sorted(shared_library_sources(version=15).items())) != self.checked_library_sources):
            raise ValueError('RHS burst checked source, program, admission or library changed')

    def materialize_scf(self):
        from .scf_commands import materialize_scf_commands
        self.revalidate()
        result = materialize_scf_commands(self.kernel)
        self.revalidate()
        return result

    def emit_llvm(self, builder, lower_call):
        from .llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        result = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return result


def burst_rhs_frontiers(kernel, adapter, *, k_chunk):
    """Propose load-first stripes of 1/2/4 K tiles without adding drains.

    Split and fission use the existing counted-loop primitives, including their
    exact equality-select and ragged-tail substitutions. Only disjoint future
    loads cross compute; current compute's increasing-K order stays intact.
    The caller must independently check source and combined epoch ownership.
    """
    if type(k_chunk) is not int or k_chunk not in (1, 2, 4):
        raise ValueError('RHS burst K chunk must be exactly 1, 2 or 4')
    if check_structure(kernel):
        raise ValueError('RHS burst requires closed actual command control')
    paths = []

    def discover(body, path=()):
        for statement in body:
            if type(statement) is not Loop:
                continue
            here = (*path, statement.var)
            if statement.body and statement.body[-1] == Call(DRAIN, ()):
                if len(statement.body) != 2 or type(statement.body[0]) is not Loop:
                    continue
                band = statement.body[0]
                ops = [adapter.project(c) for c in _calls(band.body)]
                if any(type(op) is Load for op in ops) and any(type(op) is Compute for op in ops):
                    paths.append((*here, band.var))
            else:
                discover(statement.body, here)
    discover(kernel.body)
    if not paths:
        raise ValueError('RHS burst requires an actual mixed full-K frontier')
    used = _bound_vars(kernel.body) | {a.name for a in kernel.args}
    actual, records = kernel, []

    def fresh(stem):
        while stem in used:
            stem += '_'
        used.add(stem)
        return stem

    def fission_load_first(path):
        nonlocal actual
        target = _loop_at(actual.body, path)
        suffix = fresh(target.var + '_burst_load')
        request = LoopTransform('fission', path, cut=1, names=(suffix,))
        actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
        swaps = 0
        def swap_adjacent(body):
            nonlocal swaps
            result = []
            index = 0
            while index < len(body):
                s = body[index]
                if (type(s) is Loop and s.var == target.var
                    and index + 1 < len(body) and type(body[index + 1]) is Loop
                    and body[index + 1].var == suffix):
                    result.extend((body[index + 1], s))
                    swaps += 1
                    index += 2
                    continue
                if type(s) is Loop:
                    s = replace(s, body=swap_adjacent(s.body))
                result.append(s)
                index += 1
            return tuple(result)
        actual = replace(actual, body=swap_adjacent(actual.body))
        if swaps != 1:
            raise ValueError('RHS burst fission did not expose exactly one adjacent load stream')
        return asdict(request)

    for path in paths:
        target = _loop_at(actual.body, path)
        if len(target.body) != 2 or any(type(s) is not Loop for s in target.body):
            raise ValueError('RHS burst frontier is not a complete compute then load band')
        compute_ops = [adapter.project(c) for c in _calls((target.body[0],))]
        load_ops = [adapter.project(c) for c in _calls((target.body[1],))]
        if (not compute_ops or not load_ops
            or any(type(op) not in (Preload, Compute) for op in compute_ops)
            or any(type(op) is not Load or op.role != 'weight' for op in load_ops)):
            raise ValueError('RHS burst cannot cross unknown effects or non-weight loads')
        requests = []
        if k_chunk == 1:
            reordered = replace(target, body=(target.body[1], target.body[0]))
            actual = replace(actual, body=_splice(actual.body, path, (reordered,)))
            requests.append(dict(action='swap_independent_load_compute_bands', path=path))
        elif target.extent >= k_chunk:
            outer, inner = fresh(target.var + '_burst'), fresh(target.var + '_within')
            request = LoopTransform('split', path, factor=k_chunk, names=(outer, inner))
            actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
            requests.append(asdict(request))
            requests.append(fission_load_first((*path[:-1], outer, inner)))
            if target.extent % k_chunk:
                requests.append(fission_load_first(path))
        else:
            # A short K domain is one bounded burst, with no speculative tiles.
            requests.append(fission_load_first(path))
        records.append(dict(original_k_path=path, k_tiles=target.extent,
            k_chunk=k_chunk, full_bursts=target.extent // k_chunk,
            tail_k_tiles=target.extent % k_chunk, transforms=requests))
    if check_structure(actual):
        raise ValueError('RHS burst changed command ownership or loop scope')
    check_integer_domain(actual, adapter.index_bits)
    return actual, dict(source_kernel_sha256=kernel.digest(), actual_kernel_sha256=actual.digest(),
        k_chunk=k_chunk, regions=records, issue_order='future_rhs_load_chunk_then_current_compute_chunk',
        completion='all original full-device drains retained', target_execution='not_established')


def schedule_dense_rhs_bursts(original, problem, block, resources, storage, adapter, *,
        k_chunk=2, static_budget=128, timeout_ms=30000):
    """Consume a genuine checked full-K product and recheck the burst rewrite."""
    if type(original) is not DenseRhsPanelResult:
        raise ValueError('RHS burst requires the checked immutable-configuration full-K product')
    original.revalidate()
    if block.n not in (2, 3) or block.k != problem.tiles[1] or block.m != 1:
        raise ValueError('RHS burst requires full K, one M tile and explicit N width 2 or 3')
    from .dense_segmented_pipeline import _assignment
    _assignment(problem, block, resources, storage)
    if (original.admission.hardware_identity != adapter.configuration_identity
        or original.admission.numerical_policy_identity != adapter.numerical_policy_identity
        or original.realization['block_tiles'] != [block.m, block.k, block.n]):
        raise ValueError('RHS burst original checked product differs from geometry or adapter')
    actual, rewrite = burst_rhs_frontiers(original.kernel, adapter, k_chunk=k_chunk)
    report = _verify_dense_epochs(actual, problem, resources, storage, adapter, original.admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
        ordered_accumulator_commits=True, source_phase_fission=True)
    report.update(schema='merlin.dense_rhs_bursts.v1', admission=asdict(original.admission),
        block_tiles=[block.m, block.k, block.n], rhs_slots=2, k_segmentation=False,
        rewrite=rewrite, pass_order=['checked_full_k_rhs_panels', 'split_k_chunks',
            'fission_chunk_load_compute', 'load_first_chunk_proposal',
            'whole_source_and_combined_epoch_checks', 'common_terminal_emission'])
    result = DenseRhsBurstResult(actual, report, original.source_identity, original.admission,
        original, actual.digest(), _digest(asdict(original.admission)),
        sha256(Path(__file__).read_bytes()).hexdigest(),
        tuple(sorted(shared_library_sources(version=15).items())))
    result.revalidate()
    return result
