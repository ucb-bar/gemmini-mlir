"""Weight-resident full-K M panels with two disjoint LHS windows.

Counted construction uses the ordinary encoded dense operations. The selected
owner supplies immutable load-channel and combined ordering premises. Original
source, actual physical accesses and every mixed epoch are independently checked.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, check_structure, select
from merlin.sched.ir.integer_domain import check_integer_domain
from .block_schedule import Compute, Load, Preload, Store
from .compact_dense import _digest, _edge, _emit_lhs_panel, _Group, _groups
from .dense_panel_pipeline import DRAIN, _verify_dense_epochs
from .library_sources import shared_library_sources


@dataclass(frozen=True)
class DenseLhsPanelAdmission:
    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    source_identity: str
    max_issued_compute_pairs: int
    max_issued_loads: int
    pending_compute_pairs: int
    pending_loads: int
    binding_sources: tuple
    execution: str = 'ordered_accumulator_commits_disjoint_lhs_loads_full_drain'
    load_configuration: str = 'channels_0_1_immutable_for_complete_invocation'
    descriptor: str = 'preload_immediately_paired_with_compute'
    max_issued_readouts: int = 1
    pending_readouts: int = 1
    readout_configuration: str = 'separate_drained_readout'

    def __post_init__(self):
        if (any(type(v) is not str or not v for v in (self.hardware_identity,
                self.numerical_policy_identity, self.qualification_identity, self.source_identity))
            or any(type(v) is not int or v <= 0 for v in (self.max_issued_compute_pairs,
                self.max_issued_loads, self.pending_compute_pairs, self.pending_loads))
            or type(self.binding_sources) is not tuple or not self.binding_sources
            or self.execution != 'ordered_accumulator_commits_disjoint_lhs_loads_full_drain'
            or self.load_configuration != 'channels_0_1_immutable_for_complete_invocation'
            or self.descriptor != 'preload_immediately_paired_with_compute'
            or self.max_issued_readouts != 1 or self.pending_readouts != 1
            or self.readout_configuration != 'separate_drained_readout'):
            raise ValueError('LHS panels need the explicit selected immutable-channel combined admission')
        for path, digest in self.binding_sources:
            if sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('LHS panel selected source/configuration binding changed')


@dataclass(frozen=True)
class DenseLhsPanelResult:
    kernel: Kernel
    realization: dict
    source_identity: str
    admission: DenseLhsPanelAdmission
    checked_kernel_sha256: str
    checked_admission_sha256: str
    checked_owner_sha256: str
    checked_library_sources: tuple

    def revalidate(self):
        self.admission.__post_init__()
        if (self.source_identity != self.admission.source_identity
            or self.kernel.digest() != self.checked_kernel_sha256
            or _digest(asdict(self.admission)) != self.checked_admission_sha256
            or sha256(Path(__file__).read_bytes()).hexdigest() != self.checked_owner_sha256
            or tuple(sorted(shared_library_sources(version=15).items())) != self.checked_library_sources):
            raise ValueError('LHS panel checked program, law, source or library changed')

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


def lhs_panel_assignment(problem, block, resources, storage):
    """Validate two full-K LHS slots and one RHS panel in original real views."""
    from merlin.transforms.dense_planning import dense_blocks
    if (problem.bias or block.m not in (4, 8) or not 1 <= block.n <= 4
        or block.k != problem.tiles[1] or problem.tiles[0] <= block.m
        or problem.tile_m != problem.tile_k or problem.tile_n != problem.tile_k
        or (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes) != (1, 1, 4)
        or block not in dense_blocks(problem, resources)
        or not storage.admits(problem, block, resources)):
        raise ValueError('LHS lookahead needs legal full K, multiple M groups, M width4/8 and uniform N width1..4')
    lhs_rows, rhs_rows = block.m * block.k * problem.tile_m, block.n * block.k * problem.tile_k
    slots = tuple(replace(storage.lhs, base=storage.lhs.base + slot * lhs_rows,
                          capacity=lhs_rows) for slot in (0, 1))
    rhs = replace(storage.rhs, capacity=rhs_rows)
    for view, parent in ((*((s, storage.lhs) for s in slots), (rhs, storage.rhs))):
        if (view.base < parent.base or view.base + view.capacity > parent.base + parent.capacity
            or view.base % parent.alignment):
            raise ValueError('LHS lookahead window escapes or misaligns its private assignment')
    views = (*slots, rhs, storage.accumulator)
    if storage.accumulator is None:
        raise ValueError('LHS lookahead needs original private accumulators')
    for i, left in enumerate(views):
        for right in views[i + 1:]:
            if (left.backing == right.backing
                and left.base * left.unit_bytes < (right.base + right.capacity) * right.unit_bytes
                and right.base * right.unit_bytes < (left.base + left.capacity) * left.unit_bytes):
                raise ValueError('LHS lookahead simultaneously live windows alias')
    if ((sum(s.capacity * s.unit_bytes for s in slots) + rhs.capacity * rhs.unit_bytes > resources.operand_bytes)
        or block.m * block.n * problem.tile_m * storage.accumulator.unit_bytes > resources.accumulator_bytes):
        raise ValueError('LHS lookahead two slots and retained RHS exceed original byte resources')
    return slots, rhs


def build_dense_lhs_panels(problem, block, resources, storage, adapter, *, name):
    """Count M-panel pairs with static alternating slots, peeled startup/tail.

    Each mixed epoch issues the complete FUTURE LHS panel first, then current
    increasing-K compute. Disjointness is not inferred here: the actual emitted
    accesses and full original product must pass the separate checks below.
    Every readout has its own full drain; no array STAY instruction is used.
    """
    slots, rhs = lhs_panel_assignment(problem, block, resources, storage)
    if not adapter.synchronous_completion or not adapter.exact_accumulator_continuation:
        raise ValueError('LHS lookahead needs the original exact completion and integer continuation adapter')
    serial = 0
    def variable(stem):
        nonlocal serial
        serial += 1
        return stem + str(serial)
    def call(operation):
        value = adapter.encode(operation)
        if type(value) is not Call:
            raise ValueError('LHS lookahead adapter must emit actual instruction calls')
        return value
    def nest(extents, action):
        names = tuple(variable('lhs_tile_') for _ in extents)
        body = tuple(action(*(Var(n) for n in names)))
        for n, extent in reversed(tuple(zip(names, extents))):
            body = (Loop(n, extent, body),)
        return body
    def epoch(body):
        return Loop(variable('lhs_epoch_'), 1, (*body, Call(DRAIN, ())))
    kt = problem.tiles[1]
    kg = _Group(None, 1, Const(0), kt, True)
    def lhs_loads(mg, slot):
        return _emit_lhs_panel(problem, block, replace(storage, lhs=slots[slot]), call, nest, mg, kg)
    def output_group(mg, ng, slot, *, initial=False, following=None):
        result = []
        if initial:
            result.append(epoch(lhs_loads(mg, slot)))
        def compute(k, m, n):
            rows = _edge(mg, m, problem.m, problem.tile_m)
            cols = _edge(ng, n, problem.n, problem.tile_n)
            depth = _edge(kg, k, problem.k, problem.tile_k)
            acc = storage.accumulator.base + (m * block.n + n) * problem.tile_m
            return (call(Preload(rhs.base + (k * block.n + n) * problem.tile_k,
                    (k, ng.base + n), acc, select(k, 0, 0, 1), rows, cols, depth)),
                    call(Compute(slots[slot].base + (m * kt + k) * problem.tile_m,
                    (mg.base + m, k), rows, depth, True)))
        future = () if following is None else lhs_loads(following, 1 - slot)
        result.append(epoch((*future, *nest((kt, mg.blocks, ng.blocks), compute))))
        def store(m, n):
            return (epoch((call(Store((mg.base + m) * problem.tile_m,
                (ng.base + n) * problem.tile_n, _edge(mg, m, problem.m, problem.tile_m),
                _edge(ng, n, problem.n, problem.tile_n),
                storage.accumulator.base + (m * block.n + n) * problem.tile_m)),)),)
        result.extend(nest((mg.blocks, ng.blocks), store))
        return tuple(result)
    def m_groups(ng):
        full, tail_size = divmod(problem.tiles[0], block.m)
        def group(index, blocks=block.m, *, last=False, var=None, trips=1):
            return _Group(var, trips, index * block.m, blocks, last)
        def stage(index, slot, *, initial=False, following=None, last=False):
            return output_group(group(index, last=last), ng, slot, initial=initial, following=following)
        tail = None if not tail_size else group(Const(full), tail_size, last=True)
        if full == 1:
            result = list(stage(Const(0), 0, initial=True, following=tail))
        else:
            result = list(stage(Const(0), 0, initial=True,
                following=group(Const(1), last=full == 2 and not tail_size)))
            pairs, odd = divmod(full - 2, 2)
            if pairs:
                period = variable('lhs_panel_period_')
                index = Var(period) * 2 + 1
                result.append(Loop(period, pairs, (
                    *stage(index, 1, following=group(index + 1)),
                    *stage(index + 1, 0, following=group(index + 2,
                        last=not tail_size and not odd, var=period, trips=pairs)))))
            if odd:
                index = Const(1 + 2 * pairs)
                result.extend(stage(index, 1, following=group(index + 1, last=not tail_size)))
            result.extend(stage(Const(full - 1), (full - 1) % 2, following=tail, last=not tail_size))
        if tail is not None:
            result.extend(output_group(tail, ng, full % 2))
        return tuple(result)
    body = []
    for ng in _groups(problem.tiles[2], block.n, 'outer_n'):
        def rhs_load(k, n):
            return (call(Load('weight', (k, ng.base + n), k * problem.tile_k,
                (ng.base + n) * problem.tile_n, _edge(kg, k, problem.k, problem.tile_k),
                _edge(ng, n, problem.n, problem.tile_n), rhs.base + (k * block.n + n) * problem.tile_k, 1)),)
        panel = (epoch(nest((kt, ng.blocks), rhs_load)), *m_groups(ng))
        body.extend((Loop(ng.var, ng.trips, panel),) if ng.var is not None else panel)
    kernel = Kernel(name, (TensorArg('lhs', (problem.m, problem.k), 'i8', 'read'),
        TensorArg('rhs', (problem.k, problem.n), 'i8', 'read'),
        TensorArg('output', (problem.m, problem.n), 'i32', 'write')), tuple(body))
    if check_structure(kernel):
        raise ValueError('LHS lookahead construction changed source or completion scopes')
    check_integer_domain(kernel, adapter.index_bits)
    return kernel


def unroll_lhs_compute_n(kernel, adapter):
    """Inline pairs from bounded N loops of actual paired dense operations.

    The public ordered unroll preserves every instruction instance, operand,
    completion and K order. It grants no legality: the complete transformed
    kernel must still pass the source and mixed-epoch checks below. M/K loops,
    staging and drained readout loops remain counted.
    """
    from .loop_schedule import LoopTransform, propose_loop_transforms
    paths = []
    def discover(body, path=()):
        for statement in body:
            if type(statement) is not Loop:
                continue
            here = (*path, statement.var)
            if (2 <= statement.extent <= 4 and len(statement.body) == 2
                and all(type(s) is Call for s in statement.body)
                and tuple(type(adapter.project(s)) for s in statement.body) == (Preload, Compute)):
                paths.append((here, statement.extent))
            else:
                discover(statement.body, here)
    discover(kernel.body)
    actual, records = kernel, []
    for path, extent in paths:
        request = LoopTransform('unroll', path, factor=2)
        actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
        records.append(asdict(request))
    return actual, records


def schedule_dense_lhs_panels(problem, block, resources, storage, adapter, admission, *,
        name, source_identity, external_buffers_disjoint, static_budget=128, timeout_ms=30000,
        unroll_compute_n=False):
    """Check actual future LHS writes, retained RHS and the original product."""
    if (type(unroll_compute_n) is not bool
        or type(admission) is not DenseLhsPanelAdmission or external_buffers_disjoint is not True
        or admission.source_identity != source_identity
        or admission.hardware_identity != adapter.configuration_identity
        or admission.numerical_policy_identity != adapter.numerical_policy_identity):
        raise ValueError('LHS panel law differs from original source or selected adapter')
    admission.__post_init__()
    kernel = build_dense_lhs_panels(problem, block, resources, storage, adapter, name=name)
    original_kernel_sha256 = kernel.digest()
    transforms = []
    if unroll_compute_n:
        kernel, transforms = unroll_lhs_compute_n(kernel, adapter)
    report = _verify_dense_epochs(kernel, problem, resources, storage, adapter, admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
        ordered_accumulator_commits=True, source_phase_fission=True)
    report.update(schema='merlin.dense_lhs_panel_pipeline.v1', admission=asdict(admission),
        block_tiles=[block.m, block.k, block.n], lhs_slots=2, retained_operand='weight',
        outer_order=['n', 'm'], k_segmentation=False, array_weight_retention=False,
        issue_order='future_full_k_lhs_panel_then_current_increasing_k_compute',
        ordered_compute_n_unroll=dict(enabled=unroll_compute_n, transforms=transforms,
            original_kernel_sha256=original_kernel_sha256, transformed_kernel_sha256=kernel.digest()),
        configuration='channels0/1 immutable; all readouts separately drained',
        pass_order=['full_k_weight_resident_counted_m_panel_pairs',
                    *(['bounded_ordered_compute_n_unroll'] if unroll_compute_n else []),
                    'actual_whole_source_and_combined_epoch_checks', 'common_terminal_emission'])
    result = DenseLhsPanelResult(kernel, report, source_identity, admission, kernel.digest(),
        _digest(asdict(admission)), sha256(Path(__file__).read_bytes()).hexdigest(),
        tuple(sorted(shared_library_sources(version=15).items())))
    result.revalidate()
    return result
