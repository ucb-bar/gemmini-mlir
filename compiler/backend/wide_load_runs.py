"""Bounded primitive-load loop lowering through the shared unroll primitive.

No wide MVIN is emitted. Exact original source calls, temporal order and
explicit completion epochs survive. The original source proof remains owned
by its checked schedule and is revalidated before and after actual emission.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.compile.scheduling.block_schedule import Load
from merlin.sched.ir import Call, Loop, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain


def _lower(kernel, adapter, factor):
    from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
    if type(factor) is not int or factor not in (2, 4, 8):
        raise ValueError('primitive load unroll factor must be 2, 4 or 8')
    requests = []
    def size(body):
        return sum(1 + (size(s.body) if type(s) is Loop else 0) for s in body)
    budget = 256 - size(kernel.body)
    def walk(body, path=()):
        nonlocal budget
        for statement in body:
            if type(statement) is not Loop:
                continue
            current = (*path, statement.var)
            if (len(statement.body) == 1 and type(statement.body[0]) is Call
                    and statement.body[0].instr in
                    ('gemmini_mvin_lhs', 'gemmini_mvin_rhs', 'gemmini_mvin_weight')
                    and statement.extent >= 2):
                call = statement.body[0]
                operation = adapter.project(call)
                if type(operation) is not Load or adapter.encode(operation) != call:
                    raise ValueError('load unroll requires the actual complete dense codec')
                selected = min(factor, statement.extent)
                # Partial unroll may retain a counted bulk loop and one
                # ordered tail. Reserve both before constructing the proposal.
                growth = selected + 2
                if growth <= budget:
                    requests.append(LoopTransform('unroll', current, factor=selected))
                    budget -= growth
            else:
                walk(statement.body, current)
    walk(kernel.body)
    if not requests:
        return kernel, ()
    actual = propose_loop_transforms(kernel, tuple(requests),
        index_bits=adapter.index_bits, max_static_operations=256)
    if check_structure(actual):
        raise ValueError('load unroll produced malformed counted IR')
    check_integer_domain(actual, adapter.index_bits)
    return actual, tuple(dict(kind='primitive_load_unroll', path=list(r.path),
        factor=r.factor) for r in requests)


@dataclass(frozen=True)
class DenseLoadRunResult:
    original: object
    adapter: object
    kernel: object
    factor: int
    original_digest: str
    emitted_digest: str
    changes: tuple

    @property
    def realization(self):
        return dict(self.original.realization, primitive_load_runs={
            'schema': 'dense_primitive_load_runs.v1',
            'original_kernel_sha256': self.original_digest,
            'emitted_kernel_sha256': self.emitted_digest,
            'changes': list(self.changes),
            'wide_loads': False,
            'command_order_and_explicit_epochs': 'preserved',
            'source_proof': 'original schedule revalidated before and after emission'})

    def __getattr__(self, name):
        return getattr(self.original, name)

    def revalidate(self):
        self.original.revalidate()
        if self.original.kernel.digest() != self.original_digest:
            raise ValueError('original load-run source changed')
        rebuilt, changes = _lower(self.original.kernel, self.adapter, self.factor)
        if (rebuilt != self.kernel or changes != self.changes
                or self.kernel.digest() != self.emitted_digest):
            raise ValueError('terminal load-run equivalence changed')

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        receipt = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return receipt


def select_primitive_load_runs(schedule, adapter, *, factor=4):
    """Preserve the incumbent unless a bounded pure-load loop can be unrolled.

    Invoke after source/epoch checks and before LLVM index emission, using the
    unchanged dense terminal callback. No configuration or command moves past
    an effect; only primitive load loops are unrolled through the shared exact
    substitution primitive.
    """
    schedule.revalidate()
    kernel, changes = _lower(schedule.kernel, adapter, factor)
    if not changes:
        return schedule
    result = DenseLoadRunResult(schedule, adapter, kernel, factor,
        schedule.kernel.digest(), kernel.digest(), changes)
    result.revalidate()
    return result
