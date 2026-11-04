"""Capacity-bounded residual phases with real full-completion boundaries.

Each batch owns disjoint original flattened intervals and disjoint ACC rows.
All seed writes complete before any accumulating write; all additions complete
before readout; every readout completes before the next batch reuses storage.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from merlin.sched.ir import Kernel, Var, add, mul, call, loop, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.compile.scheduling.llvm_program import (
    FnBuilderIndexAdapter, emit_index_effect_program,
)

from . import isa


def large_guarded(shape):
    """Provisional empirical region; it grants no new hardware permission.

    The validated large sources have widths 1024 and 5728. Eight complete
    primitive tiles equal two rows of the narrower source and lie below the
    smaller validated residual's 22 full tiles. The 8-tile boundary is tested
    separately; smaller standalone domains retain their original emission.
    Logical row widths must contain complete primitive column panels. Actual
    addressing remains flattened and the existing tail/alias checks still run.
    """
    from math import prod
    return (bool(shape) and shape[-1] >= isa.DIM and shape[-1] % isa.DIM == 0
            and prod(shape) // (isa.DIM * isa.DIM) >= 8)


@dataclass(frozen=True)
class ResidualBatchPlan:
    count: int
    source_identity: str
    kernel: Kernel
    ordering_identity: str
    accumulator_rows: int
    primitive: int

    def require(self):
        # Selected facts b8ff28ed...: two ACC banks, 512 rows/bank,
        # 16 signed-i32 lanes/row. These are the same facts as terminal isa.py.
        if (self.accumulator_rows != 2 * 512 or self.primitive != 16
                or isa.ACC_ROWS != self.accumulator_rows or isa.DIM != self.primitive
                or self.count < 1 or check_structure(self.kernel)):
            raise ValueError('residual batch geometry/source is not selected')
        if self.kernel != _program(self.count, self.source_identity, self.ordering_identity,
                                    self.primitive, self.accumulator_rows):
            raise ValueError('residual batch effects changed their checked original intervals')
        check_integer_domain(self.kernel, 64)


def prepare(ctx, names, count, proof):
    """Admit before mutation; uncertain external alias views keep single tiles."""
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    involved = set(names)
    if any(name in involved or (isinstance(view, dict) and view.get('source') in involved)
           for name, view in aliases.items()):
        return None
    dim, capacity = isa.DIM, isa.ACC_ROWS
    if dim != 16 or capacity != 2 * 512 or count <= dim * dim:
        return None
    record = json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering = record['ordering']
    digest = hashlib.sha256(json.dumps(ordering, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (digest != 'ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
            or record['base_facts_sha256'] != 'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
            or ordering['status'] != 'derived'):
        return None
    # No overlapping asynchronous accesses occur inside a phase. Finite queues
    # backpressure ordinary command issue; capacity here is storage, not queue
    # entries. Real full fences retain all inter-phase dependencies.
    for lane in ('load', 'store'):
        finite = record['finite_entries'][lane]
        if (finite['status'] != 'derived' or finite['value'] < 1
                or ordering['same_queue'][lane]['release'] not in ('issue', 'completion')):
            return None
    identity = hashlib.sha256(json.dumps(dict(names=names, count=count, proof=proof,
        geometry=(dim, capacity), ordering=digest), sort_keys=True).encode()).hexdigest()
    kernel = _program(count, identity, digest, dim, capacity)
    result = ResidualBatchPlan(count, identity, kernel, digest, capacity, dim)
    result.require()
    return result


def _program(count, identity, digest, dim, capacity):
    tile_elements, batch_tiles = dim * dim, capacity // dim
    batch_elements = batch_tiles * tile_elements
    groups, remainder = divmod(count, batch_elements)

    def phase(tag, base, elements):
        tiles, tail = divmod(elements, tile_elements)
        body = []
        if tiles:
            body.append(loop('tile', tiles, call(tag,
                offset=add(base, mul(Var('tile'), tile_elements)),
                local=mul(Var('tile'), dim), rows=dim, cols=dim)))
        rows, cols = divmod(tail, dim)
        if rows:
            body.append(call(tag, offset=add(base, tiles * tile_elements),
                             local=tiles * dim, rows=rows, cols=dim))
        if cols:
            body.append(call(tag, offset=add(base, tiles * tile_elements + rows * dim),
                             local=tiles * dim + rows, rows=1, cols=cols))
        return (*body, call('complete'))

    def batch(base, elements):
        return tuple(op for tag in ('seed', 'accumulate', 'readout')
                     for op in phase(tag, base, elements))

    body = [call('complete')]
    if groups:
        body.append(loop('batch', groups, *batch(mul(Var('batch'), batch_elements), batch_elements)))
    if remainder:
        body.extend(batch(groups * batch_elements, remainder))
    return Kernel('residual_capacity_batches', (), tuple(body),
                    (('source_sha256', identity), ('ordering_sha256', digest)))


def emit(plan, ctx, names):
    plan.require()
    lhs, rhs, dst = names

    def lower(tag, fields):
        offset, local = fields['offset'], fields['local']
        rows, cols = fields['rows'], fields['cols']
        acc = isa.acc_addr(local, accumulate=tag == 'accumulate', fb=ctx.fb)
        if tag == 'seed':
            ctx.emit(*isa.mvin(ctx.ptr(lhs, offset), acc, rows, cols, channel=0, fb=ctx.fb))
        elif tag == 'accumulate':
            ctx.emit(*isa.mvin(ctx.ptr(rhs, offset), acc, rows, cols, channel=1, fb=ctx.fb))
        elif tag == 'readout':
            ctx.emit(*isa.mvout(ctx.ptr(dst, offset), acc, rows, cols, fb=ctx.fb))
        else:
            raise ValueError('unknown residual batch effect ' + tag)

    receipt = emit_index_effect_program(plan.kernel, FnBuilderIndexAdapter(ctx.fb), lower,
        complete_epoch=ctx.fence, completion_instruction='complete')
    plan.require()
    return dict(receipt, source_sha256=plan.source_identity,
        ordering_sha256=plan.ordering_identity, accumulator_rows=plan.accumulator_rows,
        elements_per_batch=plan.accumulator_rows * plan.primitive,
        ownership='seed complete; accumulating load complete; readout complete before row reuse',
        numeric='unchanged exhaustive original f32/bound_lsb/ReLU screen')
