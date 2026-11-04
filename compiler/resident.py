"""Checked full-RHS placement for explicit handles with multiple consumers.

The backend owns source identity, conflicts and eviction. This helper verifies
one complete preparation/consumer program; its preparation epoch is omitted
only while that exact source and physical placement remain live.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN, verify_dense_panel_pipeline
from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
from merlin.sched.ir import Call, Loop, Var
from merlin.sched.ir.expr import select


@dataclass(frozen=True)
class ResidentResult:
    original: object
    kernel: object
    realization: dict
    admission: object
    source_sha256: str

    def revalidate(self):
        self.original.revalidate()
        if (self.kernel.digest() != self.realization['kernel_sha256']
                or sha256(Path(__file__).read_bytes()).hexdigest() != self.source_sha256):
            raise ValueError('resident RHS source or checked emitted program changed')

    def emit_resident_index_program(self, control, lower_effect, *, complete_epoch, resident_live=False):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        self.revalidate()
        kernel = replace(self.kernel, body=self.kernel.body[1:]) if resident_live else self.kernel
        report = emit_index_effect_program(kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return report


def retain_rhs(original, problem, block, resources, storage, adapter):
    edge = problem.tile_k
    facts_path = Path(__file__).parent / 'rtl_facts/facts.json'
    facts_identity = sha256(facts_path.read_bytes()).hexdigest()
    operand_rows = storage.rhs.base + storage.rhs.capacity
    resident = plan_dense_rhs_residency(problem.m, problem.k, problem.n,
        block=edge, operand_rows=operand_rows,
        accumulator_rows=storage.accumulator.capacity, rhs_base=storage.rhs.base,
        max_static_rhs_tiles=storage.rhs.capacity // edge)
    unique_rhs_tiles = ((problem.k + edge - 1) // edge) * ((problem.n + edge - 1) // edge)
    if resident is None:
        raise ValueError("admitted resident handle does not fit checked physical RHS placement")
    original_rhs_loads = original.realization['source']['rhs_loads']
    if storage.lhs.base + storage.lhs.capacity > resident.rhs_base:
        raise ValueError('LHS staging overlaps resident RHS store')

    def rewrite(body):
        changed = []
        for statement in body:
            if isinstance(statement, Loop):
                nested = rewrite(statement.body)
                if nested:
                    changed.append(replace(statement, body=nested))
            elif statement.instr == 'gemmini_weight_load':
                continue
            elif statement.instr == 'gemmini_preload':
                fields = dict(statement.args)
                fields['weight'] = resident.rhs_base + (
                    fields['tile0'] * resident.n_tiles + fields['tile1']) * edge
                changed.append(replace(statement, args=tuple(fields.items())))
            else:
                changed.append(statement)
        return tuple(changed)

    kt, nt = Var('resident_rhs_k'), Var('resident_rhs_n')
    rows = select(kt, resident.k_tiles - 1, problem.k % edge or edge, edge)
    cols = select(nt, resident.n_tiles - 1, problem.n % edge or edge, edge)
    prepare = adapter.encode(Load('weight', (kt, nt), kt * edge, nt * edge,
        rows, cols, resident.rhs_base + (kt * resident.n_tiles + nt) * edge, 1))
    preparation = Loop(kt.name, resident.k_tiles, (
        Loop(nt.name, resident.n_tiles, (Loop('resident_rhs_epoch',1,(prepare,Call(DRAIN, ()))),)),))
    body = (preparation, *rewrite(original.kernel.body))
    kernel = replace(original.kernel, name=original.kernel.name + '_resident_rhs', body=body)
    # Bounded ordered expansion removes preparation-loop dispatch overhead and
    # exposes concrete source/destination addresses for small resident panels.
    # Large panels retain counted loops; all paths use the same checked stream.
    if unique_rhs_tiles <= 16:
        from merlin.sched.primitives import Cursor, unroll
        preparation_kernel = replace(kernel, body=(preparation,))
        preparation_kernel = unroll(preparation_kernel,
            Cursor(loops=(kt.name, nt.name)))
        preparation_kernel = unroll(preparation_kernel, Cursor(loops=(kt.name,)))
        kernel = replace(kernel, body=(Loop('resident_prepare', 1,
            preparation_kernel.body), *kernel.body[1:]))
    admission = original.admission
    # Check the transformed physical program, not the precursor: all source
    # products, latest resident writes, tails and every actual epoch hazard.
    report = verify_dense_panel_pipeline(kernel, problem, block, resources, storage,
        adapter, admission, static_budget=128, timeout_ms=30000)
    report['rhs_residency'] = {
        'planner': asdict(resident), 'cross_command': True,
        'original_rhs_loads': original_rhs_loads, 'retained_rhs_loads': unique_rhs_tiles,
        'facts_sha256': facts_identity,
        'before_kernel_sha256': original.kernel.digest(),
        'after_kernel_sha256': kernel.digest(),
        'initialization': 'counted full RHS load followed by full device completion',
        'lifetime': 'resident rows immutable through all M/N/K compute epochs',
        'cross_command_law': 'reuse only while exact immutable source and fixed local rows remain live; backend invalidates on every possible conflicting writer, handle switch or eviction',
    }
    return ResidentResult(original, kernel, report, admission,
                          sha256(Path(__file__).read_bytes()).hexdigest())
