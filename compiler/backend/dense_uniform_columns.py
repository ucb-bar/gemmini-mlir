"""Source-bound proof for original DIM-wide mixed-scale dense readout tiles.

This is a numerical/coordinate admission, not a new execution law. Producers
keep their original checked dense schedule and every real readout completion.
Uniform-configuration ready/store epoch passes must admit independently.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from merlin.compiler.schedule.dense_contracts import DenseTiledIntegerReadout
from . import isa


@dataclass(frozen=True)
class DenseUniformColumnPanels:
    columns: int
    source_scale_bits: tuple[int, ...]
    tiles: tuple[tuple[int, int, tuple[int, int, int]], ...]
    readout_identity: str

    @property
    def identity(self):
        return sha256(json.dumps((self.columns, self.source_scale_bits, self.tiles,
                                  self.readout_identity)).encode()).hexdigest()

    def validate_source(self, n, attributes):
        expected = prove_uniform_column_panels(n, attributes)
        if expected != self:
            raise ValueError('dense uniform-column proof differs from original source configuration')
        return self

    @property
    def words(self):
        return tuple(command[2] for _, width, command in self.tiles for _ in range(width))


def prove_uniform_column_panels(n, attributes):
    """Keep complete source scales; refuse any unequal original aligned tile."""
    scales = attributes.get('acc_scale_by_column')
    ep = attributes.get('epilogue', ())
    if (type(n) is not int or n <= 0 or scales is None or len(scales) != n
            or attributes.get('output_dtype', 'i8') != 'i8'
            or any('pool' in str(stage).lower() for stage in ep)
            or ('requant' in ep and attributes.get('requant_shift') is not None)):
        return None
    relu = bool(attributes.get('relu', False) or 'relu' in ep)
    try:
        readout = DenseTiledIntegerReadout.from_source_scales(list(scales), columns=n,
            tile_columns=isa.DIM, relu=relu, instruction_name='mvout')
    except ValueError:
        return None
    words = tuple(isa.config_st(n, scale=scale, activation=int(relu)) for scale in scales)
    tiles = tuple((start, min(isa.DIM, n-start), words[start])
                  for start in range(0, n, isa.DIM))
    if any(any(words[c] != command for c in range(start, start+width))
           for start, width, command in tiles):
        return None
    return DenseUniformColumnPanels(n, readout.column_scale_bits, tiles, readout.identity)


def verify_uniform_column_schedule(schedule, proof, m, n, attributes):
    """Prove actual consumed store coordinates are complete original N tiles.

    The original dense verifier still owns contraction, bias, effects, aliasing
    and epochs. This additional finite symbolic query checks every actual store
    over all its loop instances, rather than assuming the planner's tile size.
    """
    import z3
    from merlin.sched.ir import Call, Const, Loop, Scaled, Select, Sum, Var
    if type(proof) is not DenseUniformColumnPanels:
        raise ValueError('direct dense panels require the immutable source proof')
    proof.validate_source(n, attributes)
    schedule.revalidate()
    queries = 0
    def expr(e, env):
        if type(e) is Const: return z3.IntVal(e.value)
        if type(e) is Var: return env[e.name]
        if type(e) is Sum: return z3.Sum([expr(t, env) for t in e.terms])
        if type(e) is Scaled: return e.factor * expr(e.expr, env)
        if type(e) is Select: return z3.If(env[e.var] == e.equals, expr(e.then, env), expr(e.other, env))
        raise ValueError('direct dense panels encountered an unsupported store index')
    def walk(nodes, env, bounds):
        nonlocal queries
        for node in nodes:
            if type(node) is Loop:
                variable = z3.Int('uniform_column_' + node.var)
                walk(node.body, dict(env, **{node.var: variable}),
                     (*bounds, variable >= 0, variable < node.extent))
            elif type(node) is Call and node.instr == 'gemmini_mvout':
                a = {key: expr(value, env) for key, value in node.args}
                col, width = a['dram_col'], a['cols']
                row, rows = a['dram_row'], a['rows']
                solver = z3.Solver(); solver.set(timeout=2000)
                solver.add(*bounds, z3.Or(col < 0, col >= n, col % isa.DIM != 0,
                    width != z3.If(n-col < isa.DIM, n-col, isa.DIM),
                    row < 0, rows <= 0, rows > isa.DIM, row+rows > m))
                if solver.check() != z3.unsat:
                    raise ValueError('direct dense readout changed an original output tile')
                queries += 1
    walk(schedule.kernel.body, {}, ())
    if not queries:
        raise ValueError('direct dense panel proof has no actual original stores')
    schedule.revalidate()
    return dict(source_identity=proof.identity, original_n_tiles=len(proof.tiles),
                actual_store_queries=queries, ready_configuration='mixed; independent admission required',
                completion='original real before-config and after-store drains retained')
