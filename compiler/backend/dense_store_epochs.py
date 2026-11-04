"""Uniform dense store epochs over the actual checked counted schedule.

Only maximal store-only loop regions change. Their original data calls and loop
order remain, while internal completion calls become real drains surrounding
the entire region. SMT checks disjoint owned ACC row intervals for every pair
of dynamic stores, including local loop backedges and all outer loop values.
Separate regions close before any producer, seed or ownership reuse. A checked
disjoint readout/compute owner may instead share its original mixed-epoch drain.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.sched.ir import Call, Const, Loop, Scaled, Select, Sum, Var, check_structure
from . import isa
from .column_batches import _admission

BEGIN = 'gemmini_dense_store_epoch_begin'
END = 'gemmini_dense_store_epoch_end'
STORE = 'gemmini_dense_store_epoch_mvout'
DRAIN = 'merlin_dense_panel_drain'


def _calls(statement):
    if type(statement) is Loop:
        return tuple(c for child in statement.body for c in _calls(child))
    return (statement,)


def _store_count(statement):
    if type(statement) is Loop:
        return statement.extent * sum(_store_count(child) for child in statement.body)
    return int(statement.instr == 'gemmini_mvout')


def _only_stores(statement):
    calls = _calls(statement)
    return any(c.instr == 'gemmini_mvout' for c in calls) and all(
        c.instr in ('gemmini_mvout', DRAIN) for c in calls)


def _strip(statement):
    if type(statement) is Loop:
        body = tuple(x for child in statement.body if (x := _strip(child)) is not None)
        return replace(statement, body=body) if body else None
    if statement.instr == DRAIN:
        return None
    return replace(statement, instr=STORE)


def _prove_region(statement, outer):
    """Prove row ownership without enumerating any outer workload dimension."""
    import z3
    stores = []
    def collect(node, local):
        if type(node) is Loop:
            if node.iter_args or node.yields or node.results:
                raise ValueError('store epochs do not rewrite completion-token loops')
            for child in node.body:
                collect(child, (*local, (node.var, node.extent)))
        elif node.instr == 'gemmini_mvout':
            stores.append((dict(node.args), local))
    collect(statement, ())
    def expr(e, env):
        if type(e) is Const:
            return z3.IntVal(e.value)
        if type(e) is Var:
            return env[e.name]
        if type(e) is Sum:
            return z3.Sum([expr(t, env) for t in e.terms])
        if type(e) is Scaled:
            return expr(e.expr, env) * e.factor
        if type(e) is Select:
            return z3.If(env[e.var] == e.equals, expr(e.then, env), expr(e.other, env))
        raise ValueError('store epoch encountered an unsupported index expression')
    shared = {name: z3.Int('outer_' + name) for name, _ in outer}
    base = [z3.And(shared[name] >= 0, shared[name] < extent) for name, extent in outer]
    def instantiate(entry, suffix):
        arguments, local = entry
        env = dict(shared)
        for name, _ in local:
            env[name] = z3.Int(suffix + name)
        domain = [z3.And(env[name] >= 0, env[name] < extent) for name, extent in local]
        return {key: expr(value, env) for key, value in arguments.items()}, env, domain
    queries = 0
    def unsat(constraints):
        nonlocal queries
        solver = z3.Solver()
        solver.set(timeout=2000)
        solver.add(*base, *constraints)
        queries += 1
        if solver.check() != z3.unsat:
            raise ValueError('store epoch ACC disjointness or capacity is unproved')
    for i, first in enumerate(stores):
        a, ea, da = instantiate(first, 'a_')
        unsat([*da, z3.Or(a['rows'] <= 0, a['rows'] > isa.DIM,
            a['cols'] <= 0, a['cols'] > isa.DIM, a['acc'] < 0,
            a['acc'] + a['rows'] > isa.ACC_ROWS)])
        for j in range(i, len(stores)):
            second = stores[j]
            b, eb, db = instantiate(second, 'b_')
            different = z3.Or([ea[name] != eb[name] for name, _ in first[1]]) if i == j else z3.BoolVal(True)
            unsat([*da, *db, different,
                   a['acc'] < b['acc'] + b['rows'], b['acc'] < a['acc'] + a['rows']])
    return dict(dynamic_stores=_store_count(statement), solver_queries=queries,
                proof='all outer/local loop values; disjoint owned ACC rows')


def _trim_completed_tail(statement):
    # BEGIN itself performs a real full drain. Move an immediately preceding
    # drain across only zero-effect singleton-loop exits, never across a loop
    # backedge or another device operation. This is the same completion cut.
    if statement == Call(DRAIN, ()):
        return None, True
    if type(statement) is Loop and statement.extent == 1 and statement.body:
        tail, removed = _trim_completed_tail(statement.body[-1])
        if removed:
            body = (*statement.body[:-1], *((tail,) if tail is not None else ()))
            return (replace(statement, body=body) if body else None), True
    return statement, False


def _transform(kernel, *, prove, disjoint_owner=False):
    regions = []
    def walk(body, outer, common_drain=False):
        result = []
        for statement in body:
            if _only_stores(statement) and _store_count(statement) > 1:
                proof = _prove_region(statement, outer) if prove else {}
                proof['completion'] = ('original mixed epoch drain after disjoint compute'
                    if common_drain else 'original separate store region drain')
                regions.append(proof)
                if disjoint_owner and result:
                    tail, coalesced = _trim_completed_tail(result[-1])
                    if coalesced:
                        result.pop()
                        if tail is not None:
                            result.append(tail)
                    proof['readiness_boundary_coalesced'] = coalesced
                result.extend((Call(BEGIN, ()), _strip(statement)))
                if not common_drain:
                    result.append(Call(END, ()))
            elif type(statement) is Loop:
                calls = _calls(statement)
                # Only the exact checked disjoint owner may keep stores pending
                # through computation. Its actual common drain closes both ACC
                # lifetimes; an inner END would serialize this admitted epoch.
                mixed = (disjoint_owner and bool(statement.body)
                    and statement.body[-1] == Call(DRAIN, ())
                    and any(c.instr == 'gemmini_mvout' for c in calls)
                    and any(c.instr == 'gemmini_compute_preloaded' for c in calls)
                    and not any(c.instr == DRAIN for c in calls[:-1]))
                result.append(replace(statement, body=walk(statement.body,
                    (*outer, (statement.var, statement.extent)), mixed)))
            else:
                result.append(statement)
        return tuple(result)
    actual = replace(kernel, body=walk(kernel.body, ()))
    if check_structure(actual):
        raise ValueError('dense store epochs produced an invalid counted program')
    return actual, regions


@dataclass(frozen=True)
class DenseStoreEpochResult:
    original: object
    kernel: object
    realization: dict
    command: tuple
    original_identity: str
    disjoint_owner: bool = False

    def revalidate(self):
        self.original.revalidate()
        expected, _ = _transform(self.original.kernel, prove=False, disjoint_owner=self.disjoint_owner)
        if self.original.kernel.digest() != self.original_identity or expected != self.kernel:
            raise ValueError('dense store epoch source or completion regions changed')

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        bridge = FnBuilderIndexAdapter(builder)
        result = emit_index_program(self.kernel, bridge, lambda instruction, arguments:
            lower_call(instruction, {key: bridge.value(value) for key, value in arguments.items()}))
        self.revalidate()
        return result


def select_store_epochs(schedule, attributes, n, dtype):
    """Return a checked consumed alternative, or unchanged source plus refusal."""
    selection = attributes.get('dense_store_epochs', knob('DENSE_STORE_EPOCHS',
        default='off', values=('off', 'on'), root=Path(__file__).resolve().parents[1]))
    if selection not in ('off', 'on'):
        raise ValueError('DENSE_STORE_EPOCHS must be off or on')
    if selection == 'off':
        return schedule, None
    scales = attributes.get('acc_scale_by_column')
    if scales is not None and len({isa.f32_bits(scale) for scale in scales}) != 1:
        return schedule, {'selection': 'on', 'applied': False, 'refusal': 'mixed column configurations'}
    ep = attributes.get('epilogue', ())
    if any('pool' in str(stage).lower() for stage in ep) or ('requant' in ep and attributes.get('requant_shift') is not None):
        return schedule, {'selection': 'on', 'applied': False, 'refusal': 'nonuniform or unsupported readout stage'}
    scale = scales[0] if scales else attributes.get('acc_scale', 1.0)
    activation = int(bool(attributes.get('relu', False) or 'relu' in ep))
    if dtype == 'i32' and (scales is not None or float(scale) != 1.0 or activation):
        return schedule, {'selection': 'on', 'applied': False, 'refusal': 'full-width epilogue is unsupported'}
    if dtype not in ('i8', 'i32'):
        return schedule, {'selection': 'on', 'applied': False, 'refusal': 'unsupported readout dtype'}
    command = isa.config_st(n * (4 if dtype == 'i32' else 1), scale=scale, activation=activation)
    schedule.revalidate()
    from merlin.compile.scheduling.dense_readout_overlap import DenseReadoutOverlapResult
    disjoint_owner = type(schedule) is DenseReadoutOverlapResult
    try:
        actual, regions = _transform(schedule.kernel, prove=True, disjoint_owner=disjoint_owner)
    except ValueError as exc:
        return schedule, {'selection': 'on', 'applied': False, 'refusal': str(exc)}
    if not regions:
        return schedule, {'selection': 'on', 'applied': False, 'refusal': 'no multi-store ACC region'}
    admission = _admission()
    admission['store_configuration_snapshot'] = 'not assumed; one uniform complete CONFIG_ST per region'
    report = dict(selection='on', applied=True, source_kernel_sha256=schedule.kernel.digest(),
        kernel_sha256=actual.digest(), uniform_config=list(command), regions=regions,
        readiness='real begin drain after all original producers and seeds',
        reuse=('original mixed-epoch full drain before either disjoint ACC window reuse; '
               'separate store regions close before the following effect'
               if disjoint_owner else
               'real closing drain before outer loop backedges or any non-store command'),
        data_projection='original store operands and loop order unchanged', admission=admission)
    result = DenseStoreEpochResult(schedule, actual,
        dict(schedule.realization, kernel_sha256=actual.digest(), store_epochs=report),
        command, schedule.kernel.digest(), disjoint_owner)
    result.revalidate()
    return result, report
