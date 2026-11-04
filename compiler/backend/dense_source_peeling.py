"""Ordered dense source-control specialization after target schedule checking.

No device effect, ownership boundary or command order changes. The library's
ordered partition certificate owns each original index exactly once. Folding
only substitutes singleton binders or drops impossible Select branches over
literal counted domains; all original call fields remain intact.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.loop_schedule import partition_counted_loop
from merlin.sched.ir import Loop, Select, Sum, Scaled, Ptr, Const, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.sched.primitives import NotApplicable, _map_calls, _subst_call, _prune_unreachable_selects
from merlin.sched.ir.events import has_carried_events


def enabled():
    return knob('DENSE_SOURCE_PEEL', default='off', values=('off', 'on'),
                root=Path(__file__).resolve().parents[1]) == 'on'


def _fold(body):
    result = []
    for node in body:
        if type(node) is not Loop:
            result.append(node)
        elif node.extent == 1:
            # Exact one-element domain, including every operand/config field.
            result.extend(_fold(_map_calls(node.body,
                lambda call: _subst_call(call, node.var, Const(0)))))
        else:
            result.append(replace(node, body=_fold(
                _prune_unreachable_selects(node.body, node.var, node.extent))))
    return tuple(result)


def _tests(value, variable):
    if type(value) is Select:
        return ((value.equals == 0 and value.var == variable)
                or _tests(value.then, variable) or _tests(value.other, variable))
    if type(value) is Sum:
        return any(_tests(t, variable) for t in value.terms)
    if type(value) is Scaled:
        return _tests(value.expr, variable)
    if type(value) is Ptr:
        return _tests(value.offset, variable)
    return False


def _first_predicate(body, path=()):
    def uses(body, variable):
        return any(uses(s.body, variable) if type(s) is Loop else
            any(_tests(v, variable) for _, v in (*s.args, *s.sets, *s.assumes))
            for s in body)
    for node in body:
        if type(node) is Loop:
            here = (*path, node.var)
            if node.extent > 1 and uses(node.body, node.var):
                return here
            nested = _first_predicate(node.body, here)
            if nested:
                return nested
    return None


def _specialize(kernel):
    return replace(kernel, body=_fold(kernel.body))


@dataclass(frozen=True)
class DenseSourcePeelResult:
    original: object
    kernel: object
    realization: dict
    custody: tuple
    source_identity: str

    def revalidate(self):
        self.original.revalidate()
        if self.original.kernel.digest() != self.source_identity:
            raise ValueError('dense source peeling source changed')
        expected = _specialize(self.original.kernel)
        for before, partition in self.custody:
            if expected != before:
                raise ValueError('dense source peeling ordered custody changed')
            partition.verify_current(before, partition.kernel)
            expected = _specialize(partition.kernel)
        if expected != self.kernel or check_structure(expected):
            raise ValueError('dense source peeling specialization changed')
        check_integer_domain(expected, 64)

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        bridge = FnBuilderIndexAdapter(builder)
        result = emit_index_program(self.kernel, bridge, lambda instruction, arguments:
            lower_call(instruction, {k: bridge.value(v) for k, v in arguments.items()}))
        self.revalidate()
        return result


def select(schedule):
    from merlin.compile.scheduling.dense_readout_overlap import DenseReadoutOverlapResult
    # The store-epoch owner currently dispatches this exact typed admission.
    # Preserve its existing terminal consumption when both options are enabled.
    if type(schedule) is DenseReadoutOverlapResult or has_carried_events(schedule.kernel.body):
        return schedule
    schedule.revalidate()
    actual = _specialize(schedule.kernel)
    custody = []
    for _ in range(16):
        path = _first_predicate(actual.body)
        if path is None:
            break
        try:
            partition = partition_counted_loop(actual, path, cuts=(1,),
                index_bits=64, max_static_operations=4096)
        except NotApplicable:
            # A bounded proposal may exceed the static body budget. Keep any
            # earlier certified partitions; never silence a failed verifier.
            break
        custody.append((actual, partition))
        actual = _specialize(partition.kernel)
    if not custody:
        return schedule
    check_integer_domain(actual, 64)
    report = dict(selection='on', applied=True,
        source_kernel_sha256=schedule.kernel.digest(), kernel_sha256=actual.digest(),
        ordered_partitions=[dict(path=list(p.request.path), cuts=list(p.request.cuts),
            regions=[dict(path=list(r.path), start=r.start, stop=r.stop) for r in p.regions])
            for _, p in custody],
        proof='library exact ordered interval custody plus literal-domain Select folding',
        device_effects='same ordered calls, operands, metadata and full completion cuts',
        static_partition_bound=16)
    result = DenseSourcePeelResult(schedule, actual,
        dict(schedule.realization, kernel_sha256=actual.digest(), source_peeling=report),
        tuple(custody), schedule.kernel.digest())
    result.revalidate()
    return result
