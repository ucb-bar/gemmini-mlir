"""Source-bound activity from consumed shared schedules, without a timing model.

Dense effects are decoded by the caller's existing reversible library adapter.
Spatial effects are observed while the shared structured emitter actually calls
the target adapter. Neither path chooses, reorders or replays a device schedule.
Intervals use logical effect ordinals, never cycles or asynchronous occupancy.
"""
from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path

MAX_EFFECTS = 100000
MAX_CELLS = 1000000
_ENABLED = ContextVar('gemmini_activity_enabled', default=False)


def enable_activity():
    """Opt in before compilation; return a token for embedding callers to reset."""
    return _ENABLED.set(True)


def reset_activity(token):
    _ENABLED.reset(token)


class ActivityUnavailable(ValueError):
    pass


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _geometry():
    from merlin.targetgen import address_space as AS
    payload = (Path(__file__).resolve().parent/'data/rtl_facts.json').read_bytes()
    space = AS.derive_address_space('gemmini', facts=json.loads(payload))
    operand, acc = AS.operand_store(space), AS.accumulator_store(space)
    return dict(facts_sha256=hashlib.sha256(payload).hexdigest(),
                operand_row_bytes=operand.store.row_bytes,
                accumulator_row_bytes=acc.store.row_bytes,
                operand_capacity_rows=operand.capacity_rows(),
                accumulator_capacity_rows=acc.capacity_rows())


def _unavailable(identity, reason):
    return dict(schema='gemmini.schedule_activity.v1', status='unavailable',
                schedule_sha256=identity, reason=str(reason), timing_authority='NONE')


class _Summary:
    def __init__(self, identity, output_dtype):
        self.identity = identity
        self.geometry = _geometry()
        self.width = 1 if str(output_dtype) in ('i8', 'int8') else 4
        self.counts, self.traffic = Counter(), Counter()
        self.ordinal = 0
        self.cells, self.finished = {}, []
        self.digest = hashlib.sha256()
        self.preload, self.weight = None, None

    def access(self, bank, row, rows, cols, *, write=False, continuation=False):
        # Lane coordinates are exact logical elements, with i8 SP/i32 ACC.
        width = 1 if bank == 'scratchpad' else 4
        row_bytes = self.geometry['operand_row_bytes' if bank == 'scratchpad' else 'accumulator_row_bytes']
        capacity = self.geometry['operand_capacity_rows' if bank == 'scratchpad' else 'accumulator_capacity_rows']
        if not (0 <= row <= row+rows <= capacity and 0 <= cols*width <= row_bytes):
            raise ActivityUnavailable('projected local access exceeds selected row geometry')
        if len(self.finished)+len(self.cells)+rows*cols > MAX_CELLS:
            raise ActivityUnavailable('local generation tracking exceeds explicit cell budget')
        for r in range(row, row+rows):
            for c in range(cols):
                key = bank, r, c
                previous = self.cells.get(key)
                if write and not continuation:
                    if previous is not None:
                        self.finished.append((key, *previous))
                    self.cells[key] = [self.ordinal, self.ordinal]
                elif previous is not None:
                    previous[1] = self.ordinal
                else:
                    raise ActivityUnavailable('local read/continuation lacks a projected producer')

    def event(self, kind, f):
        if self.ordinal >= MAX_EFFECTS:
            raise ActivityUnavailable('dynamic effects exceed explicit enumeration budget')
        self.digest.update(json.dumps([kind, f], sort_keys=True, separators=(',', ':')).encode())
        self.counts[kind] += 1
        rows, cols = f.get('rows', 0), f.get('columns', 0)
        if kind in ('load_input', 'load_weight', 'stage_input_rows', 'fill_input_rows'):
            self.access('scratchpad', f['local_row'], rows, cols, write=True)
            if kind != 'fill_input_rows' and not f.get('zero', False):
                self.traffic['external_to_scratchpad_requested_bytes'] += rows*cols
            else:
                self.traffic['zero_source_to_scratchpad_logical_bytes'] += rows*cols
        elif kind == 'initialize_bias':
            self.access('accumulator', f['accumulator_row'], rows, cols, write=True)
            self.traffic['external_to_accumulator_requested_bytes'] += rows*cols*4
        elif kind == 'preload':
            if f.get('rhs_row') is not None:
                self.access('scratchpad', f['rhs_row'], f['depth'], cols)
                self.weight = f['rhs_row'], f['depth'], cols
            elif self.weight is None:
                raise ActivityUnavailable('retained array weights lack a projected preload')
            self.preload = dict(f)
        elif kind == 'compute':
            self.access('scratchpad', f['lhs_row'], rows, f['depth'])
            if 'rhs_row' in f:  # Spatial callback fuses physical PRELOAD/COMPUTE.
                self.access('scratchpad', f['rhs_row'], f['depth'], cols)
            else:
                if self.preload is None:
                    raise ActivityUnavailable('compute lacks the actual preceding preload')
                cols = self.preload['columns']
                f = dict(f, accumulator_row=self.preload['accumulator_row'],
                         first_reduction=not self.preload['continue'])
            continued = bool(f.get('bias', False) or not f.get('first_reduction', True))
            self.access('accumulator', f['accumulator_row'], rows, cols,
                        write=True, continuation=continued)
            self.traffic['scheduled_integer_macs'] += rows*f['depth']*cols
        elif kind == 'store':
            self.access('accumulator', f['accumulator_row'], rows, cols)
            self.traffic['accumulator_to_external_requested_bytes'] += rows*cols*self.width
        elif kind not in ('completion', 'finish_raw_tile'):
            raise ActivityUnavailable('unmodeled typed effect '+kind)
        self.ordinal += 1

    def report(self):
        generations = self.finished + [(key, *value) for key, value in self.cells.items()]
        # Compact equal-lifetime adjacent lanes; rows retain explicit addresses.
        groups = {}
        for (bank, row, col), first, last in generations:
            groups.setdefault((bank, row, first, last), []).append(col)
        intervals = []
        for (bank, row, first, last), cols in sorted(groups.items()):
            start = previous = sorted(cols)[0]
            for col in sorted(cols)[1:]+[None]:
                if col is not None and col == previous+1:
                    previous = col
                    continue
                intervals.append(dict(bank=bank, row=row, first_lane=start,
                    lanes=previous-start+1, element_bytes=1 if bank=='scratchpad' else 4,
                    begin_effect=first, end_effect_exclusive=last+1))
                start = previous = col
        return dict(schema='gemmini.schedule_activity.v1', status='available',
            schedule_sha256=self.identity, geometry=self.geometry,
            basis='actual consumed typed schedule effects; exact bounded integer domains',
            denominator=dict(value=self.ordinal, unit='dynamic logical effects per observed schedule invocation',
                             includes='load/preload/compute/store/completion; excludes scalar host instructions'),
            dynamic_effect_counts=dict(sorted(self.counts.items())),
            memory_boundary_traffic=dict(sorted(self.traffic.items())),
            traffic_units='logical bytes requested by transfers, excluding protocol/cache-line traffic; MACs separately named',
            mac_scope='scheduled products, including any zero-padding products; not a useful-work utilization numerator',
            effects_sha256=self.digest.hexdigest(), local_data_intervals=intervals,
            interval_axis='serial logical effect ordinal; latest write through last logical use',
            asynchronous_reservation_lifetimes='UNKNOWN: no service completion timestamps',
            elapsed_cycles=None, capacity_normalized_activity=None, timing_authority='NONE',
            occupancy='UNKNOWN: these intervals do not measure physical occupancy or overlap')


def dense_activity(result, adapter, problem, *, output_dtype='i32', project_member=None):
    """Analyze the exact emitted library Kernel through its actual effect decoder.

    Call after successful ordinary emission. Revalidation binds the observation
    to the checked current kernel. Enumeration is bounded and never samples.
    """
    from merlin.sched.ir import Call, Loop, evaluate
    from merlin.sched.ir.expr import Expr
    from merlin.compile.scheduling.block_schedule import Load, InitializeAccumulator, Preload, Compute, Store
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
    identity = result.kernel.digest()
    if not _ENABLED.get(): return _unavailable(identity, 'optional activity analysis not requested')
    try:
        result.revalidate()
        summary = _Summary(identity, output_dtype)
        def closed(value, env):
            if isinstance(value, Expr): return evaluate(value, env)
            if isinstance(value, tuple): return tuple(closed(x, env) for x in value)
            if is_dataclass(value):
                return replace(value, **{f.name:closed(getattr(value,f.name), env) for f in fields(value)})
            return value
        def walk(body, env):
            for call in body:
                if isinstance(call, Loop):
                    if call.iter_args or call.yields or call.results:
                        raise ActivityUnavailable('carried completion requires a separate lifetime observer')
                    if call.extent > MAX_EFFECTS: raise ActivityUnavailable('counted loop exceeds activity budget')
                    for index in range(call.extent): walk(call.body, dict(env, **{call.var:index}))
                    continue
                if not isinstance(call, Call): raise ActivityUnavailable('unknown structured schedule node')
                if call.instr == DRAIN:
                    summary.event('completion', {}); continue
                actual = closed(call, env)
                op = adapter.project(actual) if project_member is None else project_member(actual)[1]
                if isinstance(op, Load):
                    if op.gather is not None: raise ActivityUnavailable('dense gather needs its original window observer')
                    summary.event('load_weight' if op.role=='weight' else 'load_input',
                        dict(local_row=op.row,rows=op.rows,columns=op.cols))
                elif isinstance(op, InitializeAccumulator):
                    if op.additional_bias: raise ActivityUnavailable('additional bias transport needs explicit lowered transfer effects')
                    summary.event('initialize_bias', dict(accumulator_row=op.row,rows=op.rows,columns=op.cols))
                elif isinstance(op, Preload):
                    summary.event('preload', dict(rhs_row=op.weight_row,accumulator_row=op.accumulator_row,
                        rows=op.rows,columns=op.cols,depth=op.weight_rows,**{'continue':op.accumulate}))
                elif isinstance(op, Compute):
                    summary.event('compute', dict(lhs_row=op.input_row,rows=op.rows,depth=op.cols))
                elif isinstance(op, Store):
                    summary.event('store',dict(accumulator_row=op.accumulator_row,rows=op.rows,columns=op.cols))
                else: raise ActivityUnavailable('unmodeled projected dense effect '+type(op).__name__)
        walk(result.kernel.body, {})
        result.revalidate()
        report = summary.report()
        report['effect_decoder'] = dict(configuration_identity=adapter.configuration_identity,
                                       numerical_policy_identity=adapter.numerical_policy_identity)
        report['logical_source_shape'] = [problem.m, problem.k, problem.n]
        return report
    except (ActivityUnavailable, ValueError, TypeError, KeyError, AttributeError) as error:
        return _unavailable(identity, error)


def _expression(value, variables):
    """Read actual xDSL scalar index definitions; tensor loads are never evaluated."""
    from xdsl.ir import SSAValue
    if value is None or type(value) in (int, bool, str): return ('literal', value)
    if not isinstance(value, SSAValue): raise ActivityUnavailable('non-scalar spatial effect operand')
    if value in variables: return ('variable', str(variables[value]))
    owner = value.owner
    name = getattr(owner, 'name', '')
    if name == 'llvm.mlir.constant': return ('literal', owner.value.value.data)
    operations = {'llvm.add':'add','llvm.sub':'sub','llvm.mul':'mul','llvm.udiv':'udiv',
                  'llvm.urem':'urem','llvm.and':'and','llvm.select':'select'}
    if name == 'llvm.icmp':
        return ('icmp', owner.predicate.value.data, *(_expression(x,variables) for x in owner.operands))
    if name in operations:
        return (operations[name], *(_expression(x,variables) for x in owner.operands))
    raise ActivityUnavailable('unmodeled spatial index definition '+name)


def _eval(expression, env):
    op, *args = expression
    if op == 'literal': return args[0]
    if op == 'variable': return env[args[0]]
    if op == 'icmp':
        predicate, left, right = args
        a,b = _eval(left,env),_eval(right,env)
        if predicate == 0: return a==b
        if predicate == 2: return a<b
        if predicate == 6: return a%(1<<64)<b%(1<<64)
        if predicate == 7: return a%(1<<64)<=b%(1<<64)
        raise ActivityUnavailable('unmodeled integer predicate')
    values = [_eval(x,env) for x in args]
    if op == 'select': return values[1] if values[0] else values[2]
    a,b = values
    if op == 'add': return a+b
    if op == 'sub': return a-b
    if op == 'mul': return a*b
    if op == 'and': return a&b
    if op == 'udiv': return (a%(1<<64))//(b%(1<<64))
    if op == 'urem': return (a%(1<<64))%(b%(1<<64))
    raise ActivityUnavailable('unmodeled index operation')


class _ControlObserver:
    def __init__(self, control):
        self.control, self.variables, self.body = control, {}, []
        self.serial, self.reason = 0, None

    def __getattr__(self, name): return getattr(self.control, name)

    def record(self, kind, fields):
        if self.reason is not None: return
        try:
            self.body.append(('effect', kind, {k:_expression(v,self.variables) for k,v in fields.items()}))
        except ActivityUnavailable as error: self.reason = str(error)

    def loop(self, start, stop, step, callback):
        parent, nested = self.body, []
        variable = self.serial; self.serial += 1
        parent.append(('loop', variable, start, stop, step, nested))
        def wrapped(index):
            # Specialized one-trip loops receive a literal SSA constant. It
            # already denotes its exact index and needs no lexical binding.
            bind = getattr(getattr(index, "owner", None), "name", None) != "llvm.mlir.constant"
            if bind: self.variables[index] = variable
            self.body = nested
            try: return callback(index)
            finally:
                self.body = parent
                if bind: del self.variables[index]
        return self.control.loop(start,stop,step,wrapped)

    def branch(self, predicate, yes, no):
        if type(predicate) is bool:
            return self.control.branch(predicate,yes,no)
        parent, branches = self.body, ([], [])
        try: expression = _expression(predicate,self.variables)
        except ActivityUnavailable as error:
            self.reason = str(error); expression = ('literal', False)
        parent.append(('branch', expression, *branches))
        def wrapped(callback, body):
            def invoke():
                self.body = body
                try: return callback()
                finally: self.body = parent
            return invoke
        return self.control.branch(predicate,wrapped(yes,branches[0]),wrapped(no,branches[1]))

    def report(self, schedule_identity, *, output_dtype='i32'):
        try:
            if self.reason: raise ActivityUnavailable(self.reason)
            summary = _Summary(schedule_identity, output_dtype)
            def walk(body, env):
                for node in body:
                    if node[0]=='effect':
                        summary.event(node[1],{k:_eval(v,env) for k,v in node[2].items()})
                    elif node[0]=='loop':
                        _,variable,start,stop,step,nested=node
                        if len(range(start,stop,step))>MAX_EFFECTS: raise ActivityUnavailable('spatial loop exceeds activity budget')
                        for i in range(start,stop,step): walk(nested,dict(env,**{str(variable):i}))
                    else:
                        walk(node[2] if _eval(node[1],env) else node[3],env)
            # Variables use string keys so copied lexical environments remain JSON-safe.
            walk(self.body,{})
            return summary.report()
        except (ActivityUnavailable, ValueError, TypeError, KeyError, AttributeError) as error:
            return _unavailable(schedule_identity,error)


def observe_control(builder, lower, complete):
    """Observe actual spatial callbacks without changing their values or effects."""
    if not _ENABLED.get():
        class Inactive:
            def report(self, identity, **kwargs):
                return _unavailable(identity, 'optional activity analysis not requested')
        return builder, lower, complete, Inactive()
    observer = _ControlObserver(builder)
    def wrapped_lower(kind, fields):
        observer.record(kind, fields)
        return lower(kind, fields)
    def wrapped_complete(kind, fields):
        observer.record('completion', {})
        return complete(kind, fields)
    return observer, wrapped_lower, wrapped_complete, observer
