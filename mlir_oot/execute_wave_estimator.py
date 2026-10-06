"""Target-derived execute wave geometry with explicit dispatch uncertainty.

Rows are mesh feed geometry, not CPU/DMA/total cycles. Source command order
does not reveal cmd.valid or hazards: paired and serial scenarios are retained.
An individual resolved wave uses the actual effective A/B/D port state, not
the preceding logical B weight tile as a substitute for the D port.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

from .tables import isa


class WaveProofError(ValueError):
    pass


def _tokens(source):
    """Narrow Scala tokenization; comments cannot create hardware facts."""
    tokens = []
    i = 0
    while i < len(source):
        if source[i].isspace():
            i += 1
        elif source.startswith('//', i):
            end = source.find('\n', i + 2)
            i = len(source) if end < 0 else end
        elif source.startswith('/*', i):
            depth = 1
            i += 2
            while i < len(source) and depth:
                if source.startswith('/*', i):
                    depth += 1
                    i += 2
                elif source.startswith('*/', i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise WaveProofError('unterminated Scala comment')
        elif source[i] == '"':
            i += 1
            while i < len(source) and source[i] != '"':
                i += 2 if source[i] == '\\' else 1
            if i >= len(source):
                raise WaveProofError('unterminated Scala string')
            i += 1
            tokens.append('<string>')
        elif source[i].isalnum() or source[i] == '_':
            end = i + 1
            while end < len(source) and (source[end].isalnum() or source[end] == '_'):
                end += 1
            tokens.append(source[i:end]); i = end
        else:
            tokens.append(source[i]); i += 1
    return tokens


def _unique(tokens, pattern):
    matches = [i for i in range(len(tokens) - len(pattern) + 1)
               if tokens[i:i + len(pattern)] == pattern]
    if len(matches) != 1:
        raise WaveProofError('hardware wave rule is absent or ambiguous')
    return matches[0]


def _body_after(tokens, pattern):
    start = _unique(tokens, _tokens(pattern)) + len(_tokens(pattern))
    depth = 1
    end = start
    while end < len(tokens) and depth:
        depth += int(tokens[end] == '{') - int(tokens[end] == '}')
        end += 1
    if depth:
        raise WaveProofError('unterminated hardware rule')
    return tokens[start:end - 1]


@dataclass(frozen=True)
class ExecuteWaveFacts:
    dimension: int
    inactive_a_rows: int
    inactive_b_rows: int
    minimum_resident_rows: int
    rtl_sha256: str
    rule: str = 'WS/no transposer/effective D garbage'

    def __post_init__(self):
        if any(type(v) is not int or not 1 <= v <= self.dimension
               for v in (self.dimension, self.inactive_a_rows, self.inactive_b_rows,
                         self.minimum_resident_rows)):
            raise WaveProofError('invalid execute wave hardware dimensions')


def derive_facts(path: Path, *, dimension: int) -> ExecuteWaveFacts:
    """Accept the exact supported controller rule, deriving its integer limits."""
    source = Path(path).read_bytes()
    tokens = _tokens(source.decode())
    _unique(tokens, _tokens('val total_rows = WireInit(block_size.U)'))
    guard = _tokens('when (current_dataflow === Dataflow.WS.id.U && d_garbage && '
                    '!a_should_be_fed_into_transposer && !b_should_be_fed_into_transposer '
                    '&& !d_should_be_fed_into_transposer) {')
    body = _body_after(tokens, ' '.join(guard))
    prefix_a = _tokens('val rows_a = Mux(a_garbage,')
    prefix_b = _tokens('val rows_b = Mux(b_garbage,')
    prefix_limit = _tokens('total_rows := maxOf(maxOf(rows_a, rows_b),')
    values = []
    for prefix, suffix in ((prefix_a, _tokens('.U, a_rows)')),
                           (prefix_b, _tokens('.U, b_rows)')),
                           (prefix_limit, _tokens('.U)'))):
        index = _unique(body, prefix) + len(prefix)
        if index >= len(body) or not body[index].isdecimal() or body[index + 1:index + 1 + len(suffix)] != suffix:
            raise WaveProofError('unsupported hardware row expression')
        values.append(int(body[index]))
    # Extra assignments or conditionals could weaken the accepted rule.
    expected = _tokens(f'val rows_a = Mux(a_garbage, {values[0]}.U, a_rows) '
        f'val rows_b = Mux(b_garbage, {values[1]}.U, b_rows) '
        f'total_rows := maxOf(maxOf(rows_a, rows_b), {values[2]}.U)')
    if body != expected:
        raise WaveProofError('hardware row rule has unsupported extra statements')
    assignments = [i for i in range(len(tokens) - 2)
                   if tokens[i:i + 3] == ['total_rows', ':', '=']
                   and (i == 0 or tokens[i - 1] != '.')]
    if len(assignments) != 1:
        raise WaveProofError('controller wave extent has extra assignments')
    # Port activation and transpose mapping are part of the same proof. In a
    # single multiply D is inactive regardless of the preceding preload.
    for statement in (
        'val start_inputting_d = WireInit(false.B)',
        'val d_garbage = d_address_rs1.is_garbage() || !start_inputting_d',
        'val a_garbage = a_address_rs1.is_garbage() || !start_inputting_a',
        'val b_garbage = b_address_rs2.is_garbage() || !start_inputting_b',
        'val a_should_be_fed_into_transposer = Mux(current_dataflow === Dataflow.OS.id.U, !a_transpose, a_transpose)',
        'val b_should_be_fed_into_transposer = current_dataflow === Dataflow.OS.id.U && bd_transpose',
        'val d_should_be_fed_into_transposer = current_dataflow === Dataflow.WS.id.U && bd_transpose',
    ):
        _unique(tokens, _tokens(statement))
    for prefix in ('when(perform_single_preload) {', '.elsewhen(perform_mul_pre) {',
                   '.elsewhen(perform_single_mul) {', '.elsewhen(DoComputes(0)) {'):
        body = _body_after(tokens, prefix)
        d_assignments = sum(body[i:i + 3] == ['start_inputting_d', ':', '=']
                            for i in range(len(body) - 2))
        if prefix.endswith('perform_single_mul) {') or prefix.endswith('DoComputes(0)) {'):
            if d_assignments:
                raise WaveProofError('single multiply activates D unexpectedly')
            for port in ('a', 'b'):
                _unique(body, _tokens(f'start_inputting_{port} := !{port}_should_be_fed_into_transposer'))
        else:
            if d_assignments != 1:
                raise WaveProofError('ambiguous D activation')
            _unique(body, _tokens('start_inputting_d := true.B'))
    return ExecuteWaveFacts(dimension, *values, hashlib.sha256(source).hexdigest())


@dataclass(frozen=True)
class WaveState:
    mode: str | None
    dataflow: int | None
    a_rows: int | None
    b_rows: int | None
    a_garbage: bool | None
    b_garbage: bool | None
    d_garbage: bool | None
    a_transposer: bool | None
    b_transposer: bool | None
    d_transposer: bool | None


def estimate_wave(facts: ExecuteWaveFacts, state: WaveState):
    """Resolve a controller feed wave or conservatively retain DIM geometry."""
    for value in (state.a_garbage, state.b_garbage, state.d_garbage,
                  state.a_transposer, state.b_transposer, state.d_transposer):
        if value is not None and type(value) is not bool:
            raise WaveProofError('port activity and transpose facts must be booleans')
    result = dict(mode=state.mode, operand_state=asdict(state), rows=facts.dimension,
                  classification='fixed_dimension', reason=None, rtl_sha256=facts.rtl_sha256)
    if state.mode not in ('single_preload', 'single_mul', 'mul_pre'):
        result.update(classification='uncertain', reason='dispatch mode unresolved'); return result
    if state.mode == 'single_mul' and state.d_garbage is False:
        raise WaveProofError('single multiply has an inactive D port')
    if state.dataflow not in (isa.WEIGHT_STATIONARY, isa.OUTPUT_STATIONARY):
        result.update(classification='uncertain', reason='dataflow unresolved'); return result
    if state.dataflow != isa.WEIGHT_STATIONARY:
        result['reason'] = 'variable row rule applies only to WS'; return result
    transposers = (state.a_transposer, state.b_transposer, state.d_transposer)
    if any(v is True for v in transposers):
        result['reason'] = 'transposer is active'; return result
    if state.d_garbage is False:
        result['reason'] = 'D port is not garbage'; return result
    if any(v is None for v in transposers) or state.d_garbage is None:
        result.update(classification='uncertain', reason='D or transposer state unresolved'); return result
    rows = []
    for value, garbage, inactive in ((state.a_rows, state.a_garbage, facts.inactive_a_rows),
                                    (state.b_rows, state.b_garbage, facts.inactive_b_rows)):
        if garbage is None or (not garbage and value is None):
            result.update(classification='uncertain', reason='active operand rows unresolved'); return result
        if garbage:
            rows.append(inactive)
        elif type(value) is not int or not 1 <= value <= facts.dimension:
            raise WaveProofError('active operand row extent exceeds hardware dimension')
        else:
            rows.append(value)
    result.update(rows=max(*rows, facts.minimum_resident_rows), classification='variable_ws_rows',
                  reason='proved WS, all transposers disabled, effective D garbage')
    return result


@dataclass(frozen=True)
class PrimitiveCommand:
    kind: str
    fields: Mapping[str, int | bool | None]


def commands_from_function(function, *, pointer_index_bits, max_steps=10000000):
    """Decode verified golden primitive IR, using core for all LLVM traversal.

    The current normal lowerer is the encoding authority. In particular,
    config_ex forwarding is checked against encoder calls; missing attributes
    do not independently justify assumed transpose settings.
    """
    from merlin.llvmlower.static_llvm_cfg import StaticInt, StaticPointer, trace_static_function
    from .ir import gemmini_dialect as G
    from .golden_device_lower import _encoded

    function.verify()
    arguments = [StaticPointer(i, StaticInt(0, pointer_index_bits))
                 for i, _ in enumerate(function.body.blocks.first.args)]
    for step in trace_static_function(function, arguments,
            observe=lambda op: isinstance(op, G._GemminiOp),
            pointer_index_bits=pointer_index_bits, max_steps=max_steps):
        op = step.operation
        fields = {}
        if isinstance(op, G.ConfigExOp):
            encoded = _encoded(op)
            choices = [(flow, ta, tb) for flow in (isa.WEIGHT_STATIONARY, isa.OUTPUT_STATIONARY)
                       for ta in (False, True) for tb in (False, True)
                       if isa.config_ex(dataflow=flow, a_transpose=ta, b_transpose=tb) == encoded]
            if len(choices) == 1:
                flow, ta, tb = choices[0]
                fields = dict(dataflow=flow, a_transpose=ta, b_transpose=tb)
        elif isinstance(op, G.PreloadOp):
            fields = {key: op.a(key) for key in ('bd', 'bd_rows', 'bd_cols', 'c', 'c_rows', 'c_cols')}
            if step.inputs:
                if len(step.inputs) != 1 or not isinstance(step.inputs[0], StaticInt):
                    raise WaveProofError('dynamic C row did not resolve to an integer')
                row = step.inputs[0].value
                if not 0 <= row <= op.a('c_max'):
                    raise WaveProofError('executed dynamic C violates declared accumulator row range')
                fields['c'] = isa.acc_addr(row,accumulate=bool(op.a('c_accumulate',0)))
        elif isinstance(op, G.ComputeOp):
            a = op.a('a')
            if step.inputs:
                if len(step.inputs) != 1 or not isinstance(step.inputs[0], StaticInt):
                    raise WaveProofError('dynamic A operand did not resolve to an integer')
                a = step.inputs[0].value
                if not 0 <= a <= op.a('a_max'):
                    raise WaveProofError('executed dynamic A violates declared scratchpad range')
            fields = dict(a=a, a_rows=op.a('a_rows'), a_cols=op.a('a_cols'),
                          bd=op.a('bd', isa.GARBAGE_ADDR), bd_rows=op.a('bd_rows', isa.DIM),
                          bd_cols=op.a('bd_cols', isa.DIM), accumulate=bool(op.a('accumulate', False)))
        yield PrimitiveCommand(op.name.removeprefix('gemmini.'), fields)


def estimate_commands(commands: Iterable[PrimitiveCommand], facts: ExecuteWaveFacts, *, retain_records=True):
    """Bound per-compute feed geometry under conditional source execute order.

    Single multiply and multiply with following preload are alternatives,
    dependent on cmd.valid and hazards. These are not complete mesh timelines:
    standalone preload waves are excluded. Consuming the input to completion is
    mandatory; errors propagate and no completed report is returned.
    """
    counts = {}

    def relevant():
        for command in commands:
            counts[command.kind] = counts.get(command.kind, 0) + 1
            if command.kind in ('config_ex', 'preload', 'compute', 'fence', 'flush'):
                yield command

    stream = iter(relevant())
    command = next(stream, None)
    records = []
    summaries = {}
    possible_total = conservative_total = 0
    configuration = None
    active_weight = pending_weight = None
    compute_index = 0
    while command is not None:
        next_command = next(stream, None)
        fields = command.fields
        if command.kind == 'config_ex':
            configuration = fields
            active_weight = pending_weight = None
        elif command.kind == 'flush':
            active_weight = pending_weight = None
        elif command.kind == 'preload':
            pending_weight = (dict(address=fields['bd'], rows=fields['bd_rows'], cols=fields['bd_cols'])
                              if type(fields.get('bd')) is int and fields['bd'] != isa.GARBAGE_ADDR
                              and type(fields.get('bd_rows')) is int and type(fields.get('bd_cols')) is int
                              and (configuration or {}).get('dataflow') == isa.WEIGHT_STATIONARY else None)
        elif command.kind == 'compute':
            if fields.get('accumulate', 0) == 0:
                active_weight, pending_weight = pending_weight, None
            cfg = configuration or {}
            # All known golden config_ex fields are forwarded below. Unknown
            # configuration must not invent an untransposed datapath.
            trans_a, trans_d = cfg.get('a_transpose'), cfg.get('b_transpose')
            flow = cfg.get('dataflow')
            no_transpose = trans_a is False and trans_d is False
            a_garbage = fields['a'] == isa.GARBAGE_ADDR if type(fields.get('a')) is int else None
            b_garbage = fields['bd'] == isa.GARBAGE_ADDR if type(fields.get('bd')) is int else None
            single = WaveState('single_mul', flow, fields.get('a_rows'), fields.get('bd_rows'),
                a_garbage, b_garbage, True, False if no_transpose else None,
                False if no_transpose else None, False if no_transpose else None)
            possible = [estimate_wave(facts, single)]
            if next_command and next_command.kind == 'preload':
                combined = WaveState('mul_pre', flow, fields.get('a_rows'), fields.get('bd_rows'),
                    a_garbage, b_garbage, (next_command.fields['bd'] == isa.GARBAGE_ADDR
                        if type(next_command.fields.get('bd')) is int else None),
                    single.a_transposer, single.b_transposer, single.d_transposer)
                possible.append(estimate_wave(facts, combined))
            record = dict(compute_index=compute_index, active_weight=active_weight,
                weight_residency_proved=active_weight is not None, possible_dispatch=possible,
                rows_min_possible=min(v['rows'] for v in possible),
                rows_conservative=max(v['rows'] for v in possible),
                dispatch_independent_rows=len({v['rows'] for v in possible}) == 1,
                classification=('uncertain_state' if any(v['classification'] == 'uncertain' for v in possible)
                    else 'uncertain_dispatch' if len({v['rows'] for v in possible}) > 1
                    else 'mode_independent_geometry'))
            if retain_records:
                records.append(record)
            key = (record['rows_min_possible'], record['rows_conservative'],
                   active_weight is not None, tuple(v['classification'] for v in possible))
            summaries[key] = summaries.get(key, 0) + 1
            possible_total += record['rows_min_possible']
            conservative_total += record['rows_conservative']
            compute_index += 1
        command = next_command
    return dict(raw_command_counts=counts, compute_records=records,
        complete_static_trace=True,
        geometry_histogram=[dict(rows_min_possible=k[0], rows_conservative=k[1],
            weight_residency_proved=k[2], wave_classifications=k[3], count=v)
            for k, v in sorted(summaries.items())],
        padded_compute_geometry=compute_index * facts.dimension,
        possible_compute_rows=possible_total,
        conservative_compute_rows=conservative_total,
        dispatch_mode_resolved=False,
        scope='Per-compute feed geometry under source execute order. Unresolved following-preload availability retained; totals exclude standalone preload waves and all CPU/DMA/stall/startup/drain costs.')
