"""Residual operand sums as scaled accumulate-on-load with a narrow readout.

Two byte tensors enter the accumulator through the load scale unit: each load
converts to f32, multiplies by its configured f32 scale with ties-to-even,
rounds to an integer and saturates to the operand byte (the contract's
``readout_semantics.operand_sum``). The first load overwrites an accumulator
tile, the second accumulates onto it, and the narrow readout applies the
activation and one f32 multiplier per column tile.

Selection is proof driven. A source region is either

* a dequantized sum ``Q(act(w_a DQ(a) + w_b DQ(b)))``, whose meaning is the
  exact rational single rounding of Merlin's reference; or
* staged byte operations: optional per-operand byte requantizations, an
  integer or f32 operand sum, then one byte readout.

Candidate register bits are accepted only when the device reproduces the
source for every operand pair of the full byte domain, column group by column
group. Anything else returns ``None`` and the existing exact integer path,
with its host readout, keeps the region. ``validate`` re-decodes the emitted
command words and checks, element by element, which source byte reached which
accumulator cell under which scale, in which order, and which store state
drained it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from fractions import Fraction
from math import lcm

import numpy as np


class Refused(ValueError):
    """The region is an operand sum this lowering cannot prove exact."""


def _bits(value) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _value(bits: int) -> np.float32:
    return np.float32(struct.unpack('<f', struct.pack('<I', int(bits)))[0])


# ---------------------------------------------------------------------------------------------
# Hardware semantics from the synced contract and the selected output path
# ---------------------------------------------------------------------------------------------

def load_semantics():
    """The declared scaled-load contract; anything unstated refuses."""
    from .readout_encoding import contract
    declared = (contract().get('readout_semantics') or {}).get('operand_sum')
    if not isinstance(declared, dict):
        raise Refused('UNKNOWN: contract declares no scaled operand-sum load')
    if (declared.get('schema') != 'operand_sum_contract_v1' or declared.get('operands') != 2
            or declared.get('scale_dtype') != 'f32' or declared.get('operand_rounding') != 'half_even'
            or declared.get('operand_saturates') is not True):
        raise Refused('UNKNOWN: scaled load semantics differ from the modeled f32 RNE saturating load')
    return dict(declared)


def readout_semantics(options, *, relu):
    """The narrow readout of the selected path; refuses unless every stage is modeled."""
    from .readout_encoding import supports_narrow_path
    path = (options or {}).get('path')
    if not isinstance(path, dict) or not path.get('exits', {}).get('requant'):
        raise Refused('output path declares no narrow readout exit')
    if not any(s.get('kind') == 'accumulate' for s in path.get('accumulator_stages') or ()):
        raise Refused('output path accumulates no loaded operand')
    if not supports_narrow_path(path):
        raise Refused('output path readout differs from the device scalar ABI')
    stages = {s['kind']: s for s in path['readout']}
    domain = None
    if relu:
        activation = stages.get('activation')
        if activation is None or activation.get('function') != 'relu':
            raise Refused('output path applies no rectifier')
        domain = activation.get('domain')
    return dict(output_dtype=stages['requant']['to'], clamp=list(stages['requant']['clamp']),
                activation_domain=domain)


def device(a, b, candidate, column, *, load, readout):
    """The emitted stream's value for operand codes ``a``/``b`` at ``column``.

    Both loads and the readout use Merlin's single f32 readout definition.
    """
    from merlin.runtime.exact_readout import f32_readout
    left = f32_readout(a, _value(candidate.load_bits[0]), load['operand_dtype'])
    right = f32_readout(b, _value(candidate.load_bits[1]), load['operand_dtype'])
    total = left.astype(np.int64) + right.astype(np.int64)
    multiplier = _value(candidate.multiplier_bits[column])
    if candidate.relu and readout['activation_domain'] == 'scaled':
        return np.maximum(f32_readout(total, multiplier, readout['output_dtype']), 0)
    return f32_readout(total, multiplier, readout['output_dtype'], relu=candidate.relu)


# ---------------------------------------------------------------------------------------------
# Source regions
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    load_bits: tuple[int, int]
    multiplier_bits: tuple[int, ...]
    relu: bool


@dataclass
class Region:
    form: str
    rounding: str
    reference: str
    left: object
    right: object
    result: object
    members: list
    rows: int
    cols: int
    source: object          # (a, b, column) -> exact source codes
    column_key: object      # column -> parameters the source reads at that column
    candidates: list


def _domain(dtype):
    info = np.iinfo({'i8': np.int8}[dtype])
    codes = np.arange(int(info.min), int(info.max) + 1, dtype=np.int64)
    a, b = np.meshgrid(codes, codes, indexing='ij')
    return a.reshape(-1), b.reshape(-1)


def _matrix(value):
    from .match import shape
    dims = shape(value)
    if not dims:
        raise Refused('operand sum needs a ranked static tensor')
    cols = dims[-1]
    return int(np.prod(dims[:-1])) if len(dims) > 1 else 1, int(cols)


def _rne(numerator, denominator):
    quotient = numerator // denominator
    remainder = numerator - quotient * denominator
    up = (2 * remainder > denominator) | ((2 * remainder == denominator) & (quotient % 2 == 1))
    return quotient + up.astype(object)


def _exact_sum(ma: Fraction, mb: Fraction, relu: bool, low: int, high: int):
    """Merlin's exact reference: one ties-to-even rounding of the rational sum."""
    denominator = lcm(ma.denominator, mb.denominator)
    wa, wb = ma.numerator * (denominator // ma.denominator), mb.numerator * (denominator // mb.denominator)

    def evaluate(a, b, _column):
        total = a.astype(object) * wa + b.astype(object) * wb
        if relu:
            total = np.maximum(total, 0)
        return np.clip(_rne(total, denominator), low, high).astype(np.int64)
    return evaluate


def dequantized_sum(op, options):
    """``Q(act(w_a DQ(a) + w_b DQ(b)))`` over identical full shapes, zero points 0."""
    from .match import analyse_body, dequant, generic_maps, identity_map, iterators, quant_sink, shape
    from merlin.runtime.exact_readout import form_multiplier
    if op.name != 'linalg.generic' or len(op.results) != 1 or any(i != 'parallel' for i in iterators(op)):
        return None
    n_in = len(op.operands) - 1
    maps = generic_maps(op)
    body = analyse_body(op)
    if body is None or not identity_map(maps[-1]) or list(op.regions[0].blocks[0].args[n_in].uses):
        return None
    terms = sorted(body.lin.terms.items())
    quantized = {i: dequant(op.operands[i]) for i, _ in terms}
    if len(terms) != 2 or any(q is None for q in quantized.values()):
        return None
    out = quant_sink(op.results[0])
    if out is None:
        return None
    if body.lin.c or any(q.zero_point or q.unsigned or q.dtype != 'i8' for q in quantized.values()):
        raise Refused('operand sum carries a constant, zero point or unsigned storage')
    if out.zero_point or out.unsigned:
        raise Refused('operand sum result carries a zero point or unsigned storage')
    (ia, wa), (ib, wb) = terms
    qa, qb = quantized[ia], quantized[ib]
    if not identity_map(maps[ia]) or not identity_map(maps[ib]) or not (
            shape(qa.value) == shape(qb.value) == shape(op.results[0])):
        raise Refused('broadcast operand sums keep the exact integer path')
    if qa.value is qb.value:
        raise Refused('a self sum has one operand, not two loads')
    ma = Fraction(wa) * Fraction(qa.scale) / Fraction(out.scale)
    mb = Fraction(wb) * Fraction(qb.scale) / Fraction(out.scale)
    if ma <= 0 or mb <= 0:
        raise Refused('a negated load minimum is unproved')
    scheme = (options or {}).get('scheme') or {}
    if scheme.get('operand_sum_rounding') == 'per_operand':
        raise Refused('per-operand rounding requires explicit per-operand requantization in the source')
    rows, cols = _matrix(op.results[0])
    largest = max(ma, mb)
    multiplier = _bits(form_multiplier(largest))
    candidates = [Candidate((_bits(form_multiplier(ma / largest)), _bits(form_multiplier(mb / largest))),
                            (multiplier,) * cols, body.relu)]
    if largest <= 1:
        candidates.append(Candidate((_bits(form_multiplier(ma)), _bits(form_multiplier(mb))),
                                    (_bits(form_multiplier(1)),) * cols, body.relu))
    members = [qa.dq, qb.dq, op, out.op]
    return Region('dequantized_sum', 'single_rounding', 'exact_rational_single_rounding',
                  qa.value, qb.value, out.value, members, rows, cols,
                  _exact_sum(ma, mb, body.relu, out.qmin, out.qmax), lambda _column: (), candidates)


# -- staged byte operations --------------------------------------------------------------------

_FLOATS = {'f32': np.float32}
_INTS = {'i8': 8, 'i16': 16, 'i32': 32, 'i64': 64}


def _wrap(values, width):
    span = 1 << width
    return ((values + (span >> 1)) % span) - (span >> 1)


def _scalar(op):
    from .match import _attr_of
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr
    attr = _attr_of(op, 'value')
    kind = str(op.results[0].type)
    if isinstance(attr, FloatAttr) and kind in _FLOATS:
        return _FLOATS[kind](attr.value.data)
    if isinstance(attr, IntegerAttr) and kind in _INTS:
        return np.int64(attr.value.data)
    raise Refused('unmodeled body constant')


def _evaluate(generic, inputs):
    """Exact elementwise semantics of a closed generic body over numpy arrays."""
    block = generic.regions[0].blocks[0]
    n_in = len(generic.operands) - 1
    if list(block.args[n_in].uses):
        raise Refused('body reads its destination')
    env = dict(zip(block.args[:n_in], inputs))
    for op in block.ops:
        name = op.name
        args = [env[v] for v in op.operands]
        kind = str(op.results[0].type) if op.results else None
        if name == 'linalg.yield':
            return env[op.operands[0]]
        if name == 'arith.constant':
            value = _scalar(op)
        elif name == 'arith.sitofp' and kind in _FLOATS:
            value = np.asarray(args[0], dtype=np.int64).astype(_FLOATS[kind])
        elif name == 'arith.extsi' and kind in _INTS:
            value = np.asarray(args[0], dtype=np.int64)
        elif name in ('arith.addi', 'arith.subi', 'arith.muli') and kind in _INTS:
            x, y = (np.asarray(v, dtype=object) for v in args)
            raw = x + y if name == 'arith.addi' else x - y if name == 'arith.subi' else x * y
            value = np.asarray(_wrap(raw, _INTS[kind]), dtype=np.int64)
        elif name in ('arith.addf', 'arith.subf', 'arith.mulf') and kind in _FLOATS:
            ufunc = {'arith.addf': np.add, 'arith.subf': np.subtract, 'arith.mulf': np.multiply}[name]
            with np.errstate(over='raise', invalid='raise'):
                value = ufunc(args[0], args[1], dtype=_FLOATS[kind])
        elif name in ('arith.maximumf', 'arith.minimumf') and kind in _FLOATS:
            value = (np.maximum if name == 'arith.maximumf' else np.minimum)(args[0], args[1]).astype(_FLOATS[kind])
        elif name in ('arith.maxsi', 'arith.minsi') and kind in _INTS:
            value = (np.maximum if name == 'arith.maxsi' else np.minimum)(args[0], args[1])
        elif name == 'math.roundeven' and kind in _FLOATS:
            value = np.rint(args[0]).astype(_FLOATS[kind])
        elif name == 'arith.fptosi' and kind in _INTS:
            source = np.asarray(args[0], dtype=np.float64)
            truncated = np.trunc(source)
            limit = 1 << (_INTS[kind] - 1)
            if not np.all(np.isfinite(source)) or np.any(truncated < -limit) or np.any(truncated >= limit):
                raise Refused('float-to-integer conversion is poison on part of the domain')
            value = truncated.astype(np.int64)
        else:
            raise Refused(f'unmodeled body operation {name}')
        if np.any(np.isnan(np.asarray(value, dtype=np.float64))):
            raise Refused('body produces NaN on the operand domain')
        env[op.results[0]] = value
    raise Refused('body has no yield')


class _Stage:
    """One rank-2 parallel generic: data inputs by position, constants by column."""

    def __init__(self, op, rows, cols):
        from .match import dense_values, generic_maps, identity_map, iterators, shape, trailing_block
        if (op.name != 'linalg.generic' or len(op.results) != 1 or len(op.regions[0].blocks) != 1
                or any(i != 'parallel' for i in iterators(op)) or len(iterators(op)) != 2):
            raise Refused('stage is not a rank-2 elementwise generic')
        maps = generic_maps(op)
        if not identity_map(maps[-1]) or shape(op.results[0]) != (rows, cols):
            raise Refused('stage result layout differs from the region')
        self.op, self.data, self.columns = op, [], {}
        for index, value in enumerate(op.operands[:-1]):
            constant = dense_values(value)
            if constant is None:
                if not identity_map(maps[index]) or shape(value) != (rows, cols):
                    raise Refused('stage data operand is broadcast or reshaped')
                self.data.append((index, value))
                continue
            stored = str(value.type.get_element_type())
            if stored not in _FLOATS and stored not in _INTS:
                raise Refused('stage constant has an unmodeled type')
            dtype = _FLOATS.get(stored, np.int64)
            if constant.ndim == 1 and trailing_block(maps[index]) == 1 and constant.size == cols:
                self.columns[index] = constant.astype(dtype)
            elif constant.size == 1 and not maps[index].results:
                self.columns[index] = np.full(cols, constant.reshape(-1)[0]).astype(dtype)
            else:
                raise Refused('stage constant varies along rows')

    def __call__(self, data, column):
        values = {}
        for (index, _), array in zip(self.data, data, strict=True):
            values[index] = array
        for index, column_values in self.columns.items():
            values[index] = column_values[column]
        return _evaluate(self.op, [values[i] for i in range(len(self.op.operands) - 1)])

    def key(self, column):
        return tuple((i, _bits(v[column]) if v.dtype == np.float32 else int(v[column]))
                     for i, v in sorted(self.columns.items()))

    def gains(self):
        floats = [v for v in self.columns.values() if v.dtype == np.float32]
        return floats[0] if len(floats) == 1 else None


def _single_user(value, user):
    from .match import users
    return users(value) == [user]


def _operand_stage(value, consumer, rows, cols):
    from .match import etype, producer
    owner = producer(value)
    if etype(value) != 'i8':
        raise Refused('summed operand is not a byte tensor')
    if owner is None or owner.name != 'linalg.generic' or not _single_user(value, consumer):
        return None
    try:
        stage = _Stage(owner, rows, cols)
    except Refused:
        return None
    if len(stage.data) != 1 or etype(stage.data[0][1]) != 'i8':
        return None
    return stage


def staged_sum(op, options):
    """Byte operands, optional per-operand requantization, a sum, one byte readout.

    ``op`` may be any member; only the first member in program order matches.
    """
    from .match import etype, producer, shape, users
    if op.name != 'linalg.generic' or len(op.results) != 1:
        return None
    readout = op
    for _ in range(3):
        if etype(readout.results[0]) == 'i8' and _readout_root(readout) is not None:
            break
        following = users(readout.results[0])
        if len(following) != 1 or following[0].name != 'linalg.generic':
            return None
        readout = following[0]
    else:
        return None
    rows_cols = shape(readout.results[0])
    if rows_cols is None or len(rows_cols) != 2:
        return None
    rows, cols = rows_cols
    final = _Stage(readout, rows, cols)
    summed = producer(final.data[0][1])
    total = _Stage(summed, rows, cols)
    if len(total.data) != 2 or etype(summed.results[0]) not in _INTS or not _single_user(summed.results[0], readout):
        raise Refused('operand sum is not a closed two-operand integer stage')
    stages = [_operand_stage(value, summed, rows, cols) for _, value in total.data]
    operands = [stage.data[0][1] if stage else value for stage, (_, value) in zip(stages, total.data)]
    if operands[0] is operands[1]:
        raise Refused('a self sum has one operand, not two loads')
    members = [s.op for s in stages if s] + [summed, readout]
    block = op.parent_block()
    order = {o: i for i, o in enumerate(block.ops)} if block is not None else {}
    if op not in members or any(order.get(m, -1) < order.get(op, -1) for m in members):
        return None
    scheme = (options or {}).get('scheme') or {}
    if scheme.get('readout_strategy') != 'f32_product_rne':
        raise Refused('staged f32 byte operations require the selected f32 readout definition')
    rounding = 'per_operand' if any(stages) else 'single_rounding'
    if scheme.get('operand_sum_rounding') not in (None, rounding):
        raise Refused('selected operand-sum rounding differs from the source boundaries')
    loads = []
    for stage in stages:
        gains = stage.gains() if stage else np.ones(cols, np.float32)
        if gains is None or len({_bits(v) for v in gains}) != 1:
            raise Refused('a per-operand load needs one per-tensor f32 scale')
        if not 0 < float(gains[0]) <= 1:
            raise Refused('operand multipliers must lie in (0, 1]')
        loads.append(_bits(gains[0]))
    multipliers = final.gains()
    if multipliers is None or np.any(multipliers <= 0) or not np.all(np.isfinite(multipliers)):
        raise Refused('readout needs one positive finite f32 multiplier per column')
    bits = tuple(_bits(v) for v in multipliers)

    def source(a, b, column):
        left = stages[0]([a], column) if stages[0] else a
        right = stages[1]([b], column) if stages[1] else b
        return np.asarray(final([total([left, right], column)], column), dtype=np.int64)

    def column_key(column):
        return tuple(s.key(column) for s in (*[s for s in stages if s], total, final))

    candidates = [Candidate(tuple(loads), bits, relu) for relu in (False, True)]
    return Region('staged_sum', rounding, 'source_f32_operations', operands[0], operands[1],
                  readout.results[0], members, rows, cols, source, column_key, candidates)


def _readout_root(op):
    """A byte readout of one integer data operand produced by another generic."""
    from .match import etype, producer
    try:
        stage = _Stage(op, *(op.results[0].type.get_shape()))
    except (Refused, TypeError, ValueError):
        return None
    if len(stage.data) != 1 or etype(stage.data[0][1]) not in _INTS or etype(stage.data[0][1]) == 'i8':
        return None
    owner = producer(stage.data[0][1])
    return owner if owner is not None and owner.name == 'linalg.generic' else None


# ---------------------------------------------------------------------------------------------
# Proof, emission and admission
# ---------------------------------------------------------------------------------------------

def prove(region, options):
    """The first candidate equal to the source on every operand pair, per column group."""
    load = load_semantics()
    a, b = _domain(load['operand_dtype'])
    expected = {}
    for candidate in region.candidates:
        if any(not 0 < float(_value(bits)) <= 1 for bits in candidate.load_bits):
            continue
        try:
            readout = readout_semantics(options, relu=candidate.relu)
        except Refused:
            continue
        groups = {}
        for column in range(region.cols):
            groups.setdefault((region.column_key(column), candidate.multiplier_bits[column]), column)
        for (key, _), column in groups.items():
            if key not in expected:
                expected[key] = region.source(a, b, column)
            if not np.array_equal(device(a, b, candidate, column, load=load, readout=readout), expected[key]):
                break
        else:
            return candidate, dict(
                schema='gemmini_scaled_operand_sum_proof_v1', status='proven', form=region.form,
                operand_sum_rounding=region.rounding, reference=region.reference,
                load_scale_bits=list(candidate.load_bits), multiplier_bits=list(candidate.multiplier_bits),
                relu=candidate.relu, activation_domain=readout['activation_domain'],
                operand_dtype=load['operand_dtype'], output_dtype=readout['output_dtype'],
                domain_pairs=int(a.size), column_groups=len(groups), rows=region.rows, cols=region.cols,
                equality='exhaustive operand-pair domain against the source and device f32 load/readout')
    raise Refused('no scaled-load candidate reproduces the source on the full operand domain')


def emit(region, candidate, proof):
    from .ops import Kernel, _begin
    from .readout_encoding import activation
    from .stream import ACC, ACC_ROWS, DIM, Arg
    rows, cols = region.rows, region.cols
    blocks = [candidate.multiplier_bits[c0:c0 + DIM] for c0 in range(0, cols, DIM)]
    if any(len(set(block)) != 1 for block in blocks):
        raise Refused('a readout multiplier changes within one column tile')
    s = _begin([Arg('a', 'i8', (rows, cols), 'input'), Arg('b', 'i8', (rows, cols), 'input'),
                Arg('y', 'i8', (rows, cols), 'output')])
    for state, bits in enumerate(candidate.load_bits):
        s.config_ld(state, cols, float(_value(bits)), shrink=True)
    act = activation(candidate.relu)
    tiles = [(r0, c0) for r0 in range(0, rows, DIM) for c0 in range(0, cols, DIM)]
    per_half = (ACC_ROWS // 2) // DIM
    current = None
    for chunk, first in enumerate(range(0, len(tiles), per_half)):
        group = tiles[first:first + per_half]
        base = (chunk & 1) * (ACC_ROWS // 2)
        slots = [(base + i * DIM, r0, c0, min(DIM, rows - r0), min(DIM, cols - c0))
                 for i, (r0, c0) in enumerate(group)]
        for row, r0, c0, height, width in slots:
            s.mvin(s.at('a', r0 * cols + c0), ACC(row), width, height, 0)
        s.fence()  # An overwriting load precedes the accumulating load of the same rows.
        for row, r0, c0, height, width in slots:
            s.mvin(s.at('b', r0 * cols + c0), ACC(row, accumulate=True), width, height, 1)
        for row, r0, c0, height, width in slots:
            bits = candidate.multiplier_bits[c0]
            if bits != current:
                s.config_st(cols, act, multiplier_bits=bits)
                current = bits
            s.mvout(s.at('y', r0 * cols + c0), ACC(row), width, height)
    s.fence()
    meta = dict(scaled_operand_sum=proof, output_dtype='i8', device_readout='scaled_operand_sum',
                device_arithmetic='f32_scaled_loads_and_readout', requantization='device',
                rows=rows, cols=cols, quantize_rounding='device_f32_RNE', fp32_product_proofs=[])
    return Kernel('scaled_operand_sum', s, meta)


def match(op, *, readout_options):
    """A proven scaled-load kernel for an operand-sum region, else ``None`` (existing path)."""
    from .match import Decline, Match
    if readout_options is None or op.name != 'linalg.generic':
        return None
    try:
        region = dequantized_sum(op, readout_options) or staged_sum(op, readout_options)
        if region is None:
            return None
        candidate, proof = prove(region, readout_options)
        kernel = emit(region, candidate, proof)
    except (Decline, Refused, ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
        return None
    root = region.members[-1]
    return Match('scaled_operand_sum', root, region.members, [('a', region.left), ('b', region.right)],
                 ('y', region.result), kernel,
                 numeric=dict(scaled_operand_sum=True, form=region.form,
                              operand_sum_rounding=region.rounding, reference=region.reference))


def validate(stream, meta):
    """Replay the actual command words against the proof; raise unless they implement it."""
    from .readout_encoding import activation, code
    from .stream import ACC_ROWS, DIM
    from .stream_proof import decode, load_model, local, _matches
    proof = meta.get('scaled_operand_sum')
    if not proof or proof.get('status') != 'proven':
        raise ValueError('scaled operand sum lacks a proof')
    if stream.readouts or stream.packs:
        raise ValueError('scaled operand sum carries a host readout')
    rows, cols = proof['rows'], proof['cols']
    loads, bits = proof['load_scale_bits'], proof['multiplier_bits']
    if len(bits) != cols or len(loads) != 2:
        raise ValueError('proof does not bind every load and output column')
    index = {arg.name: i for i, arg in enumerate(stream.args)}
    names = {i: arg.name for i, arg in enumerate(stream.args)}
    for name, role in (('a', 'input'), ('b', 'input'), ('y', 'output')):
        arg = stream.args[index[name]] if name in index else None
        if arg is None or arg.role != role or arg.dtype != 'i8' or int(np.prod(arg.shape)) != rows * cols:
            raise ValueError('kernel arguments differ from the proved operand layout')
    states, store = {}, None
    # Per accumulator cell: (source arg, element, scale bits, accumulate flag, fence epoch) per load.
    history = {}
    written = np.zeros(rows * cols, dtype=np.int64)
    epoch = 0
    expected_act = activation(proof['relu'])
    for ins in stream.ins:
        command = decode(ins)
        kind, fields = command.kind, command.fields
        if kind == 'FENCE':
            epoch += 1
            continue
        if kind == 'FLUSH':
            continue
        if kind == 'CONFIG_LD':
            states[fields['state_id']] = fields
            continue
        if kind == 'CONFIG_ST':
            store = fields
            continue
        if kind in ('MVIN', 'MVIN2', 'MVIN3'):
            state = states.get(fields['state'])
            place = local(fields['local_addr'])
            height, width = fields['num_rows'], fields['num_cols']
            if state is None:
                raise ValueError('scaled load state, width or destination differs from the proof')
            _matches(state, {**load_model(), 'shrink': 1, 'scale': state['scale'], 'dram_stride': cols},
                     ('state_id',), 'scaled load state, width or destination differs from the proof')
            if (place.garbage or not place.accumulator or place.full or width > DIM or command.arg is None):
                raise ValueError('scaled load state, width or destination differs from the proof')
            source = names[command.arg]
            if command.off % cols + width > cols:
                raise ValueError('scaled load crosses its operand row')
            for r in range(height):
                for c in range(width):
                    element = command.off + r * cols + c
                    cell = (place.row + r, c)
                    if not 0 <= place.row + r < ACC_ROWS:
                        raise ValueError('scaled load leaves the accumulator')
                    entry = (source, element, state['scale'], place.accumulate, epoch)
                    history[cell] = history.get(cell, []) + [entry] if place.accumulate else [entry]
            continue
        if kind == 'MVOUT':
            place = local(fields['local_addr'])
            height, width = fields['num_rows'], fields['num_cols']
            if (command.arg is None or names[command.arg] != 'y' or place.garbage or not place.accumulator
                    or place.accumulate or place.full or width > DIM or store is None):
                raise ValueError('readout destination, width or source differs from the proof')
            if (store['activation'] != expected_act or store['stride'] != cols or
                    store['cmd_type'] != code('config_subtype', 'CONFIG_ST') or
                    any(v for k, v in store.items() if k not in ('cmd_type', 'activation', 'stride', 'acc_scale'))):
                raise ValueError('decoded readout state differs from the proof')
            scale = store['acc_scale']
            for r in range(height):
                for c in range(width):
                    element = command.off + r * cols + c
                    if not 0 <= element < rows * cols or command.off % cols + width > cols:
                        raise ValueError('readout leaves its output row')
                    cell = history.get((place.row + r, c), [])
                    if (len(cell) != 2 or [e[:4] for e in cell] != [('a', element, loads[0], False),
                                                                    ('b', element, loads[1], True)]
                            or cell[1][4] <= cell[0][4] or scale != bits[element % cols]):
                        raise ValueError('accumulator cell does not hold the proved scaled pair')
                    written[element] += 1
            continue
        raise ValueError(f'scaled operand sum issues an unproved {ins.kind}')
    if np.any(written != 1):
        raise ValueError('readout does not cover every output element exactly once')
    return 'scaled_operand_sum'
