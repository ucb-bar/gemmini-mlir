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
  integer or f32 operand sum, then one byte readout; or
* the same staged sum with a pooled tail: f32 conversion/scale/rectifier
  stages, a channels-last view, an optional ``-inf`` border, a square max pool
  and a unit-scale quantize. One kernel loads each band's accumulator image
  and drains it through the store-path pool (``pooled_sum``).

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
    pool: object = None     # PoolTail of a pooled readout; ``source`` is then the per-pixel code


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


# -- pooled readout tail -----------------------------------------------------------------------
#
# ``Q(pool(pad(view(t_k(...t_1(sum))))))``: the closed staged sum of above, a
# chain of rank-2 elementwise f32 stages (conversion, column scale,
# rectifier), a channels-last view of the sum rows, an optional ``-inf``
# border, a square max-pool and one per-tensor quantize. The device drains the
# accumulator image of each band through the store-path pool. Per pixel the
# narrow readout reproduces ``Q(t(...))`` (the exhaustive operand-pair proof);
# ``Q`` with unit scale and zero point zero is a nondecreasing RNE clamp, so it
# commutes with the window maximum and ``-inf`` taps are the identity of max.
# A register-padded zero tap is admitted only when every proved per-pixel code
# is nonnegative.

@dataclass(frozen=True)
class PoolTail:
    image: object        # pool_plan.Image of the summed rows
    window: object       # pool_plan.Window, including the source border
    ops: tuple           # view, border insert (if any), pool, quantize
    qmin: int
    qmax: int


def _sum_stage(op):
    """``op`` as a closed integer sum of two byte data operands, else None."""
    from .match import etype, shape
    if op.name != 'linalg.generic' or len(op.results) != 1:
        return None
    dims = shape(op.results[0])
    if dims is None or len(dims) != 2 or etype(op.results[0]) not in _INTS or etype(op.results[0]) == 'i8':
        return None
    try:
        stage = _Stage(op, *dims)
    except Refused:
        return None
    if len(stage.data) != 2 or any(etype(value) != 'i8' for _, value in stage.data):
        return None
    return stage


def _reassociation(op):
    attr = op.properties.get('reassociation') or op.attributes.get('reassociation')
    try:
        return [[int(i.value.data) for i in group.data] for group in attr.data]
    except AttributeError:
        return None


def _pool_tail(value, rows, cols):
    """The elementwise stages, view, border, pool and quantize consuming ``value``, else None."""
    from . import pool_plan
    from .match import etype, quant_sink, shape, users
    from .pooled_store import _pool_window, _unpad
    tail = []
    while True:
        following = users(value)
        if len(following) != 1:
            return None
        owner = following[0]
        if owner.name != 'linalg.generic':
            break
        try:
            stage = _Stage(owner, rows, cols)
        except Refused:
            return None
        if len(stage.data) != 1 or stage.data[0][1] is not value or len(tail) == 4:
            return None
        tail.append(stage)
        value = owner.results[0]
    view = owner
    dims = shape(view.results[0]) if view.name == 'tensor.expand_shape' else None
    if (not tail or etype(value) != 'f32' or dims is None or len(dims) != 4 or dims[-1] != cols
            or dims[0] * dims[1] * dims[2] != rows or _reassociation(view) != [[0, 1, 2], [3]]):
        return None
    following = users(view.results[0])
    if len(following) != 1:
        return None
    insert = following[0] if following[0].name == 'tensor.insert_slice' else None
    if insert is not None:
        following = users(insert.results[0])
        if len(following) != 1:
            return None
    pool = following[0]
    found = _pool_window(pool)
    unpadded = None if found is None else _unpad(pool.operands[0])
    if unpadded is None or unpadded[0] is not view.results[0] or unpadded[2] is not insert:
        return None
    quantize = quant_sink(pool.results[0])
    if quantize is None:
        return None
    if quantize.scale != 1.0 or quantize.zero_point or quantize.unsigned:
        raise Refused('pooled readout quantize is not a unit-scale signed byte clamp')
    image = pool_plan.Image(*dims)
    window = pool_plan.Window(*found, tuple(unpadded[1]))
    try:
        extent = (dims[0], *window.output(image), cols)
    except pool_plan.PoolPlanRefused as exc:
        raise Refused(str(exc)) from None
    if shape(pool.results[0]) != extent or shape(quantize.value) != extent:
        raise Refused('pooled result extent differs from floor-mode windows')
    ops = (view, *([insert] if insert is not None else []), pool, quantize.op)
    return tail, PoolTail(image, window, ops, quantize.qmin, quantize.qmax), quantize.value


def pooled_sum(op, options):
    """Byte operands, optional per-operand requantization, a sum and a pooled byte readout.

    ``op`` may be any member; only the first member in program order matches.
    """
    from .match import shape, users
    if op.name != 'linalg.generic' or len(op.results) != 1:
        return None
    summed = op
    for _ in range(2):  # ``op`` may be one operand's requantization stage
        if _sum_stage(summed) is not None:
            break
        following = users(summed.results[0])
        if len(following) != 1 or following[0].name != 'linalg.generic':
            return None
        summed = following[0]
    total = _sum_stage(summed)
    if total is None:
        return None
    rows, cols = shape(summed.results[0])
    found = _pool_tail(summed.results[0], rows, cols)
    if found is None:
        return None
    tail, pool, result = found
    stages = [_operand_stage(value, summed, rows, cols) for _, value in total.data]
    operands = [stage.data[0][1] if stage else value for stage, (_, value) in zip(stages, total.data)]
    if operands[0] is operands[1]:
        raise Refused('a self sum has one operand, not two loads')
    members = [s.op for s in stages if s] + [summed] + [s.op for s in tail] + list(pool.ops)
    block = op.parent_block()
    order = {o: i for i, o in enumerate(block.ops)} if block is not None else {}
    if op not in members or any(order.get(m, -1) < order.get(op, -1) for m in members):
        return None
    scheme = (options or {}).get('scheme') or {}
    if scheme.get('readout_strategy') != 'f32_product_rne':
        raise Refused('pooled f32 byte operations require the selected f32 readout definition')
    path = (options or {}).get('path') or {}
    if not any(s.get('kind') == 'pool' and 'max' in (s.get('reductions') or ()) for s in path.get('readout') or ()):
        raise Refused('output path declares no store-path max pool')
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
    scaled = [s for s in tail if s.columns]
    if len(scaled) > 1:
        raise Refused('pooled readout applies more than one column scale')
    multipliers = scaled[0].gains() if scaled else np.ones(cols, np.float32)
    if multipliers is None or np.any(multipliers <= 0) or not np.all(np.isfinite(multipliers)):
        raise Refused('readout needs one positive finite f32 multiplier per column')
    bits = tuple(_bits(v) for v in multipliers)

    def source(a, b, column):
        left = stages[0]([a], column) if stages[0] else a
        right = stages[1]([b], column) if stages[1] else b
        value = total([left, right], column)
        for stage in tail:
            value = stage([value], column)
        value = np.asarray(value)
        if value.dtype != np.float32:
            raise Refused('pooled readout tail does not end in f32')
        # Q with unit scale and zero point zero: RNE of the exact f32 value, then the clamp.
        return np.clip(np.rint(value).astype(np.int64), pool.qmin, pool.qmax)

    def column_key(column):
        return tuple(s.key(column) for s in (*[s for s in stages if s], total, *tail))

    candidates = [Candidate(tuple(loads), bits, relu) for relu in (False, True)]
    return Region('staged_pooled_sum', rounding, 'source_f32_operations', operands[0], operands[1],
                  result, members, rows, cols, source, column_key, candidates, pool)


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
            proof = dict(
                schema='gemmini_scaled_operand_sum_proof_v1', status='proven', form=region.form,
                operand_sum_rounding=region.rounding, reference=region.reference,
                load_scale_bits=list(candidate.load_bits), multiplier_bits=list(candidate.multiplier_bits),
                relu=candidate.relu, activation_domain=readout['activation_domain'],
                operand_dtype=load['operand_dtype'], output_dtype=readout['output_dtype'],
                domain_pairs=int(a.size), column_groups=len(groups), rows=region.rows, cols=region.cols,
                equality='exhaustive operand-pair domain against the source and device f32 load/readout')
            if region.pool is not None:
                proof['pool'] = _pool_proof(region, [expected[key] for key, _ in groups])
            return candidate, proof
    raise Refused('no scaled-load candidate reproduces the source on the full operand domain')


def _bands(pool):
    """Accumulator image bands of the pooled drain; each operand element enters one band at most."""
    from . import pool_plan
    from ..tables.rtl_facts import ACC_BANK_ROWS
    try:
        bands = pool_plan.plan(pool.image, pool.window, capacity_rows=ACC_BANK_ROWS, register_padding=True)
    except pool_plan.PoolPlanRefused as exc:
        raise Refused(str(exc)) from None
    seen = set()
    for band in bands:
        rows = {(band.batch, r) for r in range(*band.in_rows)}
        if rows & seen:
            raise Refused('overlapping pool bands would load an operand element twice')
        seen |= rows
    return bands


def _pool_proof(region, codes):
    """Source-derived pooled geometry and the facts that license the device's zero taps."""
    pool = region.pool
    image, window = pool.image, pool.window
    bands = _bands(pool)
    nonnegative = all(int(np.min(c)) >= 0 for c in codes)
    if any(window.pad) and not nonnegative:
        raise Refused('a zero pool tap needs every proved per-pixel code nonnegative')
    return dict(image=[image.batch, image.rows, image.cols, image.channels], window=window.size,
                stride=window.stride, pad=list(window.pad), border='minus_inf' if any(window.pad) else 'none',
                zero_taps=bool(any(window.pad)), nonnegative_codes=nonnegative, bands=len(bands),
                quantize=dict(scale_bits=_bits(1.0), zero_point=0, clamp=[pool.qmin, pool.qmax]),
                reduction='window max after a nondecreasing per-pixel RNE clamp')


def emit(region, candidate, proof):
    from .ops import Kernel, _begin
    from .readout_encoding import activation
    from .stream import ACC, ACC_ROWS, DIM, Arg
    rows, cols = region.rows, region.cols
    blocks = [candidate.multiplier_bits[c0:c0 + DIM] for c0 in range(0, cols, DIM)]
    if any(len(set(block)) != 1 for block in blocks):
        raise Refused('a readout multiplier changes within one column tile')
    if region.pool is not None:
        return _emit_pooled(region, candidate, proof)
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


def _scaled_pair(s, runs, width, c0, cols):
    """Overwrite each run's accumulator rows with ``a``, fence, then accumulate ``b`` onto them."""
    from .stream import ACC
    for row, pixel, height in runs:
        s.mvin(s.at('a', pixel * cols + c0), ACC(row), width, height, 0)
    s.fence()  # An overwriting load precedes the accumulating load of the same rows.
    for row, pixel, height in runs:
        s.mvin(s.at('b', pixel * cols + c0), ACC(row, accumulate=True), width, height, 1)


def _emit_pooled(region, candidate, proof):
    """One kernel: per band and channel block, scaled loads into an image, then one pooled drain.

    Pixels no floor-mode window reads are still loaded once (the source sums
    every operand element) into rows that are never drained.
    """
    from ..tables.rtl_facts import ACC_BANK_ROWS, ACC_BANKS
    from .ops import Kernel, _begin
    from .pooled_store import _reset_store
    from .readout_encoding import activation
    from .stream import ACC, DIM, Arg
    rows, cols = region.rows, region.cols
    image, window = region.pool.image, region.pool.window
    out_rows, out_cols = window.output(image)
    s = _begin([Arg('a', 'i8', (rows, cols), 'input'), Arg('b', 'i8', (rows, cols), 'input'),
                Arg('y', 'i8', (image.batch, out_rows, out_cols, cols), 'output')])
    for state, bits in enumerate(candidate.load_bits):
        s.config_ld(state, cols, float(_value(bits)), shrink=True)
    act = activation(candidate.relu)
    bands = _bands(region.pool)
    units = [(band, c0, min(DIM, cols - c0)) for band in bands for c0 in range(0, cols, DIM)]
    for unit, (band, c0, width) in enumerate(units):
        base = (unit % ACC_BANKS) * ACC_BANK_ROWS
        first = (band.batch * image.rows + band.in_rows[0]) * image.cols
        pixels = band.orows * band.ocols
        _scaled_pair(s, [(base + t, first + t, min(DIM, pixels - t)) for t in range(0, pixels, DIM)],
                     width, c0, cols)
        s.config_st(cols, act, multiplier_bits=candidate.multiplier_bits[c0], pool=band.fields(window, out_cols))
        s.mvout(s.at('y', (band.batch * out_rows + band.out_rows[0]) * out_cols * cols + c0), ACC(base), width, 1)
    used = {(band.batch, r) for band in bands for r in range(*band.in_rows)}
    unread = [(n * image.rows + r) * image.cols for n in range(image.batch) for r in range(image.rows)
              if (n, r) not in used]
    runs = [(start + t, min(DIM, image.cols - t)) for start in unread for t in range(0, image.cols, DIM)]
    if runs:
        s.fence()  # Every drain completes before its bank is reloaded with unread pixels.
        per_pass = ACC_BANK_ROWS * ACC_BANKS // DIM
        for first in range(0, len(runs), per_pass):
            group = runs[first:first + per_pass]
            for c0 in range(0, cols, DIM):
                _scaled_pair(s, [(i * DIM, pixel, height) for i, (pixel, height) in enumerate(group)],
                             min(DIM, cols - c0), c0, cols)
                s.fence()
    _reset_store(s)
    s.fence()
    meta = dict(scaled_operand_sum=proof, output_dtype='i8', device_readout='scaled_operand_sum',
                device_arithmetic='f32_scaled_loads_and_readout', requantization='device',
                rows=rows, cols=cols, quantize_rounding='device_f32_RNE', fp32_product_proofs=[],
                pooled_store=dict(kind='scaled_operand_sum', image=proof['pool']['image'], window=window.size,
                                  stride=window.stride, pad=list(window.pad)))
    return Kernel('scaled_operand_sum', s, meta)


def match(op, *, readout_options):
    """A proven scaled-load kernel for an operand-sum region, else ``None`` (existing path)."""
    from .match import Decline, Match
    if readout_options is None or op.name != 'linalg.generic':
        return None
    try:
        region = (dequantized_sum(op, readout_options) or pooled_sum(op, readout_options)
                  or staged_sum(op, readout_options))
        if region is None:
            return None
        candidate, proof = prove(region, readout_options)
        kernel = emit(region, candidate, proof)
    except (Decline, Refused):
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
    if proof.get('pool') is not None:
        return _validate_pooled(stream, proof)
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


def _validate_pooled(stream, proof):
    """Decode every command word; each pooled output must drain exactly its source window.

    A window tap is either a decoded out-of-image (zero) tap at exactly a
    source border position, licensed by the proof's nonnegative codes, or an
    accumulator cell holding the proved overwrite/accumulate pair of exactly
    the source pixel and channel. Every operand element is loaded exactly once.
    """
    from ..tables.rtl_facts import ACC_BANK_ROWS
    from . import pool_plan
    from .readout_encoding import activation, code
    from .stream import ACC_ROWS, DIM
    from .stream_proof import decode, load_model, local, _matches
    pool = proof['pool']
    rows, cols = proof['rows'], proof['cols']
    loads, bits = proof['load_scale_bits'], proof['multiplier_bits']
    batch, height, width, channels = pool['image']
    size, step = pool['window'], pool['stride']
    top, left = pool['pad'][:2]
    window = pool_plan.Window(size, step, tuple(pool['pad']))
    out = (batch, *window.output(pool_plan.Image(batch, height, width, channels)), channels)
    if (len(bits) != cols or len(loads) != 2 or channels != cols or batch * height * width != rows
            or proof['relu'] is not True and pool['zero_taps']):
        raise ValueError('proof does not bind every load, output column and pool tap')
    shapes = {'a': ((rows, cols), 'input'), 'b': ((rows, cols), 'input'), 'y': (out, 'output')}
    if (stream.consts or len(stream.args) != 3 or
            any(shapes.get(arg.name) != (tuple(arg.shape), arg.role) or arg.dtype != 'i8' for arg in stream.args)):
        raise ValueError('kernel arguments differ from the proved operand and pooled result layout')
    names = {i: arg.name for i, arg in enumerate(stream.args)}
    expected_act = activation(proof['relu'])
    modelled = {'cmd_type', 'activation', 'stride', 'acc_scale', *pool_plan.POOL_FIELDS}
    states, store, history, epoch, drained_since_fence = {}, None, {}, 0, False
    loaded = {'a': np.zeros(rows * cols, np.int64), 'b': np.zeros(rows * cols, np.int64)}
    written = np.zeros(int(np.prod(out)), np.int64)
    for ins in stream.ins:
        command = decode(ins)
        kind, fields = command.kind, command.fields
        if kind == 'FENCE':
            epoch += 1
            drained_since_fence = False
            continue
        if kind == 'FLUSH':
            continue
        if kind == 'CONFIG_LD':
            states[fields['state_id']] = fields
            continue
        if kind == 'CONFIG_ST':
            if (fields['cmd_type'] != code('config_subtype', 'CONFIG_ST')
                    or any(v for k, v in fields.items() if k not in modelled)):
                raise ValueError('decoded store state sets an unmodelled field')
            store = fields
            continue
        if kind in ('MVIN', 'MVIN2', 'MVIN3'):
            state = states.get(fields['state'])
            place = local(fields['local_addr'])
            count, span = fields['num_rows'], fields['num_cols']
            if state is None:
                raise ValueError('scaled load state, width or destination differs from the proof')
            _matches(state, {**load_model(), 'shrink': 1, 'scale': state['scale'], 'dram_stride': cols},
                     ('state_id',), 'scaled load state, width or destination differs from the proof')
            if place.garbage or not place.accumulator or place.full or span > DIM or command.arg is None:
                raise ValueError('scaled load state, width or destination differs from the proof')
            source = names[command.arg]
            if source not in loaded or command.off % cols + span > cols:
                raise ValueError('scaled load crosses its operand row')
            for r in range(count):
                if not 0 <= place.row + r < ACC_ROWS:
                    raise ValueError('scaled load leaves the accumulator')
                for c in range(span):
                    element = command.off + r * cols + c
                    if not 0 <= element < rows * cols:
                        raise ValueError('scaled load leaves its operand')
                    loaded[source][element] += 1
                    entry = (source, element, state['scale'], place.accumulate, epoch)
                    cell = (place.row + r, c)
                    history[cell] = history.get(cell, []) + [entry] if place.accumulate else [entry]
            continue
        if kind == 'MVOUT':
            place = local(fields['local_addr'])
            span = fields['num_cols']
            if (command.arg is None or names[command.arg] != 'y' or place.garbage or not place.accumulator
                    or place.accumulate or place.full or span > DIM or fields['num_rows'] != 1 or store is None):
                raise ValueError('pooled readout destination, width or source differs from the proof')
            if (store['pool_stride'] != step or store['pool_size'] != size or store['activation'] != expected_act
                    or store['stride'] != cols):
                raise ValueError('decoded pooled readout state differs from the proof')
            bank = place.row // ACC_BANK_ROWS
            for porow in range(store['porows']):
                for pocol in range(store['pocols']):
                    byte = command.off + (porow * store['pool_out_dim'] + pocol) * store['stride']
                    if not 0 <= byte or byte + span > written.size:
                        raise ValueError('pooled store leaves its output')
                    n, pr, pc, c0 = (int(v) for v in np.unravel_index(byte, out))
                    if c0 + span > channels or store['acc_scale'] != bits[c0] or len(set(bits[c0:c0 + span])) != 1:
                        raise ValueError('decoded multiplier bits differ from the proved column block')
                    written[byte:byte + span] += 1
                    taps = set()
                    for wrow in range(size):
                        for wcol in range(size):
                            orow = porow * step + wrow - store['upad']
                            ocol = pocol * step + wcol - store['lpad']
                            r, c = pr * step + wrow - top, pc * step + wcol - left
                            zero = not (0 <= orow < store['orows'] and 0 <= ocol < store['ocols'])
                            if zero != (not (0 <= r < height and 0 <= c < width)):
                                raise ValueError('pool window zero taps differ from the source border')
                            if zero:
                                if not pool['nonnegative_codes']:
                                    raise ValueError('pool window reads an unproved zero tap')
                                continue
                            row = place.row + orow * store['ocols'] + ocol
                            if row // ACC_BANK_ROWS != bank or not 0 <= row < ACC_ROWS:
                                raise ValueError('pooled image leaves the accumulator bank its store is ordered against')
                            pixel = (n * height + r) * width + c
                            for ch in range(span):
                                element = pixel * cols + c0 + ch
                                cell = history.get((row, ch), [])
                                if (len(cell) != 2 or [e[:4] for e in cell] != [('a', element, loads[0], False),
                                                                                ('b', element, loads[1], True)]
                                        or cell[1][4] <= cell[0][4]):
                                    raise ValueError('pool window tap does not hold the proved scaled source pair')
                            taps.add((r, c))
                    if not taps:
                        raise ValueError('a pooled output reads no source pixel')
            drained_since_fence = True
            continue
        raise ValueError(f'scaled operand sum issues an unproved {ins.kind}')
    if np.any(written != 1):
        raise ValueError('pooled readout does not cover every output element exactly once')
    if any(np.any(counts != 1) for counts in loaded.values()):
        raise ValueError('scaled loads do not read every operand element exactly once')
    if store is None or store['pool_stride'] != 0 or drained_since_fence:
        raise ValueError('kernel leaves store-path pooling enabled or a pooled drain unfenced')
    return 'scaled_operand_sum'


def _region_at(root, options):
    """Rebuild the source region whose result ``root`` produces, from the IR alone."""
    from .match import name_of, producer
    if name_of(root) == 'quant_ext.quantize_per_tensor':
        generic = producer(root.operands[0])
        from .pooled_store import _pool_window
        if generic is not None and _pool_window(generic) is not None:
            return _pooled_region_at(root, options)
        return dequantized_sum(generic, options) if generic is not None else None
    upstream, frontier = {root}, [root]
    for _ in range(2):
        frontier = [p for op in frontier for v in op.operands
                    if (p := producer(v)) is not None and p.name == 'linalg.generic']
        upstream.update(frontier)
    block = root.parent_block()
    order = {o: i for i, o in enumerate(block.ops)} if block is not None else {}
    for op in sorted(upstream, key=lambda o: order.get(o, -1)):
        region = staged_sum(op, options)
        if region is not None and region.result is root.results[0]:
            return region
    return None


def _pooled_region_at(root, options):
    """The pooled region whose quantize is ``root``: walk its producers back to the operand loads."""
    from .match import producer
    upstream, frontier = set(), [root]
    for _ in range(12):
        frontier = [p for op in frontier for v in op.operands if (p := producer(v)) is not None
                    and p.name in ('linalg.generic', 'tensor.expand_shape', 'tensor.insert_slice')]
        upstream.update(op for op in frontier if op.name == 'linalg.generic')
    block = root.parent_block()
    order = {o: i for i, o in enumerate(block.ops)} if block is not None else {}
    for op in sorted(upstream, key=lambda o: order.get(o, -1)):
        region = pooled_sum(op, options)
        if region is not None and region.result is root.results[0]:
            return region
    return None


def _pooled_reference(region):
    """Source semantics on concrete operands: per-pixel codes, then the ``-inf``-bordered window max."""
    from .pooled_store import _window_max
    image, window = region.pool.image, region.pool.window

    def evaluate(inputs):
        a, b = (inputs[name].astype(np.int64).reshape(region.rows, region.cols) for name in ('a', 'b'))
        codes = np.stack([region.source(a[:, c], b[:, c], c) for c in range(region.cols)], axis=1)
        return _window_max(codes.reshape(image.batch, image.rows, image.cols, image.channels),
                           window.size, window.stride, window.pad)
    return evaluate


def source_reference(region):
    """``inputs -> {'y': codes}`` from the source region: per-pixel codes, pooled when the source pools."""
    if region.pool is not None:
        pooled = _pooled_reference(region)
        return lambda inputs: {'y': pooled(inputs)}

    def evaluate(inputs):
        a, b = (inputs[n].reshape(region.rows, region.cols) for n in ('a', 'b'))
        return {'y': np.stack([region.source(a[:, c], b[:, c], c) for c in range(region.cols)], axis=1)}
    return evaluate


def _pooled_differential(stream, region):
    """Execute the decoded words on the byte sweep, extremes and random operands."""
    from .pooled_store import _backstop
    from .stream_proof import ProofError, execute
    reference = _pooled_reference(region)
    flat = np.arange(region.rows * region.cols, dtype=np.int64)
    inputs = dict(a=((flat * 73 + 11) % 256 - 128).astype(np.int8).reshape(region.rows, region.cols),
                  b=((flat * 151 + 5) % 256 - 128).astype(np.int8).reshape(region.rows, region.cols))
    buffers = {i: (inputs[arg.name].copy() if arg.name in inputs else np.full(arg.shape, 0x5A, np.int8))
               for i, arg in enumerate(stream.args)}
    try:
        execute(stream, buffers)
        out = buffers[[arg.name for arg in stream.args].index('y')]
        if not np.array_equal(out.reshape(-1).astype(np.int64), reference(inputs).reshape(-1)):
            raise ValueError('emitted stream differs from the source on the differential sweep')
        _backstop(stream, 'y', reference)
    except ProofError as exc:
        raise ValueError(str(exc)) from None


def _differential(stream, region, proof):
    """Execute the emitted words on a byte sweep and compare with the source, column by column."""
    from .stream import simulate
    if region.pool is not None:
        return _pooled_differential(stream, region)
    rows, cols = region.rows, region.cols
    flat = np.arange(rows * cols, dtype=np.int64)
    a = ((flat * 73 + 11) % 256 - 128).reshape(rows, cols)
    b = ((flat * 151 + 5) % 256 - 128).reshape(rows, cols)
    shapes = {arg.name: arg.shape for arg in stream.args}
    out = simulate(stream, dict(a=a.astype(np.int8).reshape(shapes['a']),
                                b=b.astype(np.int8).reshape(shapes['b'])))['y'].reshape(rows, cols)
    for column in range(cols):
        if not np.array_equal(out[:, column].astype(np.int64), region.source(a[:, column], b[:, column], column)):
            raise ValueError('emitted stream differs from the source on the differential sweep')


def admit(match, *, readout_options):
    """Re-derive the region and its proof from ``match.root``; authenticate the match and stream.

    Nothing in the emitter's meta is trusted: the proof is rebuilt from the
    source IR, must equal the recorded one, and drives stream validation.
    """
    region = _region_at(match.root, readout_options)
    if region is None:
        raise ValueError('operand-sum root no longer forms a provable source region')
    try:
        _, proof = prove(region, readout_options)
    except Refused as exc:
        raise ValueError(str(exc)) from exc
    if match.kernel.meta.get('scaled_operand_sum') != proof:
        raise ValueError('recorded operand-sum proof differs from the source-derived proof')
    if [id(op) for op in match.consumed] != [id(op) for op in region.members] or match.root is not region.members[-1]:
        raise ValueError('replaced operations differ from the source region')
    bound = dict(match.operands)
    if (len(match.operands) != 2 or bound.get('a') is not region.left or bound.get('b') is not region.right
            or match.result[0] != 'y' or match.result[1] is not region.result):
        raise ValueError('kernel bindings differ from the source operands and result')
    device_readout = validate(match.kernel.stream, {'scaled_operand_sum': proof})
    _differential(match.kernel.stream, region, proof)
    return proof, device_readout
