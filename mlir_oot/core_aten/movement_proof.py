"""Source selector admission through the shared decoded DMA replay.

Each symbolic element records one source SSA element and its function over all
256 storage codes. Host packing and rational readouts compose those functions.
The final provenance and code function must equal the source selector, on every
legal source code. A concrete decoded execution supplies a differential backstop.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import prod

import numpy as np

from . import linmap
from .exactness import quantized
from .match import dequant, name_of, producer, quant_sink, shape, _prop_int
from .quantized_movement import QuantizedSelectors, _literal
from .stream_proof import ProofError, Replay, Tag, decode, execute, require

def _signed(value):
    return (value + 128) % 256 - 128


SIGNED_CODES = tuple(_signed(i) for i in range(256))


@lru_cache(maxsize=4096)
def _packed(codes):
    return tuple(_signed((value & 255) ^ 128) for value in codes)


def _range(q):
    return _prop_int(q.dq, 'quant_min'), _prop_int(q.dq, 'quant_max')


@dataclass(frozen=True)
class Element:
    source: int  # -1 for literals
    flat: int
    codes: tuple


@lru_cache(maxsize=4096)
def _transform(codes, factor, constant, relu, zp, low, high):
    values = [v * factor + constant for v in codes]
    return tuple(_signed(quantized(max(v, 0) if relu else v, low, high, zp))
                 for v in values)


def _selectors(match):
    """Read only source SSA, Q/DQ operands and the closed selector body."""
    qout = quant_sink(match.root.operands[0])
    require(qout is not None and qout.op is match.root, 'movement root is not its source quantizer')
    bound = dict(match.operands)
    require(len(bound) == len(match.operands) and set(bound) == {f'x{i}' for i in range(len(bound))}
            and len({id(v) for v in bound.values()}) == len(bound),
            'movement operands are not unique source selector bindings')
    sources, ids = [None] * len(bound), {}

    def leaf(value):
        if producer(value) is None or name_of(producer(value)) != 'quant_ext.dequantize_per_tensor':
            return None
        q = dequant(value)
        require(q is not None and q.dtype == 'i8', 'movement leaf has no byte Q/DQ contract')
        if value not in ids:
            positions = [i for i in range(len(bound)) if bound[f'x{i}'] is q.value]
            require(len(positions) == 1, 'movement operand is not a source selector leaf')
            ids[value] = positions[0]
            require(sources[positions[0]] is None, 'movement source has multiple dequantization contracts')
            sources[positions[0]] = q
        return ids[value], 1

    try:
        value = QuantizedSelectors(leaf, max_points=1 << 20).tensor(match.root.operands[0])
    except linmap.NotLinear as exc:
        raise ProofError('source selector is not proved: ' + str(exc)) from exc
    found = linmap.selector(value, shape(match.result[1]))
    require(found is not None, 'source movement combines input elements')
    which, indices, constants = found
    require(np.all(constants[which >= 0] == 0), 'source selector adds to an input element')
    require(all(q is not None for q in sources) and len(bound) == len(sources)
            and all(bound.get(f'x{i}') is q.value for i, q in enumerate(sources)),
            'movement operands differ from source selector leaves')
    require(match.result[0] == 'y' and match.result[1] is qout.value,
            'movement result differs from source quantizer')
    return sources, qout, which, indices, constants


class _Movement(Replay):
    def __init__(self, stream, sources):
        from .storage import numpy_storage
        super().__init__(stream, byte_widening=True)
        require(len({a.name for a in stream.args}) == len(stream.args), 'movement argument names are not unique')
        self.values = {}
        for arg in stream.args:
            require(arg.dtype in ('i8', 'i32') and arg.role in ('input', 'const', 'scratch', 'output'),
                    'movement argument has an unproved storage type or role')
            size = prod(arg.shape)
            if arg.role == 'input':
                require(arg.name in sources and arg.dtype == 'i8', 'movement has an unbound byte input')
                self.values[arg.name] = [Element(sources[arg.name], i, SIGNED_CODES) for i in range(size)]
            elif arg.role == 'const':
                data = stream.consts.get(arg.name)
                require(data is not None and data.dtype == numpy_storage(arg.dtype) and tuple(data.shape) == tuple(arg.shape),
                        'constant storage differs from its movement argument')
                self.values[arg.name] = [Element(-1, 0, (int(v),)) for v in data.flat]
            else:
                self.values[arg.name] = [None] * size
        for source, dest in stream.packs:
            target = self._argument(dest)
            require(source in sources and target.role == 'scratch' and target.dtype == 'i8'
                    and tuple(target.shape) == tuple(self._argument(source).shape), 'unproved movement packing')
            self.values[dest] = [Element(v.source, v.flat, _packed(v.codes))
                for v in self.values[source]]

    def _argument(self, name):
        args = [a for a in self.stream.args if a.name == name]
        require(len(args) == 1, 'movement argument names are not unique')
        return args[0]

    def _row_tag(self, arg, byte, width_bytes, cols, *, operand):
        tag = super()._row_tag(arg, byte, width_bytes, cols, operand=operand)
        content = tuple(self.values[arg.name][tag.flat:tag.flat + cols])
        require(len(content) == cols and all(v is not None for v in content), 'movement reads undefined source elements')
        return Tag(tag.name, tag.flat, cols, content)

    def _mvout(self, command, ins):
        require(self.st is not None and self.st['activation'] == 0 and
                self.st['acc_scale'] == self._unit_scale(), 'movement store applies an unproved conversion')
        super()._mvout(command, ins)
        store = self.stores[-1]
        require(not store.accumulator or store.full, 'movement narrows an accumulator store')
        for r, row in enumerate(store.rows):
            if store.accumulator:
                require(not row.terms and row.seed is not None, 'movement contains mesh arithmetic')
                row = row.seed
            require(row.content is not None, 'movement store lacks source provenance')
            width = 4 if store.full else 1
            flat = (command.off + r * store.config['stride']) // width
            self.values[store.name][flat:flat + store.columns] = row.content[:store.columns]

    @staticmethod
    def _unit_scale():
        from .readout_encoding import flags
        return flags()['f1']

    def _readout(self, readout):
        """Compose a one-source exact rational host conversion, at its fenced boundary."""
        from .requant import parameters
        require(not self.unfenced, 'host readout consumes unfenced DMA stores')
        require(len(readout.terms) == 1 and not readout.permutations and not readout.broadcasts,
                'movement host readout is not an elementwise one-source conversion')
        out = self._argument(readout.out)
        require(out.role in ('scratch', 'output') and out.dtype == 'i8', 'movement host readout has an invalid destination')
        name, factor = readout.terms[0]
        require(self._argument(name).dtype == 'i32', 'movement host conversion does not consume full i32')
        require(prod(self._argument(name).shape) == prod(out.shape), 'movement host conversion changes extent')
        positions = np.arange(prod(out.shape))
        if readout.placement is not None:
            require(isinstance(readout.placement, tuple), 'movement host placement is not proved')
            count, block, offset, stride = readout.placement
            require(count > 0 and block > 0 and stride > 0, 'movement host placement has invalid extents')
            logical = np.arange(count)
            positions = logical // block * stride + logical % block + offset
        require(np.all((positions >= 0) & (positions < prod(out.shape))), 'movement host placement leaves destination')
        params = [parameters(readout, v, out.shape) for v in
                  (factor, readout.constant, readout.zero_point, readout.quant_min, readout.quant_max)]
        for position in positions:
            value = self.values[name][position]
            require(value is not None, 'host movement reads an unwritten element')
            f, constant, zp, low, high = (v[position] for v in params)
            codes = _transform(value.codes, Fraction(f), Fraction(constant), readout.relu, int(zp), int(low), int(high))
            self.values[out.name][position] = Element(value.source, value.flat, codes)
            # Replay validates definition and ordering; _row_tag snapshots the
            # actual symbolic value instead of interpreting this host marker.
            self.content.setdefault(out.name, {})[int(position)] = ('host', 0, 0)

    def run(self):
        boundaries = {}
        for readout in self.stream.readouts:
            boundary = len(self.stream.ins) if readout.after is None else readout.after
            require(type(boundary) is int and 0 <= boundary <= len(self.stream.ins), 'host readout boundary leaves stream')
            boundaries.setdefault(boundary, []).append(readout)
        for index, ins in enumerate(self.stream.ins):
            for readout in boundaries.get(index, ()):
                self._readout(readout)
            self.command = decode(ins)
            require(self.command.kind in ('FENCE', 'FLUSH', 'CONFIG_LD', 'CONFIG_ST', 'MVIN', 'MVIN2', 'MVIN3', 'MVOUT'),
                    'movement contains a command outside the DMA proof')
            getattr(self, '_' + self.command.kind.lower())(self.command, ins)
        for readout in boundaries.get(len(self.stream.ins), ()):
            self._readout(readout)
        require(not self.unfenced, 'movement ends with unfenced stores')
        return self


def _expected(sources, qout, which, indices, constants):
    tables, domains = [], []
    for q in sources:
        codes = tuple(range(256)) if q.unsigned else SIGNED_CODES
        tables.append(_transform(codes, Fraction(q.scale) / Fraction(qout.scale),
            -q.zero_point * Fraction(q.scale) / Fraction(qout.scale), False, qout.zero_point, qout.qmin, qout.qmax))
        domains.append([i for i, code in enumerate(codes) if _range(q)[0] <= code <= _range(q)[1]])
    result = []
    for sid, flat, literal in zip(which, indices, constants):
        if sid < 0:
            code = quantized(_literal(literal) / Fraction(qout.scale), qout.qmin, qout.qmax, qout.zero_point)
            result.append(Element(-1, 0, (_signed(code),)))
        else:
            result.append(Element(int(sid), int(flat), tables[sid]))
    return result, domains


def _backstop(stream, sources, expected, trials=3):
    from .storage import numpy_storage
    rng = np.random.default_rng(0x5eed)
    for trial in range(trials):
        buffers, inputs = {}, {}
        for index, arg in enumerate(stream.args):
            dtype = numpy_storage(arg.dtype)
            if arg.role == 'input':
                q = sources[int(arg.name[1:])]
                data = (np.resize(np.arange(_range(q)[0], _range(q)[1] + 1), arg.shape) if trial == 0 else
                        rng.integers(_range(q)[0], _range(q)[1] + 1, arg.shape))
                data = ((data + 128) % 256 - 128).astype(dtype)
                inputs[int(arg.name[1:])] = data.reshape(-1).copy()
            elif arg.role == 'const':
                data = stream.consts[arg.name]
            else:
                data = np.full(arg.shape, 0x5a, dtype)
            buffers[index] = np.asarray(data, dtype=dtype).reshape(arg.shape).copy()
        execute(stream, buffers, host_epilogues=True)
        reference = [value.codes[0 if value.source < 0 else int(inputs[value.source][value.flat]) & 255]
                     for value in expected]
        out = next(i for i, a in enumerate(stream.args) if a.name == 'y')
        require(np.array_equal(buffers[out].reshape(-1), reference),
                f'movement decoded execution differs from the source on differential trial {trial}')


def admit(match):
    stream = match.kernel.stream
    require(sum(prod(a.shape) for a in stream.args) <= 1 << 20, 'movement provenance exceeds bounded admission capacity')
    sources, qout, which, indices, constants = _selectors(match)
    for name, value in [*match.operands, match.result]:
        args = [a for a in stream.args if a.name == name]
        require(len(args) == 1 and args[0].dtype == 'i8' and tuple(args[0].shape) == tuple(shape(value))
                and args[0].role == ('output' if name == 'y' else 'input'),
                'movement argument differs from its source binding')
    replay = _Movement(stream, {f'x{i}': i for i in range(len(sources))}).run()
    expected, domains = _expected(sources, qout, which, indices, constants)
    actual = replay.values.get('y', ())
    require(len(actual) == len(expected), 'movement output extent differs from source')
    for got, want in zip(actual, expected):
        require(got is not None and got.source == want.source and got.flat == want.flat,
                'movement source selector differs from the source operation')
        domain = [0] if want.source < 0 else domains[want.source]
        require(all(got.codes[i] == want.codes[i] for i in domain), 'movement code conversion differs from the source')
    _backstop(stream, sources, expected)
    return dict(method='decoded_source_selectors', code_domain='every_legal_storage_code', differential_trials=3)
