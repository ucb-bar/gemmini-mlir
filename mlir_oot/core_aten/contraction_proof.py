"""Admission of an integer contraction kernel from its replayed dataflow.

The expectation is derived from the source contraction alone: its operands as
byte selectors plus literal constants (``centered_regions.operand_selectors``
for i32 operands, the bound byte tensors otherwise), its seed and readout
offsets (``integer_regions.source_seed``), its readout plan (multiplier
bits per output column and the activation, or a full i32 exit) and any
composition plan (``compositions``): the dataflows it licenses and, for a
K-concat residual, the digit tiles each output must receive exactly once.

The emitted stream is replayed (``stream_proof.Replay``) into symbolic DRAM
contents. Every output element must be the final store of one accumulator
row column whose history is: exactly one seed (a source-proved constant or the
bound runtime seed, possibly through stored and re-loaded partial results),
plus, for every pair of operand planes, the products over each k in [0, K)
exactly once, at the matching (batch, row, column). The planes of each operand
must sum, element by element, to the source operand. The store that drained
it must apply the proved activation and multiplier bits. Anything else
refuses.

A differential backstop then executes the same decoded commands on random
and extreme inputs and compares bit for bit with Merlin's exact reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .source import module_of
from .stream_proof import ProofError, Replay, execute, is_zero, require


def _split(flat, shape):
    """(batch, row, column) of ``flat`` in a ``lead + (rows, cols)`` tensor."""
    rows, cols = shape[-2:]
    batch, rest = divmod(flat, rows * cols)
    return (batch, *divmod(rest, cols))


@dataclass
class Expectation:
    out: str
    out_shape: tuple
    narrow: bool
    bits: list | None
    relu: bool
    k: int
    operand_shapes: tuple  # (A shape, B shape)
    atoms: tuple           # per operand: (atom code per flat element, -1 none), (atom flat), literal constant
    atom_sources: list     # atom code -> (source SSA value, packed)
    seed: np.ndarray       # int64, out_shape (zero when absent)
    runtime_seed: object   # bound runtime seed SSA value, or None
    runtime_rows: bool
    seed_range: tuple | None
    dataflows: frozenset   # CONFIG_EX dataflow codes the source-derived plan licenses
    dataflow_plan: dict | None
    extension: object      # source-derived K-concat residual plan, or None


def expectation(match, readout_options):
    from .centered_regions import operand_selectors
    from .integer_regions import NARROW_ROUTES, contraction, planned_readout, source_seed
    from .match import shape

    op = match.root
    dims = contraction(op)
    require(dims is not None, 'admitted root is not a source integer contraction')
    from . import compositions
    module = module_of(op)
    planned, extension = planned_readout(module, op, readout_options)
    seeded = source_seed(op, dims, planned)
    planned = seeded.planned
    extension = extension if planned is not None else None
    narrow = planned is not None and planned[0].route in NARROW_ROUTES
    a_shape, b_shape = shape(op.operands[0]), shape(op.operands[1])
    if extension is None:
        dataflows, dataflow_plan = compositions.dataflows(
            module, op, readout_options, batch=len(a_shape) != 2, runtime_seed=seeded.runtime_seed is not None)
    else:
        from ..tables import isa
        dataflows, dataflow_plan = frozenset({isa.WEIGHT_STATIONARY}), None  # digit tiles are WS
    lead = tuple(a_shape[:-2])
    out_shape = lead + (dims.m, dims.n)
    atom_sources, codes = [], {}

    def code(value, packed):
        key = (id(value), packed)
        if key not in codes:
            codes[key] = len(atom_sources)
            atom_sources.append((value, packed))
        return codes[key]

    atoms = []
    if str(op.operands[0].type.get_element_type()) == 'i32':
        sources, selectors, constants, _ = operand_selectors(op)
        for (which, indices), constant in zip(selectors, constants):
            which = np.asarray(which, np.int64).reshape(-1)
            mapped = np.array([code(*sources[s]) for s in range(len(sources))], np.int64)
            atom = np.where(which >= 0, mapped[np.maximum(which, 0)], -1)
            atoms.append((atom, np.asarray(indices, np.int64).reshape(-1),
                          np.asarray(constant, np.int64).reshape(-1)))
    else:
        for value in op.operands[:2]:
            size = int(np.prod(shape(value)))
            atoms.append((np.full(size, code(value, False), np.int64), np.arange(size, dtype=np.int64),
                          np.zeros(size, np.int64)))
    seed = np.zeros(out_shape, np.int64)
    if seeded.values is not None:
        seed = np.broadcast_to(np.asarray(seeded.values, np.int64), out_shape).copy()
    bits = relu = None
    if narrow:
        proof = planned[0].decision['proof']
        bits = [int(v) for v in proof['multiplier_bits']]
        relu = 'relu' in planned[0].decision['stages']
        require(len(bits) == dims.n, 'readout multiplier bits do not cover every output column')
    return Expectation(match.result[0], out_shape, narrow, bits, bool(relu), dims.k, (a_shape, b_shape),
                       tuple(atoms), atom_sources, seed, seeded.runtime_seed, seeded.bias_rows,
                       seeded.guarded_range, dataflows, dataflow_plan, extension)


class _Check:
    def __init__(self, match, expected, replay):
        self.match, self.expected, self.replay = match, expected, replay
        self.stream = match.kernel.stream
        self.args = {a.name: a for a in self.stream.args}
        self.bound = {}
        for name, value in match.operands:
            self.bound[name] = value
        self.packed = {dest: source for source, dest in self.stream.packs}
        self.planes = [None, None]
        self.terms = {}

    # -- operand planes ----------------------------------------------------------------------
    def _plane(self, name, role):
        arg = self.args.get(name)
        require(arg is not None and tuple(arg.shape) == tuple(self.expected.operand_shapes[role]),
                'mesh operand plane differs from the source operand shape')
        return arg

    def _term(self, term):
        key = id(term)
        if key not in self.terms:
            a_tag, w_tags, depth, width = term
            require(not is_zero(a_tag) and a_tag.width >= depth, 'mesh input row reads unloaded columns')
            a_shape, b_shape = self.expected.operand_shapes
            self._plane(a_tag.name, 0)
            batch, row, k0 = _split(a_tag.flat, a_shape)
            require(k0 + depth <= a_shape[-1], 'mesh input row leaves its source row')
            first = w_tags[0]
            require(all(not is_zero(t) and t.name == first.name for t in w_tags),
                    'stationary rows mix operand planes')
            self._plane(first.name, 1)
            w_batch, w_k, w_col = _split(first.flat, b_shape)
            require(all(t.flat == first.flat + j * b_shape[-1] for j, t in enumerate(w_tags)),
                    'stationary rows are not consecutive source rows')
            require(w_batch == batch and w_k == k0, 'stationary operand differs from the source contraction')
            self.terms[key] = (term, a_tag.name, first.name, batch, row, k0, depth, w_col,
                               min(min(t.width for t in w_tags), width))
        return self.terms[key][1:]

    def _expand(self, state, c0, cols, position, coverage, leaves):
        batch, row, n0 = position
        seed = state.seed
        if seed is not None and not is_zero(seed):
            if seed.content is None:
                require(seed.width >= c0 + cols, 'accumulator seed does not cover the stored columns')
                leaves.append((seed.name, seed.flat + c0))
            else:
                entries = seed.content[c0:c0 + cols]
                index, r, column = entries[0]
                require(all(e == (index, r, column + i) for i, e in enumerate(entries)),
                        'accumulator seed re-loads a scattered partial result')
                store = self.replay.stores[index]
                require(store.accumulator and store.full and self._neutral(store.config),
                        'accumulator seed re-loads a partial result through a readout stage')
                self._expand(store.rows[r], column, cols, position, coverage, leaves)
        elif seed is not None:
            require(seed.width >= c0 + cols, 'accumulator seed does not cover the stored columns')
        for term in state.terms:
            if self._residual_term(term):
                from .compositions import digit_term
                coverage.setdefault('digits', []).append(digit_term(
                    term, self.stream.consts, residual=term[0].name, n=self.expected.out_shape[-1],
                    position=position, c0=c0, cols=cols))
                continue
            plane_a, plane_b, a_batch, a_row, k0, depth, w_col, width = self._term(term)
            require(width >= c0 + cols, 'mesh tile does not cover the stored columns')
            require(a_batch == batch and a_row == row, 'mesh row reads a different source row')
            require(w_col + c0 == n0, 'stationary columns differ from the stored output columns')
            coverage.setdefault((plane_a, plane_b), []).append((k0, k0 + depth))

    def _residual_term(self, term):
        """A mesh term reading the bound residual of a source-derived K-concat plan."""
        extension = self.expected.extension
        if extension is None:
            return False
        arg = self.args.get(term[0].name)
        return (arg is not None and arg.role == 'input' and term[0].name in self.replay.host
                and self.bound.get(term[0].name) is extension.residual)

    def _neutral(self, config):
        from ..tables import isa
        from .stream_proof import _f1
        return config['activation'] == isa.NO_ACTIVATION and config['acc_scale'] == _f1()

    def _readout(self, store, n0, cols):
        from .readout_encoding import activation
        expected = self.expected
        require(store.accumulator, 'output element is not an accumulator readout')
        require(store.full != expected.narrow, 'accumulator store width differs from the composition exit')
        if expected.narrow:
            require(store.config['activation'] == activation(expected.relu) and
                    all(store.config['acc_scale'] == b for b in expected.bits[n0:n0 + cols]),
                    'decoded readout activation or scale differs from the proved multiplier bits')
        else:
            require(self._neutral(store.config), 'full-width readout applies an unproved scale or activation')

    def _seed(self, leaves, position, cols):
        batch, row, n0 = position
        expected = self.expected
        require(len(leaves) <= 1, 'accumulator holds more than one seed')
        target = expected.seed.reshape(-1, *expected.out_shape[-2:])[batch, row, n0:n0 + cols]
        if expected.runtime_seed is not None:
            require(len(leaves) == 1, 'accumulator lacks the source runtime seed')
            name, flat = leaves[0]
            arg = self.args[name]
            require(arg.role == 'input' and self.bound.get(name) is expected.runtime_seed,
                    'accumulator seed is not the source runtime seed')
            seed_shape = tuple(arg.shape)
            if expected.runtime_rows:
                require(seed_shape == tuple(expected.out_shape), 'runtime seed shape differs from the output')
                want = (batch * expected.out_shape[-2] + row) * expected.out_shape[-1] + n0
            else:
                require(seed_shape == (expected.out_shape[-1],), 'runtime seed shape differs from one row')
                want = n0
            require(flat == want, 'accumulator seed differs from the source-proved bias slice')
            return
        if not leaves:
            require(not np.any(target), 'accumulator lacks the source-proved bias seed')
            return
        name, flat = leaves[0]
        arg = self.args[name]
        require(arg.role == 'const', 'accumulator seed is not a source-proved constant')
        values = self.stream.consts[name].reshape(-1)[flat:flat + cols].astype(np.int64)
        require(values.shape == target.shape and np.array_equal(values, target),
                'accumulator seed differs from the source-proved bias')

    def outputs(self):
        expected = self.expected
        out = self.args.get(expected.out)
        require(out is not None and out.role == 'output' and tuple(out.shape) == tuple(expected.out_shape) and
                out.dtype == ('i8' if expected.narrow else 'i32'), 'output argument differs from the source result')
        content = self.replay.content.get(out.name, {})
        size = int(np.prod(out.shape))
        require(len(content) == size and all(0 <= f < size for f in content),
                'stores do not write every output element')
        groups = {}
        for flat, (index, r, c) in content.items():
            groups.setdefault((index, r), []).append((c, flat))
        planes = None
        for (index, r), columns in groups.items():
            columns.sort()
            c0, flat0 = columns[0]
            require(all(c == c0 + i and f == flat0 + i for i, (c, f) in enumerate(columns)),
                    'stored row is split across output positions')
            cols = len(columns)
            position = _split(flat0, expected.out_shape)
            require(position[2] + cols <= expected.out_shape[-1], 'stored row leaves its output row')
            store = self.replay.stores[index]
            self._readout(store, position[2], cols)
            coverage, leaves = {}, []
            self._expand(store.rows[r], c0, cols, position, coverage, leaves)
            self._seed(leaves, position, cols)
            digits = coverage.pop('digits', [])
            if self.expected.extension is not None:
                from .compositions import require_digits
                require_digits(digits, self.expected.extension)
            row_planes = ({a for a, _ in coverage}, {b for _, b in coverage})
            require(set(coverage) == {(a, b) for a in row_planes[0] for b in row_planes[1]},
                    'an operand plane pair is missing from the reduction')
            for spans in coverage.values():
                spans.sort()
                position_k = 0
                for lo, hi in spans:
                    require(lo == position_k, 'reduction tiles do not cover K exactly once')
                    position_k = hi
                require(position_k == expected.k, 'reduction tiles do not cover K exactly once')
            planes = row_planes if planes is None else planes
            require(planes == row_planes, 'output rows use different operand planes')
        require(planes is not None, 'exact producer emitted no accumulator stores')
        self.planes = planes

    # -- plane values ------------------------------------------------------------------------
    def _atom_code(self, name):
        source = self.packed.get(name)
        value = self.bound.get(source if source is not None else name)
        for code, (want, packed) in enumerate(self.expected.atom_sources):
            if want is value and packed == (source is not None):
                return code
        raise ProofError('operand plane reads an argument bound to no source byte tensor')

    def _element(self, name, flat, cache):
        """('atom', code, flat) or ('lit', value) of one element as the stream defines it."""
        key = (name, flat)
        if key in cache:
            return cache[key]
        arg = self.args[name]
        if arg.role == 'const':
            result = ('lit', int(self.stream.consts[name].reshape(-1)[flat]))
        elif name in self.replay.host:
            from .match import shape
            code = self._atom_code(name)
            require(tuple(arg.shape) == tuple(shape(self.expected.atom_sources[code][0])) and arg.dtype == 'i8',
                    'operand argument differs from its source byte tensor')
            result = ('atom', code, flat)
        else:
            index, r, c = self.replay.content.get(name, {}).get(flat, (None, None, None))
            require(index is not None, 'operand plane element was never produced')
            store = self.replay.stores[index]
            require(not store.accumulator, 'operand plane element comes from the accumulator')
            tag = store.rows[r]
            require(c < tag.width, 'operand plane element comes from unloaded scratchpad columns')
            if is_zero(tag):
                result = ('lit', 0)
            elif tag.content is None:
                result = self._element(tag.name, tag.flat + c, cache)
            else:
                origin = tag.content[c]
                result = self._entry(origin, cache)
        cache[key] = result
        return result

    def _entry(self, entry, cache):
        index, r, c = entry
        store = self.replay.stores[index]
        require(not store.accumulator, 'operand plane element comes from the accumulator')
        tag = store.rows[r]
        require(c < tag.width, 'operand plane element comes from unloaded scratchpad columns')
        if is_zero(tag):
            return ('lit', 0)
        if tag.content is None:
            return self._element(tag.name, tag.flat + c, cache)
        return self._entry(tag.content[c], cache)

    def operands(self):
        cache = {}
        for role in (0, 1):
            atom, index, constant = self.expected.atoms[role]
            size = atom.size
            got_atom, got_index = np.full(size, -1, np.int64), np.zeros(size, np.int64)
            literal = np.zeros(size, np.int64)
            for name in sorted(self.planes[role]):
                arg = self._plane(name, role)
                if arg.role == 'const':
                    literal += self.stream.consts[name].reshape(-1).astype(np.int64)
                    continue
                if name in self.replay.host:
                    kind = self._element(name, 0, cache)
                    require(kind[0] == 'atom', 'operand plane is not a source byte tensor')
                    require(not np.any(got_atom >= 0), 'two operand planes select source bytes')
                    got_atom[:] = kind[1]
                    got_index[:] = np.arange(size)
                    continue
                for flat in range(size):
                    element = self._element(name, flat, cache)
                    if element[0] == 'lit':
                        literal[flat] += element[1]
                    else:
                        require(got_atom[flat] < 0, 'two operand planes select source bytes')
                        got_atom[flat], got_index[flat] = element[1], element[2]
            selected = atom >= 0
            require(np.array_equal(got_atom, atom) and np.array_equal(got_index[selected], index[selected]) and
                    np.array_equal(literal, constant),
                    'operand planes do not sum to the source operand')


def _backstop(match, expected, trials=3):
    """Bit-exact differential check of the decoded commands against Merlin's reference."""
    from merlin.runtime.exact_readout import f32_readout

    stream = match.kernel.stream
    bound = dict(match.operands)
    rng = np.random.default_rng(0x5eed)
    for trial in range(trials):
        buffers, sources = {}, {}
        for index, arg in enumerate(stream.args):
            dt = np.int32 if arg.dtype == 'i32' else np.int8
            if arg.role == 'const':
                data = stream.consts[arg.name]
            elif arg.role == 'input' and bound.get(arg.name) is expected.runtime_seed and expected.runtime_seed is not None:
                low, high = expected.seed_range
                low, high = max(low, -(1 << 20)), min(high, 1 << 20)
                data = rng.integers(low, high + 1, arg.shape).astype(dt)
            elif arg.role == 'input':
                data = (np.full(arg.shape, (-128, 127)[trial % 2], dt) if trial < 2 else
                        rng.integers(-128, 128, arg.shape).astype(dt))
            else:
                data = np.full(arg.shape, 0x5A if dt is np.int8 else 0x5A5A5A5A, dt)
            buffers[index] = np.ascontiguousarray(np.asarray(data).astype(dt)).reshape(arg.shape).copy()
            sources[arg.name] = buffers[index]
        index_of = {a.name: i for i, a in enumerate(stream.args)}
        for source, dest in stream.packs:
            buffers[index_of[dest]].view(np.uint8).flat[:] = buffers[index_of[source]].view(np.uint8).reshape(-1) ^ 128
        execute(stream, buffers)
        values = []
        for role in (0, 1):
            atom, flat, constant = expected.atoms[role]
            total = constant.copy()
            for code, (value, packed) in enumerate(expected.atom_sources):
                names = [n for n, v in bound.items() if v is value]
                require(names, 'reference source byte tensor has no bound argument')
                raw = sources[names[0]].reshape(-1).astype(np.int64)
                data = (raw & 0xFF) - 128 if packed else raw
                chosen = atom == code
                total[chosen] += data[flat[chosen]]
            values.append(total.reshape(expected.operand_shapes[role]))
        acc = np.matmul(values[0].astype(object), values[1].astype(object)).astype(np.int64) + expected.seed
        if expected.extension is not None:
            names = [n for n, v in bound.items() if v is expected.extension.residual]
            require(names, 'reference residual has no bound argument')
            acc = acc + expected.extension.plan.decision['coefficient'] * sources[names[0]].astype(np.int64)
        if expected.runtime_seed is not None:
            names = [n for n, v in bound.items() if v is expected.runtime_seed]
            acc = acc + np.broadcast_to(sources[names[0]].astype(np.int64), expected.out_shape)
        require(np.all((acc >= -(1 << 31)) & (acc < (1 << 31))), 'reference accumulator leaves i32')
        if expected.narrow:
            multiplier = np.array(expected.bits, dtype=np.uint32).view(np.float32)
            reference = f32_readout(acc, multiplier, 'i8', relu=expected.relu)
        else:
            reference = acc.astype(np.int32)
        actual = buffers[index_of[expected.out]]
        require(np.array_equal(actual.reshape(reference.shape), reference),
                f'instruction model differs from the exact reference on differential trial {trial}')


def admit(match, *, readout_options):
    """Return the device readout of ``match``'s emitted stream or raise ``ProofError``."""
    stream = match.kernel.stream
    require(not stream.readouts, 'exact producer carries an unproved host epilogue')
    expected = expectation(match, readout_options)
    require(not (expected.narrow and stream.packs), 'narrow readout carries host operand packing')
    meta_readout = match.kernel.meta.get('readout')
    require(meta_readout == ('f32_product_rne' if expected.narrow else 'full_i32'),
            'emitted readout exit differs from the source readout plan')
    if expected.narrow:
        require(list(match.kernel.meta.get('multiplier_bits', ())) == expected.bits,
                'readout multiplier bits differ from the composition proof')
    replay = Replay(stream, dataflows=expected.dataflows).run()
    require(not replay.unfenced, 'stream ends with unfenced stores')
    check = _Check(match, expected, replay)
    check.outputs()
    check.operands()
    _backstop(match, expected)
    return 'f32_product_rne' if expected.narrow else 'full_i32'
