"""Mandatory source/command differential backstop, shared by both catalog ABIs.

Structural proofs remain required. Small, uniform elementwise source functions
also receive exhaustive joint-domain execution; other regions retain finite trials.
References consume source SSA or source-derived plans, never emitter arithmetic.
"""

from fractions import Fraction
from math import prod
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from xdsl.dialects.arith import CMPF_COMPARISON_OPERATIONS

from . import linmap
from .stream_proof import ProofError, execute, require

TRIALS = 6
MAX_EXHAUSTIVE_BITS = 16
# xDSL's predicate table follows the arith.cmpf encoding: unordered (either NaN).
UNORDERED_FLOAT_PREDICATE = CMPF_COMPARISON_OPERATIONS.index('uno')
BOUNDARY_COUNT = 5


class SourceEvaluator(linmap.Region):
    """Concrete adapter to the existing source IR interpreter, with exact Q/DQ."""

    sum_operations = ('arith.addf', 'arith.addi')

    def __init__(self, inputs):
        super().__init__(lambda value: inputs.get(value), max_points=1 << 26)

    def tensor(self, value):
        result = super().tensor(value)
        if isinstance(result, linmap.Sym):
            require(not result.terms, 'source evaluator left symbolic data')
            return np.asarray(result.const, object)
        return result

    def _eval_op(self, op, result):
        from .match import name_of
        name = name_of(op)
        if name == 'arith.constant':
            data = np.asarray(linmap._const(op), object)
            return np.array([Fraction(float(x)) if isinstance(x, (float, np.floating)) and np.isfinite(x)
                             else x for x in data.flat], object).reshape(data.shape)
        if name in ('linalg.matmul', 'linalg.batch_matmul'):
            a, b, seed = [self.tensor(v).astype(object) for v in op.operands]
            return a @ b + seed
        if name == 'linalg.fill':
            return np.full(op.results[0].type.get_shape(), self.tensor(op.operands[0]).item(), object)
        if name.startswith('quant_ext.'):
            require(name in ('quant_ext.dequantize_per_tensor', 'quant_ext.quantize_per_tensor'),
                    'source evaluator has no quantizer semantics for ' + name)
            data, scale, zero = [self.tensor(v) for v in op.operands[:3]]
            scale, zero = Fraction(scale.item()), int(zero.item())
            if getattr(op.attributes.get('prov.quant.zero_point_dtype'), 'data', None) == 'uint8':
                zero %= 256
            storage = (getattr(op.attributes.get('prov.quant.storage_dtype'), 'data', None)
                       or getattr(op.properties.get('output_dtype'), 'data', None)
                       or getattr(op.properties.get('input_dtype'), 'data', None))
            if 'dequantize' in name:
                if storage == 'uint8' and str(op.operands[0].type.get_element_type()) == 'i8':
                    data = np.asarray(data, object) % 256
                return (np.asarray(data, object) - zero) * scale
            low, high = (op.properties[k].value.data for k in ('quant_min', 'quant_max'))
            return np.array([min(high, max(low, round(Fraction(x) / scale) + zero))
                             for x in np.asarray(data).flat], object).reshape(data.shape)
        return super()._eval_op(op, result)

    def _generic(self, op):
        value = super()._generic(op)
        return np.broadcast_to(value, op.results[0].type.get_shape()).copy()

    def _scalar(self, op, vals, env, ivs, mask):
        if op.name == 'arith.cmpf' and op.properties['predicate'].value.data == UNORDERED_FLOAT_PREDICATE:
            a, b = np.broadcast_arrays(*vals)
            return np.array([np.isnan(float(x)) or np.isnan(float(y)) for x, y in zip(a.flat, b.flat)]).reshape(a.shape)
        if op.name == 'arith.extui':
            return np.asarray(vals[0], object) % (1 << op.operands[0].type.width.data)
        if op.name == 'arith.trunci':
            bits = op.results[0].type.width.data
            return (np.asarray(vals[0], object) + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
        if op.name in ('arith.shli', 'arith.shrsi', 'arith.shrui'):
            a, b = (np.asarray(v, object) for v in vals)
            if op.name == 'arith.shrui':
                a = a % (1 << op.operands[0].type.width.data)
            return a << b if op.name == 'arith.shli' else a >> b
        if op.name == 'math.roundeven':
            return np.array([round(Fraction(x)) for x in np.asarray(vals[0]).flat], object).reshape(np.shape(vals[0]))
        return super()._scalar(op, vals, env, ivs, mask)


def _source_reference(match):
    def evaluate(inputs):
        sources = {v: np.asarray(inputs[n], object) for n, v in match.operands}
        name, value = match.result
        return {name: SourceEvaluator(sources).tensor(value)}
    return evaluate, {}


def _bitwise_reference(match):
    from .bitwise import Selectors
    from .match import etype, shape
    from .storage import itemsize, literal_bytes, numpy_storage
    bound = {value: i for i, (_, value) in enumerate(match.operands)}
    interpreter = Selectors(lambda value: (bound[value], 1) if value in bound else None,
                            max_points=1 << 26)
    name, result = match.result
    found = linmap.selector(interpreter.tensor(result), shape(result))
    require(found is not None, 'no source byte-selector evaluator')
    which, indices, constants = found
    width = itemsize(etype(result))

    def evaluate(inputs):
        raw = [inputs[n].reshape(-1).view(np.uint8).reshape(-1, width) for n, _ in match.operands]
        output = np.empty((len(which), width), np.uint8)
        for sid in range(len(raw)):
            chosen = which == sid
            output[chosen] = raw[sid][indices[chosen]]
        for position in np.flatnonzero(which < 0):
            output[position] = np.frombuffer(literal_bytes(constants[position], etype(result)), np.uint8)
        return {name: output.reshape(-1).view(numpy_storage(etype(result))).reshape(shape(result))}
    return evaluate, {}


def _reference(match, options):
    """Select an independent evaluator; metadata selects a kind, never its math."""
    meta = match.kernel.meta
    if match.kernel.stream.bitwise:
        return _bitwise_reference(match)
    if meta.get('scaled_operand_sum'):
        from .operand_sum import _region_at, source_reference
        region = _region_at(match.root, options)
        require(region is not None, 'no source evaluator for operand sum')
        return source_reference(region), {}
    if meta.get('shared_input'):
        from .shared_input import source_reference
        return source_reference(match, options), {}
    if meta.get('composition_kernel') or meta.get('pooled_store'):
        from . import pooled_store as pool
        if meta.get('composition') == 'window_mean':
            expected = pool.mean_expectation(match, options)
            return lambda inputs: {'y': pool._mean_reference(inputs['x'], expected.rows, expected.bits)}, {}
        if meta.get('composition') == 'pooled_movement':
            expected = pool.movement_expectation(match)
            window = expected.window
            return lambda inputs: {'out': pool._window_max(inputs['x'], window.size, window.stride, window.pad)}, {}
        expected = pool.fused_expectation(match, options)
        return lambda inputs: {'out': pool._fused_reference(inputs, expected)}, {}
    if meta.get('integer_producer') or meta.get('integer_readout'):
        from .contraction_proof import expectation, source_reference
        expected = expectation(match, options)
        ranges = {n: expected.seed_range for n, v in match.operands
                  if expected.runtime_seed is not None and v is expected.runtime_seed}
        return lambda inputs: {expected.out: source_reference(match, expected, inputs)}, ranges
    # The concrete interpreter refuses every unsupported source operation.
    return _source_reference(match)


def _input_ranges(match, overrides):
    """Storage codes, except source-verified runtime assertions from the reference."""
    ranges = dict(overrides)
    for name, value in match.operands:
        if name in ranges:
            continue
        width = getattr(value.type.get_element_type(), 'width', None)
        bits = width.data if width is not None else 8
        low, high = (0, 1) if bits == 1 else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        # Declared Q/DQ ranges do not enforce a runtime bound.
        ranges[name] = (low, high)
    return ranges


def _pointwise_scalar_block(block, captured, uniform_constant):
    """Scalar control flow is local; tensor accesses and iteration indices are not."""
    local = set(captured) | set(block.args)
    for scalar in block.ops:
        if not all(v in local or uniform_constant(v) for v in scalar.operands):
            return False
        if scalar.name in ('linalg.yield', 'scf.yield'):
            return True
        control = scalar.name in ('scf.if', 'scf.for')
        if not control and (scalar.regions or not scalar.name.startswith(('arith.', 'math.'))):
            return False
        for region in scalar.regions:
            if len(region.blocks) != 1 or not _pointwise_scalar_block(region.blocks[0], local, uniform_constant):
                return False
        local.update(scalar.results)
    return False


def _elementwise_domain(match, ranges=None):
    """Prove one uniform scalar function, with no reduction or lane dependence.

    Only identity-indexed parallel generics, local scalar arithmetic/control flow, splat constants
    and per-tensor Q/DQ may connect bound inputs to the result. In particular,
    index operations, gathers, broadcasts of runtime inputs and varying tensor
    constants cannot justify packing different domain points into different lanes.
    """
    from .match import generic_maps, identity_map, iterators, name_of, producer, shape, splat_value
    stream = match.kernel.stream
    if stream.bitwise:
        return None  # Movement retains its selector proof and finite byte trials.
    output_shape = shape(match.result[1])
    bound = {v for _, v in match.operands}
    ranges = _input_ranges(match, {}) if ranges is None else ranges
    domain = []
    domain_size = 1
    for name, value in match.operands:
        width = getattr(value.type.get_element_type(), 'width', None)
        if width is None or shape(value) != output_shape:
            return None
        low, high = ranges[name]
        cardinality = high - low + 1
        domain_size *= cardinality
        domain.append((name, cardinality, low))
    if not domain or domain_size > (1 << MAX_EXHAUSTIVE_BITS):
        return None

    def uniform_constant(value):
        op = producer(value)
        return (splat_value(value) is not None or
                (op is not None and op.name == 'arith.constant' and shape(value) is None))

    def pointwise(value):
        if value in bound or uniform_constant(value):
            return True
        op = producer(value)
        if op is None:
            return False
        if name_of(op) in ('quant_ext.dequantize_per_tensor', 'quant_ext.quantize_per_tensor'):
            return (shape(value) == output_shape and pointwise(op.operands[0]) and
                    all(uniform_constant(v) for v in op.operands[1:]))
        if op.name == 'tensor.cast':
            return shape(value) == shape(op.operands[0]) and pointwise(op.operands[0])
        if op.name != 'linalg.generic' or len(op.results) != 1 or shape(value) != output_shape:
            return False
        maps = generic_maps(op)
        if any(it != 'parallel' for it in iterators(op)) or not identity_map(maps[-1]):
            return False
        block = op.regions[0].blocks[0]
        if block.args[-1].uses:
            return False
        for operand, indexing in zip(op.operands[:-1], maps[:-1]):
            if not uniform_constant(operand) and not (
                    shape(operand) == output_shape and identity_map(indexing) and pointwise(operand)):
                return False
        return _pointwise_scalar_block(block, (), uniform_constant)

    inputs = {a.name for a in stream.args if a.role == 'input'}
    return domain if inputs == {n for n, _, _ in domain} and pointwise(match.result[1]) else None


def _boundary_trials(stream):
    from .storage import numpy_storage
    sizes = [int(np.prod(a.shape)) for a in stream.args if a.role == 'input' and
             (not stream.bitwise or numpy_storage(a.dtype).kind in 'ib')]
    return max(((BOUNDARY_COUNT + size - 1) // size for size in sizes if size), default=1)


def _boundaries(low, high, shape, offset):
    values = np.array([low, high, max(low, min(high, 0)), max(low, min(high, -1)),
                       max(low, min(high, 1))], np.int64)
    indices = (np.arange(np.prod(shape), dtype=np.int64) + offset) % len(values)
    return values[indices].reshape(shape)


def _buffers(stream, ranges, trial, rng, *, exhaustive=None, scale_indices=None):
    from .storage import numpy_storage
    buffers, inputs = {}, {}
    from .runtime_scale_replay import runtime_scale_refs, runtime_scale_trial
    if scale_indices is None:
        scale_indices = {ref.arg_index for ref in runtime_scale_refs(stream)}
    scale_offsets = {index: k for k, index in enumerate(sorted(scale_indices))}
    boundary_trials = _boundary_trials(stream)
    for index, arg in enumerate(stream.args):
        dtype = numpy_storage(arg.dtype)
        if index in scale_indices:
            data = runtime_scale_trial(trial, offset=scale_offsets[index])
            inputs[arg.name] = data.copy()
        elif arg.role == 'input' and exhaustive is not None:
            data = exhaustive[arg.name].astype(dtype)
            inputs[arg.name] = data.copy()
        elif (arg.role == 'input' and stream.bitwise and dtype.kind in 'ib' and
              2 <= trial < 2 + boundary_trials):
            data = _boundaries(*ranges[arg.name], arg.shape, (trial - 2) * int(np.prod(arg.shape)))
            inputs[arg.name] = data.astype(dtype)
        elif arg.role == 'input' and stream.bitwise:
            data = np.empty(arg.shape, dtype)
            raw = data.reshape(-1).view(np.uint8)
            if trial < 2:
                raw[:] = (0, 255)[trial]
            elif trial < 3 + boundary_trials:
                raw[:] = (np.arange(raw.size, dtype=np.uint64) * 73 + index * 151).astype(np.uint8)
            else:
                raw[:] = rng.integers(0, 256, raw.size, dtype=np.uint8)
            inputs[arg.name] = data.copy()
        elif arg.role == 'input':
            require(arg.name in ranges, 'semantic gate has an unbound input ' + arg.name)
            low, high = ranges[arg.name]
            if trial < 2:
                data = np.full(arg.shape, (low, high)[trial], np.int64)
            elif trial < 2 + boundary_trials:
                data = _boundaries(low, high, arg.shape, (trial - 2) * int(np.prod(arg.shape)))
            elif trial == 2 + boundary_trials:
                data = (np.arange(np.prod(arg.shape), dtype=np.int64).reshape(arg.shape) * 73 + index * 151) % (high - low + 1) + low
            else:
                data = rng.integers(low, high + 1, arg.shape, dtype=np.int64)
            inputs[arg.name] = data.astype(dtype)
        elif arg.role == 'const':
            data = stream.consts[arg.name]
        else:
            data = np.empty(arg.shape, dtype)
            data.reshape(-1).view(np.uint8)[:] = 0x5a
        buffers[index] = np.asarray(data).astype(dtype).reshape(arg.shape).copy()
    return buffers, inputs


def _uniform_dma_readout(stream):
    """Prove lane uniformity for unit DMA planes and one final host readout."""
    from dataclasses import replace
    from .movement_proof import _Movement
    from .readout_encoding import flags
    from .requant import parameters
    from .stream import DIM
    from .stream_proof import is_zero, local

    class Planes(_Movement):
        def _load(self, command, ins):
            place = local(command.fields['local_addr'])
            if place.accumulate:
                rows, cols = command.fields['num_rows'], command.fields['num_cols']
                require(place.accumulator and 0 < cols <= DIM, 'DMA plane is not a single accumulator tile')
                for row in range(place.row, place.row + rows):
                    state = self.acc.get(row)
                    require(state is not None and not state.terms and is_zero(state.seed) and
                            state.seed.width >= cols, 'DMA plane accumulates into nonzero or undefined cells')
                    state.stored = True  # Permit replacing the proved zero seed in the selector replay.
                # Zero plus a unit load is a copy, including at i32 storage width.
                command = replace(command, fields={**command.fields,
                    'local_addr': command.fields['local_addr'] & ~flags()['acc_accum']})
            super()._load(command, ins)

        _mvin = _mvin2 = _mvin3 = _load

    if len(stream.readouts) != 1:
        return False
    readout, = stream.readouts
    if (readout.after is not None or readout.placement is not None or
            readout.permutations or readout.broadcasts):
        return False
    try:
        out, = [arg for arg in stream.args if arg.role == 'output']
        if readout.out != out.name:
            return False
        values = [factor for _, factor in readout.terms] + [readout.constant, readout.zero_point,
                                                          readout.quant_min, readout.quant_max]
        for value in values:
            data = parameters(readout, value, out.shape)
            if not data.size or not np.all(data == data.flat[0]):
                return False
        require(all(arg.name for arg in stream.args), 'DMA plane has an empty argument name')
        inputs = {arg.name: i for i, arg in enumerate(stream.args) if arg.role == 'input'}
        replay = Planes(replace(stream, readouts=[]), inputs).run()
        for name, _ in readout.terms:
            plane = replay.values[name]
            if len(plane) != prod(out.shape) or not plane or plane[0] is None:
                return False
            first = plane[0]
            if any(value is None or value.source != first.source or value.flat != i or
                   value.codes != first.codes for i, value in enumerate(plane)):
                return False
        return True
    except (ValueError, KeyError, IndexError, TypeError):
        return False  # Unsupported commands or layouts retain cyclic exhaustion.


def _trial_buffers(match, ranges, *, uniform=False):
    stream = match.kernel.stream
    rng = np.random.default_rng(0x5eed)
    from .runtime_scale_replay import SCALE_TRIAL_BITS, runtime_scale_refs
    refs = runtime_scale_refs(stream)
    scale_indices = {ref.arg_index for ref in refs}
    count = max(TRIALS + _boundary_trials(stream) - 1, len(SCALE_TRIAL_BITS) if refs else 0)
    for trial in range(count):
        yield _buffers(stream, ranges, trial, rng, scale_indices=scale_indices)
    domain = _elementwise_domain(match, ranges)
    if domain is None:
        return
    size = int(np.prod(match.result[1].type.get_shape()))
    require(size > 0, 'semantic gate cannot enumerate an empty elementwise tensor')
    count = prod(cardinality for _, cardinality, _ in domain)
    # Packing needs a command proof as well as source uniformity. Otherwise
    # rotate the entire joint domain through every lane (at most 2**16 trials).
    for start in range(0, count, size if uniform else 1):
        points = (np.arange(size, dtype=np.int64) + start) % count
        inputs, stride = {}, 1
        for name, cardinality, low in domain:
            # Decode joint-domain points into full storage codes for each input.
            inputs[name] = (((points // stride) % cardinality) + low).reshape(match.result[1].type.get_shape())
            stride *= cardinality
        yield _buffers(stream, ranges, 0, rng, exhaustive=inputs, scale_indices=scale_indices)


def certify(matches, bindings, *, readout_options=None, timings=None):
    """The shared gate. Every binding executes, including deduplicated symbols."""
    require(len(matches) == len(bindings), 'semantic gate bindings differ from source regions')
    receipts = []
    for match, binding in zip(matches, bindings):
        started = perf_counter()
        stream = match.kernel.stream
        try:
            reference, overrides = _reference(match, readout_options)
            ranges = _input_ranges(match, overrides)
            uniform = False
            if match.kernel.meta.get('scaled_operand_sum'):
                # Re-derive the full source-domain proof and check every decoded
                # cell's loads/readout before sharing domain points across lanes.
                from .operand_sum import admit
                admit(match, readout_options=readout_options)
                uniform = True
            elif _elementwise_domain(match, ranges) is not None:
                uniform = _uniform_dma_readout(stream)
            outputs = {a.name: i for i, a in enumerate(stream.args) if a.role == 'output'}
            require(outputs, 'semantic gate has no output')
            for trial, (buffers, inputs) in enumerate(_trial_buffers(match, ranges, uniform=uniform)):
                expected = reference({n: v.copy() for n, v in inputs.items()})
                execute(stream, buffers, host_epilogues=True)
                require(set(expected) == set(outputs), 'source evaluator does not cover every kernel output')
                for name, values in expected.items():
                    index = outputs[name]
                    values = (np.asarray(values) if stream.bitwise else np.asarray(values, np.int64)).astype(buffers[index].dtype)
                    require(values.size == buffers[index].size and
                            np.array_equal(buffers[index].reshape(-1).view(np.uint8), values.reshape(-1).view(np.uint8)),
                            f'semantic gate: command stream differs from source on trial {trial}, output {name}')
        except (ArithmeticError, ValueError, KeyError, IndexError, TypeError, AttributeError, NotImplementedError, linmap.NotLinear) as exc:
            raise ProofError(f'semantic gate refused {binding["symbol"]}: {exc}') from exc
        if timings is not None:
            timings.append(dict(symbol=binding['symbol'], seconds=perf_counter() - started))
        receipts.append(dict(symbol=binding['symbol'], method='decoded_commands_vs_source_v1', trials=trial + 1))
    return receipts


def compile_admitted(manifest, matches, proof, compile_device, *, readout_options=None):
    """Single compilation choke point for source-bound kernels in either ABI.

    The compiler callback cannot run until structural proof authentication and
    independent command/source execution both succeed on the actual catalog.
    """
    from .catalog import authenticate_proof
    require(proof is not None, 'catalog compilation requires a source-bound numerical schedule proof')
    if manifest['schema'] == 'gemmini_core_aten_catalog_v1':
        authenticate_proof(manifest, matches, proof)
    else:
        digests = legacy_stream_digests(manifest, matches)
        require(proof.get('status') == 'proven'
                and proof.get('source_sha256') == manifest['source_sha256']
                and proof.get('bindings') == manifest['bindings']
                and {k['symbol']: k.get('schedule') for k in proof.get('kernels', [])} ==
                    {k['symbol']: k['schedule'] for k in manifest['kernels']},
                'legacy catalog compilation requires a source-bound numerical schedule proof')
        require({k['symbol']: k.get('stream_sha256') for k in proof['kernels']} == digests,
                'compiled legacy command stream differs from its admitted proof')
    manifest['semantic_gate'] = certify(matches, manifest['bindings'], readout_options=readout_options)
    manifest['numeric_schedule_proof'] = proof
    return compile_device()


def legacy_stream_digests(manifest, matches):
    """Bind legacy symbols to the same command digests at admission and compile."""
    from .catalog import _stream_digest
    return {b['symbol']: _stream_digest(m.kernel) for b, m in zip(manifest['bindings'], matches)}


def legacy_matches(source, device, manifest):
    """Adapt the legacy ABI using actual emitted commands and bound source ops."""
    from ..frontend.parse import parse_module
    from .stream import Arg
    from .trace import stream_from_module
    operations = list(parse_module(source).walk())
    matches = []
    for binding in manifest['bindings']:
        op = operations[binding['source_operation_ordinal']]
        args = [Arg(n, str(v.type.get_element_type()), tuple(v.type.get_shape()), role)
                for n, v, role in [('a', op.operands[0], 'input'), ('b', op.operands[1], 'input'),
                                   ('out', op.results[0], 'output')]]
        stream = stream_from_module(device, args, symbol=binding['symbol'])
        kernel = SimpleNamespace(stream=stream, args=args, meta={})
        matches.append(SimpleNamespace(root=op, operands=[('a', op.operands[0]), ('b', op.operands[1])],
                                       result=('out', op.results[0]), kernel=kernel))
    return matches
