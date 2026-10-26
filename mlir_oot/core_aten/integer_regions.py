"""Source-bound integer MAC and proof-selected accumulator or narrow readout.

Only a checked closed body grants semantics. Provenance describes bindings but
never substitutes for that check. Unbounded runtime seeds stay on the host:
i32 SRAM cannot implement an infinite-precision seeded sum without overflow.
Unproved epilogues retain their original full-width producer and host boundary.
"""

from dataclasses import dataclass

import numpy as np

from ..contraction_patterns import match_integer_gemm, _zero_init
from . import ops
from .exactness import require
from .match import (Match, producer, shape, dense_values, splat_value,
                    _scalar_const, identity_map, generic_maps, iterators)


def candidate_readouts(op):
    """Follow integer values only through static, bit-preserving tensor layouts.

    Grouped convolutions concatenate independently computed numerator matrices
    before a shared channel readout. These layouts stay on the host; their
    presence does not change the integer producer or move rounding boundaries.
    """
    from .views import reindex_shape
    from .match import etype
    pending, visited = [op.results[0]], set()
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        for use in value.uses:
            user = use.operation
            if getattr(user.attributes.get('prov.quant.arithmetic'), 'data', None) == 'exact':
                yield user
            if len(user.results) != 1 or etype(user.results[0]) != 'i32':
                continue
            if user.name in ('tensor.reshape', 'tensor.collapse_shape', 'tensor.expand_shape', 'tensor.cast'):
                if use.index != 0:
                    continue
                try:
                    reindex_shape(user)
                except ValueError:
                    continue
            elif user.name == 'linalg.transpose':
                if use.index != 0:
                    continue
                perm = list(user.properties['permutation'].iter_values())
                if sorted(perm) != list(range(len(shape(value)))):
                    continue
                if tuple(shape(value)[i] for i in perm) != shape(user.results[0]):
                    continue
            elif user.name == 'tensor.concat':
                axis = int(user.properties['dim'].value.data)
                output = shape(user.results[0])
                if not 0 <= axis < len(output) or any(
                        etype(v) != 'i32' or len(shape(v)) != len(output) or any(
                            a != b for i,(a,b) in enumerate(zip(shape(v),output)) if i != axis)
                        for v in user.operands):
                    continue
                if sum(shape(v)[axis] for v in user.operands) != output[axis]:
                    continue
            else:
                continue
            pending.append(user.results[0])


def readout(op):
    """Identify the source readout; retain its exact host implementation.

    The backend binds ONLY the closed integer producer. It does not reinterpret
    an explicit float bias or substitute a rounded product for the rational pair.
    """
    from .quant_schema import attributes, ratios
    for reader in candidate_readouts(op):
        if reader.name != 'linalg.generic' or len(reader.results) != 1:
            continue
        if getattr(reader.attributes.get('prov.quant.arithmetic'),'data',None) != 'exact':
            continue
        attrs = attributes(reader,'integer_readout')
        require(getattr(attrs.get('accumulator'),'data',None)=='i32','readout accumulator is not i32')
        require(getattr(attrs.get('numerics'),'data',None)=='integer_readout','readout numerics are not exact integer')
        factors = ratios(attrs)
        require(all(s != 0 for s in factors),'zero exact readout coefficient')
        channels = shape(reader.operands[0])
        require(channels is not None and len(factors) in (1,channels[-1] if channels else 1),
                'readout scale channel extent')
        if any(k.startswith('prov.quant.') for k in op.attributes):
            attributes(op,'contraction')  # validates the schema, including host_only
        return reader,factors,False
    return None


def contraction(op):
    return (match_integer_gemm(op, require_zero=False) or
            match_integer_gemm(op, require_zero=False, operand_dtype="i32"))


def has_integer_readout(module):
    return any(contraction(op) is not None and
               any(getattr(reader.attributes.get('prov.quant.arithmetic'),'data',None)=='exact'
                   for reader in candidate_readouts(op))
               for op in module.walk())


def seed_guard(value):
    """Recognize the source's retained i32 range assertion, with all SSA links checked."""
    guard = producer(value)
    if guard is None or guard.name != 'linalg.generic' or len(guard.operands) != 2:
        return None
    if any(i != 'parallel' for i in iterators(guard)) or not all(identity_map(m) for m in generic_maps(guard)):
        return None
    block = guard.regions[0].blocks[0]
    body = list(block.ops)
    if [x.name for x in body] != ['arith.extsi','arith.constant','arith.cmpi','arith.constant','arith.cmpi','arith.andi','cf.assert','linalg.yield']:
        return None
    ext,low,ge,high,le,both,assertion,yielded = body
    if not (str(block.args[0].type) == 'i32' and str(ext.results[0].type) == 'i64'
            and list(ext.operands) == [block.args[0]]
            and list(ge.operands) == [ext.results[0],low.results[0]] and ge.properties['predicate'].value.data == 5
            and list(le.operands) == [ext.results[0],high.results[0]] and le.properties['predicate'].value.data == 3
            and list(both.operands) == [ge.results[0],le.results[0]]
            and list(assertion.operands) == [both.results[0]] and list(yielded.operands) == [block.args[0]]):
        return None
    lower,upper = _scalar_const(low),_scalar_const(high)
    require(lower is not None and upper is not None and -(1<<31) <= lower <= upper < (1<<31), 'invalid seed guard range')
    return guard, guard.operands[0], (int(lower),int(upper))


def _readout_plan(module, op, options):
    if options is None or any(str(v.type.get_element_type()) != 'i8' for v in op.operands[:2]):
        return None
    from merlin.compile.composition_fusion import plan_contraction_readout
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    chain = OF.source_chain(op, OF.Values())
    if any(stage.kind == 'pool' for stage in chain.stages):
        from .pooled_store import plan_fused
        return plan_fused(module, op, options, chain)
    # Decline unsupported stages before planning: a residual chain's exhaustive
    # accumulator-cut search is costly and could never be emitted here.
    if any(stage.kind not in ('convert', 'scale', 'bias', 'relu', 'quantize') for stage in chain.stages):
        return None
    try:
        plan = plan_contraction_readout(module, op, path=options['path'], scheme=options['scheme'],
                                        facts=options['facts'])
    except ValueError:
        return None  # Retain the original integer producer and host boundary.
    from .readout_encoding import supports_narrow_path
    if plan.route == 'epilogue' and not supports_narrow_path(options['path']):
        return None
    if len(chain.ops) != len(chain.stages):
        return None  # Layout/pooling partitions have separate emission obligations.
    if chain.ops and shape(chain.ops[-1].results[0]) != shape(op.results[0]):
        return None
    proof = plan.decision['proof']
    if proof.get('output_column_partition'):
        return None
    width = proof.get('columns_per_scale')
    from .stream import DIM
    if width is not None and (width < DIM or width % DIM):
        return None
    return plan, chain


def _emit_readout_plan(kernel, op, planned, consumed):
    if planned is None:
        return op
    plan, chain = planned[:2]
    result = op
    if plan.route == 'conv_pool':
        from .pooled_store import emit_fused
        emit_fused(kernel, planned)
        result = chain.ops[-1]
        consumed.extend(chain.ops)
    elif plan.route in NARROW_ROUTES:
        from .stream import DIM
        proof = plan.decision['proof']
        relu = 'relu' in plan.decision['stages']
        width = proof.get('columns_per_scale', max(shape(op.results[0])[-1], DIM))
        width = ((width + DIM - 1) // DIM) * DIM
        kernel.stream.narrow_readout(proof['multiplier_bits'], relu=relu, columns_per_scale=width)
        result = chain.ops[-1]
        consumed.extend(chain.ops)
        kernel.meta.update(output_dtype='i8', relu=relu, readout='f32_product_rne',
                           device_readout='f32_product_rne', requantization='device',
                           multiplier_bits=proof['multiplier_bits'])
    kernel.meta['composition_plan'] = plan.record()
    kernel.meta['raw_integer'] = plan.route == 'int32_readout'
    kernel.meta['integer_readout'] |= plan.route in NARROW_ROUTES + ('conv_pool',)
    return result


#: Plans whose proved readout narrows on the device: a plain epilogue, or one
#: whose residual moved into the contraction as K-concat digit tiles.
NARROW_ROUTES = ('epilogue', 'k_concat')


def planned_readout(module, op, options):
    """The readout plan and any K-concat extension, derived from the source alone."""
    planned = _readout_plan(module, op, options)
    if planned is not None or str(op.operands[0].type.get_element_type()) != 'i8':
        return planned, None
    from . import compositions
    extension = compositions.k_extension(module, op, options, batch=len(shape(op.operands[0])) != 2)
    return (None, None) if extension is None else ((extension.readout, extension.chain), extension)


@dataclass
class SourceSeed:
    """The source accumulator initializer, with readout offsets folded in."""
    bias: bool
    values: object            # int64/int32 constant seed, [N] or the full result shape; None if runtime/zero
    bias_rows: bool           # False when ``values`` (or the runtime seed) is one row broadcast over M
    guarded_range: tuple | None
    runtime_seed: object      # source SSA value of a guarded runtime seed
    planned: object           # the readout plan, dropped when its offsets need a runtime seed
    consumed: list


def source_seed(op, dims, planned):
    """Derive the seed from the source contraction; emission and admission share it."""
    rank = len(shape(op.operands[0]))
    seed = op.operands[2]
    consumed = []
    bias = not _zero_init(seed)
    bias_rows = True
    values = None
    guarded_range = None
    runtime_seed = None
    if bias:
        seed_op = producer(seed)
        if seed_op is not None and seed_op.name == 'linalg.broadcast':
            dimensions = list(seed_op.properties['dimensions'].iter_values())
            require(dimensions == list(range(rank - 1)) and shape(seed_op.operands[0]) == (dims.n,), 'non-channel seed broadcast')
            consumed.append(seed_op)
            seed = seed_op.operands[0]
            bias_rows = False
        checked = seed_guard(seed)
        if checked is not None:
            guard_op, original_seed, guarded_range = checked
            values = dense_values(original_seed)
            if values is not None:
                require(np.all(values >= guarded_range[0]) and np.all(values <= guarded_range[1]), 'constant seed violates its range guard')
                consumed.append(guard_op)
                seed = original_seed
        else:
            values = dense_values(seed)
        if values is None:
            value = splat_value(seed)
            if value is not None:
                values = np.full(shape(seed), value, np.int64)
        require(values is not None or guarded_range is not None, 'runtime i32 seed has no nonoverflow range proof')
        if values is not None:
            require(np.all(np.isfinite(values)) and np.all(values == np.trunc(values)), 'noninteger seed')
            values = np.asarray(values, np.int32)
        else:
            runtime_seed = seed
    if planned is not None and planned[0].route == 'conv_pool' and runtime_seed is not None:
        planned = None  # pooled bands load only compile-time seeds
    if planned is not None and planned[0].route in NARROW_ROUTES + ('conv_pool',):
        offsets = np.asarray(planned[0].decision['proof']['offsets'], dtype=np.int64)
        if np.any(offsets):
            if runtime_seed is not None:
                planned = None
            else:
                values = offsets if values is None else values.astype(np.int64) + offsets
                bias = True
                if values.ndim == 1:
                    bias_rows = False
    return SourceSeed(bias, values, bias_rows, guarded_range, runtime_seed, planned, consumed)


def match_integer_region(op, *, allow_raw=False, module=None, readout_options=None):
    dims = contraction(op)
    if dims is None:
        return None
    selected = readout(op)
    planned, extension = planned_readout(module, op, readout_options)
    if selected is None:
        if planned is None and readout_options is None and (not allow_raw or not _zero_init(op.operands[2])):
            return None
        if any(k.startswith('prov.quant.') for k in op.attributes):
            from .quant_schema import attributes
            attributes(op,'contraction')
        scale, relu = None, False
    else:
        reader, scale, relu = selected
    rank = len(shape(op.operands[0]))
    batch = dims.batch if rank == 3 else 0
    seeded = source_seed(op, dims, planned)
    consumed = [op, *seeded.consumed]
    bias, values, bias_rows = seeded.bias, seeded.values, seeded.bias_rows
    guarded_range, runtime_seed, planned = seeded.guarded_range, seeded.runtime_seed, seeded.planned
    if planned is None:
        extension = None  # Its offsets needed a runtime seed: keep the host boundary.
    operands = [('a', op.operands[0]), ('b', op.operands[1])]
    if str(op.operands[0].type.get_element_type()) == 'i32':
        from .centered_regions import kernel as centered_kernel, private_producers
        kernel, operands, producers = centered_kernel(op, dims, batch, bias, values, bias_rows, guarded_range)
        consumed.extend(private_producers(producers, consumed))
    else:
        from . import compositions
        dataflow_plan = None
        if extension is None:
            dataflow_plan = compositions.output_stationary(module, op, readout_options, batch=rank != 2,
                                                           runtime_seed=runtime_seed is not None)
        recipe, inputs = compositions.recipe(dataflow_plan, extension, dims.n)
        kernel = ops.matmul('aten.bmm.default' if batch else 'aten.addmm.default' if bias else 'aten.mm.default', dims.m, dims.k, dims.n,
                            batch=batch, s_out=None, bias=bias, bias_const=values,
                            bias_rows=bias_rows, relu=relu, bias_range=guarded_range,
                            recipe=recipe, inputs=inputs)
        compositions.annotate(kernel, dataflow_plan, extension)
        if extension is not None:
            operands.append(('residual', extension.residual))
    kernel.meta.update(output_dtype='i32', scale=1, relu=False, bias=bias,
                       integer_producer=True, integer_readout=selected is not None, raw_integer=selected is None,
                       dimensions=dict(batch=dims.batch,m=dims.m,n=dims.n,k=dims.k),
                       seed=('guarded_runtime' if runtime_seed is not None else 'constant') if bias else 'zero')
    kernel.meta.update(readout='full_i32',
                       requantization='retained_source_exact_integer' if selected is not None else 'none')
    if scale is not None:
        kernel.meta.update(scale_numerator=[str(v.numerator) for v in scale],
                           scale_denominator=[str(v.denominator) for v in scale])
    result = _emit_readout_plan(kernel, op, planned, consumed)
    return Match(kernel.op, op, consumed, operands + ([('bias',runtime_seed)] if runtime_seed is not None else []),
                 ('out', result.results[0]), kernel,
                 numeric=dict(integer_producer=True, integer_readout=kernel.meta['integer_readout'], raw_integer=kernel.meta['raw_integer'], scale=scale, relu=kernel.meta['relu'], seed=None if values is None else values.tolist(), runtime_seed=runtime_seed is not None,
                              **(dict(centered_byte_contraction=kernel.meta["centered_byte_contraction"])
                                 if "centered_byte_contraction" in kernel.meta else {})))
