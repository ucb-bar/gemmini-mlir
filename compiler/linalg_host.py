"""Counted memory-backed source operations using shared host scalar semantics.

This module selects no placement. Its caller must independently establish that
host execution is allowed for the source operation before calling try_lower.
"""
from math import prod

from merlin.llvmlower.host_tensor_codegen import attr_of, elem_name, tensor_shape, is_int


_SCALARS = frozenset('''arith.addf arith.subf arith.mulf arith.divf
arith.maximumf arith.minimumf arith.maxnumf arith.minnumf arith.negf
math.powf math.exp math.erf math.rsqrt math.sqrt math.tanh math.absf math.log
arith.constant arith.extf arith.truncf arith.addi arith.subi arith.muli
arith.andi arith.ori arith.xori arith.divsi arith.divui arith.remsi arith.remui
arith.shli arith.shrsi arith.shrui arith.maxsi arith.minsi arith.negi
math.absi math.absi.default arith.sitofp arith.uitofp arith.fptosi
arith.fptoui arith.extsi arith.extui arith.trunci arith.index_cast arith.bitcast
math.roundeven arith.cmpf arith.cmpi arith.select'''.split())


def _load(host, value, index):
    dtype = elem_name(value.type.element_type)
    ptr = host.ptrs[value]
    return (host.fp.fb.load_i64(ptr, index, dtype) if is_int(dtype)
            else host.fp.load(ptr, index, dtype))


def _store(host, result, index, value):
    dtype = elem_name(result.type.element_type)
    ptr = host.ptrs[result]
    if is_int(dtype):
        host.fp.fb.store_i64(value, ptr, index, dtype)
    else:
        host.fp.store(value, ptr, index, dtype)


def _coordinates(fn, flat, shape):
    coordinates = []
    for i, extent in enumerate(shape):
        value = flat
        pitch = prod(shape[i+1:])
        if pitch != 1:
            value = fn.udiv_i(value, fn.const(pitch))
        if i:
            value = fn.urem_i(value, fn.const(extent))
        coordinates.append(value)
    return coordinates


def _linear(fn, coordinates, shape):
    offset = fn.const(0)
    for coordinate, pitch in zip(coordinates, [prod(shape[i+1:]) for i in range(len(shape))]):
        term = coordinate if pitch == 1 else fn.mul_i(coordinate, fn.const(pitch))
        offset = fn.add_i(offset, term)
    return offset


def _finish(host, op):
    for result in op.results:
        host.vals[result] = host.load_arg(host.ptrs[result], result.type)
        host.materialized_results.add(result)


def _reduce(host, op):
    # Every body instruction is checked before emitting anything. Unknown
    # effectful, tensor-reading or multi-result scalar bodies keep their caller's
    # original route; no partially emitted loop is reported as unsupported.
    if not op.results or len(op.operands) != len(op.results)*2 or len(op.regions) != 1:
        return False
    if any(value not in host.ptrs for value in (*op.operands, *op.results)):
        return False
    body = op.regions[0].block
    instructions = list(body.ops)
    if (not instructions or instructions[-1].name != 'linalg.yield'
            or len(instructions[-1].operands) != len(op.results)
            or len(body.args) != len(op.operands)
            or any(x.name not in _SCALARS or len(x.results) != 1 for x in instructions[:-1])):
        return False
    available = set(body.args)
    for instruction in instructions:
        if any(value not in available and value not in host.vals for value in instruction.operands):
            return False
        available.update(instruction.results)
    count = len(op.results)
    inputs, seeds = list(op.operands[:count]), list(op.operands[count:])
    shape = tensor_shape(inputs[0].type)
    dimensions = attr_of(op, 'dimensions')
    if dimensions is None:
        return False
    dims = tuple(map(int, dimensions.get_values()))
    if len(set(dims)) != len(dims) or any(i < 0 or i >= len(shape) for i in dims):
        return False
    output_shape = tuple(size for i, size in enumerate(shape) if i not in dims)
    if (any(tensor_shape(x.type) != shape for x in inputs)
            or any(tensor_shape(x.type) != output_shape for x in (*seeds, *op.results))):
        return False
    # Read every old value before writing any output. The source result owns
    # a separate destination from all immutable input/init buffers.
    if (len(set(host.ptrs[x] for x in op.results)) != count
            or set(host.ptrs[x] for x in op.results).intersection(host.ptrs[x] for x in op.operands)):
        return False
    fn = host.fp.fb

    def evaluate(args):
        env = dict(zip(body.args, args))

        def read(value):
            return env[value] if value in env else host.get(value)

        for instruction in instructions[:-1]:
            env[instruction.results[0]] = host._scalar_op(instruction,
                [read(value) for value in instruction.operands])
        return tuple(host._round(read(value), elem_name(result.type.element_type))
                     for result,value in zip(op.results,instructions[-1].operands))

    # Only a trailing domain makes every output's recurrence contiguous in
    # the original C-order traversal. Interleaved reductions retain memory.
    if (getattr(host, 'reduction_carried', False) and dims
            and sorted(dims) == list(range(len(shape)-len(dims),len(shape)))):
        reduction_count = prod(shape[len(output_shape):])

        def output(index):
            initial = tuple(_load(host,seed,index) for seed in seeds)

            def step(inner, accumulators):
                flat = fn.add_i(fn.mul_i(index,fn.const(reduction_count)),inner)
                args = [_load(host,source,flat) for source in inputs]
                return evaluate([*args,*accumulators])

            final = fn.for_loop_carried(0,reduction_count,1,initial,step,retain_loop=True)
            for result,value in zip(op.results,final):
                _store(host,result,index,value)

        fn.for_loop(0,prod(output_shape),1,output,retain_loop=True)
        _finish(host,op)
        return True

    def initialize(index):
        values = [_load(host, seed, index) for seed in seeds]
        for result, value in zip(op.results, values):
            _store(host, result, index, value)

    fn.for_loop(0, prod(output_shape), 1, initialize, retain_loop=True)

    def update(flat):
        coordinates = _coordinates(fn, flat, shape)
        kept = [coordinate for i, coordinate in enumerate(coordinates) if i not in dims]
        target = _linear(fn, kept, output_shape)
        args = [_load(host, source, flat) for source in inputs]
        args.extend(_load(host, result, target) for result in op.results)
        for result, value in zip(op.results, evaluate(args)):
            _store(host, result, target, value)

    # Flattened C order exactly matches HostLinalg._iter_space, including
    # noncontiguous and multiple reduction dimensions. No reassociation/tree.
    fn.for_loop(0, prod(shape), 1, update, retain_loop=True)
    _finish(host, op)
    return True


def _layout(host, op):
    if len(op.results) != 1 or op.results[0] not in host.ptrs:
        return False
    result = op.results[0]
    shape = tensor_shape(result.type)
    fn = host.fp.fb
    source = op.operands[0] if op.operands else None
    index_of = None
    if op.name in ('tensor.splat', 'linalg.fill'):
        if source is None or source not in host.vals:
            return False
        value = host.get(source)
        if not hasattr(value, 'type'):
            return False
        fn.for_loop(0, prod(shape), 1,
            lambda index: _store(host, result, index, value), retain_loop=True)
        _finish(host, op)
        return True
    if source is None or source not in host.ptrs:
        return False
    src_shape = tensor_shape(source.type)
    if source.type.element_type != result.type.element_type:
        return False
    if host.ptrs[source] == host.ptrs[result]:
        return False
    if op.name in ('tensor.expand_shape', 'tensor.collapse_shape', 'tensor.reshape', 'tensor.cast'):
        if prod(shape) != prod(src_shape):
            return False
        index_of = lambda index: index
    elif op.name == 'linalg.transpose':
        permutation = attr_of(op, 'permutation')
        if permutation is None:
            return False
        perm = tuple(map(int, permutation.get_values()))
        if sorted(perm) != list(range(len(src_shape))) or shape != tuple(src_shape[i] for i in perm):
            return False

        def index_of(index):
            target = _coordinates(fn, index, shape)
            original = [None]*len(perm)
            for i, position in enumerate(perm):
                original[position] = target[i]
            return _linear(fn, original, src_shape)
    elif op.name == 'linalg.broadcast':
        dimensions = attr_of(op, 'dimensions')
        if dimensions is None:
            return False
        dims = tuple(map(int, dimensions.get_values()))
        kept = [i for i in range(len(shape)) if i not in dims]
        if len(set(dims)) != len(dims) or any(i < 0 or i >= len(shape) for i in dims):
            return False
        if tuple(shape[i] for i in kept) != src_shape:
            return False
        index_of = lambda index: _linear(fn, [_coordinates(fn,index,shape)[i] for i in kept], src_shape)
    elif op.name == 'tensor.extract_slice':
        if len(op.operands) != 1:
            return False
        attrs = [attr_of(op, key) for key in ('static_offsets', 'static_sizes', 'static_strides')]
        if any(value is None for value in attrs):
            return False
        offsets, sizes, strides = [tuple(map(int, value.get_values())) for value in attrs]
        if (len(sizes) != len(src_shape) or len(offsets) != len(src_shape)
                or len(strides) != len(src_shape)):
            return False
        # Rank reduction may remove only unit slice dimensions; the flattened
        # element order remains exactly the original slice's C order.
        retained = 0
        for size in sizes:
            if retained < len(shape) and size == shape[retained]:
                retained += 1
            elif size != 1:
                return False
        if retained != len(shape):
            return False
        if any(o < 0 or n < 0 or step <= 0 or n and o+(n-1)*step >= extent
               for o,n,step,extent in zip(offsets,sizes,strides,src_shape)):
            return False

        def index_of(index):
            coordinates = _coordinates(fn, index, sizes)
            original = [fn.add_i(fn.mul_i(c,fn.const(step)),fn.const(o))
                        for c,step,o in zip(coordinates,strides,offsets)]
            return _linear(fn, original, src_shape)
    else:
        return False
    fn.for_loop(0, prod(shape), 1,
        lambda index: _store(host, result, index, _load(host, source, index_of(index))),
        retain_loop=True)
    _finish(host, op)
    return True


def _concat(host, op):
    if (len(op.results) != 1 or not op.operands
            or any(value not in host.ptrs for value in (*op.operands, *op.results))):
        return False
    result = op.results[0]
    shape = tensor_shape(result.type)
    dim = attr_of(op, 'dim')
    if dim is None:
        return False
    axis = int(dim.value.data)
    if axis < 0 or axis >= len(shape):
        return False
    sources = [(value, tensor_shape(value.type)) for value in op.operands]
    if (any(value.type.element_type != result.type.element_type
            or len(source_shape) != len(shape)
            or any(size != shape[i] for i, size in enumerate(source_shape) if i != axis)
            for value, source_shape in sources)
            or sum(source_shape[axis] for _, source_shape in sources) != shape[axis]
            or any(host.ptrs[value] == host.ptrs[result] for value in op.operands)):
        return False
    fn = host.fp.fb
    axis_offset = 0
    for source, source_shape in sources:
        # Capture the static source and prefix at loop construction time. Every
        # source element is copied exactly once, without scalar conversion.
        def copy(index, source=source, source_shape=source_shape, prefix=axis_offset):
            coordinates = _coordinates(fn, index, source_shape)
            if prefix:
                coordinates[axis] = fn.add_i(coordinates[axis], fn.const(prefix))
            target = _linear(fn, coordinates, shape)
            _store(host, result, target, _load(host, source, index))
        fn.for_loop(0, prod(source_shape), 1, copy, retain_loop=True)
        axis_offset += source_shape[axis]
    _finish(host, op)
    return True


def _insert_slice(host, op):
    # Static same-rank insertions preserve both source tensors. A fresh result
    # receives the entire old destination before the declared slice is replaced.
    if (len(op.operands) != 2 or len(op.results) != 1
            or any(value not in host.ptrs for value in (*op.operands, *op.results))):
        return False
    source, destination = op.operands
    result = op.results[0]
    shape = tensor_shape(result.type)
    source_shape = tensor_shape(source.type)
    if (tensor_shape(destination.type) != shape or len(source_shape) != len(shape)
            or any(value.type.element_type != result.type.element_type for value in op.operands)
            or any(host.ptrs[value] == host.ptrs[result] for value in op.operands)):
        return False
    attrs = [attr_of(op, key) for key in ('static_offsets', 'static_sizes', 'static_strides')]
    if any(value is None for value in attrs):
        return False
    offsets, sizes, strides = [tuple(map(int, value.get_values())) for value in attrs]
    if (sizes != source_shape or len(offsets) != len(shape) or len(strides) != len(shape)
            or any(o < 0 or n < 0 or step <= 0 or o > extent
                   or n and o+(n-1)*step >= extent
                   for o,n,step,extent in zip(offsets,sizes,strides,shape))):
        return False
    fn = host.fp.fb
    if prod(shape):
        fn.for_loop(0, prod(shape), 1,
            lambda index: _store(host, result, index, _load(host, destination, index)),
            retain_loop=True)

    def insert(index):
        coordinates = _coordinates(fn, index, source_shape)
        positioned = []
        for coordinate, offset, stride in zip(coordinates, offsets, strides):
            if stride != 1:
                coordinate = fn.mul_i(coordinate, fn.const(stride))
            if offset:
                coordinate = fn.add_i(coordinate, fn.const(offset))
            positioned.append(coordinate)
        target = _linear(fn, positioned, shape)
        _store(host, result, target, _load(host, source, index))

    if prod(source_shape):
        fn.for_loop(0, prod(source_shape), 1, insert, retain_loop=True)
    _finish(host, op)
    return True


def _generic_reduce(host, op):
    """Count the shared generic interpreter's exact lexicographic recurrence."""
    from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpExpr
    if (len(op.results) != 1 or not op.operands or len(op.regions) != 1
            or any(value not in host.ptrs for value in (*op.operands, *op.results))):
        return False
    iterators = tuple(attr_of(op, 'iterator_types'))
    if not any('reduction' in str(item) for item in iterators):
        return False
    if any('parallel' not in str(item) and 'reduction' not in str(item) for item in iterators):
        return False
    maps = [item.data for item in attr_of(op, 'indexing_maps')]
    rank = len(iterators)
    if (len(maps) != len(op.operands)
            or any(mapping.num_dims != rank or mapping.num_symbols for mapping in maps)):
        return False
    result, seed = op.results[0], op.operands[-1]
    shape = tensor_shape(result.type)
    if (seed.type != result.type
            or any(host.ptrs[value] == host.ptrs[result] for value in op.operands)):
        return False
    parallel_dims = [i for i,item in enumerate(iterators) if 'parallel' in str(item)]
    if (len(maps[-1].results) != len(parallel_dims)
            or any(not isinstance(expr, AffineDimExpr) or expr.position != i
                   for expr,i in zip(maps[-1].results, parallel_dims))):
        return False
    bounds = [None]*rank

    def supported(expr):
        if isinstance(expr, AffineDimExpr):
            return 0 <= expr.position < rank
        if isinstance(expr, AffineConstantExpr):
            return expr.value >= 0
        if isinstance(expr, AffineBinaryOpExpr):
            return (expr.kind.name in ('Add', 'Mul', 'FloorDiv', 'CeilDiv', 'Mod')
                    and supported(expr.lhs) and supported(expr.rhs)
                    and (expr.kind.name not in ('FloorDiv', 'CeilDiv', 'Mod')
                         or isinstance(expr.rhs, AffineConstantExpr) and expr.rhs.value > 0))
        return False

    for mapping, operand in zip(maps, op.operands):
        operand_shape = tensor_shape(operand.type)
        if len(mapping.results) != len(operand_shape):
            return False
        for expr, extent in zip(mapping.results, operand_shape):
            if not supported(expr):
                return False
            if isinstance(expr, AffineDimExpr):
                if bounds[expr.position] not in (None, extent):
                    return False
                bounds[expr.position] = extent
    # This is HostLinalg's direct-dimension bound rule. A window dimension
    # needs an explicit kernel tensor or other direct map, never an inferred
    # subtraction/division that could change the source iteration domain.
    if any(bound is None or bound < 0 for bound in bounds):
        return False
    for mapping, operand in zip(maps, op.operands):
        for expr, extent in zip(mapping.results, tensor_shape(operand.type)):
            # Nonnegative add/mul/floor-div expressions are monotone. Modulo
            # needs its divisor's full bound instead of endpoint evaluation.
            def upper(expr):
                if isinstance(expr, AffineBinaryOpExpr) and expr.kind.name == 'Mod':
                    return expr.rhs.value-1
                if isinstance(expr, AffineBinaryOpExpr):
                    left,right=upper(expr.lhs),upper(expr.rhs)
                    if expr.kind.name == 'Add': return left+right
                    if expr.kind.name == 'Mul': return left*right
                    if expr.kind.name == 'FloorDiv': return left//right
                    return (left+right-1)//right
                return bounds[expr.position]-1 if isinstance(expr,AffineDimExpr) else expr.value
            if prod(bounds) and upper(expr) >= extent:
                return False
    body = op.regions[0].block
    instructions = tuple(body.ops)
    if (len(body.args) != len(op.operands) or not instructions
            or instructions[-1].name != 'linalg.yield' or len(instructions[-1].operands) != 1):
        return False
    available = set(body.args) | set(host.vals)
    for instruction in instructions[:-1]:
        if (instruction.name not in _SCALARS or len(instruction.results) != 1
                or any(value not in available for value in instruction.operands)):
            return False
        available.update(instruction.results)
    if instructions[-1].operands[0] not in available:
        return False
    fn = host.fp.fb

    def affine(expr, coordinates):
        if isinstance(expr, AffineDimExpr): return coordinates[expr.position]
        if isinstance(expr, AffineConstantExpr): return fn.const(expr.value)
        left,right = affine(expr.lhs,coordinates),affine(expr.rhs,coordinates)
        kind=expr.kind.name
        if kind=='Add': return fn.add_i(left,right)
        if kind=='Mul': return fn.mul_i(left,right)
        if kind=='FloorDiv': return fn.udiv_i(left,right)
        if kind=='Mod': return fn.urem_i(left,right)
        return fn.udiv_i(fn.add_i(left,fn.sub_i(right,fn.const(1))),right)

    if (getattr(host, 'reduction_carried', False)
            and parallel_dims == list(range(len(parallel_dims)))):
        reduction_shape = bounds[len(parallel_dims):]

        def output(index):
            outer = _coordinates(fn,index,shape)
            initial = (_load(host,seed,index),)

            def step(inner, accumulators):
                coordinates = [*outer,*_coordinates(fn,inner,reduction_shape)]
                args=[]
                for operand,mapping,argument in zip(op.operands[:-1],maps[:-1],body.args[:-1]):
                    if argument.uses:
                        offset = _linear(fn,[affine(expr,coordinates) for expr in mapping.results],
                                         tensor_shape(operand.type))
                        args.append(_load(host,operand,offset))
                    else:
                        args.append(host.zero(elem_name(operand.type.element_type)))
                value = host._scalar_block(body,[*args,*accumulators])
                return (host._round(value,elem_name(result.type.element_type)),)

            final = fn.for_loop_carried(0,prod(reduction_shape),1,initial,step,retain_loop=True)
            _store(host,result,index,final[0])

        fn.for_loop(0,prod(shape),1,output,retain_loop=True)
        _finish(host,op)
        return True

    if prod(shape):
        fn.for_loop(0, prod(shape), 1,
            lambda index: _store(host,result,index,_load(host,seed,index)),retain_loop=True)

    def update(flat):
        coordinates = _coordinates(fn,flat,bounds)
        target = _linear(fn,[coordinates[i] for i in parallel_dims],shape)
        args=[]
        for operand,mapping,argument in zip(op.operands[:-1],maps[:-1],body.args[:-1]):
            if argument.uses:
                index = _linear(fn,[affine(expr,coordinates) for expr in mapping.results],
                                tensor_shape(operand.type))
                args.append(_load(host,operand,index))
            else:
                args.append(host.zero(elem_name(operand.type.element_type)))
        args.append(_load(host,result,target))
        value = host._scalar_block(body,args)
        _store(host,result,target,host._round(value,elem_name(result.type.element_type)))

    if prod(bounds):
        fn.for_loop(0, prod(bounds), 1, update, retain_loop=True)
    _finish(host,op)
    return True


def try_lower(host, op):
    """Return True only after fully lowering the original memory-backed source."""
    if getattr(host, 'row_alignment', 1) != 1:
        return False
    for value in (*op.operands, *op.results):
        if hasattr(value.type, 'get_shape') and any(int(d) < 0 for d in value.type.get_shape()):
            return False
    if op.name == 'linalg.generic':
        return _generic_reduce(host, op)
    if op.name == 'tensor.insert_slice':
        return _insert_slice(host, op)
    if op.name == 'tensor.concat':
        return _concat(host, op)
    if op.name == 'linalg.reduce':
        return _reduce(host, op)
    if op.name in ('tensor.splat', 'linalg.fill', 'tensor.expand_shape',
                   'tensor.collapse_shape', 'tensor.reshape', 'tensor.cast',
                   'linalg.transpose', 'linalg.broadcast', 'tensor.extract_slice'):
        return _layout(host, op)
    return False


def try_lower_quant_group(host, ops):
    """Fuse an ordered shared quant rewrite into counted scalar-region loops.

    Rank-zero helper regions (notably the rounded scale reciprocal) execute
    once before the element loop. All arithmetic remains in HostLinalg's
    scalar-region implementation, including rounding at each tensor boundary.
    """
    from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpExpr
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr

    ops = tuple(ops)
    if (not ops or ops[-1].name != 'linalg.generic'
            or getattr(host, 'row_alignment', 1) != 1):
        return False
    final = ops[-1].results[0] if len(ops[-1].results) == 1 else None
    if final is None or final not in host.ptrs:
        return False
    shape = tensor_shape(final.type)
    if any(d < 0 for d in shape):
        return False
    group = set(ops)
    available = set(host.vals)
    tensors = {}  # generated tensor -> its static shape
    plans = {}

    def affine_ok(expr, rank):
        if isinstance(expr, AffineDimExpr):
            return 0 <= expr.position < rank
        if isinstance(expr, AffineConstantExpr):
            return expr.value >= 0
        if isinstance(expr, AffineBinaryOpExpr):
            return (expr.kind.name in ('Add', 'Mul', 'FloorDiv', 'CeilDiv', 'Mod')
                    and affine_ok(expr.lhs, rank) and affine_ok(expr.rhs, rank)
                    and (expr.kind.name not in ('FloorDiv', 'CeilDiv', 'Mod')
                         or isinstance(expr.rhs, AffineConstantExpr) and expr.rhs.value > 0))
        return False

    # Preflight the entire chain: unsupported regions emit no partial code.
    for op in ops:
        if len(op.results) != 1:
            return False
        result = op.results[0]
        if (hasattr(result.type, 'get_shape') and result != final
                and any(use.operation not in group for use in result.uses)):
            return False
        if op.name == 'arith.constant':
            if not isinstance(attr_of(op, 'value'), (FloatAttr, IntegerAttr)):
                return False
            available.add(result)
            continue
        if op.name == 'tensor.empty':
            if op.operands:
                return False
            tensors[result] = tensor_shape(result.type)
            continue
        if op.name != 'linalg.generic' or len(op.regions) != 1:
            return False
        domain = tensor_shape(result.type)
        if domain not in ((), shape) or any(d < 0 for d in domain):
            return False
        if any('parallel' not in str(x) for x in attr_of(op, 'iterator_types')):
            return False
        maps = [x.data for x in attr_of(op, 'indexing_maps')]
        body = op.regions[0].block
        instructions = tuple(body.ops)
        if (len(maps) != len(op.operands) or len(body.args) != len(op.operands)
                or not instructions or instructions[-1].name != 'linalg.yield'
                or len(instructions[-1].operands) != 1 or body.args[-1].uses
                or len(maps[-1].results) != len(domain)
                or any(not isinstance(e, AffineDimExpr) or e.position != i
                       for i, e in enumerate(maps[-1].results))):
            return False
        for operand, mapping, argument in zip(op.operands, maps, body.args):
            operand_shape = tensor_shape(operand.type)
            if (mapping.num_dims != len(domain) or mapping.num_symbols
                    or len(mapping.results) != len(operand_shape)
                    or any(not affine_ok(e, len(domain)) for e in mapping.results)):
                return False
            if not argument.uses:
                continue
            if operand in tensors:
                # Register intermediates are either scalars or the current
                # element of an identically indexed full-shape producer.
                if (operand.owner.name == 'tensor.empty' or tensors[operand] not in ((), domain)
                        or any(not isinstance(e, AffineDimExpr) or e.position != i
                               for i,e in enumerate(mapping.results))):
                    return False
            elif operand not in host.ptrs:
                return False
        local = set(body.args) | available
        for instruction in instructions[:-1]:
            if (instruction.name not in _SCALARS or len(instruction.results) != 1
                    or any(value not in local for value in instruction.operands)):
                return False
            local.update(instruction.results)
        if instructions[-1].operands[0] not in local:
            return False
        tensors[result] = domain
        plans[op] = (domain, maps, body, instructions)

    fn = host.fp.fb

    def affine(expr, coordinates):
        if isinstance(expr, AffineDimExpr):
            return coordinates[expr.position]
        if isinstance(expr, AffineConstantExpr):
            return fn.const(expr.value)
        left, right = affine(expr.lhs, coordinates), affine(expr.rhs, coordinates)
        kind = expr.kind.name
        if kind == 'Add':
            return fn.add_i(left, right)
        if kind == 'Mul':
            return fn.mul_i(left, right)
        if kind == 'FloorDiv':
            return fn.udiv_i(left, right)
        if kind == 'Mod':
            return fn.urem_i(left, right)
        return fn.udiv_i(fn.add_i(left, fn.sub_i(right, fn.const(1))), right)

    scalar_tensors = {}

    def evaluate(op, coordinates, registers):
        domain, maps, body, instructions = plans[op]
        args = []
        for operand, mapping, argument in zip(op.operands, maps, body.args):
            if not argument.uses:
                value = host.zero(elem_name(operand.type.element_type))
            elif operand in registers:
                value = registers[operand]
            elif operand in scalar_tensors:
                value = scalar_tensors[operand]
            else:
                offset = _linear(fn, [affine(e, coordinates) for e in mapping.results],
                                 tensor_shape(operand.type))
                value = _load(host, operand, offset)
            args.append(value)
        env = dict(zip(body.args, args))

        def read(value):
            return env[value] if value in env else host.get(value)

        for instruction in instructions[:-1]:
            env[instruction.results[0]] = host._scalar_op(
                instruction, [read(value) for value in instruction.operands])
        return host._round(read(instructions[-1].operands[0]),
                           elem_name(op.results[0].type.element_type))

    for op in ops:
        if op.name == 'arith.constant':
            host.vals[op.results[0]] = host._const(op)
        elif op in plans and not plans[op][0]:
            scalar_tensors[op.results[0]] = evaluate(op, [], {})

    def element(index):
        coordinates = _coordinates(fn, index, shape)
        registers = {}
        for op in ops:
            if op in plans and plans[op][0]:
                registers[op.results[0]] = evaluate(op, coordinates, registers)
        _store(host, final, index,
               registers[final] if final in registers else scalar_tensors[final])

    fn.for_loop(0, prod(shape), 1, element, retain_loop=True)
    _finish(host, ops[-1])
    return True
