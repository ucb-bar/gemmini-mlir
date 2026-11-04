"""Literal ordered compare/select reductions; no max/NaN equivalence premise.

The emitted source retains the original comparisons and selected f32 values.
Only a closed recipe with static, independently bounded affine accesses is
admitted. This is source/CPU lowering, before tensor bufferization and fusion.
"""

SOURCE = r'''
def _ordered_select_pool_spec(op):
    from torch_mlir import ir
    if (op.operation.name != "linalg.generic" or len(op.operands) not in (2, 3)
        or len(op.results) != 1 or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1):
        return None
    ancestor = op.operation
    while ancestor is not None:
        if any("strictfp" in name or "strictfp" in str(ancestor.attributes[name])
               or "fenv" in name
               for name in ancestor.attributes):
            return None
        ancestor = ancestor.parent
    domain = _OrderedReductionDomains()
    if not domain.ordinary(op, {"indexing_maps", "iterator_types", "operandSegmentSizes"}):
        return None
    types = []
    for operand in op.operands:
        if not isinstance(operand.type, ir.RankedTensorType):
            return None
        typ = ir.RankedTensorType(operand.type)
        if not isinstance(typ.element_type, ir.F32Type) or any(x < 0 for x in typ.shape):
            return None
        types.append(typ)
    if op.results[0].type != op.operands[-1].type:
        return None
    block = op.regions[0].blocks[0]
    args, body = list(block.arguments), list(block.operations)
    if len(args) != len(op.operands) or len(body) != 5:
        return None
    if [child.operation.name for child in body] != [
        "arith.cmpf", "arith.cmpf", "arith.ori", "arith.select", "linalg.yield"
    ] or any(child.regions for child in body):
        return None
    if any(not domain.ordinary(child, {"predicate"} if i < 2 else ())
           for i, child in enumerate(body)):
        return None
    # cmpf predicates are the upstream arith enum: ogt=2, uno=14.
    if (int(ir.IntegerAttr(body[0].attributes["predicate"]).value) != 14
        or int(ir.IntegerAttr(body[1].attributes["predicate"]).value) != 2
        or list(body[0].operands) != [args[0], args[0]]
        or list(body[1].operands) != [args[0], args[-1]]
        or list(body[2].operands) != [body[0].results[0], body[1].results[0]]
        or list(body[3].operands) != [body[2].results[0], args[0], args[-1]]
        or list(body[4].operands) != [body[3].results[0]]
        or len(args) == 3 and list(args[1].uses)):
        return None
    iterators = [str(x) for x in op.attributes["iterator_types"]]
    parallel = "#linalg.iterator_type<parallel>"
    reduction = "#linalg.iterator_type<reduction>"
    nparallel = next((i for i, x in enumerate(iterators) if x != parallel), len(iterators))
    dims = len(iterators)
    parallel_axes = tuple(i for i, kind in enumerate(iterators) if kind == parallel)
    reduction_axes = tuple(i for i, kind in enumerate(iterators) if kind == reduction)
    if not reduction_axes or len(parallel_axes) + len(reduction_axes) != dims:
        return None
    maps = [ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != len(types) or any(m.n_dims != dims or m.n_symbols for m in maps):
        return None
    extents = [None] * dims
    def projected(affine, typ):
        positions = []
        if len(affine.results) != len(typ.shape):
            return None
        for expr, extent in zip(affine.results, typ.shape):
            if not isinstance(expr, ir.AffineDimExpr):
                return None
            pos = ir.AffineDimExpr(expr).position
            if pos in positions or pos >= dims or (extents[pos] is not None and extents[pos] != extent):
                return None
            positions.append(pos)
            extents[pos] = extent
        return positions
    outputs = projected(maps[-1], types[-1])
    if outputs is None or sorted(outputs) != list(parallel_axes):
        return None
    if len(types) == 3:
        windows = projected(maps[1], types[1])
        if windows is None or sorted(windows) != list(reduction_axes):
            return None
    else:
        if projected(maps[0], types[0]) is None:
            return None
    if any(extent is None or extent < 0 for extent in extents):
        return None
    # Affine expressions restricted to a nonnegative constant plus a sum of
    # nonnegative constant multiples of dimensions. Bounds follow independently
    # from output/kernel projected maps, never inferred by division of a window.
    def linear(expr):
        if isinstance(expr, ir.AffineDimExpr):
            pos = ir.AffineDimExpr(expr).position
            return (0, {pos: 1}) if pos < dims else None
        if isinstance(expr, ir.AffineConstantExpr):
            value = ir.AffineConstantExpr(expr).value
            return (value, {}) if value >= 0 else None
        if not isinstance(expr, (ir.AffineAddExpr, ir.AffineMulExpr)):
            return None
        binary = ir.AffineAddExpr(expr) if isinstance(expr, ir.AffineAddExpr) else ir.AffineMulExpr(expr)
        left, right = linear(binary.lhs), linear(binary.rhs)
        if left is None or right is None:
            return None
        if isinstance(expr, ir.AffineAddExpr):
            coeff = dict(left[1])
            for pos, value in right[1].items():
                coeff[pos] = coeff.get(pos, 0) + value
            return left[0] + right[0], coeff
        if isinstance(expr, ir.AffineMulExpr):
            if left[1] and right[1]:
                return None
            scale, value = (right[0], left) if left[1] else (left[0], right)
            return scale * value[0], {pos: scale * n for pos, n in value[1].items()}
        return None
    if len(maps[0].results) != len(types[0].shape):
        return None
    for expr, extent in zip(maps[0].results, types[0].shape):
        form = linear(expr)
        if form is None:
            return None
        # A zero trip iteration space performs no source access.
        if all(extents) and form[0] + sum(n * (extents[pos] - 1) for pos, n in form[1].items()) >= extent:
            return None
        # Index arithmetic must remain representable on the supported 64-bit ABI.
        if form[0] + sum(n * max(extents[pos] - 1, 0) for pos, n in form[1].items()) >= 2**63:
            return None
    # Preserve the established trailing-reduction emission byte for byte.
    # New forms use source-axis positions explicitly, including a reduction
    # before the first output iterator. Per-output folds retain source order.
    axes = (nparallel if parallel_axes == tuple(range(nparallel))
            and reduction_axes == tuple(range(nparallel, dims))
            else (parallel_axes, reduction_axes))
    return maps[0], outputs, extents, axes


def _scalarize_ordered_select_pools(ctx, module):
    from torch_mlir import ir
    todo = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _ordered_select_pool_spec(child)
                    if spec is not None:
                        todo.append((child, spec))
                    else:
                        walk(child.operation)
    walk(module.operation)
    with ctx:
        for old, (input_map, outputs, extents, nparallel) in todo:
            with old.location, ir.InsertionPoint(old):
                index, f32, tensor = ir.IndexType.get(), ir.F32Type.get(), old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return ir.Operation.create(name, operands=list(operands), results=list(results),
                                               attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index],
                                  attributes={"value": ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(n) for n in extents]
                body = list(old.regions[0].blocks[0].operations)
                def reduction(depth, acc, coordinates):
                    if depth == len(extents):
                        indices = [create("affine.apply", coordinates, [index],
                            {"map": ir.AffineMapAttr.get(ir.AffineMap.get(
                                len(extents), 0, [expr]))}).results[0]
                            for expr in input_map.results]
                        value = create("tensor.extract", [old.operands[0], *indices], [f32]).results[0]
                        args = list(old.regions[0].blocks[0].arguments)
                        bindings = {args[0]: value, args[-1]: acc}
                        for source in body[:-1]:
                            copied = create(source.operation.name,
                                [bindings[v] for v in source.operands],
                                [v.type for v in source.results],
                                {name: source.attributes[name] for name in source.attributes})
                            bindings.update(zip(source.results, copied.results))
                        return bindings[body[-1].operands[0]]
                    loop = create("scf.for", [zero, limits[depth], one, acc], [f32], regions=1)
                    block = ir.Block.create_at_start(loop.regions[0], [index, f32])
                    with ir.InsertionPoint(block):
                        result = reduction(depth + 1, block.arguments[1], [*coordinates, block.arguments[0]])
                        create("scf.yield", [result])
                    return loop.results[0]
                def output(depth, current, coordinates):
                    if depth == nparallel:
                        indices = [coordinates[pos] for pos in outputs]
                        initial = create("tensor.extract", [current, *indices], [f32]).results[0]
                        result = reduction(depth, initial, coordinates)
                        return create("tensor.insert", [result, current, *indices], [tensor]).results[0]
                    loop = create("scf.for", [zero, limits[depth], one, current], [tensor], regions=1)
                    block = ir.Block.create_at_start(loop.regions[0], [index, tensor])
                    with ir.InsertionPoint(block):
                        updated = output(depth + 1, block.arguments[1], [*coordinates, block.arguments[0]])
                        create("scf.yield", [updated])
                    return loop.results[0]
                if isinstance(nparallel, int):
                    result = output(0, old.operands[-1], [])
                else:
                    parallel_axes, reduction_axes = nparallel
                    def interleaved_reduction(depth, acc, coordinates):
                        if depth == len(reduction_axes):
                            ordered = [coordinates[pos] for pos in range(len(extents))]
                            # Invoke the existing exact scalar recipe at its
                            # leaf, where coordinates are in original map order.
                            return reduction(len(extents), acc, ordered)
                        axis = reduction_axes[depth]
                        loop = create("scf.for", [zero, limits[axis], one, acc], [f32], regions=1)
                        block = ir.Block.create_at_start(loop.regions[0], [index, f32])
                        with ir.InsertionPoint(block):
                            result = interleaved_reduction(depth + 1, block.arguments[1],
                                {**coordinates, axis: block.arguments[0]})
                            create("scf.yield", [result])
                        return loop.results[0]
                    def interleaved_output(depth, current, coordinates):
                        if depth == len(parallel_axes):
                            indices = [coordinates[pos] for pos in outputs]
                            initial = create("tensor.extract", [current, *indices], [f32]).results[0]
                            result = interleaved_reduction(0, initial, coordinates)
                            return create("tensor.insert", [result, current, *indices], [tensor]).results[0]
                        axis = parallel_axes[depth]
                        loop = create("scf.for", [zero, limits[axis], one, current], [tensor], regions=1)
                        block = ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with ir.InsertionPoint(block):
                            updated = interleaved_output(depth + 1, block.arguments[1],
                                {**coordinates, axis: block.arguments[0]})
                            create("scf.yield", [updated])
                        return loop.results[0]
                    result = interleaved_output(0, old.operands[-1], {})
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = ir.StringAttr(old.attributes["prov.transforms"]).value + "," if "prov.transforms" in old.attributes else ""
                result.owner.attributes["prov.transforms"] = ir.StringAttr.get(previous + "scalar_ordered_tensor_reductions")
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    return len(todo)
'''
