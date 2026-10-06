import sys,json
from torch_mlir import ir

def _scalar_contraction_spec(op):
    from torch_mlir import ir as _sc_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 3
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _sc_ir.RankedTensorType) for t in types):
        return None
    types = [_sc_ir.RankedTensorType(t) for t in types]
    if (any(not isinstance(t.element_type, _sc_ir.F32Type) for t in types)
            or op.results[0].type != op.operands[2].type
            or any(d < 0 for t in types for d in t.shape)):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 3 or len(body) != 3
            or [x.operation.name for x in body] != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any("fastmath" in x.attributes and str(x.attributes["fastmath"]) != "#arith.fastmath<none>" for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for x in body[:2] for name in x.attributes)):
        return None
    mul, add, yield_op = body
    args = list(block.arguments)
    if list(mul.operands) == args[:2]:
        mul_order = [0, 1]
    elif list(mul.operands) == args[1::-1]:
        mul_order = [1, 0]
    else:
        return None
    if list(add.operands) == [mul.results[0], args[2]]:
        add_order = [0, 1]
    elif list(add.operands) == [args[2], mul.results[0]]:
        add_order = [1, 0]
    else:
        return None
    if list(yield_op.operands) != [add.results[0]]:
        return None
    maps = [_sc_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 3:
        return None
    dims = maps[0].n_dims
    if dims < 2 or any(m.n_dims != dims or m.n_symbols for m in maps):
        return None
    iters = list(op.attributes["iterator_types"])
    # Enum attribute printer is the upstream binding's stable spelling.
    if ([str(x) for x in iters] !=
            ["#linalg.iterator_type<parallel>"] * (dims - 1) + ["#linalg.iterator_type<reduction>"]):
        return None
    positions = []
    extents = [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if not isinstance(expr, _sc_ir.AffineDimExpr):
                return None
            d = _sc_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if (sorted(positions[2]) != list(range(dims - 1))
            or any(dims - 1 not in pos for pos in positions[:2])
            or any(d is None for d in extents)):
        return None
    return positions, extents, mul_order, add_order


def _scalarize_tensor_contractions(ctx, module, outputs=1, reduction_unroll=0, rows=1):
    from torch_mlir import ir as _sc_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _scalar_contraction_spec(inner)
                    eligible = spec is not None and spec[1][-2] % outputs == 0
                    if eligible and rows != 1:
                        positions, extents, _, _ = spec
                        row_axis, col_axis = len(extents) - 3, len(extents) - 2
                        # A rectangular tile is a matrix schedule only when the
                        # exact projected maps prove complementary row/column
                        # reuse. No names, shapes alone or pointer facts suffice.
                        eligible = (row_axis >= 0 and extents[row_axis] % rows == 0
                                    and row_axis in positions[0] and col_axis not in positions[0]
                                    and col_axis in positions[1] and row_axis not in positions[1])
                        owner = inner.operation
                        while eligible and owner is not None:
                            if any(a in owner.attributes for a in ("strictfp", "llvm.strictfp")):
                                eligible = False
                            owner = owner.parent
                    if eligible:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)

    walk(module.operation)
    with ctx:
        for old, (positions, extents, mul_order, add_order) in todo:
            with old.location, _sc_ir.InsertionPoint(old):
                index = _sc_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sc_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                   attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sc_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(e) for e in extents]
                step = one if outputs == 1 else constant(outputs)
                offsets = [zero, *[constant(i) for i in range(1, outputs)]]
                row_offsets = [zero, *[constant(i) for i in range(1, rows)]]
                row_step = one if rows == 1 else constant(rows)

                def output_loop(depth, current, indices):
                    if depth < len(extents) - 1:
                        stride = (step if depth == len(extents) - 2 else
                                  row_step if rows != 1 and depth == len(extents) - 3 else one)
                        loop = create("scf.for", [zero, limits[depth], stride, current], [tensor], regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sc_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1], [*indices, block.arguments[0]])
                            create("scf.yield", [updated])
                        return loop.results[0]
                    if rows == 1:
                        lanes = [indices]
                        for offset in offsets[1:]:
                            column = create("arith.addi", [indices[-1], offset], [index]).results[0]
                            lanes.append([*indices[:-1], column])
                    else:
                        lanes = []
                        for row_offset in row_offsets:
                            row = (indices[-2] if row_offset == zero else
                                   create("arith.addi", [indices[-2], row_offset], [index]).results[0])
                            for offset in offsets:
                                column = (indices[-1] if offset == zero else
                                          create("arith.addi", [indices[-1], offset], [index]).results[0])
                                lanes.append([*indices[:-2], row, column])
                    out_indices = [[lane[d] for d in positions[2]] for lane in lanes]
                    initial = [create("tensor.extract", [current, *idx], [scalar]).results[0]
                               for idx in out_indices]
                    # SCF -> CF forwards LLVM-typed values; CF -> LLVM keeps the key.
                    # Use the final branch property's spelling: a retained namespaced
                    # llvm.loop_annotation is ignored by the installed translator.
                    attributes = ({"loop_annotation": _sc_ir.Attribute.parse(
                        "#llvm.loop_annotation<unroll = <count = " + str(reduction_unroll) + " : i32>>")}
                        if reduction_unroll else {})
                    lane_count = rows * outputs
                    loop = create("scf.for", [zero, limits[-1], one, *initial], [scalar] * lane_count,
                                  attributes=attributes, regions=1)
                    block = _sc_ir.Block.create_at_start(loop.regions[0], [index, *([scalar] * lane_count)])
                    with _sc_ir.InsertionPoint(block):
                        accumulated, shared = [], {}
                        for lane_number, lane in enumerate(lanes):
                            all_indices = [*lane, block.arguments[0]]
                            values = []
                            for i in range(2):
                                if rows == 1:
                                    key = i if len(extents) - 2 not in positions[i] else None
                                else:
                                    key = (i,
                                           lane_number // outputs if len(extents) - 3 in positions[i] else 0,
                                           lane_number % outputs if len(extents) - 2 in positions[i] else 0)
                                if key is not None and key in shared:
                                    value = shared[key]
                                else:
                                    value = create("tensor.extract", [old.operands[i], *[all_indices[d] for d in positions[i]]],
                                                   [scalar]).results[0]
                                    if key is not None:
                                        shared[key] = value
                                values.append(value)
                            product = create("arith.mulf", [values[d] for d in mul_order], [scalar]).results[0]
                            operands = [product, block.arguments[lane_number + 1]]
                            accumulated.append(create("arith.addf", [operands[d] for d in add_order], [scalar]).results[0])
                        create("scf.yield", accumulated)
                    for value, idx in zip(loop.results, out_indices):
                        current = create("tensor.insert", [value, current, *idx], [tensor]).results[0]
                    return current

                result = output_loop(0, old.operands[2], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _sc_ir.StringAttr(previous).value + "," if previous is not None else ""
                suffix = ("_2x4_outputs" if rows != 1 else
                          "" if outputs == 1 else "_" + str(outputs) + "_outputs")
                transforms = text + "scalar_contraction_accumulator" + suffix
                if reduction_unroll:
                    transforms += ",unroll_scalar_contraction_reduction_by_" + str(reduction_unroll)
                replacement.attributes["prov.transforms"] = _sc_ir.StringAttr.get(transforms)
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)



ctx=ir.Context()
module=ir.Module.parse(open(sys.argv[1]).read(),ctx)
records=[]
def walk(op):
 for region in op.regions:
  for block in region.blocks:
   for inner in block.operations:
    spec=_scalar_contraction_spec(inner)
    if spec is not None:
     maps,extents,mul,add=spec
     row,col=len(extents)-3,len(extents)-2
     legal=(row>=0 and extents[row]%2==0 and extents[col]%4==0 and row in maps[0] and col not in maps[0] and col in maps[1] and row not in maps[1])
     records.append(dict(positions=maps,extents=extents,mul_order=mul,add_order=add,rectangular_map_extent_eligibility=legal,types=list(map(str,(v.type for v in inner.operands)))))
    walk(inner.operation)
walk(module.operation)
print(json.dumps(records))
