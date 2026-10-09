SOURCE='/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006/prepacket.mlir'

def _pointwise_packet_spec(op):
    from torch_mlir import ir as _pp_ir
    if (op.operation.name != "linalg.generic" or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _pp_ir.RankedTensorType) or any(d < 0 for d in t.shape) for t in types):
        return None
    if len(op.operands) < 2:
        return None
    maps = [_pp_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    positions, extents = [], [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if not isinstance(expr, _pp_ir.AffineDimExpr):
                return None
            d = _pp_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if sorted(positions[-1]) != list(range(dims)) or any(d is None or d <= 0 for d in extents):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if len(block.arguments) != len(types) or not body or body[-1].operation.name != "linalg.yield" or len(body[-1].operands) != 1:
        return None
    if list(block.arguments[-1].uses):
        return None
    allowed = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
               "arith.negf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.cmpi",
               "arith.select", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi",
               "arith.muli", "arith.shli", "arith.shrsi", "arith.shrui", "arith.bitcast",
               "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extsi",
               "arith.extui", "arith.trunci", "arith.extf", "arith.truncf", "math.fma"}
    fmas = division = 0
    for inner in body[:-1]:
        if inner.operation.name not in allowed or inner.regions or len(inner.results) != 1:
            return None
        if "fastmath" in inner.attributes and str(inner.attributes["fastmath"]) != "#arith.fastmath<none>":
            return None
        if isinstance(inner.results[0].type, _pp_ir.VectorType):
            return None
        if inner.operation.name == "math.fma" and isinstance(inner.results[0].type, _pp_ir.F32Type):
            fmas += 1
        if inner.operation.name == "arith.divf" and isinstance(inner.results[0].type, _pp_ir.F32Type):
            division += 1
    return (positions, extents, body) if fmas >= 4 and division else None


def _pointwise_broadcast_axis(op, positions, extents, lanes):
    # Scalar inputs are already loop invariant and do not justify an interchange.
    # Every shared load comes from the exact same immutable tensor and coordinates.
    args = list(op.regions[0].blocks[0].arguments)
    choices = []
    for axis, extent in enumerate(extents):
        if extent < lanes:
            continue
        shared = sum(bool(pos) and axis not in pos and bool(list(arg.uses))
                     for arg, pos in zip(args[:-1], positions[:-1]))
        if shared:
            choices.append((shared, extent, -axis))
    return -max(choices)[2] if choices else None


def _packetize_pointwise(ctx, module, lanes=2, broadcast=False):
    from torch_mlir import ir as _pp_ir
    todo = []
    def walk(op):
        if any("strictfp" in name or "strictfp" in str(op.attributes[name]) for name in op.attributes):
            return
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _pointwise_packet_spec(inner)
                    if spec is not None:
                        axis = _pointwise_broadcast_axis(inner, spec[0], spec[1], lanes) if broadcast else len(spec[1]) - 1
                        if axis is not None:
                            todo.append((inner, spec, axis))
                    else:
                        walk(inner.operation)
    walk(module.operation)
    with ctx:
        for old, (positions, extents, scalar_body), axis in todo:
            with old.location, _pp_ir.InsertionPoint(old):
                index = _pp_ir.IndexType.get()
                tensor_type = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _pp_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                  attributes=attributes or {}, regions=regions)
                constants = {}
                def c(n):
                    if n not in constants:
                        constants[n] = create("arith.constant", results=[index], attributes={"value": _pp_ir.IntegerAttr.get(index, n)}).results[0]
                    return constants[n]
                # Constants dominate all subsequently constructed loops.
                for n in [0, 1, lanes, *extents, *range(lanes), extents[axis] // lanes * lanes]:
                    c(n)
                def packet(current, indices, width):
                    lane_indices = [indices]
                    for lane in range(1, width):
                        coordinate = create("arith.addi", [indices[axis], c(lane)], [index]).results[0]
                        ids = list(indices)
                        ids[axis] = coordinate
                        lane_indices.append(ids)
                    mappings = []
                    shared_extracts = {}
                    args = list(old.regions[0].blocks[0].arguments)
                    for ids in lane_indices:
                        mapped = {}
                        for arg, operand, pos in zip(args[:-1], old.operands[:-1], positions[:-1]):
                            key = (operand, *[ids[d] for d in pos])
                            if broadcast and key in shared_extracts:
                                mapped[arg] = shared_extracts[key]
                            else:
                                mapped[arg] = create("tensor.extract", [operand, *key[1:]], [arg.type]).results[0]
                                if broadcast:
                                    shared_extracts[key] = mapped[arg]
                        mappings.append(mapped)
                    # Breadth-first scalar operation order exposes independent chains.
                    for scalar_op in scalar_body[:-1]:
                        for mapped in mappings:
                            cloned = create(scalar_op.operation.name,
                                [mapped.get(v, v) for v in scalar_op.operands],
                                [r.type for r in scalar_op.results],
                                {name: scalar_op.attributes[name] for name in scalar_op.attributes})
                            mapped[scalar_op.results[0]] = cloned.results[0]
                    for ids, mapped in zip(lane_indices, mappings):
                        value = mapped.get(scalar_body[-1].operands[0], scalar_body[-1].operands[0])
                        current = create("tensor.insert", [value, current, *[ids[d] for d in positions[-1]]], [tensor_type]).results[0]
                    return current
                def nest(depth, current, indices):
                    if depth == len(extents) - 1:
                        end = extents[-1] // lanes * lanes
                        if end:
                            loop = create("scf.for", [c(0), c(end), c(lanes), current], [tensor_type], regions=1)
                            block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                            with _pp_ir.InsertionPoint(block):
                                value = packet(block.arguments[1], [*indices, block.arguments[0]], lanes)
                                create("scf.yield", [value])
                            current = loop.results[0]
                        tail = extents[-1] - end
                        if tail:
                            current = packet(current, [*indices, c(end)], tail)
                        return current
                    loop = create("scf.for", [c(0), c(extents[depth]), c(1), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = nest(depth + 1, block.arguments[1], [*indices, block.arguments[0]])
                        create("scf.yield", [value])
                    return loop.results[0]
                def broadcast_group(depth, current, coordinates, width):
                    order = [d for d in range(len(extents)) if d != axis]
                    if depth == len(order):
                        return packet(current, [coordinates[d] for d in range(len(extents))], width)
                    d = order[depth]
                    loop = create("scf.for", [c(0), c(extents[d]), c(1), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = broadcast_group(depth + 1, block.arguments[1],
                                                {**coordinates, d: block.arguments[0]}, width)
                        create("scf.yield", [value])
                    return loop.results[0]

                if broadcast:
                    current = old.operands[-1]
                    end = extents[axis] // lanes * lanes
                    loop = create("scf.for", [c(0), c(end), c(lanes), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = broadcast_group(0, block.arguments[1], {axis: block.arguments[0]}, lanes)
                        create("scf.yield", [value])
                    result = loop.results[0]
                    tail = extents[axis] - end
                    if tail:
                        result = broadcast_group(0, result, {axis: c(end)}, tail)
                else:
                    result = nest(0, old.operands[-1], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _pp_ir.StringAttr(previous).value + "," if previous is not None else ""
                transform = "scalar_pointwise_broadcast_packet_" if broadcast else "scalar_pointwise_packet_"
                replacement.attributes["prov.transforms"] = _pp_ir.StringAttr.get(text + transform + str(lanes))
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


from pathlib import Path
import json,hashlib
from torch_mlir import ir
W=Path(__file__).resolve().parent
ctx=ir.Context()
with ctx,ir.Location.unknown():
 source=ir.Module.parse(Path(SOURCE).read_text())
 candidates=[]
 def walk(operation):
  for region in operation.regions:
   for block in region.blocks:
    for op in block.operations:
     spec=_pointwise_packet_spec(op)
     if spec is not None and str(op.results[0].type)=='tensor<1x8x5632xi8>':
      candidates.append((op,spec))
     walk(op.operation)
 walk(source.operation)
 assert len(candidates)==22,len(candidates)
 original,spec=candidates[0]
 out=ir.Module.create()
 inputs=list(original.operands[:-1]);ot=original.results[0].type
 with ir.InsertionPoint(out.body):
  function=ir.Operation.create('func.func',attributes={
   'sym_name':ir.StringAttr.get('forward'),
   'function_type':ir.TypeAttr.get(ir.FunctionType.get([v.type for v in inputs],[ot]))},regions=1)
  block=function.regions[0].blocks.append(*[v.type for v in inputs])
 with ir.InsertionPoint(block):
  mapping=dict(zip(inputs,block.arguments))
  constants={}
  def resolve(value):
   if value in mapping:return mapping[value]
   owner=getattr(value.owner,'operation',value.owner)
   assert isinstance(owner,ir.Operation) and owner.name=='arith.constant',str(owner)
   assert len(owner.results)==1 and not owner.operands and not owner.regions
   if value not in constants:
    constants[value]=ir.Operation.create('arith.constant',results=[value.type],
      attributes={a:owner.attributes[a] for a in owner.attributes}).results[0]
   return constants[value]
  # Resolve exact scalar captures in the function block before the generic.
  scalar=list(original.regions[0].blocks[0].operations)
  original_args=set(original.regions[0].blocks[0].arguments)
  local=set(original_args)
  for operation in scalar:
   for value in operation.operands:
    if value not in local:resolve(value)
   local.update(operation.results)
  empty=ir.Operation.create('tensor.empty',results=[ot])
  generic=ir.Operation.create('linalg.generic',results=[ot],
    operands=[*block.arguments,empty.results[0]],
    attributes={a:original.attributes[a] for a in original.attributes},regions=1)
  inner=generic.regions[0].blocks.append(*[v.type for v in original.regions[0].blocks[0].arguments])
  imap=dict(zip(original.regions[0].blocks[0].arguments,inner.arguments))
  with ir.InsertionPoint(inner):
   for operation in scalar:
    operands=[imap[v] if v in imap else resolve(v) for v in operation.operands]
    clone=ir.Operation.create(operation.operation.name,operands=operands,
      results=[v.type for v in operation.results],
      attributes={a:operation.attributes[a] for a in operation.attributes})
    imap.update(zip(operation.results,clone.results))
  ir.Operation.create('func.return',operands=[generic.results[0]])
 assert out.operation.verify()
 (W/'source_capsule.mlir').write_text(str(out)+'\n')
 record={'source':SOURCE,'source_sha256':hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(),
  'matched_original_source_bodies':len(candidates),'selected_source_ordinal':0,
  'input_types':[str(v.type) for v in inputs],
  'indexing_maps':str(original.attributes['indexing_maps']),
  'exact_original_body':str(original.regions[0]),
  'source_capsule_sha256':hashlib.sha256((W/'source_capsule.mlir').read_bytes()).hexdigest(),
  'original_scalar_operations':len(scalar),'external_scalar_constants':len(constants),
  'selection_scope':'Experiment selects one actual source instance; production packet pass uses legality and broadcast maps only.'}
 (W/'source_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k!='exact_original_body'}),flush=True)
