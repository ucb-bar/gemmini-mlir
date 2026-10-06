"""Read-only typed source operation/use census; IDs bind evidence, never select policy."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
from torch_mlir import ir

source = Path(sys.argv[1])
ctx = ir.Context()
module = ir.Module.parse(source.read_text(), ctx)
ops = []
def walk(op):
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                ops.append(child.operation)
                walk(child.operation)
walk(module.operation)
ids = {o: i for i, o in enumerate(ops)}

def tensor(t):
    if isinstance(t, ir.RankedTensorType):
        r = ir.RankedTensorType(t)
        return dict(shape=list(r.shape), dtype=str(r.element_type), elements=math.prod(r.shape))
    return dict(scalar=str(t))

def owner_id(v):
    return ids.get(v.owner.operation if hasattr(v.owner, 'operation') else v.owner)

def body(op):
    return [s.operation for r in op.regions for b in r.blocks for s in b.operations]

def scalar_binding(op):
    if not op.regions or not op.regions[0].blocks:
        return None
    b = op.regions[0].blocks[0]
    values = {v: ('arg', i) for i, v in enumerate(b.arguments)}
    records = []
    for s in b.operations:
        o = s.operation
        rec = dict(name=o.name, inputs=[values.get(v, ('outside', str(v.type))) for v in o.operands],
                   results=[str(v.type) for v in o.results],
                   attributes={a: str(o.attributes[a]) for a in o.attributes if not a.startswith('prov.')})
        records.append(rec)
        for i, v in enumerate(o.results):
            values[v] = ('result', len(records)-1, i)
    return records

def info(op):
    return dict(id=ids[op], name=op.name, operands=[tensor(v.type) for v in op.operands],
                producers=[owner_id(v) for v in op.operands], results=[tensor(v.type) for v in op.results],
                body=[s.name for s in body(op)], scalar_binding=scalar_binding(op),
                maps=str(op.attributes['indexing_maps']) if 'indexing_maps' in op.attributes else None,
                iterators=str(op.attributes['iterator_types']) if 'iterator_types' in op.attributes else None,
                result_uses=[[dict(op=ids[u.owner], operand=u.operand_number) for u in v.uses] for v in op.results])

def ancestry(v, depth=4):
    found = {}
    def visit(value, d):
        i = owner_id(value)
        if i is None or i in found:
            return
        found[i] = info(ops[i])
        if d:
            for operand in ops[i].operands:
                visit(operand, d-1)
    visit(v, depth)
    return list(found.values())

def use_closure(v):
    seen, todo, boundaries = set(), [v], []
    while todo:
        value = todo.pop()
        for use in value.uses:
            o = use.owner
            i = ids[o]
            if i in seen:
                continue
            seen.add(i)
            # Stop at a changed representation or an externally observed call/return.
            if o.name in {'func.call', 'func.return'} or any(tensor(r.type).get('dtype') in {'i8','i32','i64','i1'} for r in o.results):
                boundaries.append(info(o))
            else:
                todo.extend(o.results)
    return dict(ids=sorted(seen), boundaries=boundaries)

norms = []
for op in ops:
    if op.name != 'linalg.reduce' or [s.name for s in body(op)] != ['arith.addf','linalg.yield']:
        continue
    p = owner_id(op.operands[0])
    if p is None:
        continue
    square = ops[p]
    ss = body(square)
    if square.name != 'linalg.generic' or [s.name for s in ss] != ['arith.mulf','linalg.yield']:
        continue
    if len(ss[0].operands) != 2 or ss[0].operands[0] != ss[0].operands[1]:
        continue
    b = square.regions[0].blocks[0]
    if ss[0].operands[0] not in b.arguments:
        continue
    source_value = square.operands[list(b.arguments).index(ss[0].operands[0])]
    inp = tensor(source_value.type)
    if inp.get('dtype') != 'f32':
        continue
    norms.append(dict(reduction=info(op), square=info(square), input_type=inp,
                      input_producer=info(ops[owner_id(source_value)]),
                      input_uses=[dict(op=ids[u.owner], operand=u.operand_number) for u in source_value.uses],
                      input_ancestry=ancestry(source_value), output_use_closure=use_closure(op.results[0])))

# Conservatively retain every ordinary call as effectful; chase all returned
# tensor/scalar dependencies. Dead unused pre-offload float copies are not work.
live, pending = set(), []
for op in ops:
    if op.name in {'func.call', 'func.return'}:
        pending.append(op)
while pending:
    op = pending.pop()
    i = ids[op]
    if i in live:
        continue
    live.add(i)
    for v in op.operands:
        producer = owner_id(v)
        if producer is not None:
            pending.append(ops[producer])
    # Captured scalar/tensor inputs in structured regions are also dependencies.
    for scalar in body(op):
        for v in scalar.operands:
            producer = owner_id(v)
            if producer is not None and producer != i:
                pending.append(ops[producer])

families = collections.Counter()
live_families = collections.Counter()
for op in ops:
    if op.name.startswith('linalg.') and op.name not in {'linalg.yield','linalg.index'}:
        key = (op.name, tuple(str(v.type) for v in op.operands), tuple(s.name for s in body(op)))
        families[key] += 1
        if ids[op] in live:
            live_families[key] += 1
record = dict(schema='typed_host_device_coverage_source_census_v1', source_path=str(source),
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              source_mutated=False, source_operation_count=len(ops),
              operation_counts=dict(collections.Counter(o.name for o in ops)),
              families=[dict(name=k[0], operand_types=k[1], body=k[2], count=n) for k,n in families.items()],
              conservatively_live_operation_ids=sorted(live),
              live_families=[dict(name=k[0], operand_types=k[1], body=k[2], count=n) for k,n in live_families.items()],
              norms=norms,
              tensor_graph=[info(o) for o in ops if any(isinstance(v.type,ir.RankedTensorType) for v in list(o.operands)+list(o.results))],
              scope='Read-only typed SSA/type/map/use census, not a numerical/offload proof. Provenance/operation IDs identify evidence only; no workload selection. Actual current2004 lowering/source/object bindings separately qualified.',
              token_usage_available=False)
Path(sys.argv[2]).write_text(json.dumps(record, indent=2)+'\n')
print('TYPED_SOURCE_CENSUS',len(ops),len(norms),len(families),flush=True)
