"""Private host-segment fusion, preserving original source SSA ownership.

The shared live float-to-integer pass is used unchanged.  The extra private
chain rule uses its same pure pointwise admission, and only joins identity
bridges over equal static domains.  This is the no-extra-work condition of
``fusion_guard``: broadcasting other inputs never repeats a producer body.
No scalar expression is simplified, reassociated, or duplicated.
"""
from __future__ import annotations

import hashlib
from math import prod

from xdsl.dialects import builtin, func
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from merlin.llvmlower.host_pointwise_fusion import (
    _pointwise, _transparent, fuse_live_pointwise,
)
from merlin.llvmlower.automatic_rounding import _container
from merlin.xdsl_dialects._common import text


def _fuse_private_chains(module, *, max_scalar_operations=512, max_fusions=4096):
    """Inline a sole-use f32 producer once into an equal-domain consumer."""
    rows = []
    # Restart after each mutation so a newly fused chain can itself be consumed.
    while len(rows) < max_fusions:
        changed = False
        for consumer in tuple(module.walk()):
            if not _pointwise(consumer):
                continue
            rank = len(consumer.iterator_types)
            identity = AffineMap.identity(rank)
            if not rank or consumer.indexing_maps.data[-1].data != identity:
                continue
            for index, source in enumerate(consumer.inputs):
                producer = source.owner
                if (not isinstance(producer, GenericOp) or not _pointwise(producer)
                        or source.type.get_element_type() != builtin.f32
                        or producer.parent is not consumer.parent
                        or len(producer.iterator_types) != rank
                        or producer.indexing_maps.data[-1].data != identity
                        or consumer.indexing_maps.data[index].data != identity
                        or source.type.get_shape() != consumer.results[0].type.get_shape()
                        or len(tuple(source.uses)) != 1
                        or len(tuple(producer.body.block.ops)) + len(tuple(consumer.body.block.ops)) > max_scalar_operations):
                    continue
                block = consumer.parent
                operations = tuple(block.ops)
                start, end = operations.index(producer), operations.index(consumer)
                if start >= end or any(not _transparent(op) for op in operations[start+1:end]):
                    continue
                inputs = [*producer.inputs, *(v for i, v in enumerate(consumer.inputs) if i != index)]
                outputs = list(consumer.outputs)
                body = Block(arg_types=[v.type.get_element_type() for v in (*inputs, *outputs)])
                values = dict(zip(producer.body.block.args[:-1], body.args[:len(producer.inputs)], strict=True))
                # Admission proves the producer's destination argument unused.
                for scalar in tuple(producer.body.block.ops)[:-1]:
                    body.add_op(scalar.clone(value_mapper=values))
                yielded = producer.body.block.last_op.operands[0]
                floating = values.get(yielded, yielded)
                cursor = len(producer.inputs)
                for i, argument in enumerate(consumer.body.block.args[:-1]):
                    values[argument] = floating if i == index else body.args[cursor]
                    cursor += i != index
                values[consumer.body.block.args[-1]] = body.args[-1]
                for scalar in tuple(consumer.body.block.ops)[:-1]:
                    body.add_op(scalar.clone(value_mapper=values))
                yielded = consumer.body.block.last_op.operands[0]
                body.add_op(YieldOp(values.get(yielded, yielded)))
                maps = [*producer.indexing_maps.data[:-1], *(
                    m for i, m in enumerate(consumer.indexing_maps.data[:-1]) if i != index
                ), consumer.indexing_maps.data[-1]]
                fused = GenericOp(inputs, outputs, Region(body), maps, consumer.iterator_types,
                                  result_types=[v.type for v in outputs])
                fused.attributes.update(consumer.attributes)
                fused.attributes['prov.fused_producer'] = builtin.DictionaryAttr(dict(producer.attributes))
                fused.attributes['prov.host_pointwise_fused'] = builtin.UnitAttr()
                block.insert_op_before(fused, consumer)
                consumer.results[0].replace_all_uses_with(fused.results[0])
                block.erase_op(consumer)
                block.erase_op(producer)
                shape = source.type.get_shape()
                rows.append({'kind': 'private_chain', 'shape': list(shape),
                             'shared_elements': prod(shape), 'eliminated_read_bytes': 4*prod(shape),
                             'eliminated_write_bytes': 4*prod(shape)})
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    module.verify()
    return rows


def prepare_segment(ops, outputs):
    """Clone a closed segment and retain all actual external observations.

    Return the cloned executable operations, external argument bindings,
    original/live cloned result pairs, and a JSON-safe transformation receipt.
    The caller lowers the clone and publishes results under original SSA keys.
    Original operations and all of their users remain entirely untouched.
    """
    nested = {inner for op in ops for inner in op.walk()}
    defined = {v for op in nested for v in op.results}
    defined.update(arg for op in nested for region in op.regions for block in region.blocks for arg in block.args)
    external = list(dict.fromkeys(v for op in ops for inner in op.walk()
                                  for v in inner.operands if v not in defined))
    observed = {v for v, _ in outputs}
    live = [v for op in ops for v in op.results
            if v in observed or any(use.operation not in nested for use in v.uses)]
    block = Block(arg_types=[v.type for v in external])
    values = dict(zip(external, block.args, strict=True))
    for index, op in enumerate(ops):
        clone = op.clone(value_mapper=values)
        if isinstance(clone, GenericOp):
            clone.attributes['prov.host_segment_index'] = builtin.IntegerAttr(index, builtin.i64)
        block.add_op(clone)
    returned = func.ReturnOp(*(values[v] for v in live))
    block.add_op(returned)
    module = builtin.ModuleOp([func.FuncOp('host_segment',
        ([v.type for v in external], [v.type for v in live]), Region(block))])
    module.verify()
    before = text(module, generic=True)
    # A synthetic function must not erase an original enclosing restriction.
    # Recheck the shared source/container contract before admitting any rewrite.
    admitted = all(_container(op) for op in ops if isinstance(op, GenericOp))
    rows = _fuse_private_chains(module) if admitted else []
    live_rows = fuse_live_pointwise(module) if admitted else []
    for op in module.walk():
        if isinstance(op, GenericOp) and 'prov.fused_producer' in op.attributes:
            op.attributes['prov.host_pointwise_fused'] = builtin.UnitAttr()
    rows.extend(dict(row, kind='live_float_integer') for row in live_rows)
    after = text(module, generic=True)
    report = {'schema': 'merlin.gemmini_host_pointwise_fusion.v1', 'rewrites': len(rows),
              'segment_op_count': len(ops), 'regions': rows,
              'source_segment_sha256': hashlib.sha256(before.encode()).hexdigest(),
              'output_segment_sha256': hashlib.sha256(after.encode()).hexdigest()}
    return (list(block.ops)[:-1], list(zip(external, block.args, strict=True)),
            list(zip(live, returned.operands, strict=True)), report)
