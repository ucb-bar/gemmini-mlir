"""Keep source select reductions in ordered scalar SSA before bufferization.

This lowers storage, not arithmetic: every scalar operation, comparison predicate,
operand order and attribute is cloned unchanged. Affine window accesses and all
iteration bounds are checked on the current typed source. No non-NaN, reassociation,
floating-environment, pointer alias or physical storage permission is supplied.
"""
from __future__ import annotations

import hashlib

from xdsl.dialects import affine, arith, builtin, scf, tensor
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import (
    AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr,
    AffineMap,
)

FEATURE = "scalar_window_reductions"


def _linear(expression, rank):
    """Return constant plus coefficients, refusing non-linear/division maps."""
    if isinstance(expression, AffineConstantExpr):
        return expression.value, [0] * rank
    if isinstance(expression, AffineDimExpr) and 0 <= expression.position < rank:
        coefficients = [0] * rank
        coefficients[expression.position] = 1
        return 0, coefficients
    if not isinstance(expression, AffineBinaryOpExpr):
        return None
    left, right = _linear(expression.lhs, rank), _linear(expression.rhs, rank)
    if left is None or right is None:
        return None
    if expression.kind == AffineBinaryOpKind.Add:
        return left[0] + right[0], [a + b for a, b in zip(left[1], right[1])]
    if expression.kind == AffineBinaryOpKind.Mul:
        if not any(left[1]):
            return left[0] * right[0], [left[0] * c for c in right[1]]
        if not any(right[1]):
            return right[0] * left[0], [right[0] * c for c in left[1]]
    return None


def _spec(op):
    if not isinstance(op, GenericOp) or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    if op.results[0].type != op.outputs[0].type:
        return None
    ancestor = op
    while ancestor is not None:
        if any("strictfp" in name or "fenv" in name or "strictfp" in str(value) or "fenv" in str(value)
               for name, value in ancestor.attributes.items()):
            return None
        ancestor = ancestor.parent_op()
    # Only normal source generics; no library/effect/distribution override.
    if op.attributes and any(not name.startswith("prov.") for name in op.attributes):
        return None
    if any(name not in {"indexing_maps", "iterator_types", "operandSegmentSizes"}
           for name in op.properties):
        return None
    kinds = [iterator.data.value for iterator in op.iterator_types]
    parallel = kinds.count("parallel")
    if not parallel or kinds != ["parallel"] * parallel + ["reduction"] * (len(kinds) - parallel):
        return None
    if parallel == len(kinds):
        return None
    rank = len(kinds)
    maps = [attribute.data for attribute in op.indexing_maps]
    if maps[-1] != AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(parallel))):
        return None
    values = list(op.operands)
    if any(not isinstance(value.type, builtin.TensorType)
           or not isinstance(value.type.encoding, builtin.NoneAttr)
           or any(extent <= 0 for extent in value.type.get_shape()) for value in values):
        return None
    try:
        bounds = tuple(op.get_static_loop_ranges())
    except (ValueError, NotImplementedError, AssertionError):
        return None
    if len(bounds) != rank or any(type(bound) is not int or bound <= 0 for bound in bounds):
        return None
    for value, mapping in zip(values, maps, strict=True):
        if mapping.num_dims != rank or mapping.num_symbols or len(mapping.results) != len(value.type.get_shape()):
            return None
        for expression, extent in zip(mapping.results, value.type.get_shape(), strict=True):
            linear = _linear(expression, rank)
            if linear is None:
                return None
            constant, coefficients = linear
            low = constant + sum(min(0, c * (bound - 1)) for c, bound in zip(coefficients, bounds))
            high = constant + sum(max(0, c * (bound - 1)) for c, bound in zip(coefficients, bounds))
            if low < 0 or high >= extent:
                return None
    body = op.body.block
    scalars = tuple(body.ops)
    if not scalars or scalars[-1].name != "linalg.yield" or len(scalars[-1].operands) != 1:
        return None
    if scalars[-1].properties or any(not name.startswith("prov.") for name in scalars[-1].attributes):
        return None
    # These select recurrences retain source NaN payloads and signed zeros.
    # Deliberately do not admit an arithmetic reduction under a new domain proof.
    allowed = {"arith.constant", "arith.cmpf", "arith.cmpi", "arith.andi", "arith.ori", "arith.select"}
    if not any(scalar.name == "arith.select" for scalar in scalars[:-1]):
        return None
    local = set(body.args)
    for scalar in scalars[:-1]:
        if scalar.name not in allowed or scalar.regions or any(operand not in local for operand in scalar.operands):
            return None
        if any(not name.startswith("prov.") for name in scalar.attributes):
            return None
        if any(name not in {"value", "predicate", "fastmath"} for name in scalar.properties):
            return None
        if "fastmath" in scalar.properties and scalar.properties["fastmath"].data:
            return None
        local.update(scalar.results)
    if scalars[-1].operands[0] not in local or not body.args[-1].uses:
        return None
    return parallel, bounds, maps


def lower_window_reductions(module):
    """Mutate verified private tensor IR; return source-derived realized regions."""
    module.verify()
    rows = []
    for op in tuple(module.walk()):
        spec = _spec(op)
        if spec is None:
            continue
        parallel, bounds, maps = spec
        prefix = []
        def constant(value):
            result = arith.ConstantOp.from_int_and_width(value, builtin.IndexType())
            prefix.append(result)
            return result.result
        zero, one = constant(0), constant(1)
        limits = [constant(bound) for bound in bounds]
        scalar_type = op.outputs[0].type.get_element_type()

        def reduction(depth, coordinates, current):
            if depth < len(bounds):
                block = Block(arg_types=[builtin.IndexType(), scalar_type])
                loop = scf.ForOp(zero, limits[depth], one, [current], Region(block))
                child_ops, result = reduction(depth + 1, [*coordinates, block.args[0]], block.args[1])
                block.add_ops([*child_ops, scf.YieldOp(result)])
                return [loop], loop.results[0]
            operations, mapper = [], {op.body.block.args[-1]: current}
            for value, mapping, argument in zip(op.inputs, maps[:-1], op.body.block.args[:-1], strict=True):
                if not argument.uses:
                    continue
                indices = []
                for expression in mapping.results:
                    apply = affine.ApplyOp(coordinates, builtin.AffineMapAttr(AffineMap(len(bounds), 0, (expression,))))
                    operations.append(apply)
                    indices.append(apply.result)
                load = tensor.ExtractOp(value, indices, argument.type)
                operations.append(load)
                mapper[argument] = load.result
            for original in tuple(op.body.block.ops)[:-1]:
                operations.append(original.clone(value_mapper=mapper))
            return operations, mapper[op.body.block.last_op.operands[0]]

        def output(depth, coordinates, current):
            if depth < parallel:
                block = Block(arg_types=[builtin.IndexType(), current.type])
                loop = scf.ForOp(zero, limits[depth], one, [current], Region(block))
                child_ops, result = output(depth + 1, [*coordinates, block.args[0]], block.args[1])
                block.add_ops([*child_ops, scf.YieldOp(result)])
                return [loop], loop.results[0]
            seed = tensor.ExtractOp(current, coordinates, scalar_type)
            operations, result = reduction(parallel, coordinates, seed.result)
            store = tensor.InsertOp(result, current, coordinates)
            return [seed, *operations, store], store.result

        operations, result = output(0, [], op.outputs[0])
        operations[0].attributes.update(op.attributes)
        op.parent.insert_ops_before([*prefix, *operations], op)
        op.results[0].replace_all_uses_with(result)
        op.parent.erase_op(op)
        rows.append({"parallel_axes": parallel, "reduction_axes": len(bounds) - parallel,
                     "bounds": list(bounds), "scalar_body": "cloned_unchanged",
                     "axis_order": "source_lexicographic", "seed": "source_output_tensor"})
    module.verify()
    return rows


def prepare_scalar_window_reductions(source, *, enabled=False):
    """Source-bound opt-in transport shared by ordinary and adapter lowering."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    if type(enabled) is not bool:
        raise ValueError("window reduction selection requires a Boolean")
    digest = hashlib.sha256(source.encode()).hexdigest()
    report = dict(schema="merlin.scalar_window_reductions.v1", rewrites=0, regions=[],
                  source_sha256=digest, output_sha256=digest, hardware_cycles="not_measured")
    if not enabled:
        return source, report
    module = parse_mlir_text(source)
    rows = lower_window_reductions(module)
    selected = text(module, generic=True) if rows else source
    report.update(rewrites=len(rows), regions=rows, output_sha256=hashlib.sha256(selected.encode()).hexdigest())
    return selected, report
