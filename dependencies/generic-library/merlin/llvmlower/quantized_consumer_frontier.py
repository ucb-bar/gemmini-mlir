"""Retain the typed source observations after a floating tensor producer.

This closure analysis grants no approximation or numeric certificate. A provider
must prove every retained integer and floating observation, validate the source
witness, and preserve the source conversion, extrema, scale and rounding DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ordered_fma_groups import (
    _context_snapshot,
    _snapshot,
    _source_pure_body,
    _supported,
)


@dataclass(frozen=True)
class QuantizedConsumerFrontier:
    source_values: tuple
    integer_output: object
    operations: tuple
    observation_outputs: tuple
    floating_escapes: tuple
    unquantized_source_escapes: tuple
    assembly_boxes: tuple
    _witness: tuple = field(repr=False)
    _contexts: tuple = field(repr=False)
    _source_types: tuple = field(repr=False)
    _source_uses: tuple = field(repr=False)

    @property
    def source_uses_closed(self) -> bool:
        """Necessary typed-use condition; not a numerical admission decision."""
        return not self.unquantized_source_escapes


def _strict(operation) -> bool:
    return any(
        "strictfp" in name or "strictfp" in str(value)
        for name, value in (*operation.attributes.items(), *operation.properties.items())
    )


def _allowed(operation) -> bool:
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg.ops import TransposeOp

    if _strict(operation):
        return False
    if isinstance(operation, arith.ConstantOp):
        return True
    if isinstance(operation, (tensor.EmptyOp, tensor.SplatOp)):
        return all(
            isinstance(value.type, TensorType) and all(n > 0 for n in value.type.get_shape())
            for value in operation.results
        )
    if isinstance(operation, tensor.InsertSliceOp):
        return True  # Complete static ownership is checked as a whole chain.
    if isinstance(operation, TransposeOp):
        if not _source_pure_body(operation):
            return False
        source, destination = operation.operands
        if not isinstance(source.type, TensorType) or not isinstance(destination.type, TensorType):
            return False
        permutation = tuple(operation.permutation.get_values())
        shape = source.type.get_shape()
        return (
            sorted(permutation) == list(range(len(shape)))
            and tuple(shape[index] for index in permutation) == destination.type.get_shape()
            and source.type.element_type == destination.type.element_type
            and all(n > 0 for n in shape)
        )
    return _supported(operation)


def _complete_assembly(operation, sources) -> tuple:
    """Prove all carrier elements written exactly once by static source slices."""
    import math

    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import TensorType

    chain = []
    current = operation
    while isinstance(current, tensor.InsertSliceOp):
        chain.append(current)
        current = current.operands[1].owner
    if not isinstance(current, tensor.EmptyOp):
        raise ValueError("consumer assembly requires fresh empty carrier")
    carrier = operation.results[0].type
    if not isinstance(carrier, TensorType):
        raise ValueError("consumer assembly requires a tensor carrier")
    shape = carrier.get_shape()
    boxes = []
    for insert in reversed(chain):
        if len(insert.operands) != 2 or insert.operands[0] not in sources:
            raise ValueError("consumer assembly has a dynamic or unbound slice")
        if insert.operands[1].type != carrier or insert.results[0].type != carrier:
            raise ValueError("consumer assembly carrier type changed")
        offsets = tuple(insert.static_offsets.get_values())
        sizes = tuple(insert.static_sizes.get_values())
        strides = tuple(insert.static_strides.get_values())
        if (
            len(offsets) != len(shape)
            or len(sizes) != len(shape)
            or strides != (1,) * len(shape)
            or insert.operands[0].type != TensorType(carrier.element_type, sizes)
            or any(o < 0 or n <= 0 or o + n > bound for o, n, bound in zip(offsets, sizes, shape, strict=True))
        ):
            raise ValueError("consumer assembly lacks exact static coordinates")
        boxes.append((insert.operands[0], offsets, sizes))
    for index, (_, offsets, sizes) in enumerate(boxes):
        for _, previous_offsets, previous_sizes in boxes[:index]:
            if all(
                o < po + pn and po < o + n
                for o, n, po, pn in zip(offsets, sizes, previous_offsets, previous_sizes, strict=True)
            ):
                raise ValueError("consumer assembly slices overlap")
    if sum(math.prod(sizes) for _, _, sizes in boxes) != math.prod(shape):
        raise ValueError("consumer assembly does not write every carrier element")
    return tuple(boxes)


def analyze_quantized_consumer_frontier(integer_output, *, source_values) -> QuantizedConsumerFrontier:
    """Close an actual integer observation back to explicit floating producers.

    Static views, pure typed pointwise bodies and registered reductions retain
    all original arithmetic and coordinate maps. Other live observations remain
    explicit, including floating scales and any residual/unquantized escape.
    No operation is moved, cloned, erased or selected by a source label.
    """
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import AnyFloat, IntegerType, TensorType
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir import OpResult

    sources = tuple(source_values)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("distinct floating source values required")
    if (
        not isinstance(integer_output, OpResult)
        or not isinstance(integer_output.type, TensorType)
        or not isinstance(integer_output.type.element_type, IntegerType)
    ):
        raise ValueError("integer tensor observation required")
    block = integer_output.owner.parent
    if block is None or block.parent_op().name != "func.func":
        raise ValueError("single source function block required")
    for value in sources:
        if (
            not isinstance(value.type, TensorType)
            or not isinstance(value.type.element_type, AnyFloat)
            or any(n <= 0 for n in value.type.get_shape())
            or not isinstance(value, OpResult)
            or value.owner.parent is not block
        ):
            raise ValueError("static floating producer in the same block required")
    positions = {operation: index for index, operation in enumerate(block.ops)}
    needed, reached = set(), set()

    def need(value):
        if value in sources:
            reached.add(value)
            return
        if not isinstance(value, OpResult) or value.owner.parent is not block:
            raise ValueError("consumer has an unbound or cross-block input")
        operation = value.owner
        if operation in needed:
            return
        if not _allowed(operation):
            raise ValueError("consumer has unsupported numeric or effect semantics")
        needed.add(operation)
        for index, operand in enumerate(operation.operands):
            # Fully parallel, pure writers need no unread output initializer.
            if (
                isinstance(operation, GenericOp)
                and index >= len(operation.inputs)
                and not tuple(operation.body.block.args[index].uses)
                and isinstance(operand.owner, tensor.EmptyOp)
            ):
                needed.add(operand.owner)
                continue
            need(operand)

    need(integer_output)
    if reached != set(sources):
        raise ValueError("integer observation does not depend on every source value")
    operations = tuple(sorted(needed, key=positions.__getitem__))
    assemblies = tuple(operation for operation in operations if isinstance(operation, tensor.InsertSliceOp))
    # Intermediate insertions may be incomplete; only complete terminal carriers
    # are read by numerical/view operations or leave this chain.
    boxes = []
    for operation in assemblies:
        uses = tuple(operation.results[0].uses)
        if any(not isinstance(use.operation, tensor.InsertSliceOp) or use.index != 1 for use in uses):
            boxes.extend(_complete_assembly(operation, sources))
    descendants = {child for operation in operations for child in operation.walk()}
    memo = {}

    def depends(value):
        if value in sources:
            return True
        if value in memo:
            return memo[value]
        memo[value] = False
        if isinstance(value, OpResult) and value.owner in needed:
            memo[value] = any(depends(operand) for operand in value.owner.operands)
        return memo[value]

    def views_only(value):
        if value in sources:
            return True
        if not isinstance(value, OpResult) or value.owner not in needed:
            return False
        operation = value.owner
        if isinstance(operation, GenericOp):
            scalars = tuple(operation.body.block.ops)
            if len(scalars) == 1 and scalars[0].name == "linalg.yield":
                returned = scalars[0].operands[0]
                for index, argument in enumerate(operation.body.block.args[: len(operation.inputs)]):
                    if returned is argument:
                        return views_only(operation.inputs[index])
        return (
            isinstance(
                operation,
                (
                    tensor.InsertSliceOp,
                    tensor.ExtractSliceOp,
                    tensor.ExpandShapeOp,
                    tensor.CollapseShapeOp,
                    tensor.CastOp,
                ),
            )
            or operation.name == "linalg.transpose"
        ) and any(views_only(operand) for operand in operation.operands)

    escapes = []
    for value in (*sources, *(result for operation in operations for result in operation.results)):
        if value is integer_output or not depends(value):
            continue
        if any(use.operation not in descendants for use in value.uses):
            escapes.append(value)
    outputs = (integer_output, *escapes)
    floating = tuple(
        value
        for value in escapes
        if isinstance(value.type, TensorType) and isinstance(value.type.element_type, AnyFloat)
    )
    contexts = []
    current = block.parent_op()
    while current is not None:
        if _strict(current):
            raise ValueError("consumer enclosing strictfp requires separate effect proof")
        contexts.append(_context_snapshot(current))
        current = current.parent_op()
    witnesses = tuple(
        _snapshot(child)
        for operation in (*dict.fromkeys(value.owner for value in sources), *operations)
        for child in operation.walk()
    )
    return QuantizedConsumerFrontier(
        sources,
        integer_output,
        operations,
        outputs,
        floating,
        tuple(value for value in escapes if views_only(value)),
        tuple(boxes),
        witnesses,
        tuple(contexts),
        tuple(value.type for value in sources),
        tuple(frozenset((use.operation, use.index) for use in value.uses) for value in sources),
    )


def validate_quantized_consumer_frontier(frontier: QuantizedConsumerFrontier) -> None:
    """Refuse changed uses, operation bodies, coordinates, types or context."""
    if (
        tuple(value.type for value in frontier.source_values) != frontier._source_types
        or tuple(frozenset((use.operation, use.index) for use in value.uses) for value in frontier.source_values)
        != frontier._source_uses
        or any(_snapshot(w.operation) != w for w in frontier._witness)
        or any(_context_snapshot(c.operation) != c for c in frontier._contexts)
    ):
        raise ValueError("quantized consumer source changed after analysis")


def quantized_consumer_semantic_sha256(frontier: QuantizedConsumerFrontier) -> str:
    """Fingerprint the complete live typed DAG and every observation, no clone.

    Local SSA references include scalar block arguments and operand order. Only
    provenance is omitted; numerical attributes/properties, coordinates, source
    types and each live observation remain. Ancestor numeric context is retained
    independently by the validated source witness.
    """
    import hashlib
    import json

    validate_quantized_consumer_frontier(frontier)
    values = {value: ["source", index] for index, value in enumerate(frontier.source_values)}
    next_value = 0

    def reference(value):
        if value not in values:
            raise ValueError("consumer fingerprint has an unbound source value")
        return values[value]

    def attribute_items(items):
        return [[name, str(value)] for name, value in sorted(items) if not name.startswith("prov.")]

    def describe(operation):
        nonlocal next_value
        operands = [reference(value) for value in operation.operands]
        results = []
        for value in operation.results:
            values[value] = ["value", next_value]
            next_value += 1
            results.append(dict(reference=reference(value), type=str(value.type)))
        regions = []
        for region in operation.regions:
            blocks = []
            for block in region.blocks:
                arguments = []
                for value in block.args:
                    values[value] = ["value", next_value]
                    next_value += 1
                    arguments.append(dict(reference=reference(value), type=str(value.type)))
                blocks.append(dict(arguments=arguments, operations=[describe(child) for child in block.ops]))
            regions.append(blocks)
        return dict(
            operation=operation.name,
            operands=operands,
            results=results,
            attributes=attribute_items(operation.attributes.items()),
            properties=attribute_items(operation.properties.items()),
            regions=regions,
        )

    body = [describe(operation) for operation in frontier.operations]
    record = dict(
        source_types=[str(value.type) for value in frontier.source_values],
        operations=body,
        observations=[reference(value) for value in frontier.observation_outputs],
        unquantized_source_escapes=[reference(value) for value in frontier.unquantized_source_escapes],
    )
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def find_quantized_consumer_frontiers(module, *, source_values) -> tuple[QuantizedConsumerFrontier, ...]:
    """Discover source observations by live dependencies, never producer order.

    This read-only query requires explicit source producer values. Unsupported
    candidate paths remain unselected; each returned frontier retains any extra
    source escape for the caller to refuse. Integer observations downstream of
    unknown calls or additional contractions are not inferred to be equivalent.
    """
    from xdsl.dialects.builtin import IntegerType, TensorType
    from xdsl.ir import OpResult

    sources = tuple(source_values)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("distinct explicit source values required")
    source_set = set(sources)
    memo = {}

    def origins(value):
        if value in source_set:
            return frozenset((value,))
        if value in memo:
            return memo[value]
        memo[value] = frozenset()
        if isinstance(value, OpResult) and _allowed(value.owner):
            memo[value] = frozenset().union(*(origins(operand) for operand in value.owner.operands))
        return memo[value]

    found = []
    for operation in module.walk():
        for result in operation.results:
            if not isinstance(result.type, TensorType) or not isinstance(result.type.element_type, IntegerType):
                continue
            dependencies = origins(result)
            if not dependencies:
                continue
            try:
                frontier = analyze_quantized_consumer_frontier(
                    result, source_values=tuple(value for value in sources if value in dependencies)
                )
            except ValueError:
                continue
            found.append(frontier)
    return tuple(found)
