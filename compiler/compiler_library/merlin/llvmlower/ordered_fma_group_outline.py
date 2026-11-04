"""Partition a closed live source group without changing its arithmetic.

This explicit preparation utility moves original operations into a function.
It proves its tensor frontier and call placement; it grants no approximation,
provider ABI, numeric certificate, target route or physical no-alias permission.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ordered_fma_groups import OrderedFMAGroup, validate_group_source


@dataclass(frozen=True)
class OutlinedOrderedFMAGroup:
    function: object
    call: object
    inputs: tuple
    outputs: tuple
    source_operations: tuple
    omitted_unread_initializers: tuple
    internalized_constants: tuple


def read_group_inputs(group: OrderedFMAGroup) -> tuple:
    """Find actual external tensor reads using live registered source bodies.

    Only unused output-init block arguments of proved pure all-parallel generics
    can be omitted. Reduction seeds and every other tensor operand remain reads.
    """
    from xdsl.dialects.linalg import ops as linalg

    validate_group_source(group)
    external = set(group.inputs)
    read = set()
    for operation in group.operations:
        for index, value in enumerate(operation.operands):
            if value not in external:
                continue
            if (
                isinstance(operation, linalg.GenericOp)
                and index >= len(operation.inputs)
                and not tuple(operation.body.block.args[index].uses)
            ):
                continue
            read.add(value)
    return tuple(value for value in group.inputs if value in read)


def outline_ordered_fma_group(module, group: OrderedFMAGroup, symbol: str) -> OutlinedOrderedFMAGroup:
    """Move one proved closed group into an ordinary source-exact function.

    Every retained operation is moved, never cloned. Unread pointwise output
    initializers become fresh local tensor.empty values. Exact scalar constant
    tensor seeds are materialized locally with their source attributes. All casts,
    reductions, coefficients, operand order and source function attributes remain.
    The regular upstream pipeline owns any resulting tensor bufferization.
    """
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region

    module.verify()
    validate_group_source(group)
    if not group.closed_bf16_endpoints:
        raise ValueError("source group must have closed BF16 endpoints")
    if not symbol or any(op.name == "func.func" and op.sym_name.data == symbol for op in module.body.block.ops):
        raise ValueError("distinct nonempty group function symbol required")
    block = group.operations[0].parent
    if block is None or any(operation.parent is not block for operation in group.operations):
        raise ValueError("group operations must share one source block")
    owner = block.parent_op()
    if not isinstance(owner, func.FuncOp) or len(owner.body.blocks) != 1:
        raise ValueError("single-block source function required")
    positions = {operation: index for index, operation in enumerate(block.ops)}
    operations = set(group.operations)
    descendants = {child for operation in group.operations for child in operation.walk()}
    anchor = group.operations[-1]
    endpoint_set = set(group.bf16_outputs)
    read_inputs = read_group_inputs(group)
    constants = {}
    for value in read_inputs:
        seed = value.owner
        if not isinstance(seed, (tensor.SplatOp, linalg.FillOp)):
            continue
        constant = seed.operands[0].owner
        if (
            isinstance(constant, arith.ConstantOp)
            and len(seed.operands) == (1 if isinstance(seed, tensor.SplatOp) else 2)
            and value.type.element_type == constant.result.type
        ):
            constants[value] = constant
    inputs = tuple(value for value in read_inputs if value not in constants)
    all_external = set(group.inputs)
    for operation in group.operations:
        if any(value.owner not in operations and value not in all_external for value in operation.operands):
            raise ValueError("unsupported non-tensor or unbound group input")
        for value in operation.results:
            for use in value.uses:
                if use.operation in descendants:
                    continue
                if value not in endpoint_set or use.operation.parent is not block:
                    raise ValueError("group has a non-endpoint or cross-block escape")
                if positions[use.operation] <= positions[anchor]:
                    raise ValueError("endpoint use precedes the source group call placement")
    for value in inputs:
        if isinstance(value.owner, Block):
            if value.owner is not block:
                raise ValueError("group input block argument does not dominate the source call")
        elif value.owner.parent is not block or positions[value.owner] > positions[anchor]:
            raise ValueError("group input producer does not dominate the source call")
    if any(not isinstance(value.type, TensorType) for value in (*inputs, *group.bf16_outputs)):
        raise ValueError("source group boundary requires tensor values")

    # All selection and domination checks above precede any source mutation.
    body = Block(arg_types=[value.type for value in inputs])
    mapping = dict(zip(inputs, body.args, strict=True))
    for value, constant in constants.items():
        scalar = constant.clone()
        filled = tensor.SplatOp(scalar.result, [], value.type)
        filled.attributes.update(value.owner.attributes)
        body.add_ops([scalar, filled])
        mapping[value] = filled.result
    omitted = tuple(value for value in group.inputs if value not in mapping)
    for value in omitted:
        empty = tensor.EmptyOp([], value.type)
        body.add_op(empty)
        mapping[value] = empty.tensor
    call = func.CallOp(symbol, inputs, [value.type for value in group.bf16_outputs])
    block.insert_op_before(call, anchor)
    for value, replacement in zip(group.bf16_outputs, call.results, strict=True):
        for use in tuple(value.uses):
            if use.operation not in descendants:
                operands = list(use.operation.operands)
                operands[use.index] = replacement
                use.operation.operands = operands
    for operation in group.operations:
        for child in operation.walk():
            child.operands = tuple(mapping.get(value, value) for value in child.operands)
        block.detach_op(operation)
        body.add_op(operation)
    body.add_op(func.ReturnOp(*group.bf16_outputs))
    function = func.FuncOp(
        symbol,
        ([value.type for value in inputs], [value.type for value in group.bf16_outputs]),
        Region(body),
        visibility="private",
    )
    function.attributes.update(owner.attributes)
    module.body.block.add_op(function)
    module.verify()
    return OutlinedOrderedFMAGroup(
        function, call, inputs, group.bf16_outputs, group.operations, omitted, tuple(constants)
    )
