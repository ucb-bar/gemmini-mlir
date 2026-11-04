"""Prove that a closed tensor observation can partition one parallel domain.

The proof preserves each row's complete scalar DAG, reduction order and other
dimensions. It supplies no numerical relaxation, storage ownership, target
callback, profitability or physical fallback permission.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .observation_boundary import ObservationBoundary, analyze_observation_boundary, validate_observation_boundary
from .prepared_operand_effects import validate_pure_tensor_function


@dataclass(frozen=True)
class RowBlock:
    offset: int
    size: int


@dataclass(frozen=True)
class RowPartitionPlan:
    rows: int
    block_rows: int
    blocks: tuple[RowBlock, ...]

    def validate(self) -> None:
        if type(self.rows) is not int or type(self.block_rows) is not int or min(self.rows, self.block_rows) <= 0:
            raise ValueError("positive static row and block dimensions required")
        expected = tuple(RowBlock(i, min(self.block_rows, self.rows - i)) for i in range(0, self.rows, self.block_rows))
        if self.blocks != expected:
            raise ValueError("partition must cover original rows once in order, including its tail")


def plan_row_partition(rows: int, block_rows: int) -> RowPartitionPlan:
    if type(rows) is not int or type(block_rows) is not int or min(rows, block_rows) <= 0:
        raise ValueError("positive static row and block dimensions required")
    plan = RowPartitionPlan(
        rows, block_rows, tuple(RowBlock(i, min(block_rows, rows - i)) for i in range(0, rows, block_rows))
    )
    plan.validate()
    return plan


@dataclass(frozen=True)
class RowValue:
    value: object
    axis: int | None
    uniform: bool = False


@dataclass(frozen=True)
class RowOperation:
    operation: object
    parallel_dimension: int | None
    reduction_dimensions: tuple[int, ...] = ()


@dataclass(frozen=True)
class ClosedRowDomain:
    boundary: ObservationBoundary
    rows: int
    input_axes: tuple[int | None, ...]
    output_axes: tuple[int, ...]
    values: tuple[RowValue, ...]
    operations: tuple[RowOperation, ...]
    _function: object = field(repr=False)
    _functions: tuple = field(repr=False)
    _selection: tuple = field(repr=False)

    @property
    def row_invariant_operations(self) -> tuple:
        return tuple(w.operation for w in self.operations if w.parallel_dimension is None)


def _shape(value):
    from xdsl.dialects.builtin import TensorType

    if not isinstance(value.type, TensorType) or any(n <= 0 for n in value.type.get_shape()):
        raise ValueError("positive static tensor shape required")
    return value.type.get_shape()


def _axis(value, axis, rows):
    shape = _shape(value)
    if type(axis) is not int or not 0 <= axis < len(shape) or shape[axis] != rows:
        raise ValueError("row axis must select exactly the original row extent")


def _contains(expr, dim):
    from xdsl.ir.affine import AffineDimExpr

    return any(isinstance(child, AffineDimExpr) and child.position == dim for child in expr.dfs())


def _generic(operation, states, rows):
    from xdsl.dialects.linalg.ops import IndexOp, IteratorType
    from xdsl.ir.affine import AffineDimExpr

    maps = tuple(attr.data for attr in operation.indexing_maps)
    if any(isinstance(scalar, IndexOp) for scalar in operation.body.block.ops):
        raise ValueError("source index observations require an independent partition-offset proof")
    if len(maps) != len(operation.operands) or any(m.num_symbols for m in maps):
        raise ValueError("complete static symbol-free operand maps required")
    # Empty destination tensors provide shapes, not uniform initialized values.
    # They are safe seeds only when the original scalar body never reads them.
    unobserved_seeds = tuple(
        index >= len(operation.inputs) and not tuple(operation.body.block.args[index].uses)
        for index in range(len(operation.operands))
    )
    selected = set()
    for value, amap in zip(operation.operands, maps, strict=True):
        state = states[value]
        if state.axis is not None:
            expr = amap.results[state.axis]
            if not isinstance(expr, AffineDimExpr):
                raise ValueError("row indexing must be an exact parallel dimension")
            selected.add(expr.position)
    if len(selected) > 1:
        raise ValueError("operand row paths disagree")
    if not selected:
        uniform = all(
            states[v].uniform or unused for v, unused in zip(operation.operands, unobserved_seeds, strict=True)
        )
        return tuple(RowValue(v, None, uniform) for v in operation.results), RowOperation(operation, None)
    dim = selected.pop()
    if tuple(operation.iterator_types)[dim].data != IteratorType.PARALLEL:
        raise ValueError("row dimension may not be reduced")
    for value, amap, unused in zip(operation.operands, maps, unobserved_seeds, strict=True):
        state = states[value]
        uses = [i for i, expr in enumerate(amap.results) if _contains(expr, dim)]
        if state.axis is not None:
            if uses != [state.axis]:
                raise ValueError("row dimension may not mix into another tensor axis")
        elif uses:
            if not state.uniform and not unused:
                raise ValueError("row invariant operand is indexed by the row dimension")
            for axis in uses:
                if amap.results[axis] != AffineDimExpr(dim):
                    raise ValueError("uniform row seed requires an exact row dimension")
                _axis(value, axis, rows)
    result_states = []
    for result, amap in zip(operation.results, maps[len(operation.inputs) :], strict=True):
        axes = [i for i, expr in enumerate(amap.results) if _contains(expr, dim)]
        if len(axes) != 1 or amap.results[axes[0]] != AffineDimExpr(dim):
            raise ValueError("every result must preserve one complete row axis")
        _axis(result, axes[0], rows)
        result_states.append(RowValue(result, axes[0]))
    reductions = tuple(i for i, kind in enumerate(operation.iterator_types) if kind.data == IteratorType.REDUCTION)
    return tuple(result_states), RowOperation(operation, dim, reductions)


def _reshape(operation, source, rows):
    from xdsl.dialects import tensor

    result = operation.results[0]
    if source.axis is None:
        return RowValue(result, None, source.uniform)
    groups = tuple(tuple(v.value.data for v in group) for group in operation.reassociation)
    if isinstance(operation, tensor.ExpandShapeOp):
        group = groups[source.axis]
        shape = _shape(result)
        candidates = [i for i in group if shape[i] == rows]
        if len(candidates) != 1 or any(shape[i] != 1 for i in group if i != candidates[0]):
            raise ValueError("row expansion may introduce unit axes only")
        axis = candidates[0]
    else:
        axis = next((i for i, group in enumerate(groups) if source.axis in group), None)
        if axis is None or any(_shape(operation.src)[i] != 1 for i in groups[axis] if i != source.axis):
            raise ValueError("row collapse may merge unit axes only")
    _axis(result, axis, rows)
    return RowValue(result, axis)


def analyze_closed_row_domain(*, inputs, outputs, input_axes, output_axes, functions=None) -> ClosedRowDomain:
    """Retain complete row observations using exact typed maps and shapes.

    ``None`` inputs are invariant with respect to the explicitly supplied row
    domain. Other axes, including heads/channels and reduction lanes, remain
    inside every row. Physical views, prepared epochs and private publication
    require separate ownership and fallback proofs before any replacement.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg.ops import GenericOp, ReduceOp

    inputs, outputs, input_axes, output_axes = map(tuple, (inputs, outputs, input_axes, output_axes))
    if len(inputs) != len(input_axes) or len(outputs) != len(output_axes) or not any(a is not None for a in input_axes):
        raise ValueError("complete input/output row axis declarations required")
    rows_set = set()
    for value, axis in zip(inputs, input_axes, strict=True):
        if axis is not None:
            shape = _shape(value)
            if type(axis) is not int or not 0 <= axis < len(shape):
                raise ValueError("valid input row axis required")
            rows_set.add(shape[axis])
    if len(rows_set) != 1:
        raise ValueError("input row extents disagree")
    rows = rows_set.pop()
    for value, axis in zip(outputs, output_axes, strict=True):
        _axis(value, axis, rows)
    boundary = analyze_observation_boundary(inputs=inputs, outputs=outputs)
    if not boundary.requested_boundary_closed:
        raise ValueError("row domain has external observations")
    function = boundary._block.parent_op()
    functions = dict(functions or {function.sym_name.data: function})
    validate_pure_tensor_function(function, functions)
    from xdsl.utils.exceptions import VerifyException

    try:
        function.verify()
    except VerifyException as exc:
        raise ValueError("row domain source verification failed") from exc
    states = {v: RowValue(v, a) for v, a in zip(inputs, input_axes, strict=True)}
    witnesses = []
    for operation in boundary.operations:
        witness = RowOperation(operation, None)
        if isinstance(operation, arith.ConstantOp):
            if any(isinstance(v.type, TensorType) for v in operation.results):
                raise ValueError("nonuniform tensor constants require a separate row proof")
            result_states = tuple(RowValue(v, None, True) for v in operation.results)
        elif isinstance(operation, tensor.SplatOp):
            if operation.dynamicSizes or not states[operation.input].uniform:
                raise ValueError("static uniform splat required")
            result_states = (RowValue(operation.result, None, True),)
        elif isinstance(operation, tensor.EmptyOp):
            result_states = tuple(RowValue(v, None) for v in operation.results)
        elif isinstance(operation, GenericOp):
            result_states, witness = _generic(operation, states, rows)
        elif isinstance(operation, ReduceOp):
            source, initial = states[operation.input], states[operation.init]
            dimensions = tuple(operation.dimensions.get_values())
            if source.axis is not None and source.axis in dimensions:
                raise ValueError("row dimension may not be reduced")
            axis = None if source.axis is None else source.axis - sum(d < source.axis for d in dimensions)
            if not initial.uniform and initial.axis != axis:
                raise ValueError("reduction initializer row path disagrees")
            result = operation.results[0]
            if axis is not None:
                _axis(result, axis, rows)
            result_states = (RowValue(result, axis, source.uniform and initial.uniform),)
            witness = RowOperation(operation, source.axis, dimensions)
        elif isinstance(operation, (tensor.ExpandShapeOp, tensor.CollapseShapeOp)):
            operation.verify()
            result_states = (_reshape(operation, states[operation.src], rows),)
            witness = RowOperation(operation, result_states[0].axis)
        elif isinstance(operation, tensor.ExtractSliceOp):
            source = states[operation.source]
            offsets, sizes, strides = (
                tuple(a.get_values())
                for a in (operation.static_offsets, operation.static_sizes, operation.static_strides)
            )
            shape = _shape(operation.source)
            if (
                operation.offsets
                or operation.sizes
                or operation.strides
                or sizes != _shape(operation.result)
                or len(shape) != len(_shape(operation.result))
                or len(offsets) != len(shape)
                or any(v < 0 for v in offsets)
                or any(s <= 0 for s in sizes)
                or any(s <= 0 for s in strides)
                or any(
                    o + (n - 1) * s >= extent for o, n, s, extent in zip(offsets, sizes, strides, shape, strict=True)
                )
            ):
                raise ValueError("complete rank-preserving static slice required")
            if source.axis is not None and (offsets[source.axis], sizes[source.axis], strides[source.axis]) != (
                0,
                rows,
                1,
            ):
                raise ValueError("source slice must preserve all original rows")
            result_states = (RowValue(operation.result, source.axis, source.uniform),)
            witness = RowOperation(operation, source.axis)
        elif isinstance(operation, tensor.CastOp):
            source = states[operation.source]
            if _shape(operation.source) != _shape(operation.dest):
                raise ValueError("row cast must preserve complete static shape")
            result_states = (RowValue(operation.dest, source.axis, source.uniform),)
            witness = RowOperation(operation, source.axis)
        else:
            raise ValueError("unsupported operation in closed row domain")
        for state in result_states:
            states[state.value] = state
        witnesses.append(witness)
    if tuple(states[v].axis for v in outputs) != output_axes:
        raise ValueError("derived output row paths disagree with observations")
    values = tuple(states.values())
    operation_witnesses = tuple(witnesses)
    selection = (rows, input_axes, output_axes, values, operation_witnesses)
    return ClosedRowDomain(
        boundary,
        rows,
        input_axes,
        output_axes,
        values,
        operation_witnesses,
        function,
        tuple(functions.items()),
        selection,
    )


def validate_closed_row_domain(domain: ClosedRowDomain) -> None:
    validate_observation_boundary(domain.boundary)
    if (domain.rows, domain.input_axes, domain.output_axes, domain.values, domain.operations) != domain._selection:
        raise ValueError("row domain contents changed after analysis")
    fresh = analyze_closed_row_domain(
        inputs=domain.boundary.inputs,
        outputs=domain.boundary.requested_outputs,
        input_axes=domain.input_axes,
        output_axes=domain.output_axes,
        functions=dict(domain._functions),
    )
    if fresh._selection != domain._selection:
        raise ValueError("row domain source changed after analysis")
