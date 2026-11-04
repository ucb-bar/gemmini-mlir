"""Explicit bounded integer-position lookup for source-evaluated constants.

The caller proves the position interval and owns the table's numerical/source
identity. This builder neither evaluates transcendental functions nor changes
mathematical operators automatically. Tables can preserve a selected source
backend while avoiding runtime libm differences.
"""

from __future__ import annotations


def build_position_table_lookup(positions, table, *, position_bounds: tuple[int, int], table_min: int):
    """Return detached ops and tensor<BxLx1xD>, preserving table element bits.

    ``position_bounds`` is an inclusive proven range, not an observed range from
    sample inputs. Unsupported types/shapes and incomplete table coverage refuse.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IndexType, IntegerAttr, TensorType, f32, i64
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if not isinstance(positions.type, TensorType) or positions.type.element_type != i64:
        raise TypeError("positions must be an i64 tensor")
    if not isinstance(table.type, TensorType) or table.type.element_type != f32:
        raise TypeError("table must be an f32 tensor")
    shape = list(positions.type.get_shape())
    table_shape = list(table.type.get_shape())
    if len(shape) != 2 or len(table_shape) != 2 or any(n <= 0 for n in [*shape, *table_shape]):
        raise ValueError("static positive rank-two positions and table required")
    if len(position_bounds) != 2 or any(type(x) is not int for x in [*position_bounds, table_min]):
        raise TypeError("explicit integer position interval required")
    low, high = position_bounds
    if low > high or low < table_min or high >= table_min + table_shape[0]:
        raise ValueError("table does not cover the proven position interval")
    if not -(1 << 63) <= table_min <= low <= high < (1 << 63) or high - table_min >= (1 << 63):
        raise ValueError("position subtraction must fit signed i64")
    output_type = TensorType(f32, [*shape, 1, table_shape[1]])
    empty = tensor.EmptyOp((), output_type)
    body = Block(arg_types=[i64, f32])
    minimum = arith.ConstantOp(IntegerAttr(table_min, i64))
    offset = arith.SubiOp(body.args[0], minimum.result)
    index = arith.IndexCastOp(offset.result, IndexType())
    lane = linalg.IndexOp(3)
    value = tensor.ExtractOp(table, [index.result, lane.result], f32)
    body.add_ops([minimum, offset, index, lane, value, linalg.YieldOp(value.result)])
    op = linalg.GenericOp(
        inputs=(positions,),
        outputs=(empty.tensor,),
        body=Region(body),
        indexing_maps=ArrayAttr(
            [
                AffineMapAttr(AffineMap(4, 0, (AffineDimExpr(0), AffineDimExpr(1)))),
                AffineMapAttr(AffineMap.identity(4)),
            ]
        ),
        iterator_types=ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 4),
        result_types=(output_type,),
    )
    return [empty, op], op.results[0]
