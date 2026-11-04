"""Logical dense row scatter and explicit prepared-consumer quota plans.

These plans describe indexing and complete coverage. Physical allocation,
no-alias/lifetime, current immutable epoch, numerical and failure-before-public-
publication permissions remain independent mandatory obligations.
"""

import math
from dataclasses import dataclass

from .closed_row_domain import ClosedRowDomain, RowPartitionPlan, validate_closed_row_domain


@dataclass(frozen=True)
class RowScatter:
    block: int
    source_offset: int
    destination_offset: int
    elements: int


@dataclass(frozen=True)
class DenseRowScatterPlan:
    shape: tuple[int, ...]
    axis: int
    partition: RowPartitionPlan
    block_shapes: tuple[tuple[int, ...], ...]
    copies: tuple[RowScatter, ...]

    @property
    def output_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def private_scatter_elements(self) -> int:
        return sum(copy.elements for copy in self.copies)

    def validate(self) -> None:
        fresh = plan_dense_row_scatter(self.shape, self.axis, self.partition)
        if fresh != self:
            raise ValueError("row scatter contents do not match exact complete dense coverage")


def plan_dense_row_scatter(shape, axis: int, partition: RowPartitionPlan) -> DenseRowScatterPlan:
    """Copy each private packed row block into one complete private result.

    The returned destination offsets are logical elements in a dense result.
    This does not permit writing a public result before every block succeeds.
    """
    shape = tuple(shape)
    partition.validate()
    if (
        not shape
        or any(type(d) is not int or d <= 0 for d in shape)
        or type(axis) is not int
        or not 0 <= axis < len(shape)
    ):
        raise ValueError("positive static dense shape and valid row axis required")
    if shape[axis] != partition.rows:
        raise ValueError("scatter row extent differs from original domain")
    prefixes, suffixes = math.prod(shape[:axis]), math.prod(shape[axis + 1 :])
    shapes, copies = [], []
    for index, block in enumerate(partition.blocks):
        shapes.append((*shape[:axis], block.size, *shape[axis + 1 :]))
        for prefix in range(prefixes):
            copies.append(
                RowScatter(
                    index,
                    prefix * block.size * suffixes,
                    (prefix * partition.rows + block.offset) * suffixes,
                    block.size * suffixes,
                )
            )
    return DenseRowScatterPlan(shape, axis, partition, tuple(shapes), tuple(copies))


def plan_closed_row_scatters(domain: ClosedRowDomain, partition: RowPartitionPlan) -> tuple[DenseRowScatterPlan, ...]:
    validate_closed_row_domain(domain)
    partition.validate()
    if partition.rows != domain.rows:
        raise ValueError("partition differs from closed original domain")
    return tuple(
        plan_dense_row_scatter(value.type.get_shape(), axis, partition)
        for value, axis in zip(domain.boundary.requested_outputs, domain.output_axes, strict=True)
    )


@dataclass(frozen=True)
class PartitionedConsumerLeases:
    public_consumers: int
    blocks: int
    inner_consumers: int
    ordinal_limit: int

    def inner_ordinal(self, public: int, block: int) -> int:
        self.validate()
        if (
            type(public) is not int
            or type(block) is not int
            or not 0 <= public < self.public_consumers
            or not 0 <= block < self.blocks
        ):
            raise ValueError("consumer and block must be within the explicit ordered lease")
        return public * self.blocks + block

    def validate(self) -> None:
        fresh = plan_partitioned_consumer_leases(self.public_consumers, self.blocks, ordinal_limit=self.ordinal_limit)
        if fresh != self:
            raise ValueError("prepared lease contents changed")


def plan_partitioned_consumer_leases(
    public_consumers: int, blocks: int, *, ordinal_limit: int
) -> PartitionedConsumerLeases:
    """Expand an explicit immutable-owner lease without silently adding uses."""
    if any(type(n) is not int or n <= 0 for n in (public_consumers, blocks, ordinal_limit)):
        raise ValueError("positive explicit consumer/block/ordinal bounds required")
    if public_consumers > ordinal_limit // blocks:
        raise ValueError("expanded consumer quota exceeds its declared ordinal bound")
    return PartitionedConsumerLeases(public_consumers, blocks, public_consumers * blocks, ordinal_limit)
