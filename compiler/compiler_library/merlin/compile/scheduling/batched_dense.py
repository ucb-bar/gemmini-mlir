"""Independent matrix products over explicit shared batch/plane views.

Batch axes describe logical products, not physical accelerator dimensions.
Omitting a logical axis from a read view reuses that original operand. Every
output retains every logical axis; no product or numerical plane is combined.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod


@dataclass(frozen=True)
class BatchMatrixView:
    """Leading physical tensor axes select named logical batch axes.

    The last two physical dimensions hold a matrix. ``transposed`` changes
    their interpretation, without claiming a physical transpose or copy.
    """

    shape: tuple[int, ...]
    batch_axes: tuple[int, ...]
    transposed: bool = False

    def __post_init__(self):
        if (
            type(self.shape) is not tuple or not 2 <= len(self.shape) <= 8
            or any(type(n) is not int or n <= 0 for n in self.shape)
            or prod(self.shape) * 4 >= 1 << 63
            or type(self.batch_axes) is not tuple
            or len(self.batch_axes) != len(self.shape) - 2
            or any(type(a) is not int or a < 0 for a in self.batch_axes)
            or len(set(self.batch_axes)) != len(self.batch_axes)
            or type(self.transposed) is not bool
        ):
            raise ValueError("batch view requires bounded static dimensions and distinct explicit axes")

    @property
    def matrix_shape(self):
        return self.shape[-2:][::-1] if self.transposed else self.shape[-2:]


@dataclass(frozen=True)
class BatchedDenseProblem:
    """Complete Cartesian product with independently projected read operands.

    For example, A[lp,head,M,K] and B[rp,head,K,N] use axes (0,2)
    and (1,2), producing Y[lp,rp,head,M,N]. All plane combinations remain.
    Accessible allocations, nonalias and primitive effects need separate checks.
    """

    batch_shape: tuple[int, ...]
    lhs: BatchMatrixView
    rhs: BatchMatrixView

    def __post_init__(self):
        if (
            type(self.batch_shape) is not tuple or not 1 <= len(self.batch_shape) <= 6
            or any(type(n) is not int or n <= 0 for n in self.batch_shape)
            or type(self.lhs) is not BatchMatrixView or type(self.rhs) is not BatchMatrixView
        ):
            raise ValueError("independent products require explicit positive bounded batch axes")
        for view in (self.lhs, self.rhs):
            view.__post_init__()
            if any(axis >= len(self.batch_shape) for axis in view.batch_axes):
                raise ValueError("operand batch projection is outside the logical product")
            if tuple(self.batch_shape[a] for a in view.batch_axes) != view.shape[:-2]:
                raise ValueError("operand batch projection changes an original extent")
        if self.lhs.matrix_shape[1] != self.rhs.matrix_shape[0]:
            raise ValueError("independent products must preserve the complete reduction extent")
        if prod(self.output_shape) * 4 >= 1 << 63:
            raise ValueError("batched output exceeds the signed index range")

    @property
    def m(self):
        return self.lhs.matrix_shape[0]

    @property
    def k(self):
        return self.lhs.matrix_shape[1]

    @property
    def n(self):
        return self.rhs.matrix_shape[1]

    @property
    def output_shape(self):
        return (*self.batch_shape, self.m, self.n)

    @property
    def products(self):
        return prod(self.batch_shape)

    def record(self):
        self.__post_init__()
        return {"schema": "merlin.batched_dense_problem.v1", **asdict(self),
                "output_shape": self.output_shape, "products": self.products,
                "numeric_law": "independent zero-seeded signed-i8 products, modular-i32 accumulation",
                "target_effects": "not established by geometry"}
