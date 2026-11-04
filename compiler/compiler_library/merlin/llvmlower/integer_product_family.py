"""Exact integer contraction families sharing immutable plane-major operands.

Each output is an explicit sum of selected signed-byte matrix products. Bounds
cover every intermediate prefix independently of reduction or pair order. This
contract permits sharing operands across outputs; hardware storage and command
ordering belong to the backend. No radix or floating source policy is inferred.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class IntegerProductGroup:
    pairs: tuple[tuple[int, int], ...]
    absolute_bound: int


@dataclass(frozen=True)
class IntegerProductFamily:
    rows: int
    columns: int
    reduction_length: int
    lhs_planes: int
    rhs_planes: int
    groups: tuple[IntegerProductGroup, ...]
    lhs_magnitude_bound: int = 128
    rhs_magnitude_bound: int = 128
    output_plane_stride: int | None = None

    def __post_init__(self):
        dimensions = (self.rows, self.columns, self.reduction_length, self.lhs_planes, self.rhs_planes)
        if any(type(value) is not int or not 0 < value <= (1 << 31) - 1 for value in dimensions):
            raise ValueError("positive signed-int dimensions and plane counts required")
        if any(
            type(value) is not int or not 0 <= value <= 128
            for value in (self.lhs_magnitude_bound, self.rhs_magnitude_bound)
        ):
            raise ValueError("explicit signed-byte magnitude bounds required")
        if type(self.groups) is not tuple or not self.groups:
            raise ValueError("nonempty immutable output groups required")
        for group in self.groups:
            if not isinstance(group, IntegerProductGroup) or type(group.pairs) is not tuple or not group.pairs:
                raise ValueError("nonempty immutable pair lists required")
            for pair in group.pairs:
                if (
                    type(pair) is not tuple
                    or len(pair) != 2
                    or any(type(value) is not int for value in pair)
                    or not 0 <= pair[0] < self.lhs_planes
                    or not 0 <= pair[1] < self.rhs_planes
                ):
                    raise ValueError("pair index outside declared operand planes")
            required = len(group.pairs) * self.reduction_length * self.lhs_magnitude_bound * self.rhs_magnitude_bound
            if type(group.absolute_bound) is not int or not required <= group.absolute_bound < 1 << 31:
                raise ValueError("group bound must cover every signed-i32 prefix")
        stride = self.output_plane_stride
        if stride is not None and (type(stride) is not int or stride < self.rows * self.columns):
            raise ValueError("disjoint complete output planes required")
        if max(self.lhs_elements, self.rhs_elements, 4 * self.output_elements) > (1 << 63) - 1:
            raise ValueError("complete storage extent exceeds supported pointer domain")

    @property
    def plane_stride(self) -> int:
        return self.rows * self.columns if self.output_plane_stride is None else self.output_plane_stride

    @property
    def lhs_elements(self) -> int:
        return self.lhs_planes * self.rows * self.reduction_length

    @property
    def rhs_elements(self) -> int:
        return self.rhs_planes * self.reduction_length * self.columns

    @property
    def output_elements(self) -> int:
        return (len(self.groups) - 1) * self.plane_stride + self.rows * self.columns


def family_from_radix_plan(plan, *, rows: int, columns: int, output_plane_stride: int | None = None):
    """Delegate exact radix eligibility to its existing source-owned proof."""
    from .radix_product_groups import RadixProductPlan, plan_radix_product_groups

    if not isinstance(plan, RadixProductPlan):
        raise ValueError("complete radix product plan required")
    derived = plan_radix_product_groups(
        radix_bits=plan.radix_bits, digits=plan.digits, reduction_length=plan.reduction_length
    )
    if plan != derived or plan.radix_bits > 7:
        raise ValueError("radix proof changed or digits exceed signed-byte domain")
    magnitude = (1 << plan.radix_bits) - 1
    return IntegerProductFamily(
        rows,
        columns,
        plan.reduction_length,
        plan.digits,
        plan.digits,
        tuple(IntegerProductGroup(group.pairs, group.accumulator_bound) for group in plan.groups),
        magnitude,
        magnitude,
        output_plane_stride,
    )


def reference_outputs(family: IntegerProductFamily, lhs: Sequence[int], rhs: Sequence[int]):
    """Complete Python integer oracle, independent of any target schedule."""
    if len(lhs) != family.lhs_elements or len(rhs) != family.rhs_elements:
        raise ValueError("complete plane-major operands required")
    for values, magnitude in ((lhs, family.lhs_magnitude_bound), (rhs, family.rhs_magnitude_bound)):
        if any(type(value) is not int or not -128 <= value <= 127 or abs(value) > magnitude for value in values):
            raise ValueError("operand does not satisfy declared signed-byte domain")
    m, n, k = family.rows, family.columns, family.reduction_length
    return tuple(
        tuple(
            sum(lhs[(a * m + row) * k + z] * rhs[(b * k + z) * n + col] for a, b in group.pairs for z in range(k))
            for row in range(m)
            for col in range(n)
        )
        for group in family.groups
    )


def c_header(family: IntegerProductFamily, *, symbol: str) -> str:
    """One checked callback over full operands and all private output planes.

    This mechanical wrapper checks span and output nonaliasing. The caller must
    independently bind the callback implementation to this exact family and
    prove fully initialized outputs, source domains, lifetimes and completion.
    A return value is not an implementation or numerical-equivalence proof.
    """
    if not symbol or not symbol.isascii() or not symbol.isidentifier():
        raise ValueError("explicit ASCII C identifier required")
    return f"""#include <stddef.h>
#include <stdint.h>
typedef int (*merlin_integer_product_family_callback)(void*,const int8_t*,const int8_t*,int32_t*,int,int,int,size_t);
static inline int {symbol}(merlin_integer_product_family_callback callback,void *opaque,
    const int8_t *a,size_t a_bytes,const int8_t *b,size_t b_bytes,int32_t *c,size_t c_elements) {{
  if({family.lhs_elements}ULL>SIZE_MAX||{family.rhs_elements}ULL>SIZE_MAX||
     {family.output_elements}ULL>SIZE_MAX/sizeof(int32_t))return 0;
  const size_t ae={family.lhs_elements}ULL,be={family.rhs_elements}ULL,ce={family.output_elements}ULL;
  if(!callback||!a||!b||!c||a_bytes<ae||b_bytes<be||c_elements<ce)return 0;
  if((uintptr_t)a>UINTPTR_MAX-ae||(uintptr_t)b>UINTPTR_MAX-be||
     ce>SIZE_MAX/sizeof(int32_t)||(uintptr_t)c>UINTPTR_MAX-ce*sizeof(int32_t))return 0;
  const uintptr_t ab=(uintptr_t)a,bb=(uintptr_t)b,cb=(uintptr_t)c,cz=cb+ce*sizeof(int32_t);
  if(!(cz<=ab||ab+ae<=cb)||!(cz<=bb||bb+be<=cb))return 0;
  return callback(opaque,a,b,c,{family.rows},{family.columns},{family.reduction_length},{family.plane_stride}ULL)==1;
}}
"""
