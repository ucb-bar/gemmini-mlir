"""Exact finite-domain binary64 enclosure conversion capability.

This permission is stronger than arbitrary outward enclosure: the provider must
return the immediately adjacent binary32 floor/ceiling, preserving signed zero,
without changing the source rounding environment. It is used only after the
existing bound producer proves finite ordered endpoints within binary32 range.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExactBoundConversionContract:
    exact_binary32_floor_ceil: bool
    preserves_signed_zero: bool
    source_rounding_unchanged: bool
    nontrapping: bool
    exception_flags_unobserved: bool

    def validate(self) -> None:
        if any(type(x) is not bool or not x for x in self.__dict__.values()):
            raise ValueError("complete exact finite bound-conversion capability required")


def emit_exact_bound_conversion_permission(contract: ExactBoundConversionContract) -> str:
    """Select separately supplied mathematical hooks; no CPU ISA is emitted."""
    if not isinstance(contract, ExactBoundConversionContract):
        raise ValueError("typed exact bound-conversion contract required")
    contract.validate()
    return (
        "#define MERLIN_ENABLE_EXACT_BOUND_CONVERSION 1\n"
        "#if !defined(MERLIN_F32_EXACT_FLOOR_FROM_F64) || !defined(MERLIN_F32_EXACT_CEIL_FROM_F64)\n"
        '#error "explicit exact bound conversion provider required"\n'
        "#endif\n"
    )
