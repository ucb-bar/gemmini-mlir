"""Explicit compiler builtins for an admitted source numeric component.

This is not a global runtime/interposition policy. Normal provider compilation
must pin this prefix and compiler flags, and separately close actual target ISA.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceNumericContract:
    round_to_nearest_even: bool
    fused_single_rounding: bool
    errno_unobserved: bool
    nontrapping: bool
    exception_flags_unobserved: bool
    standard_bitcast_copy: bool
    copy_interposition_unobserved: bool
    standard_fp_classification: bool = False
    classification_interposition_unobserved: bool = False
    standard_absolute_value: bool = False
    absolute_value_interposition_unobserved: bool = False
    absolute_value_nan_payload_unobserved: bool = False
    standard_min_max: bool = False
    min_max_interposition_unobserved: bool = False
    min_max_signed_zero_nan_payload_unobserved: bool = False
    standard_floor_values: bool = False
    floor_interposition_unobserved: bool = False


def emit_source_numeric_capability(
    contract: SourceNumericContract,
    *,
    inline_fma: bool = False,
    inline_bitcasts: bool = False,
    inline_classification: bool = False,
    inline_absolute_values: bool = False,
    inline_min_max: bool = False,
    inline_floor: bool = False,
) -> str:
    """Select separately proved builtins; default returns no prefix bytes."""
    if any(
        type(value) is not bool
        for value in (
            inline_fma,
            inline_bitcasts,
            inline_classification,
            inline_absolute_values,
            inline_min_max,
            inline_floor,
        )
    ):
        raise ValueError("explicit boolean capability choices required")
    if not any(
        (inline_fma, inline_bitcasts, inline_classification, inline_absolute_values, inline_min_max, inline_floor)
    ):
        return ""
    if not isinstance(contract, SourceNumericContract):
        raise ValueError("source numeric observation contract required")
    for value in vars(contract).values():
        if type(value) is not bool:
            raise ValueError("explicit source obligations required")
    if inline_fma and not (
        contract.round_to_nearest_even
        and contract.fused_single_rounding
        and contract.errno_unobserved
        and contract.nontrapping
        and contract.exception_flags_unobserved
    ):
        raise ValueError("source FMA observation contract is incomplete")
    if inline_bitcasts and not (contract.standard_bitcast_copy and contract.copy_interposition_unobserved):
        raise ValueError("standard object-representation copy contract required")
    if inline_classification and not (
        contract.standard_fp_classification
        and contract.classification_interposition_unobserved
        and contract.nontrapping
        and contract.exception_flags_unobserved
    ):
        raise ValueError("standard nontrapping classification observation contract required")
    if inline_absolute_values and not (
        contract.standard_absolute_value
        and contract.absolute_value_interposition_unobserved
        and contract.absolute_value_nan_payload_unobserved
        and contract.errno_unobserved
        and contract.nontrapping
        and contract.exception_flags_unobserved
    ):
        raise ValueError("standard absolute-value observation contract required")
    if inline_min_max and not (
        contract.standard_min_max
        and contract.min_max_interposition_unobserved
        and contract.min_max_signed_zero_nan_payload_unobserved
        and contract.errno_unobserved
        and contract.nontrapping
        and contract.exception_flags_unobserved
    ):
        raise ValueError("standard min/max observation contract required")
    if inline_floor and not (
        contract.standard_floor_values
        and contract.floor_interposition_unobserved
        and contract.errno_unobserved
        and contract.nontrapping
        and contract.exception_flags_unobserved
    ):
        raise ValueError("standard nontrapping floor-value observation contract required")
    text = [
        "#ifndef MERLIN_ADMITTED_SOURCE_NUMERIC_CAPABILITY_H",
        "#define MERLIN_ADMITTED_SOURCE_NUMERIC_CAPABILITY_H",
        "#ifdef MERLIN_SOURCE_F32_MATH_H",
        '#error "source numeric capability must precede numeric headers"',
        "#endif",
        "#if !defined(__clang__) && !defined(__GNUC__)",
        '#error "source numeric builtin capability unavailable"',
        "#endif",
    ]
    if inline_fma:
        text.extend(
            [
                "#if !defined(__has_builtin)",
                '#error "elementwise FMA compiler capability unavailable"',
                "#elif !__has_builtin(__builtin_elementwise_fma)",
                '#error "elementwise FMA compiler capability unavailable"',
                "#endif",
                "#define MERLIN_SOURCE_F32_FMA(a,b,c) __builtin_elementwise_fma((a),(b),(c))",
            ]
        )
    if inline_bitcasts:
        text.append("#define MERLIN_SOURCE_BITCAST_COPY(d,s,n) __builtin_memcpy((d),(s),(n))")
    if inline_classification:
        text.extend(
            [
                "#if !defined(__has_builtin)",
                '#error "finite classification compiler capability unavailable"',
                "#elif !__has_builtin(__builtin_isfinite)",
                '#error "finite classification compiler capability unavailable"',
                "#endif",
                "#define MERLIN_SOURCE_ISFINITE(x) __builtin_isfinite((x))",
            ]
        )
    if inline_absolute_values:
        text.extend(
            [
                "#if !defined(__has_builtin)",
                '#error "absolute-value compiler capability unavailable"',
                "#elif !__has_builtin(__builtin_fabsf) || !__has_builtin(__builtin_fabs)",
                '#error "absolute-value compiler capability unavailable"',
                "#endif",
                "#define MERLIN_SOURCE_F32_ABS(x) __builtin_fabsf((x))",
                "#define MERLIN_SOURCE_F64_ABS(x) __builtin_fabs((x))",
            ]
        )
    if inline_min_max:
        text.extend(
            [
                "#if !defined(__has_builtin)",
                '#error "min/max compiler capability unavailable"',
                "#elif !__has_builtin(__builtin_fminf) || !__has_builtin(__builtin_fmaxf) || !__has_builtin(__builtin_fmin) || !__has_builtin(__builtin_fmax)",
                '#error "min/max compiler capability unavailable"',
                "#endif",
                "#define MERLIN_SOURCE_F32_MIN(a,b) __builtin_fminf((a),(b))",
                "#define MERLIN_SOURCE_F32_MAX(a,b) __builtin_fmaxf((a),(b))",
                "#define MERLIN_SOURCE_F64_MIN(a,b) __builtin_fmin((a),(b))",
                "#define MERLIN_SOURCE_F64_MAX(a,b) __builtin_fmax((a),(b))",
            ]
        )
    if inline_floor:
        text.extend(
            [
                "#if !defined(__has_builtin)",
                '#error "floor-value compiler capability unavailable"',
                "#elif !__has_builtin(__builtin_floorf) || !__has_builtin(__builtin_isfinite)",
                '#error "floor-value compiler capability unavailable"',
                "#endif",
                "#include <math.h>",
                "static inline float merlin_source_f32_floor_finite(float value) {",
                "  return __builtin_isfinite(value) ? __builtin_floorf(value) : floorf(value);",
                "}",
                "#define MERLIN_SOURCE_F32_FLOOR(x) merlin_source_f32_floor_finite((x))",
            ]
        )
    return "\n".join([*text, "#endif", ""])
