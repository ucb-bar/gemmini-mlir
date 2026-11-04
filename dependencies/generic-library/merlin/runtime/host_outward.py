"""Explicit shared host CPU implementation of directed certificate arithmetic.

Source rounding remains unchanged. Nontrapping arithmetic and unobserved flags,
original numerical eligibility and the actual selected host capability remain
independent caller obligations. This emitter selects no source operation,
accelerator, graph transform or approximation policy.
"""

from __future__ import annotations


def emit_fixed_outward_f64_header(
    *, name: str, host_isa: str, narrow_f32: bool = False, exact_bound_f32: bool = False
) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name[0].isdigit()
        or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in name)
    ):
        raise ValueError("explicit C identifier required")
    if host_isa != "rv64gc":
        raise ValueError("explicit supported host CPU capability required")
    if type(narrow_f32) is not bool:
        raise ValueError("explicit boolean narrowing capability required")
    if type(exact_bound_f32) is not bool:
        raise ValueError("explicit boolean exact bound capability required")
    guard = name.upper() + "_OUTWARD_F64_H"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "#ifdef MERLIN_ORDERED_FMA_BOUNDS_H",
        '#error "outward capability must precede every ordered-FMA header include"',
        "#endif",
        "#if defined(MERLIN_F64_OUTWARD_ADD_UP) || defined(MERLIN_F64_OUTWARD_ADD_DOWN)"
        " || defined(MERLIN_F64_OUTWARD_MUL_UP)",
        '#error "outward scalar capability already selected"',
        "#endif",
        "/* Explicit IEEE binary64 directed arithmetic; source rounding is unchanged.",
        " * Nontrapping arithmetic and unobserved exception flags are required.",
        " * Include before ordered_fma_bounds.h; no accelerator instruction. */",
    ]
    for operation, instruction, mode in (
        ("add_up", "fadd.d", "rup"),
        ("add_down", "fadd.d", "rdn"),
        ("mul_up", "fmul.d", "rup"),
    ):
        lines.extend(
            [
                f"static inline double {name}_{operation}(double a, double b) {{",
                "  double result;",
                f'  __asm__("{instruction} %0, %1, %2, {mode}"',
                '          : "=f"(result) : "f"(a), "f"(b));',
                "  return result;",
                "}",
            ]
        )
    lines.extend(
        [
            f"#define MERLIN_F64_OUTWARD_ADD_UP(a,b) {name}_add_up((a),(b))",
            f"#define MERLIN_F64_OUTWARD_ADD_DOWN(a,b) {name}_add_down((a),(b))",
            f"#define MERLIN_F64_OUTWARD_MUL_UP(a,b) {name}_mul_up((a),(b))",
        ]
    )
    if narrow_f32 or exact_bound_f32:
        lines.extend(
            [
                "#if defined(MERLIN_F32_OUTWARD_FROM_F64_DOWN) || defined(MERLIN_F32_OUTWARD_FROM_F64_UP)",
                '#error "outward narrowing capability already selected"',
                "#endif",
            ]
        )
        for operation, mode in (("cast_down", "rdn"), ("cast_up", "rup")):
            lines.extend(
                [
                    f"static inline float {name}_{operation}(double value) {{",
                    "  float result;",
                    f'  __asm__("fcvt.s.d %0, %1, {mode}" : "=f"(result) : "f"(value));',
                    "  return result;",
                    "}",
                ]
            )
        if narrow_f32:
            lines.extend(
                [
                    f"#define MERLIN_F32_OUTWARD_FROM_F64_DOWN(value) {name}_cast_down((value))",
                    f"#define MERLIN_F32_OUTWARD_FROM_F64_UP(value) {name}_cast_up((value))",
                ]
            )
        if exact_bound_f32:
            lines.extend(
                [
                    "#if defined(MERLIN_F32_EXACT_FLOOR_FROM_F64) || defined(MERLIN_F32_EXACT_CEIL_FROM_F64)",
                    '#error "exact bound narrowing capability already selected"',
                    "#endif",
                    f"#define MERLIN_F32_EXACT_FLOOR_FROM_F64(value) {name}_cast_down((value))",
                    f"#define MERLIN_F32_EXACT_CEIL_FROM_F64(value) {name}_cast_up((value))",
                ]
            )
    return "\n".join([*lines, "#endif", ""])
