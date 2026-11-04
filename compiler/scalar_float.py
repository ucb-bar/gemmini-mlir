"""Strict floating-point extrema expressions for scalar C lowering."""


def emit_float_extremum(op, dtype, a, b):
    """Emit an f32/f64 extremum, including NaNs and signed zero.

    Operands must be side-effect-free scalar values: the expression may read
    each operand more than once. Compile without unsafe floating-point flags.
    Requires the code generator's gm_scalar_select[_double] bitwise selection
    helper. Classification operations are Clang/GCC builtins.
    """
    if dtype not in ("f32", "f64"):
        raise ValueError("floating extremum requires f32 or f64: " + str(dtype))
    if op not in (
        "arith.maximumf", "arith.minimumf", "arith.maxnumf", "arith.minnumf"
    ):
        raise ValueError("unsupported floating extremum: " + str(op))

    a, b = "(" + a + ")", "(" + b + ")"
    select = "gm_scalar_select" if dtype == "f32" else "gm_scalar_select_double"
    maximum = op in ("arith.maximumf", "arith.maxnumf")
    comparison = ">" if maximum else "<"
    # For a zero tie, choose +0 for maximum and -0 for minimum. Selecting
    # operands by their sign avoids dependence on the active rounding mode.
    zero = (
        f"{select}(__builtin_signbit({a}), {b}, {a})" if maximum else
        f"{select}(__builtin_signbit({a}), {a}, {b})"
    )
    ordered = (
        f"{select}((({a} == 0) & ({b} == 0)), {zero}, "
        f"{select}({a} {comparison} {b}, {a}, {b}))"
    )
    if op in ("arith.maximumf", "arith.minimumf"):
        return (
            f"{select}(__builtin_isnan({a}), {a}, "
            f"{select}(__builtin_isnan({b}), {b}, {ordered}))"
        )
    return (
        f"{select}(__builtin_isnan({a}), {b}, "
        f"{select}(__builtin_isnan({b}), {a}, {ordered}))"
    )
