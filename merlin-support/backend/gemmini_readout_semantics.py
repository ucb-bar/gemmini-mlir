"""Readout semantic contract derived from the exact target's scalar ABI header."""

import hashlib
from pathlib import Path

from merlin.targetgen.capability_discovery import parse_c_header


def narrow_readout_contract(params_header: Path) -> dict:
    text = params_header.read_text()
    parsed = parse_c_header(params_header)
    types = {alias: underlying for alias, underlying, _ in parsed.typedefs}
    if types.get("acc_t") != "int32_t" or types.get("elem_t") != "int8_t" or types.get("acc_scale_t") != "float":
        raise ValueError("unsupported target readout scalar ABI")
    scale = parsed.macro("ACC_SCALE")
    rounding = parsed.macro("ROUND_NEAR_EVEN")
    if scale is None or rounding is None:
        raise ValueError("missing readout/rounding source")
    expected_scale = (
        "({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y);})"
    )
    expected_rounding = (
        "({ const SAME_TYPE(x) x_ = (x); const long long i = x_; const long long next = x_ < 0 ? x_ - 1 : x_ + 1; "
        "SAME_TYPE(x) rem = x_ - i; rem = rem < 0 ? -rem : rem; SAME_TYPE(x) result = rem < 0.5 ? i : (rem > 0.5 ? next : ( "
        "i % 2 == 0 ? i : next)); result; })"
    )
    if " ".join(scale.body.split()) != expected_scale or " ".join(rounding.body.split()) != expected_rounding:
        raise ValueError("unrecognized target scale/rounding implementation")
    # Widths come from the verified C typedefs, not an unrelated target constant.
    accumulator_bits = int(types["acc_t"].removeprefix("int").removesuffix("_t"))
    output_bits = int(types["elem_t"].removeprefix("int").removesuffix("_t"))
    return {
        "schema": "scalar_narrow_readout_contract_v1",
        "accumulator_dtype": f"i{accumulator_bits}",
        "output_dtype": f"i{output_bits}",
        "scale_dtype": "f32",
        "clamp_min": -(1 << (output_bits - 1)),
        "clamp_max": (1 << (output_bits - 1)) - 1,
        "provenance": {
            "params_header_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "macros": ["ACC_SCALE", "ROUND_NEAR_EVEN"],
            "scope": "exact existing target scalar ABI; independent RTL arithmetic proof not supplied",
        },
    }


def operand_sum_contract(params_header: Path) -> dict | None:
    """What this design's LOAD path does to an operand on its way into the accumulator.

    A sum of two tensors is two loads into one accumulator row range, the second accumulating, and
    one readout. What makes that an integer ``residual_add`` and not just data movement is that the
    load itself multiplies: the parameter header states whether it does, in what type, and the
    exact arithmetic. ``None`` when the design was generated without a scaled load.

    The arithmetic matters as much as its existence. The verified construction rounds half to even
    AND saturates each operand to the element type before anything is added, so this unit computes
    ``sat(round(a*sa)) + sat(round(b*sb))``: a different function from the single-rounding
    reference, by at most one step per pair of operands while no operand saturates, and by an
    unbounded amount once one does. Both facts are returned; a caller that admits a sum has to
    hold the multipliers to the range where the second cannot happen.
    """
    text = params_header.read_text()
    parsed = parse_c_header(params_header)
    if parsed.macro("HAS_MVIN_SCALE") is None:
        return None
    types = {alias: underlying for alias, underlying, _ in parsed.typedefs}
    scale = parsed.macro("MVIN_SCALE")
    if types.get("scale_t") != "float" or types.get("elem_t") != "int8_t" or scale is None:
        raise ValueError("unsupported target load-scale ABI")
    expected = (
        "({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (elem_t)y);})"
    )
    if " ".join(scale.body.split()) != expected:
        raise ValueError("unrecognized target load-scale implementation")
    element_bits = int(types["elem_t"].removeprefix("int").removesuffix("_t"))
    return {
        "schema": "operand_sum_contract_v1",
        # The vendor's own sequence accumulates two loads; more would be the same mechanism and is
        # not claimed until a program has been run that way.
        "operands": 2,
        "operand_dtype": f"i{element_bits}",
        "scale_dtype": "f32",
        "operand_rounding": "half_even",
        "operand_saturates": True,
        "provenance": {
            "params_header_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "macros": ["HAS_MVIN_SCALE", "MVIN_SCALE"],
            "scope": "exact existing target scalar ABI; independent RTL arithmetic proof not supplied",
        },
    }
