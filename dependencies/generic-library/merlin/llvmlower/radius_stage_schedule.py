"""Explicit scheduling of the generated admitted radius-coordinate consumer.

The enclosing source emitter already binds the producer's immutable private row
and complete output ownership. This optional transform recognizes its exact
consumer statement; it grants no new mathematical or buffer admission.
"""

from merlin.llvmlower.independent_lane_schedule import (
    LaneEffects,
    LaneOperation,
    independent_lane_schedule,
)


def stage_separable_radius_columns(source: str, *, effects: LaneEffects) -> str:
    """Stage eight columns, retaining checked scalar tails and refusal."""
    operations = (
        LaneOperation("product", ("factor", "maximum"), "f64", "rup"),
        LaneOperation("radius", ("product", "eta"), "f64", "rup"),
        LaneOperation("negative_radius", ("radius",), "f64", "exact"),
        LaneOperation("low", ("center", "negative_radius"), "f64", "rdn"),
        LaneOperation("high", ("center", "radius"), "f64", "rup"),
        LaneOperation("lower", ("low",), "f32", "rdn"),
        LaneOperation("upper", ("high",), "f32", "rup"),
    )
    independent_lane_schedule(
        operations,
        ("factor", "maximum", "eta", "center"),
        lanes=8,
        effects=effects,
        stage_major=True,
    )
    old = "if(radius_plan.valid){if(!merlin_fma_separable_radius_apply(&radius_plan,j,&lo[r*n+j],&hi[r*n+j]))return 0;continue;}"
    new = "if(radius_plan.valid){if(n-j>=8){if(!merlin_fma_separable_radius_apply_eight(&radius_plan,j,&lo[r*n+j],&hi[r*n+j]))return 0;j+=7;}else if(!merlin_fma_separable_radius_apply(&radius_plan,j,&lo[r*n+j],&hi[r*n+j]))return 0;continue;}"
    if source.count(old) != 1 or "#define MERLIN_ENABLE_EXACT_BOUND_CONVERSION 1" not in source:
        raise ValueError("one generated exact-conversion radius consumer required")
    return source.replace(old, new)
