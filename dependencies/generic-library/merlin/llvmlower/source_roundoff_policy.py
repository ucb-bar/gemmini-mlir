"""Explicit approximate source-roundoff policy for owned source products.

The estimate is not a rigorous enclosure or an exact observer certificate. Its
permission is independent of the typed source/representation/effect witnesses.
No model, captured values or accuracy oracle selects this policy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ApproximateSourceRoundoffPolicy:
    """Permission for the fixed RMS4 source-error estimate experiment.

    Every field is required. A provider must preserve original source fallback,
    deterministic prefix safety, representation error and nonlinear/order rules.
    Independent original output validation remains required before deployment;
    this policy supplies no probability or deterministic accuracy guarantee.
    """

    approximate_output_permitted: bool
    independent_output_validation_required: bool
    representation_error_unscaled: bool
    source_prefix_safety_checked: bool
    original_source_fallback_retained: bool
    stable_rne: bool
    nontrapping: bool
    exception_flags_unobserved: bool

    @property
    def numerical_policy(self) -> str:
        return "approximate_source_roundoff_rms4"

    def validate(self) -> None:
        if any(type(value) is not bool or not value for value in vars(self).values()):
            raise ValueError("complete explicit approximate source-roundoff permission required")


def prepare_source_roundoff_estimates(source: str, *, policy: ApproximateSourceRoundoffPolicy) -> str:
    """Insert the estimate before the unchanged rigorous private fallback.

    Only the regular emitter's complete encoded-row product signature is
    accepted. The estimate cannot replace representation uncertainty or source
    overflow safety. A refused batch executes the original rigorous producer;
    provider refusal still selects the retained original source writer.
    """
    if not isinstance(policy, ApproximateSourceRoundoffPolicy):
        raise ValueError("typed approximate source-roundoff policy required")
    policy.validate()
    signature = "static int dot_bounds(const float*a,const float*alo,const float*ahi,const float*b,const double*ar,const double*br,const double*center,float*lo,float*hi,int m,int n,int k,struct product_scratch *scratch,const merlin_encoded_row_equality *aproof,const merlin_encoded_row_equality *bproof){"
    if source.count(signature) != 1 or source.count("scratch->uncertainty") < 1:
        raise ValueError("complete owned encoded-row bound producer required")
    if "source_rms_point_products.h" in source:
        raise ValueError("source-roundoff policy already selected")
    prefix = '#include "source_rms_point_products.h"\n'
    selected = (
        signature
        + "\n merlin_fma_bound rms_environment=merlin_fma_bound_begin();\n if(merlin_source_rms4_point_product_estimates(&rms_environment,aproof,bproof,a,alo,ahi,b,ar,br,center,lo,hi,m,n,k,scratch->uncertainty,CHUNK))return 1;\n"
    )
    return (
        "/* Explicit APPROXIMATE RMS4 source-roundoff policy: no rigorous source or exact-observer certificate. */\n"
        + source.replace(signature, prefix + selected)
    )
