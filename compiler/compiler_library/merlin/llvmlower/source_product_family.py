"""Explicit complete integer families at an owned source reconstruction seam."""

from dataclasses import dataclass

from .integer_product_family import c_header, family_from_radix_plan
from .radix_product_groups import plan_radix_product_groups


@dataclass(frozen=True)
class SourceProductFamilyContract:
    exact_all_integer_products: bool
    complete_output_planes: bool
    immutable_input_planes: bool
    private_disjoint_output_planes: bool
    synchronous_completion_and_drain: bool
    preserves_host_fenv: bool

    def validate(self) -> None:
        if any(type(value) is not bool or not value for value in vars(self).values()):
            raise ValueError("complete exact source product-family contract required")


def source_product_families(plan):
    """Derive every actual source product shape and its explicit plane stride."""
    from .source_attention_frontier import SourceAttentionFrontierPlan

    if not isinstance(plan, SourceAttentionFrontierPlan):
        raise ValueError("typed complete source-attention plan required")
    plan.validate()
    shapes = (
        (plan.query_rows, plan.chunk, plan.depth),
        (plan.query_rows, plan.depth, plan.segment),
        (plan.query_rows, plan.depth, plan.chunk - 2 * plan.segment),
    )
    return tuple(
        family_from_radix_plan(
            plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=k),
            rows=m,
            columns=n,
            output_plane_stride=plan.query_rows * plan.chunk,
        )
        for m, n, k in dict.fromkeys(shapes)
    )


def prepare_source_product_family(source: str, *, plan, contract: SourceProductFamilyContract) -> str:
    """One complete callback; all five readout planes and reconstruction retained.

    The physical caller must supply the separately typed family callback. The
    degree callback ABI is not compatible. Storage/lifetime and implementation
    qualification are obligations outside this source-owned specialization.
    """
    if not isinstance(contract, SourceProductFamilyContract):
        raise ValueError("typed source product-family contract required")
    contract.validate()
    if "merlin_attention_complete_product_family" in source:
        raise ValueError("source product family already selected")
    families = source_product_families(plan)
    original = """ const int32_t *planes[MERLIN_RADIX_FUSED_INTEGER_GROUPS];
 for(int degree=0;degree<MERLIN_RADIX_FUSED_INTEGER_GROUPS;degree++){
  planes[degree]=w->readout[degree];
  if(!product(opaque,w->ap,@RHS@,w->readout[degree],m,n,k,degree))return 0;
 }
 merlin_radix_integer_fused_exact_f64(w->center,planes,(size_t)m*n);
"""
    candidates = [(rhs, original.replace("@RHS@", rhs)) for rhs in ("w->bp", "bp")]
    matches = [(rhs, text) for rhs, text in candidates if source.count(text) == 1]
    storage = " int32_t readout[MERLIN_RADIX_FUSED_INTEGER_GROUPS][ROWS*CHUNK];"
    if len(matches) != 1 or source.count(storage) != 1:
        raise ValueError("owned complete fused readout seam changed; refused")
    rhs, selected = matches[0]
    headers = []
    choices = []
    for index, family in enumerate(families):
        symbol = f"merlin_attention_product_family_{index}"
        headers.append(c_header(family, symbol=symbol))
        choices.append(
            f" if(m=={family.rows}&&n=={family.columns}&&k=={family.reduction_length})"
            f"return {symbol}(product,opaque,a,a_bytes,b,b_bytes,c,c_elements);\n"
        )
    dispatch = (
        "static inline int merlin_attention_complete_product_family("
        "merlin_integer_product_family_callback product,void*opaque,"
        "const int8_t*a,size_t a_bytes,const int8_t*b,size_t b_bytes,"
        "int32_t*c,size_t c_elements,int m,int n,int k){\n" + "".join(choices) + " return 0;\n}\n"
    )
    replacement = (
        " const int32_t *planes[MERLIN_RADIX_FUSED_INTEGER_GROUPS];\n"
        f" if(!merlin_attention_complete_product_family(product,opaque,w->ap,(size_t)3*m*k,{rhs},(size_t)3*n*k,"
        "w->readout,sizeof(w->readout)/sizeof(int32_t),m,n,k))return 0;\n"
        " for(int degree=0;degree<MERLIN_RADIX_FUSED_INTEGER_GROUPS;degree++)"
        "planes[degree]=w->readout+(size_t)degree*ROWS*CHUNK;\n"
        " merlin_radix_integer_fused_exact_f64(w->center,planes,(size_t)m*n);\n"
    )
    if "merlin_attention_product product" not in source:
        raise ValueError("explicit source degree-callback ABI missing; refused")
    return (
        "".join(headers)
        + dispatch
        + source.replace(selected, replacement)
        .replace(storage, " int32_t readout[MERLIN_RADIX_FUSED_INTEGER_GROUPS*ROWS*CHUNK];")
        .replace("merlin_attention_product product", "merlin_integer_product_family_callback product")
    )
