"""Exact point observations for the owned BF16 row quantizer.

This permission concerns a pure source value operation, not removal of calls to
an arbitrary interposed C function. It enables no approximation or routing.
"""

from dataclasses import dataclass

from merlin.common.paths import data_path


@dataclass(frozen=True)
class FrontierPointCellsContract:
    stable_rne: bool
    nontrapping: bool
    exception_flags_unobserved: bool
    integer_value_only_observation: bool
    pure_quantizer_no_library_effects: bool
    immutable_endpoints: bool
    private_disjoint_outputs: bool

    def validate(self) -> None:
        if any(type(value) is not bool or not value for value in vars(self).values()):
            raise ValueError("complete pure finite-point observation contract required")


def c_header(contract: FrontierPointCellsContract) -> str:
    """Specialize the source-owned row implementation; refuse template drift.

    The ordinary row validates finite BF16 endpoints and the source scale before
    any writes. With the same scale, equal finite endpoints have the same final
    clamped integer observation. Opposite signed zeros also both yield the
    clamped zero value: the source adds +0 before its integer conversion.
    This also covers clamp ranges that exclude zero. Subnormals, source
    epsilon and saturation keep the original scale DAG. Nonpoint endpoints and
    all unstable-scale rows retain the original quantizer checks.
    """
    if not isinstance(contract, FrontierPointCellsContract):
        raise ValueError("typed finite-point observation contract required")
    contract.validate()
    source = data_path("runtime", "c", "bf16_quant_frontier.h").read_text()
    start_marker = "static inline int merlin_frontier_row("
    end_marker = "\n/* Explicit caller-owned coordination"
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError("owned frontier row structure changed; refused")
    row = source[source.index(start_marker) : source.index(end_marker)]
    original = (
        "int ambiguous=stable?merlin_frontier_quant_prepared(low[i],inverse,plan)"
        "!=merlin_frontier_quant_prepared(high[i],inverse,plan):"
    )
    if row.count(original) != 1:
        raise ValueError("owned frontier observation structure changed; refused")
    row = row.replace(start_marker, "static inline int merlin_frontier_row_finite_points(", 1)
    row = row.replace(
        original,
        "int ambiguous=stable?(low[i]==high[i]?0:"
        "merlin_frontier_quant_prepared(low[i],inverse,plan)"
        "!=merlin_frontier_quant_prepared(high[i],inverse,plan)):",
        1,
    )
    return (
        '#include "bf16_quant_frontier.h"\n'
        "/* Explicit pure integer observations; default quantizer unchanged. */\n" + row + "\n"
    )


def prepare_frontier_point_cells(source: str, *, contract: FrontierPointCellsContract) -> str:
    """Select only the complete owned source-attention row observation seam."""
    header = c_header(contract)
    original = "merlin_frontier_row(w->rowlo,w->rowhi,w->rowcandidate,HEADS*DEPTH,quant_plan,w->pending,&scale,&count)"
    if "merlin_frontier_row_finite_points" in source:
        raise ValueError("finite-point observation already selected")
    if source.count(original) != 1:
        raise ValueError("owned source frontier observation seam changed; refused")
    replacement = original.replace("merlin_frontier_row(", "merlin_frontier_row_finite_points(")
    return header + source.replace(original, replacement)
