"""Independent shared-input components; original unmatched source stays intact.

Only existing exact source outlining and borrowed writer checks authorize each
rewrite. This supplies neither placement nor target effects or execution proof.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SiblingComponent:
    core: object
    writer: object
    receipt: dict


@dataclass(frozen=True)
class SiblingComponents:
    module: object
    components: tuple
    receipt: dict


def prepare_sibling_components(module, *, effects, max_members=8, max_components=256):
    """Select disjoint same-SSA-LHS clusters using current semantic proofs.

    Unsupported/nonzero roots and clusters crossing unknown effects remain
    untouched for ordinary placement. Every member keeps its original host
    observers, including distinct per-column quantizers and external outputs.
    """
    from xdsl.dialects import func

    from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.passes_xdsl import lower_quant_ext
    from merlin.llvmlower.source_region import outline_sibling_contractions
    from merlin.llvmlower.source_region_host import bind_sibling_source_outline
    from merlin.transforms.tensor_views import canonicalize_tensor_views
    from merlin.xdsl_dialects._common import text

    if (
        type(effects) is not BorrowedBufferEffects
        or type(max_members) is not int
        or not 2 <= max_members <= 8
        or type(max_components) is not int
        or max_components < 1
    ):
        raise ValueError("source components need typed effects and finite explicit limits")
    effects.__post_init__()
    module.verify()
    before = text(module, generic=True)
    current = module.clone()
    canonicalize_tensor_views(current)
    lower_quant_ext(current)
    components, attempts = [], []
    while len(components) < max_components:
        functions = [
            op for op in current.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
        ]
        if len(functions) != 1 or len(functions[0].body.blocks) != 1:
            raise ValueError("source components require one original counted forward block")
        runs = {}
        for op in functions[0].body.block.ops:
            if op.name not in {"linalg.generic", "linalg.matmul"} or not op.results:
                continue
            try:
                shape = read_zero_seeded_integer_matmul(op)
            except ValueError:
                continue
            runs.setdefault((op.operands[0], shape[:2]), []).append(op)
        success = False
        for roots in runs.values():
            for start in range(0, len(roots), max_members):
                selected = tuple(roots[start : start + max_members])
                if len(selected) < 2:
                    continue
                symbol = "source_shared_component_" + str(len(components))
                try:
                    outline = outline_sibling_contractions(current, selected, symbol=symbol)
                except ValueError as error:
                    attempts.append({"selected": False, "reason": str(error)})
                    continue
                bound = bind_sibling_source_outline(current, outline, symbol=symbol, effects=effects)
                components.append(SiblingComponent(bound.core, bound.writer, bound.receipt))
                current = bound.module
                success = True
                break
            if success:
                break
        if not success:
            break
    current.verify()
    if text(module, generic=True) != before:
        raise ValueError("source component selection changed original source")
    return SiblingComponents(
        current,
        tuple(components),
        {
            "schema": "merlin.original_sibling_components.v1",
            "components": [c.receipt for c in components],
            "refused_proposals": attempts,
            "unmatched_source_retained": True,
            "all_original_epilogues_and_outputs_preserved": True,
            "effect_or_placement_grant": False,
            "component_limit_reached": len(components) == max_components,
            "max_members": max_members,
            "max_components": max_components,
        },
    )
