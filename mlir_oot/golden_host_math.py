"""Exact CPU math support for the explicitly selected stock RV64GC target."""
from xdsl.dialects.builtin import Float32Type
from xdsl.dialects import func
from xdsl.ir import Region

SYMBOL = 'gemmini_golden_roundevenf'


def rewrite_roundeven(module) -> int:
    """Route scalar f32 roundeven to the target's RNE helper; f64 stays upstream."""
    selected = [op for op in module.walk() if op.name == 'math.roundeven'
                and len(op.results) == 1 and isinstance(op.results[0].type, Float32Type)]
    if not selected:
        return 0
    if any(op.name == 'func.func' and op.sym_name.data == SYMBOL for op in module.walk()):
        raise ValueError('host rounding helper symbol already exists')
    for op in selected:
        replacement = func.CallOp(SYMBOL, list(op.operands), [op.results[0].type])
        replacement.attributes.update(op.attributes)
        op.parent.insert_op_before(replacement, op)
        op.results[0].replace_all_uses_with(replacement.results[0])
        op.parent.erase_op(op)
    ty = Float32Type()
    module.body.block.add_op(func.FuncOp(SYMBOL, ([ty], [ty]), Region(), visibility='private'))
    module.verify()
    return len(selected)
